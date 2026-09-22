"""S4 · Triton NF4 GEMV 的验收测试。

对应 [reports/s4-nf4-gemv.md](../reports/s4-nf4-gemv.md)：

| 测试 | 判据 |
|------|------|
| `test_extract_matches_bnb_dequant` | 自写解包与 `bnb.functional.dequantize_4bit` **逐位一致** |
| `test_nf4_linear_matches_bnb_decode` | M=1 时 `NF4Linear` vs `Linear4bit` 的差 < 10 ULP(fp16) |
| `test_nf4_linear_matches_bnb_prefill` | M=7 时同上（prefill 路径） |
| `test_convert_to_nf4_replaces_modules` | `convert_to_nf4` 只换自己那棵树上的 `Linear4bit` |

跑法：`& .\\.venv\\Scripts\\python.exe -m pytest tests -q`
"""

from __future__ import annotations

import pytest
import torch

from nova.kernels import triton_available
from nova.quant import NF4Linear, convert_to_nf4, extract_nf4

pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton 不可用")

SHAPES = [(256, 512), (512, 256), (1024, 64), (64, 1024)]
FP16_ULP = 2.0 ** -10


def _make_bnb(n: int, k: int, seed: int = 0):
    import bitsandbytes as bnb

    torch.manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.05
    lin = bnb.nn.Linear4bit(k, n, bias=True, compute_dtype=torch.float16,
                            quant_type="nf4", compress_statistics=True).to("cuda")
    with torch.no_grad():
        lin.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                       compress_statistics=True, quant_storage=torch.float16)
        lin.weight = lin.weight.to("cuda")
        lin.bias.data.normal_(0, 0.01)
        _ = lin(torch.zeros(1, 1, k, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
    return lin


def _torch_dequant(packed: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, n: int, k: int):
    """用纯 torch 按同样的布局复算一遍反量化（布局正确性由它与 bnb 对齐来保证）。"""
    idx = torch.arange(k, device="cuda")
    hi = (packed.long() >> 4)                    # [N, K//2] -> 偶数元素
    lo = (packed.long() & 0x0F)                  # [N, K//2] -> 奇数元素
    vals = torch.empty(n, k, device="cuda", dtype=torch.float32)
    vals[:, 0::2] = code[hi]
    vals[:, 1::2] = code[lo]
    del idx
    # absmax 是 [K//64, N]，转回 [N, K//64] 再按 64 展开
    am = absmax.t().repeat_interleave(64, dim=1)
    return (vals * am).to(torch.float16)


@pytest.mark.parametrize("n,k", SHAPES)
def test_extract_matches_bnb_dequant(n, k):
    """自写解包必须与 bnb 的 dequantize_4bit 逐位一致。"""
    import bitsandbytes as bnb

    lin = _make_bnb(n, k)
    qs = lin.weight.quant_state
    packed, absmax, code, (nn_, kk) = extract_nf4(lin.weight.data, qs)
    assert (nn_, kk) == (n, k)
    assert packed.shape == (n, k // 2) and absmax.shape == (k // 64, n)

    ref = bnb.functional.dequantize_4bit(lin.weight.data, qs).reshape(n, k)
    mine = _torch_dequant(packed, absmax, code, n, k)
    assert torch.equal(mine, ref), f"max|diff| = {(mine.float() - ref.float()).abs().max().item():.3e}"


@pytest.mark.parametrize("n,k", SHAPES)
def test_nf4_linear_matches_bnb_decode(n, k):
    """M=1：自写 kernel 与 bnb 的差应在 fp16 舍入级。"""
    lin = _make_bnb(n, k)
    q = NF4Linear.from_bnb(lin)
    x = torch.randn(1, k, device="cuda", dtype=torch.float16) * 0.5
    with torch.inference_mode():
        ref, got = lin(x), q(x)
    assert got.shape == ref.shape
    scale = ref.abs().max().item()
    diff = (got.float() - ref.float()).abs().max().item()
    assert diff <= 10 * FP16_ULP * scale, f"max|diff|={diff:.3e} vs 允许 {10 * FP16_ULP * scale:.3e}"


@pytest.mark.parametrize("n,k", [(512, 256), (256, 512)])
def test_nf4_linear_matches_bnb_prefill(n, k):
    """M>1（prefill 路径）也必须对。"""
    lin = _make_bnb(n, k)
    q = NF4Linear.from_bnb(lin)
    x = torch.randn(7, k, device="cuda", dtype=torch.float16) * 0.5
    with torch.inference_mode():
        ref, got = lin(x), q(x)
    scale = ref.abs().max().item()
    diff = (got.float() - ref.float()).abs().max().item()
    assert diff <= 10 * FP16_ULP * scale, f"max|diff|={diff:.3e} vs 允许 {10 * FP16_ULP * scale:.3e}"


def test_convert_to_nf4_replaces_modules():
    """`convert_to_nf4` 只替换传入的那棵模块树，不动原对象。"""
    import torch.nn as nn

    import bitsandbytes as bnb

    lin = _make_bnb(64, 128)
    holder = nn.Sequential(nn.Identity(), lin)   # holder[1] 是 Linear4bit
    other = nn.Sequential(lin)                   # 另一棵树共用同一个对象

    n_done = convert_to_nf4(holder)
    assert n_done == 1
    assert isinstance(holder[1], NF4Linear)
    assert isinstance(other[0], bnb.nn.Linear4bit), "不能动到别的模块树"

    x = torch.randn(1, 128, device="cuda", dtype=torch.float16) * 0.5
    with torch.inference_mode():
        a, b = holder[1](x), other[0](x)
    scale = b.abs().max().item()
    assert (a.float() - b.float()).abs().max().item() <= 10 * FP16_ULP * scale