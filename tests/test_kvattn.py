r"""int8 融合注意力核 · 验收测试（对应 [reports/kv-int8-fused-attn.md](../reports/kv-int8-fused-attn.md)）。

| 测试 | 判据 |
|------|------|
| `test_ulp_against_dequant_then_sdpa` | **主判据**：与"先还原再算"（`dequantize_int8` → SDPA math）**逐元素 ≤ 2 ULP** |
| `test_ulp_under_channel_outliers` | 同上，但 K 带**通道级离群**（真模型 K 的形态） |
| `test_two_split_settings_agree` | `chunk` 不同（split 数不同 ⇒ 归约顺序不同）两条路都 ≤ 2 ULP |
| `test_gqa_head_mapping` | V 按 KV 头取常数时，第 h 个 Q 头必须回吐出 `h // GQ` 那头的常数（头序不能错） |
| `test_used_ignores_unwritten_slots` | `used < L` 时尾部垃圾**不影响**输出（逐位一致） |
| `test_pack_matches_quantizer` | `pack_int8_kv` 与 `kvquant.quantize_int8(..., "token")` 逐位同源（不许另写量化器） |
| `test_cuda_graph_capture` | 预分配 scratch/out 后能进 CUDA Graph，重放结果与 eager **逐位一致** |
| `test_rejects_bad_inputs` | 形状/分组/幂次不对时报错，不静默算错 |

跑法：`& .\.venv\Scripts\python.exe -m pytest tests -q`

## 判据为什么是两级（修订记录，见 D42 与报告）

原稿只写了"逐元素 ≤ 2 ULP"。真模型 K/V 复核（`probe_kvattn_real.py`）发现：
**输出里有些元素是"大数相消"出来的**（该行 `Σ|p·v| ≈ 11`，而元素本身只有 `0.01`），
此时任何 fp32 归约顺序差异都会被放大成几十个 ULP —— 哪怕核本身没有引入新误差。
所以判据补一条**兜底**：

> 逐元素满足 **≤ 2 ULP**，**或** 绝对偏差 **≤ 1e-5**。

为什么第二条是**绝对**阈值而不是相对阈值：核与参考的差异来自 **fp32 归约顺序**，它的量级由
"归约搬运过的量"决定（本机实测全层最大 **3.9e-6**），**与元素本身多小无关**。
拿小元素的 ULP 去卡一个绝对噪声，等于要求两种求和顺序**逐位相同** —— 做不到，也不代表更准。
1e-5 这个门槛比该层自身的 fp16 舍入噪声（O(0.1) 输出上是 ~1e-4）低一个数量级。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nova.kernels import triton_available
from nova.kvattn import GROUP_T, PAD_M, int8_attn_decode, pack_int8_kv
from nova.kvquant import dequantize_int8, quantize_int8

pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton 不可用")

HQ, H_KV, D = 32, 8, 128
FP16_TINY = torch.finfo(torch.float16).tiny


def _make(used: int, seed: int = 0, outliers: bool = False):
    """造一份 fp16 K/V 与一个 Q。`outliers=True` 时给 K 加通道级离群（真模型第 0 层的形态）。"""
    torch.manual_seed(seed)
    k = torch.randn(1, H_KV, used, D, device="cuda", dtype=torch.float16) * 0.7
    if outliers:
        # 每头挑 3 个通道放大 65x —— 与 D34 记的"第 0 层 K 通道离群 65x"同量级
        scale = torch.ones(D, device="cuda")
        scale[torch.tensor([3, 71, 120])] = 65.0
        k = (k.float() * scale).to(torch.float16)
    v = torch.randn(1, H_KV, used, D, device="cuda", dtype=torch.float16) * 0.7
    q = torch.randn(1, HQ, 1, D, device="cuda", dtype=torch.float16) * 0.7
    return q, k, v


def _reference(q, kq, kmn, kst, vq, vmn, vst, used: int):
    """**先还原再算**：`dequantize_int8` → fp16 → SDPA(math 后端)。就是现状代码路径的数值。"""
    k = dequantize_int8(kq[:, :, :used].contiguous(), kmn, kst, GROUP_T, "token")
    v = dequantize_int8(vq[:, :, :used].contiguous(), vmn, vst, GROUP_T, "token")
    gq = HQ // H_KV
    k = k[:, :, None, :, :].expand(1, H_KV, gq, used, D).reshape(1, HQ, used, D)
    v = v[:, :, None, :, :].expand(1, H_KV, gq, used, D).reshape(1, HQ, used, D)
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)


def _ulp(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """按**真实 fp16 间距**（nextafter）算每个元素的 ULP 偏差。"""
    up = (torch.nextafter(want, torch.full_like(want, float("inf"))).float() - want.float()).abs()
    up = up.clamp_min(FP16_TINY * 2 ** -10)  # 0 的间距 = 最小次正规数
    return (got.float() - want.float()).abs() / up


ABS_TOL = 1e-5  # fp32 归约顺序噪声的绝对上界（实测真数据全层最大 3.9e-6）


def _assert_within_criterion(got, want, tag: str = "") -> dict:
    ulp = _ulp(got, want)
    diff = (got.float() - want.float()).abs()
    bad = (ulp > 2.0) & (diff > ABS_TOL)
    assert int(bad.sum().item()) == 0, (
        f"{tag} 有 {int(bad.sum().item())} 个元素既超 2 ULP 又超 {ABS_TOL:g} 绝对阈值"
        f"（最大 ULP {ulp.max().item():.2f}，最大绝对差 {diff.max().item():.2e}）"
    )
    return {"max_ulp": ulp.max().item(),
            "max_abs": diff.max().item(),
            "exact": (got == want).float().mean().item()}


@pytest.mark.parametrize("used", [7, 63, 64, 65, 511, 512, 513, 1024, 2049, 4096])
def test_ulp_against_dequant_then_sdpa(used: int):
    """主判据：与"先还原再算"逐元素比 ULP。长度覆盖分组边界（63/64/65）与分块边界（511/512/513）。"""
    q, k, v = _make(used)
    packed = pack_int8_kv(k, v)
    want = _reference(q, *packed, used)
    got = int8_attn_decode(q, *packed, used=used)
    st = _assert_within_criterion(got, want, f"used={used}")
    assert st["max_ulp"] <= 2.0, f"used={used} 最大 {st['max_ulp']:.2f} ULP"
    # 参考统计量：逐位相同率（随元素数下降，不作为判据 —— 见模块头与报告的说明）
    assert (got == want).float().mean().item() > 0.97


@pytest.mark.parametrize("used", [300, 2049])
def test_ulp_under_channel_outliers(used: int):
    """真模型的 K 有通道级离群 —— 分组尺子必须扛得住，核的偏差也必须不变。"""
    q, k, v = _make(used, seed=1, outliers=True)
    packed = pack_int8_kv(k, v)
    want = _reference(q, *packed, used)
    got = int8_attn_decode(q, *packed, used=used)
    _assert_within_criterion(got, want, f"离群 used={used}")


@pytest.mark.parametrize("chunk", [64, 512, 4096])
def test_two_split_settings_agree(chunk: int):
    """split 数改变归约顺序 ⇒ 数值会动，但必须都在判据内。"""
    used = 1024
    q, k, v = _make(used, seed=2)
    packed = pack_int8_kv(k, v)
    want = _reference(q, *packed, used)
    got = int8_attn_decode(q, *packed, used=used, chunk=chunk)
    _assert_within_criterion(got, want, f"chunk={chunk}")


def test_gqa_head_mapping():
    """V 每个 KV 头取常数：注意力权重归一化后，Q 头 h 的输出必须就是第 `h // GQ` 头的常数。"""
    used = 256
    torch.manual_seed(3)
    k = torch.randn(1, H_KV, used, D, device="cuda", dtype=torch.float16)
    v = torch.arange(H_KV, device="cuda", dtype=torch.float16).view(1, H_KV, 1, 1).expand(1, H_KV, used, D).contiguous()
    q = torch.randn(1, HQ, 1, D, device="cuda", dtype=torch.float16)
    got = int8_attn_decode(q, *pack_int8_kv(k, v), used=used)
    want = torch.arange(HQ, device="cuda", dtype=torch.float16) // (HQ // H_KV)
    assert torch.equal(got[0, :, 0, 0], want), (got[0, :, 0, 0].tolist(), want.tolist())


def test_used_ignores_unwritten_slots():
    """`used` 之后的槽位是垃圾也必须看不见（尾部用大数填充，输出要逐位一致）。"""
    used, capacity = 300, 640
    q, k, v = _make(used, seed=4)
    kq, kmn, kst, vq, vmn, vst = pack_int8_kv(k, v)
    # 把 [used, capacity) 填成垃圾（尺子也一起填）
    big_k = torch.cat([kq, torch.full((1, H_KV, capacity - used, D), 255, dtype=torch.uint8, device="cuda")], dim=2)
    big_v = torch.cat([vq, torch.full((1, H_KV, capacity - used, D), 255, dtype=torch.uint8, device="cuda")], dim=2)
    # ⚠️ 尺子的第二维是**组数** ceil(capacity/64)，不是 token 数
    pad_g = -(-capacity // GROUP_T) - kmn.shape[2]
    big_kmn = torch.cat([kmn, torch.full((1, H_KV, pad_g, D), 300.0, dtype=torch.float16, device="cuda")], dim=2)
    big_kst = torch.cat([kst, torch.full((1, H_KV, pad_g, D), 9.0, dtype=torch.float16, device="cuda")], dim=2)
    big_vmn = torch.cat([vmn, torch.full((1, H_KV, pad_g, D), 300.0, dtype=torch.float16, device="cuda")], dim=2)
    big_vst = torch.cat([vst, torch.full((1, H_KV, pad_g, D), 9.0, dtype=torch.float16, device="cuda")], dim=2)

    want = int8_attn_decode(q, kq, kmn, kst, vq, vmn, vst, used=used)
    got = int8_attn_decode(q, big_k, big_kmn, big_kst, big_v, big_vmn, big_vst, used=used)
    assert torch.equal(got, want)


def test_zero_used_returns_zeros():
    q, k, v = _make(64)
    out = int8_attn_decode(q, *pack_int8_kv(k, v), used=0)
    assert torch.equal(out, torch.zeros_like(out))


def test_pack_matches_quantizer():
    """融合核的输入必须与 E3/E5 量过精度的那个量化器**逐位同源**，不许另写一份。"""
    _, k, v = _make(200, seed=5)
    kq, kmn, kst, vq, vmn, vst = pack_int8_kv(k, v)
    for got, want in zip((kq, kmn, kst), quantize_int8(k, GROUP_T, "token")):
        assert torch.equal(got, want.contiguous())
    for got, want in zip((vq, vmn, vst), quantize_int8(v, GROUP_T, "token")):
        assert torch.equal(got, want.contiguous())


def test_cuda_graph_capture():
    """解码路径是 CUDA Graph（D29/D35）—— 核必须能进图，且重放结果与 eager 逐位一致。"""
    used, chunk = 1024, 512
    q, k, v = _make(used, seed=6)
    packed = pack_int8_kv(k, v)
    want = int8_attn_decode(q, *packed, used=used, chunk=chunk)

    n_splits = -(-used // chunk)
    pm = torch.empty((H_KV, n_splits, PAD_M), dtype=torch.float32, device="cuda")
    pl = torch.empty_like(pm)
    pa = torch.empty((H_KV, n_splits, PAD_M, D), dtype=torch.float32, device="cuda")
    out = torch.empty((1, HQ, 1, D), dtype=torch.float16, device="cuda")
    args = dict(used=used, chunk=chunk, scratch=(pm, pl, pa), out=out)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):  # 预热：Triton 编译不能发生在捕获期
            int8_attn_decode(q, *packed, **args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        int8_attn_decode(q, *packed, **args)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)


def test_rejects_bad_inputs():
    q, k, v = _make(64)
    packed = pack_int8_kv(k, v)
    with pytest.raises(ValueError, match="解码一步"):
        int8_attn_decode(q.expand(1, HQ, 2, D).contiguous(), *packed, used=64)
    with pytest.raises(ValueError, match="chunk"):
        int8_attn_decode(q, *packed, used=64, chunk=100)
    with pytest.raises(ValueError, match="group"):
        int8_attn_decode(q, *packed, used=64, group=100)
    with pytest.raises(ValueError, match="used"):
        int8_attn_decode(q, *packed, used=65)
    with pytest.raises(ValueError, match="尺子"):
        int8_attn_decode(q, *packed, used=64, group=32)
