r"""探针：把 bnb 的 NF4 存储格式（含 double quant）**逐位复现**。

自写 GEMV 的前提是能自己算出 `dequantize_4bit` 的结果。本脚本先用纯 torch
实现一份参考，再与 `bnb.functional.dequantize_4bit` 对照。

要确认的细节：
  1. `absmax` 是 uint8（double quant）时怎么还原成 fp32
  2. 一个字节里**低半字节**对应第 2i 个元素还是第 2i+1 个
  3. `quant_state.offset` 加在哪一步

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_nf4_format.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])

import torch

N, K = 256, 512
BLOCK = 64


def unpack_nibbles(packed: torch.Tensor, order: str) -> torch.Tensor:
    """packed: [nbytes] uint8 -> [2*nbytes] long（索引到 codebook）。"""
    lo = (packed & 0x0F).long()
    hi = ((packed >> 4) & 0x0F).long()
    if order == "lo_first":
        return torch.stack([lo, hi], dim=1).reshape(-1)
    return torch.stack([hi, lo], dim=1).reshape(-1)


def my_dequant(packed: torch.Tensor, qs, order: str = "lo_first") -> torch.Tensor:
    n, k = qs.shape
    nblocks = packed.numel() * 2 // qs.blocksize

    if qs.nested:
        s2 = qs.state2
        qa = qs.absmax.long().reshape(-1)
        nsuper = (qa.numel() + s2.blocksize - 1) // s2.blocksize
        a2 = s2.absmax.reshape(-1)[:nsuper]
        a2 = a2.repeat_interleave(s2.blocksize)[: qa.numel()]
        absmax = s2.code.reshape(-1)[qa] * a2
        absmax = absmax + qs.offset
    else:
        absmax = qs.absmax.reshape(-1).float()

    assert absmax.numel() == nblocks, (absmax.numel(), nblocks)

    idx = unpack_nibbles(packed.reshape(-1), order)
    code = qs.code.reshape(-1)
    vals = code[idx] * absmax.repeat_interleave(qs.blocksize)
    return vals.reshape(n, k).to(torch.float16)


def main() -> None:
    import bitsandbytes as bnb

    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.7)

    lin = bnb.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.float16,
                            quant_type="nf4", compress_statistics=True).to("cuda")
    with torch.no_grad():
        lin.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                       compress_statistics=True, quant_storage=torch.float16)
        lin.weight = lin.weight.to("cuda")
        _ = lin(torch.zeros(1, 1, K, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()

    qs = lin.weight.quant_state
    packed = lin.weight.data.reshape(-1).view(torch.uint8)
    print("packed:", tuple(packed.shape), packed.dtype, " 元素数:", packed.numel())
    print("qs.shape:", tuple(qs.shape), "blocksize:", qs.blocksize, "nested:", qs.nested,
          "dtype:", qs.dtype, "offset:", getattr(qs, "offset", None))
    print("qs.code:", qs.code.tolist())
    print("absmax:", tuple(qs.absmax.shape), qs.absmax.dtype, "min/max:",
          qs.absmax.min().item(), qs.absmax.max().item())
    if qs.nested:
        s2 = qs.state2
        print("state2: absmax", tuple(s2.absmax.shape), s2.absmax.dtype,
              "blocksize", s2.blocksize, "code", s2.code.tolist(), "dtype", s2.dtype)

    ref = bnb.functional.dequantize_4bit(lin.weight.data, qs).reshape(N, K)

    for order in ("lo_first", "hi_first"):
        mine = my_dequant(packed, qs, order)
        d = (mine.float() - ref.float()).abs()
        print(f"\norder={order:9s} max|diff|={d.max().item():.3e}  "
              f"mean|diff|={d.mean().item():.3e}  ref|mean|={ref.abs().mean().item():.3e}")

    # 也确认一下偏置项
    print("\n--- 无 offset 时的差异（确认 offset 加在哪）---")
    saved = qs.offset
    try:
        qs.offset = 0.0
        print("offset=0 时的 bnb 结果 vs 带 offset 的 bnb 结果:",
              (bnb.functional.dequantize_4bit(lin.weight.data, qs).reshape(N, K).float() - ref.float()).abs().max().item())
    finally:
        qs.offset = saved


if __name__ == "__main__":
    main()
