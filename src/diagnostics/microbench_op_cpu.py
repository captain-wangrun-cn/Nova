r"""单算子 CPU 派发开销微基准：Nova 每 token 的 62ms 到底花在哪。

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\microbench_op_cpu.py

方法：**不 sync** 连续调用 N 次。测到的 `cpu` 就是"CPU 把算子塞进队列"的时间；
如果 `cpu` 接近 `wall`，该算子就是 CPU 受限。见 [reports/s2-speed-diagnosis.md](../../reports/s2-speed-diagnosis.md) 第八节。
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
DT = torch.float16
HID = 2560
INT = 9728


def cpu_bound_us(fn, n=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t_cpu = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    t_wall = (time.perf_counter() - t0) / n * 1e6
    return t_cpu, t_wall


def report(name, fn, n=200):
    c, w = cpu_bound_us(fn, n=n)
    print(f"  {name:40s} cpu={c:8.2f}us  wall={w:8.2f}us")
    return c


def main():
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")

    x1 = torch.randn(1, 1, HID, device=DEV, dtype=DT)
    x8 = torch.randn(1, 8, HID, device=DEV, dtype=DT)

    print("\n--- 基线：一个'什么都不做'的算子要多少 CPU ---")
    report("x1.view(1,1,2560)", lambda: x1.view(1, 1, HID))
    report("x1.to(fp16)  [no-op]", lambda: x1.to(DT))
    report("x1 * 2.0", lambda: x1 * 2.0)
    report("x1.contiguous()", lambda: x1.contiguous())

    print("\n--- fp16 nn.Linear (2560x2560) ---")
    lin = nn.Linear(HID, HID, bias=False).to(DEV, DT)
    with torch.no_grad():
        report("nn.Linear fp16  M=1", lambda: lin(x1))
        report("nn.Linear fp16  M=8", lambda: lin(x8))

    print("\n--- bnb Linear4bit NF4 (2560x2560) ---")
    try:
        import bitsandbytes as bnb

        q = bnb.nn.Linear4bit(
            HID, HID, bias=False, compute_dtype=DT, quant_type="nf4", compress_statistics=True
        ).to(DEV)
        with torch.no_grad():
            for _ in range(5):
                q(x1)
            torch.cuda.synchronize()
        report("bnb Linear4bit  M=1", lambda: q(x1))
        report("bnb Linear4bit  M=8", lambda: q(x8))
        qs = q.weight.quant_state
        print(f"    quant_state: shape={tuple(q.weight.shape)} blocksize={qs.blocksize} "
              f"dtype={qs.dtype} nested={qs.nested} absmax={tuple(qs.absmax.shape)} "
              f"absmax.dtype={qs.absmax.dtype} code={tuple(qs.code.shape) if qs.code is not None else None}")
        if qs.nested:
            print(f"    state2: absmax={tuple(qs.state2.absmax.shape)} dtype={qs.state2.absmax.dtype} "
                  f"blocksize={qs.state2.blocksize}")
    except Exception as exc:  # noqa: BLE001
        print(f"  bnb 不可用: {type(exc).__name__}: {exc}")

    print("\n--- 归一化 ---")
    w = torch.ones(HID, device=DEV, dtype=DT)
    report("eager rms_norm (手写)", lambda: _eager_rms(x1, w))
    report("F.rms_norm", lambda: F.rms_norm(x1, (HID,), w, 1e-6))
    try:
        from nova.kernels import fused_rms_norm

        with torch.no_grad():
            report("triton fused_rms_norm", lambda: fused_rms_norm(x1, w, 1e-6))
    except Exception as exc:  # noqa: BLE001
        print(f"  triton 不可用: {type(exc).__name__}: {exc}")

    print("\n--- 注意力 / 其它 ---")
    qq = torch.randn(1, 32, 1, 128, device=DEV, dtype=DT)
    kk = torch.randn(1, 8, 1, 128, device=DEV, dtype=DT)
    vv = torch.randn(1, 8, 1, 128, device=DEV, dtype=DT)
    report("SDPA M=1 (gqa)", lambda: F.scaled_dot_product_attention(qq, kk, vv, is_causal=False, enable_gqa=True))
    report("x1 @ x1.T (matmul)", lambda: x1 @ x1.transpose(-1, -2))
    report("torch.cat([x1,x1],-1)", lambda: torch.cat([x1, x1], dim=-1))
    report("torch.stack([x1,x1]).mean(0)", lambda: torch.stack([x1, x1], dim=0).mean(dim=0))


def _eager_rms(h, weight, eps=1e-6):
    h32 = h.to(torch.float32)
    var = h32.pow(2).mean(-1, keepdim=True)
    h32 = h32 * torch.rsqrt(var + eps)
    return weight * h32.to(h.dtype)


if __name__ == "__main__":
    main()
