r"""CUDA Graph 探针 3：隔离"Triton kernel 进图后变慢"这个矛盾。

探针 2 的 E 组：7 个 Linear4bit（图内 GPU 154us）+ 1 个 Triton RMSNorm，
wall 却变成 1184us。本脚本把三块拆开单独进图，定位是谁。

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_cuda_graph3.py
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

DEV = "cuda"
DT = torch.float16
HID = 2560


def timed(fn, n=60, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    cpu = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n * 1e6
    return cpu, wall


def capture(fn, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def show(name, fn):
    c, w = timed(fn, n=60, warm=10)
    print(f"  {name:34s} eager cpu={c:9.2f}us wall={w:9.2f}us", end="")
    try:
        g = capture(fn)
        cg, wg = timed(g.replay, n=60, warm=20)
        print(f" | graph cpu={cg:8.2f}us wall={wg:9.2f}us")
    except Exception as exc:  # noqa: BLE001
        print(f" | 捕获失败 {type(exc).__name__}: {exc}")


def main():
    import bitsandbytes as bnb
    from nova.kernels import fused_rms_norm

    x = torch.randn(1, 1, HID, device=DEV, dtype=DT)
    nw = torch.ones(HID, device=DEV, dtype=DT)
    linears = [
        bnb.nn.Linear4bit(HID, HID, bias=False, compute_dtype=DT, quant_type="nf4",
                          compress_statistics=True).to(DEV)
        for _ in range(7)
    ]
    with torch.no_grad():
        for _ in range(5):
            linears[0](x)
        torch.cuda.synchronize()

    print("=== 拆开看 ===")
    with torch.no_grad():
        show("1) Triton RMSNorm only", lambda: fused_rms_norm(x, nw, 1e-6))
        show("2) x + 1.0 (纯 aten)", lambda: x + 1.0)

        def seven():
            a = x
            for lin in linears:
                a = lin(a)
            return a

        show("3) 7x Linear4bit only", seven)

        def rms_then_seven():
            a = fused_rms_norm(x, nw, 1e-6)
            for lin in linears:
                a = lin(a)
            return x + a

        show("4) RMSNorm + 7x Linear4bit", rms_then_seven)

        def two_rms():
            a = fused_rms_norm(x, nw, 1e-6)
            return fused_rms_norm(a, nw, 1e-6)

        show("5) 2x Triton RMSNorm", two_rms)

    print("\n=== Triton launch 是否同步？===")
    with torch.no_grad():
        for _ in range(20):
            fused_rms_norm(x, nw, 1e-6)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            fused_rms_norm(x, nw, 1e-6)
        t_launch = (time.perf_counter() - t0) / 200 * 1e6
        torch.cuda.synchronize()
        t_wall = (time.perf_counter() - t0) / 200 * 1e6
        print(f"  200x triton rms: launch={t_launch:.2f}us wall={t_wall:.2f}us")
        big = torch.randn(1, 1, HID, device=DEV, dtype=DT)
        for _ in range(5):
            fused_rms_norm(big, nw, 1e-6)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            fused_rms_norm(big, nw, 1e-6)
        t_launch = (time.perf_counter() - t0) / 200 * 1e6
        torch.cuda.synchronize()
        print(f"  launch 时间是否随 GPU 队列增长: {t_launch:.2f}us")


if __name__ == "__main__":
    main()
