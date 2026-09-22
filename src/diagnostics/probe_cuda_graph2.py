r"""CUDA Graph 探针 2：把 GPU 时间与 CPU 时间彻底分开。

探针 1 出现矛盾：单个 bnb Linear4bit 进图后 wall 只有 26.7us，
但"7 个 Linear4bit + 2 个 norm"进图后 wall 却是 1136us。必须查清。

方法：把一段计算捕获成图后，replay 的 **wall** 就是纯 GPU 时间（CPU 只发一次）。
用这个尺子量：
  - 单层（7 Linear4bit）的纯 GPU 时间
  - 换成 fp16 nn.Linear 后是多少（判断 bnb 是否在偷偷 dequant 到 fp16 workspace）
  - Linear4bit 的 bs=1 GPU 时间随 out_features 是否线性

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_cuda_graph2.py
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

DEV = "cuda"
DT = torch.float16
HID = 2560


def timed(fn, n=100, warm=20):
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


def main():
    x = torch.randn(1, 1, HID, device=DEV, dtype=DT)

    print("=== A) bnb Linear4bit 单算子：eager cpu/wall vs graph wall ===")
    import bitsandbytes as bnb

    q = bnb.nn.Linear4bit(HID, HID, bias=False, compute_dtype=DT, quant_type="nf4",
                          compress_statistics=True).to(DEV)
    with torch.no_grad():
        for _ in range(5):
            q(x)
        torch.cuda.synchronize()
        c, w = timed(lambda: q(x), n=200, warm=20)
        print(f"  eager : cpu={c:8.2f}us  wall={w:8.2f}us")
        gq = capture(lambda: q(x))
        cg, wg = timed(gq.replay, n=200, warm=50)
        print(f"  graph : cpu={cg:8.2f}us  wall={wg:8.2f}us   <- wall 即纯 GPU 时间")

    print("\n=== B) 同样形状的 fp16 nn.Linear（对照组）===")
    lin = nn.Linear(HID, HID, bias=False).to(DEV, DT)
    with torch.no_grad():
        c, w = timed(lambda: lin(x), n=200, warm=20)
        print(f"  eager : cpu={c:8.2f}us  wall={w:8.2f}us")
        gl = capture(lambda: lin(x))
        cg, wg = timed(gl.replay, n=200, warm=50)
        print(f"  graph : cpu={cg:8.2f}us  wall={wg:8.2f}us")

    print("\n=== C) 逐个线性层数进图，看 wall 如何增长 ===")
    linears = [
        bnb.nn.Linear4bit(HID, HID, bias=False, compute_dtype=DT, quant_type="nf4",
                          compress_statistics=True).to(DEV)
        for _ in range(8)
    ]
    with torch.no_grad():
        for k in (1, 2, 4, 8):
            def run(k=k):
                a = x
                for i in range(k):
                    a = linears[i](a)
                return a

            for _ in range(3):
                run()
            torch.cuda.synchronize()
            c, w = timed(run, n=60, warm=10)
            gg = capture(run)
            cg, wg = timed(gg.replay, n=60, warm=20)
            print(f"  {k:2d} 个 Linear4bit : eager cpu={c:8.2f}us wall={w:8.2f}us | graph cpu={cg:7.2f}us wall={wg:8.2f}us")

    print("\n=== D) bnb 是否在 dequant 到 fp16 workspace？看显存增量 ===")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            q(x)
        torch.cuda.synchronize()
    print(f"  单次 q(x) 峰值显存增量: {torch.cuda.max_memory_allocated()/1024**2:.2f} MiB")
    print(f"    （fp16 权重 2560x2560 = {HID*HID*2/1024**2:.1f} MiB；若峰值接近它，说明确实 dequant 到 fp16）")

    print("\n=== E) 整层（7 线性 + 2 norm）eager 的 cpu 与 wall ===")
    from nova.kernels import fused_rms_norm

    nw = torch.ones(HID, device=DEV, dtype=DT)
    h = x.clone()
    ls = linears[:7]
    with torch.no_grad():
        def layer():
            a = fused_rms_norm(h, nw, 1e-6)
            for lin_ in ls:
                a = lin_(a)
            return h + a

        for _ in range(5):
            layer()
        torch.cuda.synchronize()
        c, w = timed(layer, n=60, warm=10)
        print(f"  eager : cpu={c:8.2f}us  wall={w:8.2f}us")
        glay = capture(layer)
        cg, wg = timed(glay.replay, n=60, warm=20)
        print(f"  graph : cpu={cg:8.2f}us  wall={wg:8.2f}us")


if __name__ == "__main__":
    main()
