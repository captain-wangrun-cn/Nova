r"""CUDA Graph 可行性探针。

背景：`microbench_op_cpu.py` 实测本机**单次 kernel 启动约 15us CPU**（连 `x*2` 都要 16us）。
Nova 单通路每 token 有 ~6500 次启动 -> 光启动就 62ms。若能把整步捕获成一张 CUDA Graph，
每步只剩 1 次 replay，启动开销从 ~62ms 掉到 ~0.02ms。

本脚本回答三个问题：
  1. 启动开销是否随算子数线性增长（确认它就是瓶颈）
  2. 纯 aten 算子能否被捕获 / replay 多快
  3. **bnb Linear4bit 与 Triton kernel 能否被捕获**（这是能不能走这条路的前提）

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_cuda_graph.py
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


def timed(fn, n=100, warm=10):
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


def main():
    x = torch.randn(1, 1, HID, device=DEV, dtype=DT)

    print("=== 1) 启动开销是否线性 ===")
    for k in (1, 4, 16, 64):
        cpu, wall = timed(lambda k=k: [x.mul(2.0) for _ in range(k)])
        print(f"  {k:3d} 个 mul  cpu={cpu:8.2f}us  -> 每次 {cpu/k:6.2f}us")

    print("\n=== 2) 纯 aten 算子：捕获 vs 不捕获 ===")
    N = 256
    cpu_eager, _ = timed(lambda: [x.mul(2.0) for _ in range(N)], n=20, warm=3)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(N):
                x.mul(2.0)
    torch.cuda.current_stream().wait_stream(s)
    try:
        with torch.cuda.graph(g):
            for _ in range(N):
                x.mul(2.0)
        cpu_graph, wall_graph = timed(g.replay, n=100, warm=20)
        print(f"  eager  {N} ops : cpu={cpu_eager:8.2f}us")
        print(f"  graph  {N} ops : cpu={cpu_graph:8.2f}us  wall={wall_graph:8.2f}us   -> 提速 {cpu_eager/max(cpu_graph,1e-9):6.1f}x")
    except Exception as exc:  # noqa: BLE001
        print(f"  捕获失败: {type(exc).__name__}: {exc}")

    print("\n=== 3) bnb Linear4bit 能否进图 ===")
    try:
        import bitsandbytes as bnb

        q = bnb.nn.Linear4bit(HID, HID, bias=False, compute_dtype=DT, quant_type="nf4",
                              compress_statistics=True).to(DEV)
        with torch.no_grad():
            for _ in range(5):
                q(x)
            torch.cuda.synchronize()
        out = torch.empty_like(x)
        cpu_eager, _ = timed(lambda: out.copy_(q(x)), n=100, warm=10)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out.copy_(q(x))
        torch.cuda.current_stream().wait_stream(s)
        gq = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gq):
            out.copy_(q(x))
        cpu_g, wall_g = timed(gq.replay, n=100, warm=20)
        print(f"  bnb Linear4bit eager : cpu={cpu_eager:8.2f}us")
        print(f"  bnb Linear4bit graph : cpu={cpu_g:8.2f}us wall={wall_g:8.2f}us")
        gq.replay()
        torch.cuda.synchronize()
        ref = q(x)
        print(f"  replay 数值一致性: max|diff| = {(out - ref).abs().max().item():.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"  bnb 进图失败: {type(exc).__name__}: {exc}")

    print("\n=== 4) Triton RMSNorm 能否进图 ===")
    try:
        from nova.kernels import fused_rms_norm

        w = torch.ones(HID, device=DEV, dtype=DT)
        y = fused_rms_norm(x, w, 1e-6)
        out2 = torch.empty_like(x)
        cpu_eager, _ = timed(lambda: out2.copy_(fused_rms_norm(x, w, 1e-6)), n=100, warm=10)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out2.copy_(fused_rms_norm(x, w, 1e-6))
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out2.copy_(fused_rms_norm(x, w, 1e-6))
        cpu_g, wall_g = timed(gr.replay, n=100, warm=20)
        print(f"  triton rms eager : cpu={cpu_eager:8.2f}us")
        print(f"  triton rms graph : cpu={cpu_g:8.2f}us wall={wall_g:8.2f}us")
        gr.replay()
        torch.cuda.synchronize()
        print(f"  replay 数值一致性: max|diff| = {(out2 - y).abs().max().item():.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"  triton 进图失败: {type(exc).__name__}: {exc}")

    print("\n=== 5) 模拟一层解码：7 个 Linear4bit + 2 个 norm ===")
    try:
        import bitsandbytes as bnb
        from nova.kernels import fused_rms_norm

        linears = [
            bnb.nn.Linear4bit(HID, HID, bias=False, compute_dtype=DT, quant_type="nf4",
                              compress_statistics=True).to(DEV)
            for _ in range(7)
        ]
        nw = torch.ones(HID, device=DEV, dtype=DT)
        h = x.clone()

        def layer(h):
            a = fused_rms_norm(h, nw, 1e-6)
            for lin in linears:
                a = lin(a)
            return h + a

        with torch.no_grad():
            for _ in range(5):
                layer(h)
            torch.cuda.synchronize()
        cpu_eager, _ = timed(lambda: layer(h), n=100, warm=10)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                layer(h)
        torch.cuda.current_stream().wait_stream(s)
        gl = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gl):
            layer(h)
        cpu_g, wall_g = timed(gl.replay, n=100, warm=20)
        print(f"  eager : cpu={cpu_eager:8.2f}us/token")
        print(f"  graph : cpu={cpu_g:8.2f}us/token wall={wall_g:8.2f}us/token")
        print(f"  -> 36 层外推: eager {cpu_eager*36/1000:.1f} ms/token, graph {wall_g*36/1000:.1f} ms/token")
    except Exception as exc:  # noqa: BLE001
        print(f"  模拟层失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
