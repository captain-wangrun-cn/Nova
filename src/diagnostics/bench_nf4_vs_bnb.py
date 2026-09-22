r"""NF4 GEMV 最终结论用的一组测量：
  1. 本机 DRAM 读带宽（决定"4-bit GEMV 最多能省多少"）
  2. 7 个真实投影形状：bnb Linear4bit vs 自写 NF4Linear（同一 harness、同一权重）
  3. 每层 / 每 token 的线性层合计

用法：& .\.venv\Scripts\python.exe src\diagnostics\bench_nf4_vs_bnb.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
import torch


def gt(fn, iters=100, reps=7, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e6)
    return sorted(ts)[len(ts) // 2]      # 中位数


# Qwen3-VL-4B 文本塔：每层 7 个投影
LAYER = [("q_proj", 4096, 2560, 1), ("k_proj", 1024, 2560, 1), ("v_proj", 1024, 2560, 1),
         ("o_proj", 2560, 4096, 1), ("gate_proj", 9728, 2560, 1), ("up_proj", 9728, 2560, 1),
         ("down_proj", 2560, 9728, 1)]


def dram_bandwidth():
    """用大张量归约量本机 DRAM 读带宽（远超 L2，32MB）。"""
    n = 512 * 1024 * 1024 // 4          # 512MB fp32
    x = torch.empty(n, device="cuda", dtype=torch.float32).normal_()
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    fn = lambda: torch.sum(x, dim=0, keepdim=True, out=out)
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    del x
    torch.cuda.empty_cache()
    return n * 4 / dt / 1e9, dt * 1e3


def main():
    from diagnostics.bench_nf4_gemv import make_bnb_linear
    from nova.kernels import nf4_launch_config
    from nova.quant import NF4Linear

    bw, ms = dram_bandwidth()
    print(f"DRAM 读带宽（512MB 归约）: {bw:.0f} GB/s  ({ms:.2f} ms)\n")
    print(f"NF4 launch config: {nf4_launch_config()}\n")

    print(f"{'shape':<11s} {'N':>6s} {'K':>6s} {'4bit MB':>8s} {'bnb us':>8s} {'nf4 us':>8s} "
          f"{'nf4/bnb':>8s} {'bnb GB/s':>9s} {'nf4 GB/s':>9s}")
    tot_bnb = tot_nf4 = tot_mb = 0.0
    for tag, N, K, cnt in LAYER:
        lin = make_bnb_linear(N, K)
        q = NF4Linear.from_bnb(lin)
        x = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.5
        with torch.inference_mode():
            t_bnb = gt(lambda: lin(x))
            t_nf4 = gt(lambda: q(x))
        mb = (N * K / 2 + N * (K / 64) * 4) / 1e6
        tot_bnb += t_bnb * cnt; tot_nf4 += t_nf4 * cnt; tot_mb += mb * cnt
        print(f"{tag:<11s} {N:>6d} {K:>6d} {mb:8.2f} {t_bnb:8.1f} {t_nf4:8.1f} "
              f"{t_nf4/t_bnb:7.2f}x {mb*1e6/(t_bnb*1e-6)/1e9:9.0f} {mb*1e6/(t_nf4*1e-6)/1e9:9.0f}")
        del lin, q
        torch.cuda.empty_cache()

    print(f"\n每层合计: 4bit 权重 {tot_mb:.1f} MB | bnb {tot_bnb:.0f} us | nf4 {tot_nf4:.0f} us")
    print(f"36 层合计: bnb {tot_bnb*36/1000:.2f} ms | nf4 {tot_nf4*36/1000:.2f} ms")
    print(f"DRAM 下界（只读 4-bit，{tot_mb*36/1000:.2f} GB @ {bw:.0f} GB/s）: {tot_mb*36/1e3/bw*1000:.2f} ms")
    print(f"bnb 距下界 {tot_bnb*36/1000/(tot_mb*36/1e3/bw*1000):.2f}x ; nf4 距下界 {tot_nf4*36/1000/(tot_mb*36/1e3/bw*1000):.2f}x")


if __name__ == "__main__":
    main()