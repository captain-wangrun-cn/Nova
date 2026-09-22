r"""lm_head 形状（151936 x 2560）上 NF4 kernel vs fp16 的对比。
lm_head 列数极多（2374 个 program @ BLOCK_N=64），occupancy 好，是 gather 延迟最容易被藏住的形状。
用法：& .\.venv\Scripts\python.exe src\diagnostics\bench_nf4_lmhead.py
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


def gt(fn, iters=40, reps=5, warm=2):
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
    return sorted(ts)[len(ts) // 2]


def main():
    import bitsandbytes as bnb
    from nova.kernels import set_nf4_launch_config
    from nova.quant import NF4Linear

    N, K = 151936, 2560
    print(f"lm_head 形状: N={N} K={K}")
    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.02)
    lin = bnb.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.float16,
                            quant_type="nf4", compress_statistics=True).to("cuda")
    with torch.no_grad():
        lin.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                       compress_statistics=True, quant_storage=torch.float16)
        lin.weight = lin.weight.to("cuda")
        _ = lin(torch.zeros(1, 1, K, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
    del w
    torch.cuda.empty_cache()

    fp = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
    x = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.5

    with torch.inference_mode():
        t_bnb = gt(lambda: lin(x))
        t_fp = gt(lambda: fp(x))
        print(f"  fp16 nn.Linear : {t_fp:8.1f} us   ({N*K*2/1e6/(t_fp*1e-6)/1e9:5.0f} GB/s, {N*K*2/1e6:.0f} MB)")
        print(f"  bnb Linear4bit : {t_bnb:8.1f} us")
        q = NF4Linear.from_bnb(lin)
        del lin
        torch.cuda.empty_cache()
        best = None
        for bn in (32, 64, 128, 256):
            for nw in (2, 4, 8):
                set_nf4_launch_config(block_n=bn, block_m=1, num_warps=nw, num_stages=2)
                try:
                    t = gt(lambda: q(x), iters=30, reps=3)
                except Exception:
                    continue
                mb = (N*K/2 + N*(K/64)*4) / 1e6
                print(f"  nf4 bn={bn:<3d} nw={nw} : {t:8.1f} us   ({mb/(t*1e-6)/1e9:5.0f} GB/s, {mb:.0f} MB)  vs fp16 {t_fp/t:5.2f}x")
                if best is None or t < best[0]:
                    best = (t, bn, nw)
        print(f"  最佳: {best[0]:.1f} us (bn={best[1]} nw={best[2]}) -> 相对 fp16 加速 {t_fp/best[0]:.2f}x, "
              f"省 {(t_fp-best[0])/1000:.2f} ms/token")


if __name__ == "__main__":
    main()
