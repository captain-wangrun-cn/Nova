r"""NF4 kernel 调参实验：只跑一个形状，快速试 launch 参数。

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\exp_nf4_launch.py
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


def graph_time(fn, iters=60, reps=3, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / iters * 1e6


def main():
    from nova.kernels import set_nf4_launch_config, nf4_launch_config
    from nova.quant import NF4Linear
    from diagnostics.bench_nf4_gemv import make_bnb_linear

    shapes = [("2560x2560", 2560, 2560), ("9728x2560", 9728, 2560)]
    prepared = []
    for tag, n, k in shapes:
        lin = make_bnb_linear(n, k)
        q = NF4Linear.from_bnb(lin)
        xi = torch.randn(1, k, device="cuda", dtype=torch.float16) * 0.5
        with torch.inference_mode():
            t_bnb = graph_time(lambda: lin(xi))
        prepared.append((tag, n, k, q, xi, t_bnb))
        print(f"{tag}: bnb = {t_bnb:.1f} us")

    print(f"\n{'bn':>5s} {'nw':>4s} {'ns':>4s} " + " ".join(f"{t:>12s}" for t, _, _, _, _, _ in prepared))
    best = None
    for bn in (32, 64, 128, 256):
        for nw in (2, 4, 8):
            for ns in (2, 3, 4):
                set_nf4_launch_config(block_n=bn, block_m=1, num_warps=nw, num_stages=ns)
                row = []
                ok = True
                for tag, n, k, q, xi, t_bnb in prepared:
                    try:
                        with torch.inference_mode():
                            row.append(graph_time(lambda: q(xi)))
                    except Exception as exc:  # noqa: BLE001
                        row.append(float("nan"))
                        ok = False
                cells = " ".join(f"{v:12.1f}" for v in row)
                print(f"{bn:>5d} {nw:>4d} {ns:>4d} {cells}")
                if ok and best is None or (ok and sum(row) < best[0]):
                    best = (sum(row), bn, nw, ns, list(row))
    print(f"\nbest: block_n={best[1]} num_warps={best[2]} num_stages={best[3]} -> {best[4]}")


if __name__ == "__main__":
    main()