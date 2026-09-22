r"""NF4 GEMV 微基准：自写 Triton kernel vs bnb `Linear4bit` vs fp16 `nn.Linear`。

三个指标：
  1. **数值**：`NF4Linear` 的输出 vs bnb 的输出，`max|diff|` 应该落在 fp16 舍入级
  2. **纯 GPU 时间**：把 N 次调用**捕获进一张 CUDA Graph** 再 replay —— 这样量到的
     是不含 CPU 派发开销的真 GPU 时间（本机单次 kernel 启动 CPU 侧要 13.5us，
     直接循环计时会把 20us 的算子量成 33us）
  3. **调参**：`--sweep` 扫 BLOCK_N / BLOCK_M / num_warps / num_stages

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\bench_nf4_gemv.py
    & .\.venv\Scripts\python.exe src\diagnostics\bench_nf4_gemv.py --sweep
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))

import torch  # noqa: E402

# Qwen3-VL-4B 文本塔的真实形状：(N, K) = (out_features, in_features)
SHAPES = [
    ("q_proj   ", 4096, 2560),
    ("kv_proj  ", 1024, 2560),
    ("o_proj   ", 2560, 4096),
    ("gate/up  ", 9728, 2560),
    ("down_proj", 2560, 9728),
]


def make_bnb_linear(n: int, k: int, seed: int = 0):
    import bitsandbytes as bnb

    torch.manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.05)
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


def graph_time(fn, iters: int = 200, reps: int = 5, warm: int = 3) -> float:
    """把 `iters` 次调用捕获进一张图，replay `reps` 次取平均。返回单次 **微秒**。"""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="扫 launch 参数")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    import bitsandbytes as bnb
    from nova.kernels import nf4_launch_config, set_nf4_launch_config
    from nova.quant import NF4Linear

    print(f"launch config: {nf4_launch_config()}")
    print()

    x1 = torch.randn(1, 2560, device="cuda", dtype=torch.float16)

    print("=== 数值对照（M=1，随机权重）===")
    print(f"{'shape':<12s} {'N':>6s} {'K':>6s}  {'max|diff|':>10s}  {'fp16 分辨率':>11s}  {'判定':>6s}")
    for name, n, k in SHAPES:
        lin = make_bnb_linear(n, k)
        q = NF4Linear.from_bnb(lin)
        xi = torch.randn(1, k, device="cuda", dtype=torch.float16) * 0.5
        with torch.inference_mode():
            ref = lin(xi)
            got = q(xi)
        d = (ref.float() - got.float()).abs().max().item()
        scale = ref.abs().mean().item()
        ulp = 2.0 ** (torch.log2(torch.tensor(max(scale, 1e-6))).floor().item() - 10)
        print(f"{name:<12s} {n:>6d} {k:>6d}  {d:10.3e}  {ulp:11.3e}  "
              f"{'OK' if d <= 4 * ulp else 'FAIL':>6s}")
    print()

    print("=== 纯 GPU 时间（CUDA Graph 内 replay，M=1）===")
    print(f"{'shape':<12s} {'N':>6s} {'K':>6s} {'bnb us':>9s} {'fp16 us':>9s} {'nf4 us':>9s} "
          f"{'vs bnb':>8s} {'带宽 GB/s':>10s}")
    tot_bnb = tot_nf4 = 0.0
    for name, n, k in SHAPES:
        lin = make_bnb_linear(n, k)
        q = NF4Linear.from_bnb(lin)
        fp = torch.nn.Linear(k, n, bias=True, device="cuda", dtype=torch.float16)
        xi = torch.randn(1, k, device="cuda", dtype=torch.float16) * 0.5

        with torch.inference_mode():
            t_bnb = graph_time(lambda: lin(xi), iters=args.iters)
            t_fp16 = graph_time(lambda: fp(xi), iters=args.iters)
            t_nf4 = graph_time(lambda: q(xi), iters=args.iters)

        nbytes = n * k / 2 + n * (k / 64) * 4 + k * 2 + n * 2
        bw = nbytes / (t_nf4 * 1e-6) / 1e9
        tot_bnb += t_bnb
        tot_nf4 += t_nf4
        print(f"{name:<12s} {n:>6d} {k:>6d} {t_bnb:9.1f} {t_fp16:9.1f} {t_nf4:9.1f} "
              f"{t_bnb / t_nf4:7.2f}x {bw:10.0f}")
    print(f"{'合计':<12s} {'':>6s} {'':>6s} {tot_bnb:9.1f} {'':>9s} {tot_nf4:9.1f} "
          f"{tot_bnb / tot_nf4:7.2f}x")
    print()

    if args.sweep:
        print("=== 调参扫描（(2560,2560) 与 (9728,2560)）===")
        print(f"{'block_n':>8s} {'block_m':>8s} {'warps':>6s} {'stages':>7s} "
              f"{'2560x2560':>10s} {'9728x2560':>10s}")
        base = nf4_launch_config()
        for bn in (32, 64, 128, 256):
            for bm in (1, 2, 4):
                for nw in (2, 4, 8):
                    for ns in (2, 3, 4):
                        set_nf4_launch_config(block_n=bn, block_m=bm, num_warps=nw, num_stages=ns)
                        try:
                            times = []
                            for name, n, k in (("a", 2560, 2560), ("b", 9728, 2560)):
                                lin = make_bnb_linear(n, k)
                                q = NF4Linear.from_bnb(lin)
                                xi = torch.randn(1, k, device="cuda", dtype=torch.float16) * 0.5
                                with torch.inference_mode():
                                    times.append(graph_time(lambda: q(xi), iters=60, reps=3))
                            print(f"{bn:>8d} {bm:>8d} {nw:>6d} {ns:>7d} "
                                  f"{times[0]:10.1f} {times[1]:10.1f}")
                        except Exception as exc:  # noqa: BLE001
                            print(f"{bn:>8d} {bm:>8d} {nw:>6d} {ns:>7d}  FAILED: {type(exc).__name__}")
        set_nf4_launch_config(**base)


if __name__ == "__main__":
    main()