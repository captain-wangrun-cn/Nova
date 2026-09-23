r"""D42 前置 · int8 融合注意力核：解码一步的延迟与有效带宽。

**比的是同一件事**：解码一步、单层、`q_len=1`，KV 长度 L。三条路：

| 路 | 说明 | 读多少字节 |
|---|---|---|
| `fp16-sdpa` | **现状**：`LeanAttention` 把 fp16 cache 交给 SDPA（`repeat_kv` 展平成 32 头） | `2·H_q·L·D·2` |
| `int8-sdpa` | **模拟版**：`dequantize_int8` 还原成 fp16 再喂 SDPA（`QuantRoundTripCache` 的写法） | 同上 + 还原 |
| `int8-fused` | **本核**：直接读 int8，GQA 由 8 个 KV 头摊开，K/V 每字节只读一次 | `2·H_kv·L·D·1` + 尺子 |

⚠️ 两条路的差距来自**两个**独立原因，别混着算：
1. **int8 存储**：1.88x 记账（D38）；
2. **GQA 不重复读**：`repeat_kv` 会把 8 个 KV 头展平成 32 个（4x 字节），融合核按 KV 头切，不展平。
有效带宽那一列是**按各自实际读的字节**算的，所以两者可以横向比"带宽打得满不满"。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\bench_int8_attn.py --lens 1024 4096 8192 16384
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from nova.kvattn import int8_attn_decode, pack_int8_kv  # noqa: E402
from nova.kvquant import dequantize_int8  # noqa: E402

PEAK = 232.5  # GiB/s：侧会话实测纯读上限（见 reports/kv-unpack-bench.md）


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def bench(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def graph_time(fn, reps: int) -> float:
    """把 `fn` 捕进 CUDA Graph 再重放 —— 量的是**纯 GPU 时间**。

    为什么必须这么量：解码一步的核只有几十微秒，Python 侧（Triton 启动器 + 3 次分配）的
    开销比核本身还大，eager 循环量出来的根本不是核的性能。而项目的解码路径本来就是
    **CUDA Graph**（D29 / D35），所以 graph 重放才是"接进去之后会看到什么"的答案。
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # 先在旁路预热（Triton JIT 编译不能发生在捕获期）
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """与 `nova.layers.repeat_kv` 同义（现状路径要展平成 32 头才能进 SDPA）。"""
    b, h, n, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, n, d).reshape(b, h * n_rep, n, d)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="int8 融合注意力核的解码延迟")
    ap.add_argument("--lens", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    hq, h_kv, d = args.heads, args.kv_heads, args.head_dim
    dev = "cuda"
    torch.manual_seed(0)
    print(f"解码一步 · 单层 · {hq} Q 头 / {h_kv} KV 头 / D={d} · "
          f"chunk={args.chunk} warps={args.warps} stages={args.stages} · clocks.sm {clock_sm()}")
    print(f"{'L':>7s} {'路':>11s} {'读':>9s} {'eager':>9s} {'graph':>9s} "
          f"{'graph带宽':>10s} {'占上限':>7s} {'vs 现状':>8s}")

    for length in args.lens:
        k = (torch.randn(1, h_kv, length, d, device=dev, dtype=torch.float16) * 0.7)
        v = (torch.randn(1, h_kv, length, d, device=dev, dtype=torch.float16) * 0.7)
        q = (torch.randn(1, hq, 1, d, device=dev, dtype=torch.float16) * 0.7)
        kq, kmn, kst, vq, vmn, vst = pack_int8_kv(k, v)
        k16 = repeat_kv(k, hq // h_kv)
        v16 = repeat_kv(v, hq // h_kv)
        kd = repeat_kv(dequantize_int8(kq, kmn, kst, 64, "token"), hq // h_kv)
        vd = repeat_kv(dequantize_int8(vq, vmn, vst, 64, "token"), hq // h_kv)
        scale = d ** -0.5

        bytes_fp16 = 2 * hq * length * d * 2
        bytes_int8 = 2 * h_kv * length * d * 1 + 2 * 2 * 2 * (-(-length // 64)) * d

        def run_fused():
            int8_attn_decode(q, kq, kmn, kst, vq, vmn, vst, used=length,
                             chunk=args.chunk, num_warps=args.warps, num_stages=args.stages)

        def run_int8_sdpa():
            F.scaled_dot_product_attention(q, kd, vd, scale=scale)

        def run_fp16_sdpa():
            F.scaled_dot_product_attention(q, k16, v16, scale=scale)

        n_splits = -(-length // args.chunk)
        pm = torch.empty((h_kv, n_splits, 16), dtype=torch.float32, device=dev)
        pl = torch.empty_like(pm)
        pa = torch.empty((h_kv, n_splits, 16, d), dtype=torch.float32, device=dev)
        out = torch.empty((1, hq, 1, d), dtype=torch.float16, device=dev)

        def run_fused_graph():
            int8_attn_decode(q, kq, kmn, kst, vq, vmn, vst, used=length,
                             chunk=args.chunk, num_warps=args.warps, num_stages=args.stages,
                             scratch=(pm, pl, pa), out=out)

        base = graph_time(run_fp16_sdpa, args.reps)
        for name, fn, nbytes in (("fp16-sdpa", run_fp16_sdpa, bytes_fp16),
                                 ("int8-sdpa", run_int8_sdpa, bytes_fp16),
                                 ("int8-fused", run_fused, bytes_int8),
                                 ("int8-fused*", run_fused_graph, bytes_int8)):
            eager = bench(fn, args.reps)
            dt = graph_time(fn, args.reps)
            gib = nbytes / 1024 ** 3
            bw = gib / dt
            print(f"{length:>7d} {name:>11s} {gib:>7.3f}G {eager * 1000:>7.3f} ms "
                  f"{dt * 1000:>7.3f} ms {bw:>8.1f} GiB/s {bw / PEAK:>6.1%} {base / dt:>7.2f}x")
        del k16, v16, kd, vd, pm, pl, pa, out

    print(f"\n上限 {PEAK} GiB/s（侧会话实测纯读）· clocks.sm {clock_sm()}")
    print("* = 预分配 scratch/out 后捕进 CUDA Graph 重放（纯 GPU 时间，接进解码路径的形态）")


if __name__ == "__main__":
    main()
