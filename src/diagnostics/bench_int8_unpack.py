r"""E4 · 窄化证伪：解包 + 反量化能不能打满显存带宽？

**背景**：int8 / int4 的显存收益只有在**融合核**里才兑现（`kvquant.py` 的模拟版不省显存）。
而 D30 的教训是"访存省了、但算术没融合好 ⇒ 比 bnb 还慢"。所以先写**最小**的核：
**读 → 解包 → 反量化 → 乘一个固定查询向量再 reduce**，量它能打多少 GB/s。

判据（与 232.5 GiB/s 的纯读上限比）：
- **≥ 50%** ⇒ 融合核有戏，值得继续写完整的 attention 核；
- **< 25%** ⇒ 结案，量化只作为**存储格式**保留（收益靠别的路）。

三种模式共用同一个访存模式（`M=1`，即 decode 一步的读法）：
| 模式 | 读什么 | 元素数 |
|---|---|---|
| `fp16` | fp16 KV 原样（基线） | N |
| `int8` | uint8 + 每 32 元素一组 `min`/`step`（fp16） | N |
| `int4` | 打包 uint8（2 个 nibble/字节）+ 每组 `min`/`step` | N/2 |

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\bench_int8_unpack.py --tokens 65536
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
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

GROUP = 32  # 与 kvquant.GROUP_K 一致


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


@triton.jit
def _reduce_fp16(x_ptr, q_ptr, out_ptr, total, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    qv = tl.load(q_ptr + (offs % N), mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(x * qv))


@triton.jit
def _reduce_int8(q_ptr, mn_ptr, st_ptr, x_ptr, out_ptr, total, n_groups,
                 N: tl.constexpr, BLOCK: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    q = tl.load(q_ptr + offs, mask=m, other=0).to(tl.float32)
    g = offs // G
    mn = tl.load(mn_ptr + g, mask=m & (g < n_groups), other=0.0).to(tl.float32)
    st = tl.load(st_ptr + g, mask=m & (g < n_groups), other=0.0).to(tl.float32)
    x = q * st + mn
    qv = tl.load(x_ptr + (offs % N), mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(x * qv))


@triton.jit
def _reduce_int4(p_ptr, mn_ptr, st_ptr, x_ptr, out_ptr, total_packed, n_groups,
                 N: tl.constexpr, BLOCK: tl.constexpr, G: tl.constexpr):
    """读打包字节 -> 拆两个 nibble -> 各自反量化 -> 各自乘查询再求和。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)          # 打包字节下标
    m = offs < total_packed
    b = tl.load(p_ptr + offs, mask=m, other=0).to(tl.int32)
    lo = (b & 0x0F).to(tl.float32)
    hi = ((b >> 4) & 0x0F).to(tl.float32)
    e_lo = 2 * offs
    e_hi = 2 * offs + 1
    g_lo = e_lo // G
    mn = tl.load(mn_ptr + g_lo, mask=m & (g_lo < n_groups), other=0.0).to(tl.float32)
    st = tl.load(st_ptr + g_lo, mask=m & (g_lo < n_groups), other=0.0).to(tl.float32)
    x_lo = lo * st + mn
    x_hi = hi * st + mn                                # 同一个组（G 是偶数）
    q_lo = tl.load(x_ptr + (e_lo % N), mask=m, other=0.0).to(tl.float32)
    q_hi = tl.load(x_ptr + (e_hi % N), mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(x_lo * q_lo + x_hi * q_hi))


def bench(fn, reps: int = 30) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="解包+反量化的带宽微基准")
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    n, h, d = args.tokens, args.kv_heads, args.head_dim
    N = h * d
    total = n * N
    dev = "cuda"

    torch.manual_seed(0)
    x16 = torch.randn(total, dtype=torch.float16, device=dev)
    qv = torch.randn(N, dtype=torch.float16, device=dev)
    # int8：q + 每 G 个元素一组 min/step
    ng = (total + GROUP - 1) // GROUP
    q8 = torch.randint(0, 256, (total,), dtype=torch.uint8, device=dev)
    mn8 = torch.randn(ng, dtype=torch.float16, device=dev) * 0.1
    st8 = (torch.rand(ng, dtype=torch.float16, device=dev) * 0.01 + 1e-4)
    # int4：打包（两 nibble 一字节），元素数减半
    total_p = total // 2
    q4 = torch.randint(0, 256, (total_p,), dtype=torch.uint8, device=dev)
    ng4 = (total + GROUP - 1) // GROUP
    mn4 = torch.randn(ng4, dtype=torch.float16, device=dev) * 0.1
    st4 = (torch.rand(ng4, dtype=torch.float16, device=dev) * 0.01 + 1e-4)

    grid = (triton.cdiv(total, args.block),)
    out = torch.zeros(grid[0], dtype=torch.float32, device=dev)
    grid4 = (triton.cdiv(total_p, args.block),)
    out4 = torch.zeros(grid4[0], dtype=torch.float32, device=dev)

    def run_fp16():
        _reduce_fp16[grid](x16, qv, out, total, N=N, BLOCK=args.block, num_warps=args.warps)

    def run_int8():
        _reduce_int8[grid](q8, mn8, st8, qv, out, total, ng, N=N, BLOCK=args.block,
                           G=GROUP, num_warps=args.warps)

    def run_int4():
        _reduce_int4[grid4](q4, mn4, st4, qv, out4, total_p, ng4, N=N, BLOCK=args.block,
                            G=GROUP, num_warps=args.warps)

    bytes16 = total * 2
    bytes8 = total * 1 + ng * 4
    bytes4 = total_p * 1 + ng4 * 4

    print(f"KV 形状：{n} token × {h} 头 × {d} 维（单层）· BLOCK {args.block} · warps {args.warps}"
          f" · clocks.sm {clock_sm()}")
    print(f"{'模式':>6s} {'每次读':>10s} {'耗时':>10s} {'有效带宽':>10s} {'相对上限':>9s}")
    res = {}
    for name, fn, byt in (("fp16", run_fp16, bytes16), ("int8", run_int8, bytes8), ("int4", run_int4, bytes4)):
        dt = bench(fn, args.reps)
        gib = byt / 1024 ** 3
        bw = gib / dt
        res[name] = (gib, dt, bw)
        print(f"{name:>6s} {gib:>8.3f} GiB {dt * 1000:>8.3f} ms {bw:>8.1f} GiB/s {bw / 232.5:>8.1%}")

    print(f"\n上限：232.5 GiB/s（侧会话实测纯读，本机峰值 97.7%）")
    base = res["fp16"][2]
    for name in ("int8", "int4"):
        gib, dt, bw = res[name]
        print(f"  {name:>4s} 相对 fp16 基线：带宽 {bw / base:.2f}x · "
              f"同一份 KV 的耗时 {res['fp16'][1] / dt:.2f}x")
    print(f"\n判据：int8/int4 的**有效带宽** ≥ 50% 上限 ⇒ 融合核有戏；< 25% ⇒ 结案。")
    print(f"clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
