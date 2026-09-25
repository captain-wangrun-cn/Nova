r"""② · 记忆段加载的**端到端**延迟：`safe_open` 逐张量 vs `SegmentPrefetcher` 双缓冲。

比的是同一件事：**把 `.safetensors` 里的记忆从磁盘搬到显存、可用的那一刻**。

| 路 | 读法 | 盘读与 H2D |
|---|---|---|
| `safe_open` | 逐张量 `get_tensor()` | 串行 |
| `prefetch` | 整段原始字节（pin + `readinto` + 双缓冲 + `non_blocking`）+ 显存内零拷贝视图 | **重叠** |

## 为什么要造大记忆来测

S4 demo 的记忆只有几百 KB，端到端时间被**固定开销**（打开文件、解析头、建 store）吃满，
两条路的差别看不出来。真正要量的是**搬运带宽**，所以本脚本按"记忆 = 240 KiB/token"
（HANDOFF 第七节的实测值）造合成记忆：**1 万 token ≈ 2.3 GiB**、**1000 token ≈ 230 MiB**。

## 判据

1. **正确性**：预取路径读出的张量与 `safe_open` 路径**逐位相同**（不是"接近"——两条路都没算数）；
2. **带宽**：`prefetch` 的 GiB/s 应接近 D41 的 4.49 GiB/s 上限（盘读 5.09 / PCIe 12.2）；
3. **速度**：`prefetch` 应至少不慢于 `safe_open`，大记忆上应有明确倍数。

⚠️ 速度数字必须带 `clocks.sm`（本机 GPU 空闲会停在 ~780 MHz，上限 3105，同一条命令差 1.9x）。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\bench_memory_load.py
    & .\.venv\Scripts\python.exe -u src\diagnostics\bench_memory_load.py --tokens 1000 4000 --reps 5
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

from nova.memory import MemorySchema, MemoryStore  # noqa: E402

# 真实形状：**双通路** 60 个 cache 槽位 × 8 KV 头 × 128 head_dim × (k+v) × fp16
# = 240 KiB/token —— 与 HANDOFF 第七节记的实测量级一致（单通路 36 槽是 144 KiB）。
# ⚠️ 用 36 槽会低估一半，量出来的带宽还好看，但换算到真实记忆上就偏了。
LAYERS, KV_HEADS, HEAD_DIM, NUM_PATHS = 60, 8, 128, 2
BYTES_PER_TOKEN = LAYERS * KV_HEADS * HEAD_DIM * 2 * 2


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def make_memory(path: Path, tokens: int, items: int) -> MemoryStore:
    """造一条 ~`tokens` 个 token 的合成记忆（分 `items` 项），写盘。"""
    schema = MemorySchema(hidden_size=2560, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                          num_cache_layers=LAYERS, num_paths=NUM_PATHS)
    store = MemoryStore(schema, fingerprint="bench")
    per = max(1, tokens // items)
    gen = torch.Generator().manual_seed(0)
    for i in range(items):
        n = per if i < items - 1 else max(1, tokens - per * (items - 1))
        # 直接给 CPU 张量：`save()` 本来就要 `.cpu().contiguous()`，这里省一次大拷贝
        k = torch.randn(LAYERS, KV_HEADS, n, HEAD_DIM, dtype=torch.float16, generator=gen)
        v = torch.randn(LAYERS, KV_HEADS, n, HEAD_DIM, dtype=torch.float16, generator=gen)
        store.add(k, v, label=f"item{i}")
    store.save(path)
    return store


def timed(fn, reps: int) -> tuple[float, object]:
    """跑 `reps` 次，返回**最快一次**的毫秒数与最后一次的返回值。"""
    best, out = float("inf"), None
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1000, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1000, 4000, 10000])
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seg-mib", type=float, default=4.0)
    ap.add_argument("--ring", type=int, default=4)
    ap.add_argument("--demo", type=str, default="", help="额外量一个真实记忆文件（S4 demo）")
    args = ap.parse_args()

    outdir = ROOT / ".tmp" / "mem-bench"
    outdir.mkdir(parents=True, exist_ok=True)
    seg_bytes = int(args.seg_mib * 1024 * 1024)
    print(f"clocks.sm {clock_sm()} · 段长 {args.seg_mib} MiB · ring {args.ring} · "
          f"每 token {BYTES_PER_TOKEN / 1024:.0f} KiB\n")

    cases: list[tuple[str, Path]] = []
    for tokens in args.tokens:
        path = outdir / f"synth-{tokens}.safetensors"
        if not path.exists() or path.stat().st_size < tokens * BYTES_PER_TOKEN * 0.9:
            print(f"造合成记忆 {tokens} token（{tokens * BYTES_PER_TOKEN / 1024 ** 3:.2f} GiB）…")
            make_memory(path, tokens, args.items)
        cases.append((f"合成 {tokens} tok", path))
    if args.demo:
        cases.append(("S4 demo", Path(args.demo)))

    print()
    print(f"{'文件':>16s} {'大小':>9s} {'safe_open':>11s} {'prefetch':>11s} "
          f"{'倍数':>6s} {'prefetch 带宽':>13s}  逐位一致")
    for name, path in cases:
        size_mib = path.stat().st_size / 1024 ** 2
        t_open, a = timed(lambda: MemoryStore.load(path, device="cuda"), args.reps)
        t_pf, b = timed(lambda: MemoryStore.load_prefetched(
            path, device="cuda", seg_bytes=seg_bytes, ring=args.ring), args.reps)
        same = all(torch.equal(x.k, y.k) and torch.equal(x.v, y.v)
                   for x, y in zip(a.items, b.items))
        gib = size_mib / 1024
        bw = gib / (t_pf / 1000) if t_pf else 0.0
        print(f"{name:>16s} {size_mib:8.1f}M {t_open:9.1f}ms {t_pf:9.1f}ms "
              f"{t_open / t_pf:5.2f}x {bw:10.2f} GiB/s  {same}")
        if not same:
            print(f"  ⚠️ {name} 两条路读出的张量不一致 —— 这条结果是废的")
        del a, b
        torch.cuda.empty_cache()
    print(f"\nclocks.sm {clock_sm()}（结束时采样）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
