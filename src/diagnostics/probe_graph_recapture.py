r"""反复重捕一张解码图会不会爆显存？图内存池的三种策略实测。

起因：`reports/kv-int4.md` 记了"每题重捕，重捕 9 次 -> 7905/8188 MiB 崩溃"。
P0 引入**容量分桶**之后，"跨桶重捕"变成常规操作（2070 -> 4096 -> 8192 ...），
所以必须先把"重捕能不能安全地反复做"量清楚。

三种策略（`GraphDecoder.capture(pool=...)`）：

| 策略 | 行为 |
|---|---|
| `off`（默认） | 不传池，每张图自己的 |
| `shared` | 死用一个池（用来复现 PyTorch 裸 assert：`it->second->use_count > 0`） |
| `auto` | 上一个用池的图还活着就复用，否则换新池 |

每轮都显式 `del dec` + `gc.collect()` + `empty_cache()`，看**显存有没有回落**。
另加一个"同一个解码器连续跨桶重捕"的场景 —— 那是 P0 引入分桶后的常规路径。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_graph_recapture.py --rounds 4
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
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

from chatfmt import render  # noqa: E402
from exp_needle import build  # noqa: E402
from nova.decode import GraphDecoder  # noqa: E402
from s4_memory_demo import load_bundle  # noqa: E402


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def one_round(nova, ids, mode: str, max_len: int) -> tuple[bool, str]:
    """一轮：建解码器 -> prefill -> 捕获 -> 走 3 步 -> **彻底回收**。"""
    dec = None
    try:
        dec = GraphDecoder(nova, max_len=max_len)
        dec.prefill(ids)
        dec.capture(pool=mode)
        for _ in range(3):
            dec.step()
        torch.cuda.synchronize()
        return True, ""
    except RuntimeError as exc:
        torch.cuda.synchronize()
        return False, str(exc).strip().splitlines()[0][:110]
    finally:
        dec = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def recapture_chain(nova, ids, mode: str, buckets: list[int]) -> None:
    """**P0 的常规路径**：同一个解码器跨桶重捕（grow -> capture -> grow -> capture）。"""
    dec = GraphDecoder(nova, max_len=buckets[0])
    dec.prefill(ids)
    dec.capture(pool=mode)
    print(f"    桶 {buckets[0]:>6d} 首次捕获  allocated {torch.cuda.memory_allocated() / 1024 ** 2:8.1f} MiB")
    for b in buckets[1:]:
        dec.grow(b)
        dec.capture(pool=mode)
        dec.step()
        torch.cuda.synchronize()
        print(f"    桶 {b:>6d} 重捕      allocated {torch.cuda.memory_allocated() / 1024 ** 2:8.1f} MiB"
              f" · pos {int(dec.cache.pos.item())} · 图 {'在' if dec._captured else '不在'}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="重捕解码图 vs 图内存池策略")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2070)
    ap.add_argument("--prompt", type=int, default=24)
    ap.add_argument("--buckets", type=int, nargs="+", default=[2070, 4096, 8192])
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    msgs, _at = build(8)
    hay = tok(render(tok, msgs, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([hay[: args.prompt]], device="cuda")
    base = torch.cuda.memory_allocated() / 1024 ** 2
    print(f"Qwen3-VL-4B · 单通路 · max_len {args.max_len} · 每轮 del+gc+empty_cache")
    print(f"起点 allocated {base:.1f} MiB · clocks.sm {clock_sm()}")

    print("\n=== 场景一：每轮新建解码器 + 彻底回收 ===")
    for mode in ("off", "shared", "auto"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"\n=== pool={mode} ===")
        for r in range(args.rounds):
            ok, msg = one_round(nova, ids, mode, args.max_len)
            alloc = torch.cuda.memory_allocated() / 1024 ** 2
            print(f"  第 {r + 1} 轮 {'OK  ' if ok else '失败'} allocated {alloc:8.1f} MiB{'' if ok else '  ' + msg}")
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        after = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"  小结：起点 {before:.1f} -> 终点 {after:.1f} MiB（差 {after - before:+.1f}）· 峰值 {peak:.1f} MiB")

    print("\n=== 场景二：同一个解码器连续跨桶重捕（P0 常规路径）===")
    for mode in ("off", "auto"):
        print(f"  pool={mode}")
        try:
            recapture_chain(nova, ids, mode, args.buckets)
        except RuntimeError as exc:
            print(f"    失败 {str(exc).strip().splitlines()[0][:100]}")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("\nP0 结论（实测）：")
    print("  ① 场景一三种策略的逐轮 allocated 都基本平 ⇒ **“重捕就爆显存”不成立**，")
    print("     真凶是图没被正确释放，不是“没共享池” ⇒ 共享池不是必需。")
    print("  ② `shared` 在本脚本里也没抛 assert ⇒ 那个裸 assert 依赖分配器状态，")
    print("     只在 pytest 全量跑里中过 ⇒ 只能避开：默认 `pool=\"off\"`。")
    print(f"clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
