r"""decode 每 token 耗时到底由"实际长度"还是"`max_len`"决定？

起因：侧会话实测显存纯读 **232.5 GiB/s**（峰值 97.7%），据此推断"decode 是带宽受限、带宽已到顶"。
但同文里又写"当前有效读仅 ~60 GiB/s" —— 两者矛盾。若只用到峰值的 ~26%，那 decode **不是**带宽受限，
而是被"读满 `max_len` 个槽位"这类浪费卡住。

本脚本把 `used`（真实写进去的 token 数）与 `max_len`（cache 预留长度）**分开**测：

    (used=2048, max_len=18432) vs (used=7291, max_len=18432)   -> 耗时是否随 used 变？
    (used=7291, max_len=7323)  vs (used=7291, max_len=18432)   -> 耗时是否随 max_len 变？

判据：`GraphDecoder._body()` 传的 mask 覆盖整个 `max_len`，而 `q_len == 1` 时
`layers.py` 里 `is_causal = False` ⇒ SDPA 会读满 `max_len` 个槽位 ⇒ **耗时应当只跟 `max_len` 走**。

跑法（同轮内交替，两遍取小值）：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_decode_maxlen.py
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="decode 耗时 vs used vs max_len")
    ap.add_argument("--used", type=int, nargs="+", default=[2048, 7291])
    ap.add_argument("--max-len", type=int, nargs="+", default=[2070, 7323, 18432])
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    n_rounds = max(8, int(round(max(args.used) / 36)) + 8)
    msgs, _at = build(n_rounds)
    hay = tok(render(tok, msgs, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    print(f"模型：Qwen3-VL-4B-Instruct · 单通路 · 干草堆可用 {len(hay)} token · clocks.sm {clock_sm()}")

    # 只测 fp16：这里要问的是"读多少个槽位"，与量化无关
    best: dict[tuple[int, int], float] = {}
    for _round in range(args.rounds):
        for used in args.used:
            for ml in args.max_len:
                if ml < used + 8:
                    continue
                dec = GraphDecoder(nova, max_len=ml)
                ids = torch.tensor([hay[:used]], device="cuda")
                dec.prefill(ids)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.reps):
                    dec._body()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / args.reps * 1000.0
                key = (used, ml)
                best[key] = dt if key not in best else min(best[key], dt)
                del dec, ids
                torch.cuda.empty_cache()

    print(f"\n{'used':>6s} {'max_len':>8s} {'ms/token':>9s} {'等效读量':>10s} {'有效带宽':>10s}")
    print(f"{'':>6s} {'':>8s} {'':>9s} {'GiB/step':>10s} {'GiB/s':>10s}")
    for (used, ml), dt in sorted(best.items()):
        kv = ml * 36 * 2 * 1 * 8 * 128 * 2 / 1024 ** 3  # 按 max_len 读
        print(f"{used:>6d} {ml:>8d} {dt:>9.1f} {kv:>10.2f} {kv / (dt / 1000):>10.1f}")

    print("\n判据：若耗时只随 max_len 变、几乎不随 used 变 -> decode 被「读满 max_len 槽位」卡住，"
          "而不是被真实上下文长度卡住。")
    print(f"clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
