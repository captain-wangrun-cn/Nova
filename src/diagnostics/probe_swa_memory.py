r"""E2 第二条：**滑窗下的 S4 记忆取回**（注入的记忆位于窗口外时还找不找得到）。

记忆是**注在最前面**的（位置 `0..m`），而局部层只看最近 W 个 token ⇒ 上下文一长，
注入的记忆就掉出局部层的窗口，只剩**全局层**还看得见它。
这正是"滑窗 + 内部记忆"最容易翻车的地方，必须单独测。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_swa_memory.py --swa 0 2048 --turns 120
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

from chatfmt import EOS_ID, EOS_ID_ALT, find_span, render  # noqa: E402
from nova.loader import build_nova, enable_lm_head_4bit, load_hf_base  # noqa: E402
from nova.memory import MemorySession, MemoryStore  # noqa: E402
from s4_memory_demo import FACTS, QUESTIONS, build_filler  # noqa: E402

STOP = {EOS_ID, EOS_ID_ALT}


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
    ap = argparse.ArgumentParser(description="滑窗下的 S4 记忆取回")
    ap.add_argument("--swa", type=int, nargs="+", default=[0, 2048])
    ap.add_argument("--global-every", type=int, default=4)
    ap.add_argument("--turns", type=int, default=120, help="写入之后插入多少轮无关对话")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=6144)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    hf = load_hf_base()

    def make(window: int):
        m = build_nova(hf, norm_impl="triton", num_paths=1,
                       swa_window=window, swa_global_every=args.global_every)
        enable_lm_head_4bit(m)
        return m

    nova0 = make(0)
    store = MemoryStore.for_model(nova0)
    wtext = render(tok, [{"role": "user", "content": " ".join(f[1] for f in FACTS)}],
                   add_generation_prompt=False)
    wids = tok(wtext, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    for label, sentence in FACTS:
        store.write(nova0, wids, find_span(tok, wtext, sentence), label=label)
    print(f"记忆写入：{store}（第 1 轮 {wids.shape[1]} token，之后从可见历史里去掉）")
    inj = store.inject_length(store.items)

    filler = build_filler(args.turns)
    hist_ids = tok(render(tok, filler, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    hist_t = torch.tensor([hist_ids], device="cuda")
    print(f"可见历史 {len(hist_ids)} token = {args.turns} 轮无关对话 · 每 {args.global_every} 层 1 个全局层")
    print(f"{'窗口':>6s} {'注入':>5s}  答对/8   明细     耗时")

    for window in args.swa:
        nova = nova0 if window == 0 else make(window)
        sess = MemorySession(store, nova, max_len=args.max_len, top_k=1)
        hit = 0
        detail = []
        t_all = 0.0
        for qtext, want, accept in QUESTIONS:
            full_text = render(tok, filler + [{"role": "user", "content": qtext}], add_generation_prompt=True)
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
            qspan = find_span(tok, full_text, qtext)
            cur_t = torch.tensor([full_ids[len(hist_ids):]], device="cuda")
            t0 = time.perf_counter()
            info = sess.prefill(hist_t, cur_t, use_memory=True, query_span=qspan, place="turn")
            out = sess.generate(args.max_new, STOP)
            t_all += time.perf_counter() - t0
            text = tok.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
            ok = all(k in text for k in accept)
            hit += int(ok)
            detail.append("O" if ok else "X")
            if not ok:
                print(f"    ↳ 窗口 {window} 问「{qtext}」（期望 {want}）答：{text[:44]}")
        print(f"{window:>6d} {inj:>5d}  "
              f"{hit}/8     {' '.join(detail)}  {t_all:.1f}s", flush=True)

    print(f"\n判据：滑窗下 8/8 不能掉（注入的记忆在窗口外时只剩全局层看得见）。clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
