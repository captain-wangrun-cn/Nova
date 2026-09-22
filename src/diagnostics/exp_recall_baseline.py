r"""对照实验：**原版 4B 在长对话里到底会不会忘？**

S4 演示里的 `off = 0/8` 有两种可能解释，必须分开，否则会得出错误结论：

  (a) 模型**看到过**第 1 轮，但 20 轮之后**忘了**；
  (b) 第 1 轮被从可见历史里**删掉**了（S4 演示的做法），模型**从未见过**。

S4 演示测的是 (b) —— 它是"记忆确实起了作用"的**因果对照**，**不是**"原版模型会忘"的证据。
本脚本测 (a)：把第 1 轮**留在**可见历史里，只在后面追加 N 轮无关对话，**不注入任何记忆**。

**实测（2026-09-22，单通路 triton）：原版模型 20 / 60 / 120 轮全部 8/8，没有忘。**
该模型上下文上限 **262144 token**，而 120 轮闲聊只有 3785 token —— 远没到需要"记忆"的程度。

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\exp_recall_baseline.py --turns 20 60 120

结论：S4 验证的是**记忆的读写通路**（抓取 / 寻址 / 注入 / 存盘）能不能工作，
**不是**"原版模型做不到"。要证明记忆有收益，得换到上下文装不下的场景（显存或窗口）。
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))

import torch  # noqa: E402

from chatfmt import EOS_ID, EOS_ID_ALT, render  # noqa: E402
from nova.memory import MemorySession, MemoryStore  # noqa: E402
from s4_memory_demo import FACTS, QUESTIONS, build_filler, load_bundle  # noqa: E402

STOP = {EOS_ID, EOS_ID_ALT}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="原版模型长对话回忆对照实验")
    ap.add_argument("--turns", type=int, nargs="+", default=[20, 60, 120])
    ap.add_argument("--paths", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()

    nova, tok = load_bundle(args.paths, "triton")
    ctx = getattr(nova.model.config, "max_position_embeddings", "?")
    print(f"模型：Qwen3-VL-4B-Instruct · 通路数 {args.paths} · 上下文上限 {ctx} token\n")

    sess = MemorySession(MemoryStore.for_model(nova), nova, max_len=args.max_len)
    turn1 = [{"role": "user", "content": " ".join(f[1] for f in FACTS)}]

    print(f"{'可见历史':>28s} {'token':>7s}  答对  逐题")
    for turns in args.turns:
        msgs = turn1 + build_filler(turns)
        hist_ids = tok(render(tok, msgs, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
        hist_t = torch.tensor([hist_ids], device="cuda")

        hits, detail = 0, []
        for qtext, _want, accept in QUESTIONS:
            full_text = render(tok, msgs + [{"role": "user", "content": qtext}], add_generation_prompt=True)
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
            assert full_ids[: len(hist_ids)] == hist_ids, "历史不是前缀"
            cur_t = torch.tensor([full_ids[len(hist_ids) :]], device="cuda")
            sess.prefill(hist_t, cur_t, use_memory=False)
            out = sess.generate(args.max_new, STOP)
            text = tok.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
            ok = all(k in text for k in accept)
            hits += int(ok)
            detail.append("O" if ok else "X")
            if not ok:
                print(f"        ↳ 没答出「{accept[0]}」：{text[:60]}")
        print(f"{f'第1轮 + {turns} 轮':>28s} {len(hist_ids):>7d}  {hits}/{len(QUESTIONS)}  {' '.join(detail)}")

    print("\n注：全程不注入任何记忆，第 1 轮一直在可见历史里。")
    print("    S4 演示的 off=0/8 是「第 1 轮被移出上下文」的因果对照，不是「模型会忘」。")


if __name__ == "__main__":
    main()
