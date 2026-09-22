"""S4 诊断 ⑤：修正查询窗口后，寻址到底行不行 + 记忆该放在哪。

诊断 ①③④ 的共同错误：查询取的是 prompt **最后 4 个 token**，而那是
`<|im_start|>assistant\\n`（生成提示，**没有内容**）。改成"问题本身的最后 4 个 token"重测。

三种布局（都用真实位置旋转 Q/K，softmax 只在候选之间归一）：
- `no_hist_front`：没有历史，记忆在位置 0..Σm，问题紧随其后（距离 ~40）
- `hist_front`：20 轮无关历史，记忆在**最前面**（距离 ~600）
- `hist_turn`：20 轮无关历史，记忆插在**当前轮之前**（距离 ~10）

跑法：`& .\.venv\Scripts\python.exe src\diagnostics\probe_memory_addressing3.py`
"""

from __future__ import annotations

import os
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

from chatfmt import find_span, render  # noqa: E402
from nova.layers import apply_rotary_pos_emb  # noqa: E402
from nova.loader import load_nova  # noqa: E402
from nova.memory import _rope_cos_sin, capture_qkv  # noqa: E402

FACTS = [
    ("红裙子", "我今天换了一条红裙子，是上周在巴黎买的。"),
    ("橘猫", "我养的猫叫雷纳德，是一只橘猫。"),
    ("花瓶", "我把阳台上的蓝色花瓶打碎了。"),
]
QUESTIONS = [
    ("裙子颜色", "我今天穿的裙子是什么颜色的？", "红裙子"),
    ("猫名字", "我养的猫叫什么名字？", "橘猫"),
    ("打碎的东西", "我把什么东西打碎了？", "花瓶"),
    ("无关问题", "请用一句话解释什么是潮汐。", None),
]
FILLER = [
    ("今天天气怎么样？", "今天多云转晴，风不大。"),
    ("三加五等于几？", "三加五等于八。"),
    ("你会下棋吗？", "会一点，但不擅长。"),
    ("推荐一部电影吧。", "可以看《星际穿越》。"),
    ("现在几点了？", "我没有时钟，看不到时间。"),
]
QUERY_LAST = 4
N_FILLER_TURNS = 20


def rotate(x_pre, start, hidden, rotary):
    cos, sin = _rope_cos_sin(rotary, x_pre, start, x_pre.shape[2], hidden)
    return apply_rotary_pos_emb(x_pre, x_pre, cos, sin)[0]


def grouped_logits(q, k, head_dim):
    groups = q.shape[1] // k.shape[1]
    qg = q.view(q.shape[0], k.shape[1], groups, q.shape[2], q.shape[3])
    out = torch.einsum("lhgtc,lhmc->lhgtm", qg, k) * (head_dim ** -0.5)
    return out.reshape(q.shape[0], q.shape[1], q.shape[2], k.shape[2])


def mass(logits):
    """logits [层,头,t,Σm] -> 每项的注意力质量 [候选数]。"""
    w = torch.softmax(logits, dim=-1)
    out, start = [], 0
    for m in M_SPANS:
        out.append(w[..., start : start + m].sum(-1).mean(-1).mean(-1).float())
        start += m
    return torch.stack(out, dim=-1).mean(0)


def build_filler_ids(tok):
    msgs = []
    for i in range(N_FILLER_TURNS):
        u, a = FILLER[i % len(FILLER)]
        msgs.append({"role": "user", "content": f"{u}（第 {i + 2} 轮）"})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def main() -> None:
    global M_SPANS
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nova, _hf, tok = load_nova(norm_impl="exact")
    tm = nova.model
    hidden, head_dim, rotary = int(nova.config.hidden_size), int(nova.config.head_dim), tm.rotary_emb

    # 写入轮：抓三条事实的 K（RoPE 之前）
    wtext = render(tok, [{"role": "user", "content": " ".join(f[1] for f in FACTS)}], add_generation_prompt=False)
    wids = tok(wtext, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    wspans = [find_span(tok, wtext, f[1]) for f in FACTS]
    _q, wk, _v = capture_qkv(tm, wids, (0, wids.shape[1]))
    M_SPANS = [s1 - s0 for s0, s1 in wspans]

    filler = build_filler_ids(tok)

    print("\n布局对比（top-1；✓=对，·=无关问题）")
    for layout in ("no_hist_front", "hist_front", "hist_turn"):
        hist = [] if layout == "no_hist_front" else filler
        rows = []
        for qlabel, qtext, want in QUESTIONS:
            msgs = list(hist) + [{"role": "user", "content": qtext}]
            qp = render(tok, msgs, add_generation_prompt=True)
            qids = tok(qp, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            t = qids.shape[1]
            qspan = find_span(tok, qp, qtext)
            q0 = max(qspan[0], qspan[1] - QUERY_LAST)
            q_pre, _k, _v = capture_qkv(tm, qids, (0, t))

            # 记忆的放置位置 + 查询的真实位置
            if layout == "hist_turn":
                hist_text = render(tok, hist, add_generation_prompt=False) if hist else ""
                h = len(tok(hist_text, add_special_tokens=False)["input_ids"]) if hist else 0
                mem_start, q_start = h, h + sum(M_SPANS) + q0
            else:
                mem_start, q_start = 0, sum(M_SPANS) + q0

            q = rotate(q_pre.float()[:, :, q0:qspan[1], :], q_start, hidden, rotary)
            ks, start = [], mem_start
            for (s0, s1) in wspans:
                ks.append(rotate(wk[:, :, s0:s1, :].float(), start, hidden, rotary))
                start += s1 - s0
            logits = torch.cat([grouped_logits(q, k, head_dim) for k in ks], dim=-1)
            sc = mass(logits)
            top = FACTS[int(sc.argmax())][0]
            rows.append(f"{qlabel}:{top}{'✓' if top == want else ('·' if want is None else '✗')}")
        print(f"    {layout:14s} " + "  ".join(rows))
    print()


if __name__ == "__main__":
    main()
