"""S4 诊断 ⑥：查询向量怎么算才稳（矩阵扫描）。

诊断 ⑤ 的结论：没有历史时寻址 3/3 全对；**一旦塞进 20 轮无关历史**就退化成恒选第一条。
怀疑是"查询的语境和记忆的语境不匹配"（记忆是在短语境里抓的，查询却是在长语境里算的）。

这里扫三个维度：
- 查询窗口：问题最后 4 个 token / 整个问题取平均 / 问题前 4 个 token
- 是否按真实位置旋转（position-aware vs position-free）
- 查询语境：**真实长语境**（带 20 轮历史）vs **规范短语境**（只把问题单独编码一次）

跑法：`& .\.venv\Scripts\python.exe src\diagnostics\probe_memory_addressing4.py`
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
]
FILLER = [
    ("今天天气怎么样？", "今天多云转晴，风不大。"),
    ("三加五等于几？", "三加五等于八。"),
    ("你会下棋吗？", "会一点，但不擅长。"),
    ("推荐一部电影吧。", "可以看《星际穿越》。"),
    ("现在几点了？", "我没有时钟，看不到时间。"),
]
N_FILLER_TURNS = 20
WINDOWS = {"last4": ("last", 4), "mean_all": ("all", 0), "first4": ("first", 4)}


def rotate(x_pre, start, hidden, rotary):
    cos, sin = _rope_cos_sin(rotary, x_pre, start, x_pre.shape[2], hidden)
    return apply_rotary_pos_emb(x_pre, x_pre, cos, sin)[0]


def grouped_logits(q, k, head_dim):
    groups = q.shape[1] // k.shape[1]
    qg = q.view(q.shape[0], k.shape[1], groups, q.shape[2], q.shape[3])
    out = torch.einsum("lhgtc,lhmc->lhgtm", qg, k) * (head_dim ** -0.5)
    return out.reshape(q.shape[0], q.shape[1], q.shape[2], k.shape[2])


def score(logits, spans):
    w = torch.softmax(logits, dim=-1)
    out, start = [], 0
    for m in spans:
        out.append(w[..., start : start + m].sum(-1).mean(-1).mean(-1).float())
        start += m
    return torch.stack(out, dim=-1).mean(0)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nova, _hf, tok = load_nova(norm_impl="exact")
    tm = nova.model
    hidden, head_dim, rotary = int(nova.config.hidden_size), int(nova.config.head_dim), tm.rotary_emb

    wtext = render(tok, [{"role": "user", "content": " ".join(f[1] for f in FACTS)}], add_generation_prompt=False)
    wids = tok(wtext, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    wspans = [find_span(tok, wtext, f[1]) for f in FACTS]
    _q, wk, _v = capture_qkv(tm, wids, (0, wids.shape[1]))
    spans = [s1 - s0 for s0, s1 in wspans]

    filler = []
    for i in range(N_FILLER_TURNS):
        u, a = FILLER[i % len(FILLER)]
        filler.append({"role": "user", "content": f"{u}（第 {i + 2} 轮）"})
        filler.append({"role": "assistant", "content": a})

    def query_q(msgs, qtext, window):
        """在给定语境里编码问题，返回 (Q 段 [层,头,t,D], 该段起点, 段长)。"""
        prompt = render(tok, msgs, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        span = find_span(tok, prompt, qtext)
        mode, k = WINDOWS[window]
        if mode == "last":
            s0 = max(span[0], span[1] - k)
            s1 = span[1]
        elif mode == "first":
            s0, s1 = span[0], min(span[1], span[0] + k)
        else:
            s0, s1 = span
        q_pre, _kk, _vv = capture_qkv(tm, ids, (0, ids.shape[1]))
        return q_pre.float()[:, :, s0:s1, :], s0, s1

    def run(context, window, aware):
        ok = 0
        picked = []
        for qlabel, qtext, want in QUESTIONS:
            msgs = list(filler) + [{"role": "user", "content": qtext}] if context == "real" else [{"role": "user", "content": qtext}]
            qseg, s0, s1 = query_q(msgs, qtext, window)
            if aware:
                if context == "real":
                    h = len(tok(render(tok, filler, add_generation_prompt=False), add_special_tokens=False)["input_ids"])
                    q_start = h + sum(spans) + s0
                else:
                    q_start = sum(spans) + s0
                q = rotate(qseg, q_start, hidden, rotary)
                ks, start = [], 0
                for (a, b) in wspans:
                    ks.append(rotate(wk[:, :, a:b, :].float(), start, hidden, rotary))
                    start += b - a
            else:
                q = qseg
                ks = [wk[:, :, a:b, :].float() for a, b in wspans]
            sc = score(torch.cat([grouped_logits(q, k, head_dim) for k in ks], dim=-1), spans)
            top = FACTS[int(sc.argmax())][0]
            picked.append(f"{qlabel}:{top}{'✓' if top == want else '✗'}")
            ok += int(top == want)
        return ok, picked

    print("\n矩阵（3 个问题，命中数越高越好）")
    print(f"    {'语境':6s} {'窗口':9s} {'位置':7s} 结果")
    for context in ("real", "canonical"):
        for window in WINDOWS:
            for aware in (True, False):
                ok, picked = run(context, window, aware)
                print(f"    {context:6s} {window:9s} {'aware' if aware else 'free':7s} {ok}/3  " + "  ".join(picked))
    print()


if __name__ == "__main__":
    main()
