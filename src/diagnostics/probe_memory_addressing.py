"""S4 诊断 ②：寻址到底该怎么打分。

诊断 ① 的结论：pre-RoPE 的裸 Q·K（位置无关）几乎不区分内容 —— 三个记忆项的打分顺序
在所有问题下都一样（红裙子 > 橘猫 > 花瓶），top-1 只有 5/15。

这里试另外两种：
- **A 同上下文**：事实和问题在**同一个 prompt** 里（不注入），按真实位置旋转 Q/K，
  算问题位置对每个事实的注意力质量 —— 这是"模型的注意力本来会不会区分内容"的上界。
- **B 注入到最前面**：记忆放在位置 0..Σm，问题在其后，按注入后的真实位置旋转。

两者的差别只有"距离"（A 近，B 远）。跑法：
```powershell
& .\.venv\Scripts\python.exe src\diagnostics\probe_memory_addressing.py
```
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
QUERY_LAST = 4


def rotate(x_pre: torch.Tensor, start: int, hidden_size: int, rotary_emb) -> torch.Tensor:
    """按真实位置旋转一段 Q 或 K（形状 [层, 头, t, D]）。"""
    cos, sin = _rope_cos_sin(rotary_emb, x_pre, start, x_pre.shape[2], hidden_size)
    return apply_rotary_pos_emb(x_pre, x_pre, cos, sin)[0]


def grouped_logits(q: torch.Tensor, k: torch.Tensor, head_dim: int) -> torch.Tensor:
    """q [层,Hq,t,D] · k [层,Hkv,m,D] -> [层,Hq,t,m]（GQA 分组）。"""
    groups = q.shape[1] // k.shape[1]
    qg = q.view(q.shape[0], k.shape[1], groups, q.shape[2], q.shape[3])
    out = torch.einsum("lhgtc,lhmc->lhgtm", qg, k) * (head_dim ** -0.5)
    return out.reshape(q.shape[0], q.shape[1], q.shape[2], k.shape[2])


def mass(q, k_list, head_dim: int, per_layer: bool = False):
    """softmax 只在候选之间归一，返回每项的注意力质量。"""
    logits = torch.cat([grouped_logits(q, k, head_dim) for k in k_list], dim=-1)  # [层,Hq,t,Σm]
    w = torch.softmax(logits, dim=-1)
    out, start = [], 0
    for k in k_list:
        m = k.shape[2]
        out.append(w[..., start : start + m].sum(-1).mean(-1).mean(-1).float())  # [层]
        start += m
    stacked = torch.stack(out, dim=-1)  # [层, n_items]
    return stacked if per_layer else stacked.mean(0)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nova, _hf, tok = load_nova(norm_impl="exact")
    text_model = nova.model
    hidden = int(nova.config.hidden_size)
    head_dim = int(nova.config.head_dim)
    rotary = text_model.rotary_emb

    print("\n[A] 同一上下文：事实 + 问题在同一个 prompt 里")
    facts_text = " ".join(f[1] for f in FACTS)
    for qlabel, qtext, want in QUESTIONS:
        text = render(tok, [{"role": "user", "content": facts_text + qtext}], add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        spans = [find_span(tok, text, f[1]) for f in FACTS]
        t = ids.shape[1]
        q_pre, k_pre, _v = capture_qkv(text_model, ids, (t - QUERY_LAST, t))
        q = rotate(q_pre.float(), t - QUERY_LAST, hidden, rotary)
        ks = [rotate(k_pre.float()[:, :, s0:s1, :], s0, hidden, rotary) for s0, s1 in spans]
        sc = mass(q, ks, head_dim)
        top = int(torch.argmax(sc))
        ok = "OK" if FACTS[top][0] == want else ("" if want is None else "MISS")
        print(f"    {qlabel:8s}" + "".join(f"{f[0]}={float(v):.3f} " for f, v in zip(FACTS, sc)) + f" -> {FACTS[top][0]} {ok}")

    print("\n[B] 注入到最前面：记忆在位置 0..Σm，问题在其后（距离 ~40+）")
    for qlabel, qtext, want in QUESTIONS:
        wtext = render(tok, [{"role": "user", "content": facts_text}], add_generation_prompt=False)
        wids = tok(wtext, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        wspans = [find_span(tok, wtext, f[1]) for f in FACTS]
        _q, wk, _v = capture_qkv(text_model, wids, (0, wids.shape[1]))
        qtext_full = render(tok, [{"role": "user", "content": qtext}], add_generation_prompt=True)
        qids = tok(qtext_full, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        tq = qids.shape[1]
        q_pre, _k2, _v2 = capture_qkv(text_model, qids, (tq - QUERY_LAST, tq))

        ks, starts, offset = [], [], 0
        for s0, s1 in wspans:
            ks.append(wk[:, :, s0:s1, :].float())
            starts.append(offset)
            offset += s1 - s0
        q = rotate(q_pre.float(), offset + tq - QUERY_LAST, hidden, rotary)
        rot_ks = [rotate(k, s, hidden, rotary) for k, s in zip(ks, starts)]
        sc = mass(q, rot_ks, head_dim)
        top = int(torch.argmax(sc))
        ok = "OK" if FACTS[top][0] == want else ("" if want is None else "MISS")
        print(f"    {qlabel:8s}" + "".join(f"{f[0]}={float(v):.3f} " for f, v in zip(FACTS, sc)) + f" -> {FACTS[top][0]} {ok}")

    print("\n[C] 每层分别看（注入到最前面）：命中层占比")
    for qlabel, qtext, want in QUESTIONS:
        if want is None:
            continue
        wtext = render(tok, [{"role": "user", "content": facts_text}], add_generation_prompt=False)
        wids = tok(wtext, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        wspans = [find_span(tok, wtext, f[1]) for f in FACTS]
        _q, wk, _v = capture_qkv(text_model, wids, (0, wids.shape[1]))
        qtext_full = render(tok, [{"role": "user", "content": qtext}], add_generation_prompt=True)
        qids = tok(qtext_full, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        tq = qids.shape[1]
        q_pre, _k2, _v2 = capture_qkv(text_model, qids, (tq - QUERY_LAST, tq))
        ks, starts, offset = [], [], 0
        for s0, s1 in wspans:
            ks.append(wk[:, :, s0:s1, :].float())
            starts.append(offset)
            offset += s1 - s0
        q = rotate(q_pre.float(), offset + tq - QUERY_LAST, hidden, rotary)
        rot_ks = [rotate(k, s, hidden, rotary) for k, s in zip(ks, starts)]
        per_layer = mass(q, rot_ks, head_dim, per_layer=True)  # [层, 3]
        hit = (per_layer.argmax(-1) == [f[0] for f in FACTS].index(want)).float().mean().item()
        print(f"    {qlabel:8s} 命中层占比 {hit:.2f}（{int(hit * per_layer.shape[0])}/{per_layer.shape[0]} 层）")
    print()


if __name__ == "__main__":
    main()
