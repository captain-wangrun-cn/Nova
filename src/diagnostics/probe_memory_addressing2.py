"""S4 诊断 ③：寻址信号到底在哪些 (层, 头) 上。

诊断 ② 的结论：把所有层、所有头平均之后，打分顺序在所有问题下都一样（红裙子恒赢），
怀疑是"少数检索头 + 大量噪声头"的经典问题。这里做两件事：

1. **内容 vs 位置**：把三条事实在 prompt 里换顺序，看赢家跟内容还是跟位置。
2. **逐 (层, 头) 扫描**：找出"三个问题全部答对"的 (层, 头)，看信号是不是集中在少数头上。

跑法：`& .\.venv\Scripts\python.exe src\diagnostics\probe_memory_addressing2.py`
"""

from __future__ import annotations

import itertools
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

FACTS = {
    "红裙子": "我今天换了一条红裙子，是上周在巴黎买的。",
    "橘猫": "我养的猫叫雷纳德，是一只橘猫。",
    "花瓶": "我把阳台上的蓝色花瓶打碎了。",
}
QUESTIONS = {
    "裙子颜色": "我今天穿的裙子是什么颜色的？",
    "猫名字": "我养的猫叫什么名字？",
    "打碎的东西": "我把什么东西打碎了？",
}
QUERY_LAST = 4


def rotate(x_pre, start, hidden_size, rotary):
    cos, sin = _rope_cos_sin(rotary, x_pre, start, x_pre.shape[2], hidden_size)
    return apply_rotary_pos_emb(x_pre, x_pre, cos, sin)[0]


def grouped_logits(q, k, head_dim):
    groups = q.shape[1] // k.shape[1]
    qg = q.view(q.shape[0], k.shape[1], groups, q.shape[2], q.shape[3])
    out = torch.einsum("lhgtc,lhmc->lhgtm", qg, k) * (head_dim ** -0.5)
    return out.reshape(q.shape[0], q.shape[1], q.shape[2], k.shape[2])


def scores_for(nova, tok, order, question, hidden, head_dim, rotary, gen_prompt=True):
    """[层, Q头, 候选数]：查询位置对每条事实的 max Q·K（按真实位置旋转）。"""
    facts_text = " ".join(FACTS[k] for k in order)
    text = render(tok, [{"role": "user", "content": facts_text + question}], add_generation_prompt=gen_prompt)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    spans = [find_span(tok, text, FACTS[k]) for k in order]
    t = ids.shape[1]
    # ⚠️ 必须抓**整段**：查询要 Q（最后 4 个位置），事实要 K（各自的区间）
    q_pre, k_pre, _v = capture_qkv(nova.model, ids, (0, t))
    q = rotate(q_pre.float()[:, :, t - QUERY_LAST :, :], t - QUERY_LAST, hidden, rotary)
    out = []
    for s0, s1 in spans:
        k = rotate(k_pre.float()[:, :, s0:s1, :], s0, hidden, rotary)
        out.append(grouped_logits(q, k, head_dim).amax(dim=-1).mean(dim=-1))  # [层, 头]
    return torch.stack(out, dim=-1)  # [层, 头, 候选数]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nova, _hf, tok = load_nova(norm_impl="exact")
    hidden = int(nova.config.hidden_size)
    head_dim = int(nova.config.head_dim)
    rotary = nova.model.rotary_emb

    print("\n[1] 内容 vs 位置：换事实顺序，看赢家跟谁走（查询 = 问题最后 4 个 token，不含生成提示）")
    for order in itertools.permutations(FACTS, 3):
        row = []
        for qlabel, qtext in QUESTIONS.items():
            sc = scores_for(nova, tok, order, qtext, hidden, head_dim, rotary, gen_prompt=False)
            mean_sc = sc.mean(dim=(0, 1))
            row.append(f"{qlabel}->{order[int(mean_sc.argmax())]}")
        print(f"    顺序 {'/'.join(order):20s} " + "  ".join(row))

    print("\n[2] 逐 (层, 头) 扫描：哪些头三个问题全对？（事实顺序固定）")
    order = tuple(FACTS)
    per_q = {}
    for qlabel, qtext in QUESTIONS.items():
        per_q[qlabel] = scores_for(nova, tok, order, qtext, hidden, head_dim, rotary, gen_prompt=False)
    n_layers, n_heads = next(iter(per_q.values())).shape[:2]
    good = []
    for l in range(n_layers):
        for h in range(n_heads):
            ok = True
            for qi, (qlabel, _) in enumerate(QUESTIONS.items()):
                sc = per_q[qlabel][l, h]
                ok &= order[int(sc.argmax())] == list(QUESTIONS)[qi].replace("裙子颜色", "红裙子").replace("猫名字", "橘猫").replace("打碎的东西", "花瓶")
            if ok:
                good.append((l, h))
    print(f"    全对的头：{len(good)}/{n_layers * n_heads}")
    print(f"    {good[:40]}")
    if good:
        idx = torch.tensor(good)
        acc = []
        for qlabel in QUESTIONS:
            sub = per_q[qlabel][idx[:, 0], idx[:, 1]]  # [n_good, 3]
            acc.append(order[int(sub.mean(0).argmax())])
        print(f"    只用这些头的平均打分 -> {acc}")

    print("\n[3] 每层的最优头（只取该层里三个问题全对的头数）")
    per_layer_good = []
    for l in range(n_layers):
        cnt = sum(1 for (ll, _h) in good if ll == l)
        per_layer_good.append(cnt)
    print("    " + " ".join(f"{l}:{c}" for l, c in enumerate(per_layer_good) if c))
    print()


if __name__ == "__main__":
    main()
