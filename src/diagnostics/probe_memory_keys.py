"""S4 诊断 ④：检索键该用什么。

诊断 ③ 的结论：模型自己的 K 只在"同一上下文"里能区分内容；跨上下文（问题在**看不到事实**的
语境里算出来）时区分度很差（top-1 5/15）。而 docs/03 本来就是设计**单独的 key 张量**
（`memory_keys: [128, 512]`），所以这里比几种候选键：

| 键 | 含义 |
|----|------|
| `k`  | 模型自己的注意力键（诊断 ③ 用的） |
| `hid_last` | 最后一层 hidden state，在区间上取平均 |
| `hid_mid`  | 中段（第 20 个槽位）hidden state 平均 |
| `embed`    | 输入 embedding（词表向量）平均 —— 本质是词面匹配 |

打分统一为**对候选集去均值后的余弦**（去均值 = 减掉"所有记忆共有的公共分量"）。

跑法：`& .\.venv\Scripts\python.exe src\diagnostics\probe_memory_keys.py`
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
from nova.loader import load_nova  # noqa: E402

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


def collect(text_model, ids, spans, hooks_out):
    """跑一次前向，返回 {(键名): [候选数, m, D] 的张量列表}。"""
    for h in hooks_out.values():
        h["buf"] = None
    with torch.inference_mode():
        text_model(input_ids=ids, cross_mode="off")
    embed = text_model.embed_tokens(ids)[0]  # [t, D]
    out = {}
    for name, buf in hooks_out.items():
        hidden = buf["buf"][0]  # [t, D]
        out[name] = [hidden[s0:s1].float() for s0, s1 in spans]
    out["embed"] = [embed[s0:s1].float() for s0, s1 in spans]
    return out


def centered_cos(query, cands):
    """候选集去均值后的余弦（query [D]，cands [n, m, D]）。"""
    flat = torch.cat(cands, dim=0)  # [Σm, D]
    mu = flat.mean(dim=0, keepdim=True)
    q = (query - mu).flatten()
    out = []
    for c in cands:
        sim = torch.nn.functional.cosine_similarity(c - mu, q.unsqueeze(0), dim=-1)  # [m]
        out.append(sim.amax())  # 区间里最像的那个 token
    return torch.stack(out)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nova, _hf, tok = load_nova(norm_impl="exact")
    text_model = nova.model

    # 抓 hidden state 的钩子：最后一层（norm 之后）与中段某一层
    hooks = {
        "hid_last": {"buf": None, "handle": text_model.norm.register_forward_hook(
            lambda m, i, o: hooks["hid_last"].__setitem__("buf", o.detach()))},
        "hid_mid": {"buf": None, "handle": text_model.path_layers[0][14].register_forward_hook(
            lambda m, i, o: hooks["hid_mid"].__setitem__("buf", o.detach()))},
    }

    wtext = render(tok, [{"role": "user", "content": " ".join(f[1] for f in FACTS)}], add_generation_prompt=False)
    wids = tok(wtext, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    wspans = [find_span(tok, wtext, f[1]) for f in FACTS]
    keys = collect(text_model, wids, wspans, hooks)
    print(f"写入轮 {wids.shape[1]} token，区间 {wspans}")

    results = {name: [] for name in keys}
    for qlabel, qtext, want in QUESTIONS:
        qp = render(tok, [{"role": "user", "content": qtext}], add_generation_prompt=True)
        qids = tok(qp, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        t = qids.shape[1]
        qkeys = collect(text_model, qids, [(t - QUERY_LAST, t)], hooks)
        line = []
        for name, cands in keys.items():
            q = qkeys[name][0].mean(dim=0)  # 查询侧：最后 4 个 token 平均
            sc = centered_cos(q, cands)
            top = FACTS[int(sc.argmax())][0]
            results[name].append((qlabel, want, top, [round(float(v), 3) for v in sc]))
            line.append(f"{name}:{top}{'✓' if top == want else ('·' if want is None else '✗')}")
        print(f"    {qlabel:8s} " + "  ".join(line))

    print("\n命中统计（3 个可判定的问题）")
    for name, rows in results.items():
        ok = sum(1 for _, want, top, _ in rows if want is not None and want == top)
        n = sum(1 for _, want, _, _ in rows if want is not None)
        print(f"    {name:9s} {ok}/{n}   " + " | ".join(f"{lab}:{sc}" for lab, want, top, sc in rows if want))

    for h in hooks.values():
        h["handle"].remove()
    print()


if __name__ == "__main__":
    main()
