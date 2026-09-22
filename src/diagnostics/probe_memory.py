"""S4 诊断：记忆的捕获 / 注入 / 联想检索能不能用。

三件事：
1. **捕获正确性**：把整段 prompt 抓成记忆再注入，KV cache 必须与"真实 prefill"**逐位一致**。
2. **通路 0/1 是否相同**（门控关闭时理论上应逐位相同）—— 决定要不要去重。
3. **检索区分度**：N 条记忆 × M 个问题的相似度矩阵，看 top-1 对不对。

跑法：
```powershell
& .\.venv\Scripts\python.exe src\diagnostics\probe_memory.py
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
from nova.decode import GraphDecoder  # noqa: E402
from nova.loader import load_nova  # noqa: E402
from nova.memory import MemoryStore, capture_kv, query_vectors  # noqa: E402

FACTS = [
    ("红裙子", "我今天换了一条红裙子，是上周在巴黎买的。"),
    ("橘猫", "我养的猫叫雷纳德，是一只橘猫。"),
    ("花瓶", "我把阳台上的蓝色花瓶打碎了。"),
]
QUESTIONS = [
    ("裙子颜色", "我今天穿的裙子是什么颜色的？"),
    ("猫名字", "我养的猫叫什么名字？"),
    ("打碎的东西", "我把什么东西打碎了？"),
    ("无关问题", "请用一句话解释什么是潮汐。"),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nova, _hf, tok = load_nova(norm_impl="exact")
    text_model = nova.model

    turn = " ".join(f[1] for f in FACTS)
    prompt = render(tok, [{"role": "user", "content": turn}], add_generation_prompt=False)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    print(f"写入轮 prompt：{ids.shape[1]} token")

    store = MemoryStore.for_model(nova)
    for label, sentence in FACTS:
        span = find_span(tok, prompt, sentence)
        idx = store.write(nova, ids, span, label=label)
        print(f"  写入 [{idx}] {label:6s} span={span} -> {store.items[-1].n_tokens} token")

    print()
    print(store)
    for line in store.describe():
        print("  " + line)

    # ---- 1. 捕获 / 注入的正确性：整段 prompt 当记忆，注入后 cache 应与真实 prefill 逐位一致 ----
    dec_ref = GraphDecoder(nova, max_len=256)
    dec_ref.prefill(ids)
    whole_k, whole_v = capture_kv(text_model, ids, (0, ids.shape[1]))

    from nova.memory import MemoryItem

    dec_mem = GraphDecoder(nova, max_len=256)
    n_prefix = store.inject(dec_mem.cache, [MemoryItem(whole_k, whole_v)], text_model.rotary_emb)
    print(f"\n[1] 整段注入：前缀 {n_prefix} token（= prompt 长度 {ids.shape[1]}）")
    worst = 0.0
    for slot in range(text_model.num_cache_layers):
        for name, a, b in (
            ("k", dec_ref.cache.key_cache[slot], dec_mem.cache.key_cache[slot]),
            ("v", dec_ref.cache.value_cache[slot], dec_mem.cache.value_cache[slot]),
        ):
            d = (a.float() - b.float()).abs().max().item()
            worst = max(worst, d)
    print(f"    cache 与真实 prefill 的最大差异 = {worst:.3e}  ({'逐位一致' if worst == 0 else '不一致'})")

    # ---- 2. 通路 0/1 是否逐位相同 ----
    p0, p1 = nova.config.num_prefix_layers, nova.config.num_path_layers
    same = True
    for i in range(p0, p0 + p1):
        same &= bool(torch.equal(whole_k[i], whole_k[i + p1]))
    print(f"[2] 通路 0 与通路 1 的捕获是否逐位相同：{same}")

    # ---- 3. 检索区分度 ----
    expect = {"裙子颜色": "红裙子", "猫名字": "橘猫", "打碎的东西": "花瓶", "无关问题": None}
    for mode in ("mass", "logit"):
        print(f"\n[3] 打分矩阵 mode={mode}（行 = 问题，列 = 记忆项）")
        print("    " + "n_last  " + "".join(f"{f[0]:>10s}" for f in FACTS) + "     top-1")
        n_ok = n_tot = 0
        for qlabel, qtext in QUESTIONS:
            qp = render(tok, [{"role": "user", "content": qtext}], add_generation_prompt=True)
            qids = tok(qp, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            for n_last in (1, 4, 8, 12, 16):
                q = query_vectors(text_model, qids, n_last=n_last)
                sc = store.scores(q, mode=mode)
                vals = "".join(f"{float(v):>10.4f}" for v in sc)
                top = int(torch.argmax(sc))
                want = expect[qlabel]
                if want is not None:
                    n_tot += 1
                    n_ok += int(store.items[top].label == want)
                mark = "OK" if store.items[top].label == want else ("" if want else "(无关)")
                print(f"    {n_last:>5d}  {qlabel:8s}{vals}   -> {store.items[top].label} {mark}")
        if n_tot:
            print(f"    top-1 命中 {n_ok}/{n_tot}")
    print()


if __name__ == "__main__":
    main()
