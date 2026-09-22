"""S4 · 记忆最小实现的验收测试。

对应 [HANDOFF.md](../HANDOFF.md) 第七节与 [reports/s4-memory-min.md](../reports/s4-memory-min.md)：

| 测试 | 判据 |
|------|------|
| `test_schema_is_frozen` | 表示空间指纹稳定；同模型同 schema |
| `test_save_load_roundtrip` | 存盘 -> 读回逐位一致 |
| `test_load_rejects_other_schema` | 表示空间不匹配 **拒绝加载**（D09） |
| `test_load_rejects_other_model` | 模型指纹不匹配拒绝；`allow_foreign` 才放行 |
| `test_injection_reproduces_prefill` | 整段注入后的 KV cache 与真实 prefill **逐位一致** |
| `test_paths_are_identical` | 门控关闭时通路 0/1 捕获逐位相同（去重的依据） |
| `test_retrieval_top1` | 6 条候选、8 个问题（含改写）top-1 全中 |
| `test_memory_after_20_turns` | **写入 -> 20 轮无关对话 -> 提问答对**（验收标准） |
| `test_without_memory_cannot_answer` | 同一 prompt 不注入 -> 答不出（对照） |
| `test_swap_changes_answer` | 注入另一条 -> 答案跟着变（因果证据） |
| `test_memory_budget` | 峰值显存 < 7GB（D17） |

跑法：`& .\\.venv\\Scripts\\python.exe -m pytest tests -q`
"""

from __future__ import annotations

import pytest
import torch

from chatfmt import find_span, render
from nova.decode import GraphDecoder
from nova.memory import (
    MemoryFormatError,
    MemoryItem,
    MemorySchema,
    MemorySchemaMismatch,
    MemorySession,
    MemoryStore,
    capture_kv,
    model_fingerprint,
)
from s4_memory_demo import FACTS, FILLER, QUESTIONS, build_filler

MAX_LEN = 1024
TURNS = 20


# ---------------------------------------------------------------- 工具


@pytest.fixture(scope="module")
def facts_ids(bundle):
    """写入轮：第 1 轮里那 6 件事（整段 prompt + 每条的 token 区间）。"""
    _, _, tok = bundle
    text = render(tok, [{"role": "user", "content": " ".join(f[1] for f in FACTS)}], add_generation_prompt=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    spans = [find_span(tok, text, sentence) for _, sentence in FACTS]
    return text, ids, spans


@pytest.fixture(scope="module")
def loaded_store(bundle, facts_ids):
    """写入 6 条记忆（模块内共享，避免每条测试都重写一遍）。"""
    nova, _, _ = bundle
    _text, ids, spans = facts_ids
    store = MemoryStore.for_model(nova)
    for (label, _sentence), span in zip(FACTS, spans):
        store.write(nova, ids, span, label=label)
    return store


@pytest.fixture(scope="module")
def ask_prompt(bundle):
    """20 轮无关历史 + 每个问题的完整 prompt（历史是前缀，当前轮单独切出来）。"""
    _, _, tok = bundle
    filler = build_filler(TURNS)
    hist_ids = tok(render(tok, filler, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    out = {}
    for qtext, want, accept in QUESTIONS:
        full_text = render(tok, filler + [{"role": "user", "content": qtext}], add_generation_prompt=True)
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(hist_ids)] == hist_ids, "历史不是前缀（chat 模板变了？）"
        out[qtext] = {
            "hist": torch.tensor([hist_ids], device="cuda"),
            "cur": torch.tensor([full_ids[len(hist_ids) :]], device="cuda"),
            "span": find_span(tok, full_text, qtext),
            "want": want,
            "accept": accept,
        }
    return out


def run_condition(bundle, store, ask_prompt, qtext, cond, max_new=32):
    """跑一个条件（auto / off / swap），返回 (答案文本, info)。"""
    from chatfmt import EOS_ID, EOS_ID_ALT

    nova, _, tok = bundle
    item = ask_prompt[qtext]
    force = None
    if cond == "swap":
        force = ([f[0] for f in FACTS].index(item["want"]) + 1) % len(FACTS)
    sess = MemorySession(store, nova, max_len=MAX_LEN)
    info = sess.prefill(
        item["hist"], item["cur"],
        use_memory=(cond != "off"),
        force_index=force,
        query_span=item["span"],
    )
    out = sess.generate(max_new, {EOS_ID, EOS_ID_ALT})
    return tok.decode(out, skip_special_tokens=True), info


# ---------------------------------------------------------------- 表示空间（D07 / D09）


def test_schema_is_frozen(bundle):
    nova, _, _ = bundle
    schema = MemorySchema.for_model(nova)
    assert schema.num_cache_layers == nova.model.num_cache_layers == 60
    assert schema.num_paths == 2 and schema.num_kv_heads == 8 and schema.head_dim == 128
    assert schema.digest == MemorySchema.for_model(nova).digest, "同模型必须得到同一个 schema 指纹"
    assert model_fingerprint(nova) == model_fingerprint(nova)


def test_save_load_roundtrip(bundle, loaded_store, tmp_path):
    nova, _, _ = bundle
    path = loaded_store.save(tmp_path / "mem.safetensors")
    assert path.stat().st_size > 0
    back = MemoryStore.load(path, schema=MemorySchema.for_model(nova), fingerprint=model_fingerprint(nova))
    assert len(back) == len(loaded_store)
    for a, b in zip(loaded_store.items, back.items):
        assert torch.equal(a.k, b.k) and torch.equal(a.v, b.v)
        assert a.label == b.label and a.n_tokens == b.n_tokens
    assert not back.foreign


def test_load_rejects_other_schema(bundle, loaded_store, tmp_path):
    """表示空间不一致必须**拒绝加载** —— 宁可"想不起来"，也不要读串味（D09）。"""
    nova, _, _ = bundle
    path = loaded_store.save(tmp_path / "mem.safetensors")
    other = MemorySchema.for_model(nova)
    other = MemorySchema(**{**other.to_dict(), "num_paths": 1})
    with pytest.raises(MemorySchemaMismatch):
        MemoryStore.load(path, schema=other)


def test_load_rejects_other_model(bundle, loaded_store, tmp_path):
    nova, _, _ = bundle
    path = loaded_store.save(tmp_path / "mem.safetensors")
    with pytest.raises(MemorySchemaMismatch):
        MemoryStore.load(path, schema=MemorySchema.for_model(nova), fingerprint="deadbeef" * 4)
    back = MemoryStore.load(
        path, schema=MemorySchema.for_model(nova), fingerprint="deadbeef" * 4, allow_foreign=True
    )
    assert back.foreign and len(back) == len(loaded_store)


def test_load_rejects_non_memory_file(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "not-memory.safetensors"
    save_file({"x": torch.zeros(2)}, str(path))
    with pytest.raises(MemoryFormatError):
        MemoryStore.load(path)


# ---------------------------------------------------------------- 写入 / 注入


def test_injection_reproduces_prefill(bundle, facts_ids):
    """**最关键的一条机制测试。**

    把整段 prompt 抓成记忆、再按位置注入，KV cache 必须与"真实 prefill"**逐位一致** ——
    同时验证了：捕获抓的是 RoPE 之前的值、槽位映射没错、注入时按新位置重新旋转是对的。
    """
    nova, _, _ = bundle
    _text, ids, _spans = facts_ids
    dec_ref = GraphDecoder(nova, max_len=256)
    dec_ref.prefill(ids)

    k, v = capture_kv(nova.model, ids, (0, ids.shape[1]))
    dec_mem = GraphDecoder(nova, max_len=256)
    store = MemoryStore.for_model(nova)
    n_prefix = store.inject(dec_mem.cache, [MemoryItem(k, v)], nova.model.rotary_emb)
    assert n_prefix == ids.shape[1]

    worst = 0.0
    for slot in range(nova.model.num_cache_layers):
        for a, b in (
            (dec_ref.cache.key_cache[slot], dec_mem.cache.key_cache[slot]),
            (dec_ref.cache.value_cache[slot], dec_mem.cache.value_cache[slot]),
        ):
            worst = max(worst, (a.float() - b.float()).abs().max().item())
    assert worst == 0.0, f"注入后的 cache 与真实 prefill 有差异：max|diff|={worst:.3e}"


def test_paths_are_identical(bundle, facts_ids):
    """门控关闭时两条通路的捕获逐位相同 —— 这是"文件里可以只存一份"的依据。"""
    nova, _, _ = bundle
    _text, ids, _spans = facts_ids
    k, _v = capture_kv(nova.model, ids, (0, ids.shape[1]))
    cfg = nova.config
    for i in range(cfg.num_prefix_layers, cfg.num_prefix_layers + cfg.num_path_layers):
        assert torch.equal(k[i], k[i + cfg.num_path_layers]), f"槽位 {i} 与 {i + cfg.num_path_layers} 不同"


# ---------------------------------------------------------------- 联想检索


def test_retrieval_top1(bundle, loaded_store, ask_prompt):
    """6 条候选、8 个问题（其中 3 个是改写问法）—— 实测 top-1 全中，这里锁死。

    走 `MemorySession.rank`，与真实检索**同一条路径**（含把 `query_span` 截成末尾
    `query_last` 个 token）。教训：测试里另手搓一遍打分就会漂移 —— 首版忘了截，
    命中率立刻从 8/8 掉到 7/8。
    """
    nova, _, _ = bundle
    hits, detail = 0, []
    sess = MemorySession(loaded_store, nova, max_len=MAX_LEN)
    for qtext, want, _accept in QUESTIONS:
        item = ask_prompt[qtext]
        scores = sess.rank(item["hist"], item["cur"], item["span"])
        top = loaded_store.items[int(scores.argmax())].label
        hits += int(top == want)
        detail.append(f"{qtext}->{top}{'✓' if top == want else '✗'}")
    assert hits == len(QUESTIONS), "检索命中率下降：" + " ".join(detail)


# ---------------------------------------------------------------- 验收标准


def test_memory_after_20_turns(bundle, loaded_store, ask_prompt):
    """**S4 的验收标准**：第 1 轮写入 -> 20 轮无关对话 -> 提问答对。"""
    for qtext in ("我今天穿的裙子是什么颜色的？", "我养的猫叫什么名字？"):
        text, info = run_condition(bundle, loaded_store, ask_prompt, qtext, "auto")
        assert info.used_memory, "没取回任何记忆"
        accept = ask_prompt[qtext]["accept"]
        assert all(k in text for k in accept), f"答案里没有 {accept}：{text!r}"


def test_without_memory_cannot_answer(bundle, loaded_store, ask_prompt):
    """对照：同一个 prompt、同一条历史，不注入记忆就答不出来。"""
    qtext = "我今天穿的裙子是什么颜色的？"
    text, info = run_condition(bundle, loaded_store, ask_prompt, qtext, "off")
    assert not info.used_memory
    assert not all(k in text for k in ask_prompt[qtext]["accept"]), f"没注入也答对了？{text!r}"


def test_swap_changes_answer(bundle, loaded_store, ask_prompt):
    """因果证据：故意注入另一条记忆，答案就不再是原来那条。"""
    qtext = "我养的猫叫什么名字？"
    text, info = run_condition(bundle, loaded_store, ask_prompt, qtext, "swap")
    assert info.forced is not None and info.prefix_len > 0
    assert "雷纳德" not in text, f"注入了别的记忆却还是答出了原答案：{text!r}"


def test_memory_budget(bundle, loaded_store, ask_prompt):
    """D17：显存预算 < 7GB（含 1024 槽位的 KV cache 与记忆）。"""
    torch.cuda.reset_peak_memory_stats()
    run_condition(bundle, loaded_store, ask_prompt, "我养的猫叫什么名字？", "auto", max_new=8)
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    assert peak_gib < 7.0, f"峰值显存 {peak_gib:.2f} GiB 超出 D17 预算"
