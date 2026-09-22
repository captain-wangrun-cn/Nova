"""S3 · 双通路骨架的验收测试。

对应 [HANDOFF.md](../HANDOFF.md) 第六节的 4 条判据 + 显存验收：

| 测试 | 判据 |
|------|------|
| `test_forward_shapes` | 前向 shape 正确 |
| `test_gating_off_matches_baseline` | **门控关闭时 ≈ 基线**（最重要；两边都钉在 math 后端，测的是架构保真度） |
| `test_fused_attention_agrees_on_tokens` | 展平 KV 走融合内核时，贪心 token 与 math 路径一致 |
| `test_cross_isolation` | 交叉注意力隔离：切断一条通路不影响另一条 |
| `test_generate_20_tokens` | 生成 20 token 不崩 |
| `test_memory_budget` | 显存 < 7GB（D17） |

跑法：`& .\\.venv\\Scripts\\python.exe -m pytest tests -q`
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from nova.generate import greedy_generate

CROSS_INDICES = [0, 4, 8, 12, 16, 20]


def test_config_split(bundle):
    """层切分必须自洽：prefix + path + suffix == 总层数。"""
    nova, _, _ = bundle
    cfg = nova.config
    assert cfg.num_prefix_layers + cfg.num_path_layers + cfg.num_suffix_layers == cfg.num_hidden_layers
    assert cfg.path_layer_indices() == CROSS_INDICES
    # KV cache 槽位必须够，且各通路不重叠
    assert nova.model.num_cache_layers == 60
    slots = {nova.model.cache_slot_path(p, i) for p in range(cfg.num_paths) for i in range(cfg.num_path_layers)}
    assert len(slots) == cfg.num_paths * cfg.num_path_layers


def test_forward_shapes(bundle, prompt_ids):
    nova, _, _ = bundle
    with torch.inference_mode():
        logits = nova(input_ids=prompt_ids, cross_mode="off")
    assert logits.shape == (1, prompt_ids.shape[1], nova.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_gating_off_matches_baseline(bundle, prompt_ids):
    """**S3 最关键的一条。** 门控关闭时，Nova 必须与单通路基线逐位一致。

    prefill 与 decode 都要测：只测 prefill 会漏掉位置编码 / KV cache 槽位类的错误
    （开发过程中确实漏过一次 —— 见 [reports/s3-dual-path-skeleton.md](../reports/s3-dual-path-skeleton.md)）。

    ⚠️ **两边都钉在 math 后端**（`sdpa_kernel(MATH)`）。这条测的是**架构与权重的保真度**，
    不该受"SDPA 恰好 dispatch 到哪个内核"影响：本机没有 flash，GQA 会让 SDPA 退回 math，
    而 Nova 现在默认先展平 KV 走融合内核（见 [reports/long-context-attention.md](../reports/long-context-attention.md)）。
    不钉后端的话，这条会因为内核不同而假失败 —— 那是内核差异，不是实现错误。
    """
    nova, hf, _ = bundle
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        base = hf(input_ids=prompt_ids, use_cache=False).logits
        got = nova(input_ids=prompt_ids, cross_mode="off")
    assert torch.equal(base, got), f"prefill 不一致，max|diff|={(base.float()-got.float()).abs().max().item():.3e}"

    cache_a, cache_b = DynamicCache(), DynamicCache()
    cur = prompt_ids
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for step in range(8):
            la = hf(input_ids=cur, past_key_values=cache_a, use_cache=True).logits[:, -1]
            lb = nova(input_ids=cur, past_key_values=cache_b, cross_mode="off")[:, -1]
            assert torch.equal(la, lb), (
                f"decode 第 {step} 步不一致，max|diff|={(la.float()-lb.float()).abs().max().item():.3e}"
            )
            cur = la.argmax(dim=-1, keepdim=True)


def test_fused_attention_agrees_on_tokens(bundle, prompt_ids):
    """展平 KV（走融合内核）与 math 路径的 **token 级**一致率。

    两条路都正确（对手写 fp32 参考的相对误差相同，4.07e-04），原始 logits 会有 fp16 累加级差异，
    所以判据是**贪心 token 一致**，不是逐位 —— 与 lm_head 4-bit 当年同一标准（24/24、32/32）。
    实测 32/32。

    这里**不钉后端**：`False` 展平后 SDPA 自己会挑融合内核，`True` 则因为 GQA 头数不等而回退 math,
    两者正是要对比的两条真实路径。
    """
    nova, _, _ = bundle

    attns = [
        layer.self_attn
        for group in (list(nova.model.prefix_layers), *nova.model.path_layers, list(nova.model.suffix_layers))
        for layer in group
    ]
    outs = {}
    for flag in (False, True):  # False = 展平（默认，融合内核）；True = GQA 留 SDPA（回退 math）
        for attn in attns:
            attn.gqa_in_sdpa = flag
        with torch.inference_mode():
            outs[flag] = greedy_generate(nova, prompt_ids, max_new_tokens=16, cross_mode="off")
    for attn in attns:  # 复原默认
        attn.gqa_in_sdpa = False
    got, want = outs[False].tolist(), outs[True].tolist()
    n = min(len(got), len(want))
    same = sum(1 for a, b in zip(got[:n], want[:n]) if a == b)
    assert same == n, f"展平 KV 后 token 不一致：{same}/{n}"


def test_cross_isolation(bundle, prompt_ids):
    """交叉注意力隔离：门控关闭时，扰动通路 1 **不应**改变通路 0 的状态。"""
    nova, _, _ = bundle
    cfg = nova.config
    assert cfg.num_paths == 2

    def path0_state(perturb: bool):
        handle = None
        if perturb:
            def hook(_m, _i, out):
                return out + 1.0

            handle = nova.model.path_layers[1][0].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                _, paths = nova.model(input_ids=prompt_ids, cross_mode="off", return_paths=True)
            return paths[0].clone()
        finally:
            if handle is not None:
                handle.remove()

    clean = path0_state(perturb=False)
    dirty = path0_state(perturb=True)
    assert torch.equal(clean, dirty), "门控关闭时通路 0 被通路 1 影响了 —— 隔离失败"

    # 反过来：门控打开时，扰动**必须**传导过去，否则说明交叉模块根本没接上
    def path0_with_cross(perturb: bool):
        handle = None
        if perturb:
            def hook(_m, _i, out):
                return out + 1.0

            handle = nova.model.path_layers[1][0].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                _, paths = nova.model(input_ids=prompt_ids, cross_mode="on", return_paths=True)
            return paths[0].clone()
        finally:
            if handle is not None:
                handle.remove()

    a = path0_with_cross(perturb=False)
    b = path0_with_cross(perturb=True)
    assert not torch.equal(a, b), "门控打开时通路 0 完全没变 —— 交叉模块没接上"


def test_generate_20_tokens(bundle, prompt_ids):
    nova, _, _ = bundle
    eos = nova.config.__dict__.get("eos_token_id", 151645)
    out = greedy_generate(nova, prompt_ids, max_new_tokens=20, cross_mode="off", eos_token_id=eos)
    assert out.shape[0] == 1
    assert 1 <= out.shape[1] <= 20
    assert out.dtype == torch.long


def test_memory_budget(bundle, prompt_ids):
    """D17：显存预算 < 7GB。"""
    nova, _, _ = bundle
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        nova(input_ids=prompt_ids, cross_mode="on")
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    assert peak_gib < 7.0, f"峰值显存 {peak_gib:.2f} GiB 超出 D17 预算"


@pytest.mark.parametrize("impl", ["exact", "triton"])
def test_norm_impls_close(bundle, prompt_ids, impl):
    """两种 RMSNorm 实现的输出差异必须很小（triton 路径不是逐位一致，但要有界）。"""
    from nova.norm import LeanRMSNorm

    nova, _, _ = bundle
    ref = nova.model.norm
    alt = LeanRMSNorm.from_hf(_hf_norm_of(ref), impl).cuda()

    x = torch.randn(1, 4, nova.config.hidden_size, device="cuda", dtype=torch.float16)
    a, b = ref(x), alt(x)
    rel = ((a.float() - b.float()).abs().max() / a.abs().max()).item()
    if impl == "exact":
        assert torch.equal(a, b)
    else:
        assert rel < 5e-3, f"triton 与 exact 相对差异 {rel:.2e} 过大"


def _hf_norm_of(lean_norm):
    """把 LeanRMSNorm 包成一个能喂给 `from_hf` 的临时对象。"""
    from types import SimpleNamespace

    return SimpleNamespace(weight=lean_norm.weight.detach(), variance_epsilon=lean_norm.eps)
