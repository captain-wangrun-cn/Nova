"""速度路径 ① · lm_head 4-bit 副本的验收测试。

对应 [reports/speed-path1-nf4-gemv.md](../reports/speed-path1-nf4-gemv.md) 第四节：

| 测试 | 判据 |
|------|------|
| `test_lm_head_forward_uses_embed_tokens` | 没装 4-bit 副本时走 fp16 `embed_tokens.weight`（**锁死无限递归 bug**） |
| `test_lm_head4_greedy_matches_fp16` | 装 4-bit 副本后 **24/24 个贪心 token 与 fp16 一致** |
| `test_lm_head4_logits_close` | logits 的 mean\\|diff\\| < 0.5（4-bit 的正常量化误差） |
| `test_lm_head4_memory_cost` | 额外显存 < 0.30 GiB |
| `test_convert_to_nf4_skips_lm_head4` | `convert_to_nf4` **不碰** `lm_head4`（自写 kernel 在 lm_head 形状上更慢） |

跑法：`& .\\.venv\\Scripts\\python.exe -m pytest tests -q`
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

N_NEW = 24


def _small_bnb(n: int = 64, k: int = 64):
    """造一个小尺寸、**已完成量化**的 bnb `Linear4bit`（给 `convert_to_nf4` 的测试用）。"""
    import bitsandbytes as bnb

    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.05
    lin = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float16,
                            quant_type="nf4", compress_statistics=True).to("cuda")
    with torch.no_grad():
        lin.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                       compress_statistics=True, quant_storage=torch.float16)
        lin.weight = lin.weight.to("cuda")
        _ = lin(torch.zeros(1, 1, k, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
    return lin


def _greedy(nova, prompt_ids, n):
    """返回 `(tokens [1, n], 最后一步的 logits)`。"""
    cache = DynamicCache()
    ids = []
    logits = None
    with torch.inference_mode():
        logits = nova(input_ids=prompt_ids, past_key_values=cache, cross_mode="off")
        cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids.append(cur.clone())
        for _ in range(n - 1):
            logits = nova(input_ids=cur, past_key_values=cache, cross_mode="off")
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
            ids.append(cur.clone())
    return torch.cat(ids, dim=1), logits


def test_lm_head_forward_uses_embed_tokens(bundle):
    """`lm_head4 is None` 时必须真的走 fp16 `embed_tokens.weight`。

    **锁死开发中踩到的 bug**：一次全局字符串替换把 fallback 那行也换成了
    `self.lm_head_forward(hidden)`，导致无限递归 —— 只有这条断言能抓住它。
    """
    nova, _, _ = bundle
    assert nova.lm_head4 is None, "前置条件：本测试要求未装 4-bit lm_head"

    W = nova.model.embed_tokens.weight
    torch.manual_seed(0)
    hidden = torch.randn(1, 1, W.shape[1], device="cuda", dtype=torch.float16)
    got = nova.lm_head_forward(hidden)
    assert torch.equal(got, F.linear(hidden, W))


def test_lm_head4_greedy_matches_fp16(bundle, prompt_ids):
    """**主验收**：4-bit lm_head 的 24 个贪心 token 与 fp16 逐个相同。"""
    nova, _, _ = bundle
    assert nova.lm_head4 is None, "前置条件：本测试要求未装 4-bit lm_head"

    ref, _ = _greedy(nova, prompt_ids, N_NEW)

    from nova.loader import enable_lm_head_4bit

    try:
        enable_lm_head_4bit(nova)
        got, _ = _greedy(nova, prompt_ids, N_NEW)
    finally:
        nova.lm_head4 = None  # 还原，别污染同会话的其它测试

    n_same = int((got == ref).sum().item())
    assert n_same == N_NEW, f"4-bit lm_head 只有 {n_same}/{N_NEW} 个 token 与 fp16 一致"


def test_lm_head4_logits_close(bundle, prompt_ids):
    """4-bit 量化误差的量级：mean|diff| 应远小于 logits 自身的尺度。"""
    nova, _, _ = bundle
    assert nova.lm_head4 is None, "前置条件：本测试要求未装 4-bit lm_head"

    _, ref_logits = _greedy(nova, prompt_ids, 4)

    from nova.loader import enable_lm_head_4bit

    try:
        enable_lm_head_4bit(nova)
        _, got_logits = _greedy(nova, prompt_ids, 4)
    finally:
        nova.lm_head4 = None

    d = (ref_logits.float() - got_logits.float()).abs()
    assert d.mean().item() < 0.5, f"mean|diff| = {d.mean().item():.4f} 偏大"
    assert torch.equal(ref_logits.argmax(-1), got_logits.argmax(-1))


def test_lm_head4_memory_cost(bundle):
    """D17：这份额外副本只值约 0.19 GiB。"""
    nova, _, _ = bundle
    assert nova.lm_head4 is None, "前置条件：本测试要求未装 4-bit lm_head"

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    from nova.loader import enable_lm_head_4bit

    try:
        enable_lm_head_4bit(nova)
        torch.cuda.synchronize()
        delta_gib = (torch.cuda.memory_allocated() - before) / 1024**3
    finally:
        nova.lm_head4 = None

    assert 0.0 < delta_gib < 0.30, f"lm_head4 占用 {delta_gib:.3f} GiB，超出预期"


def test_convert_to_nf4_skips_lm_head4():
    """`convert_to_nf4` 必须跳过 `lm_head4`（实测自写 kernel 在该形状上慢 1.34x）。

    用**合成的小模块树**验证 —— 不能在会话级 `bundle` 上真跑 `convert_to_nf4`，
    那会把整棵 Nova 永久换掉，污染同会话的其它测试。
    """
    from nova.kernels import triton_available

    if not triton_available():
        return

    import bitsandbytes as bnb
    import torch.nn as nn

    from nova.quant import NF4Linear, convert_to_nf4

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _small_bnb()
            self.lm_head4 = _small_bnb()

    m = Holder()
    n_done = convert_to_nf4(m)

    assert n_done == 1, f"应只替换 1 个，实际 {n_done} 个"
    assert isinstance(m.inner, NF4Linear)
    assert isinstance(m.lm_head4, bnb.nn.Linear4bit), "lm_head4 被误换成了 NF4Linear"
