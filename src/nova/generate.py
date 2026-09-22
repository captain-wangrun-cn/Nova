"""最小贪心解码。

S3 只需要"能连续生成不崩"，**不是性能路径** —— 速度问题见 D27，独立立项。
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    cross_mode: str = "off",
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """返回**新生成**的 token（不含 prompt）。"""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须 >= 1")

    cache = DynamicCache()
    logits = model(input_ids=input_ids, past_key_values=cache, cross_mode=cross_mode)
    cur = logits[:, -1].argmax(dim=-1, keepdim=True)
    out = [cur]

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and bool((cur == eos_token_id).all()):
            break
        logits = model(input_ids=cur, past_key_values=cache, cross_mode=cross_mode)
        cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        out.append(cur)

    return torch.cat(out, dim=1)
