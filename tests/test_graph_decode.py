"""S3 · CUDA Graph 解码的验收测试。

对应 [reports/s3-graph-decode.md](../reports/s3-graph-decode.md)：

| 测试 | 判据 |
|------|------|
| `test_static_cache_writes_in_place` | 静态 KV cache 的写入位置正确 |
| `test_prefill_does_not_duplicate_last_token` | **prefill 之后输入缓冲是"第一个生成 token"，不是最后一个 prompt token** |
| `test_graph_decode_matches_eager` | 图解码与 eager 解码 **token 逐个相同** |
| `test_graph_memory_budget` | 显存 < 7GB（D17） |

跑法：`& .\\.venv\\Scripts\\python.exe -m pytest tests -q`
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from nova.cache import StaticKVCache
from nova.decode import GraphDecoder

MAX_LEN = 128
N_NEW = 12


def test_static_cache_writes_in_place():
    """`update()` 必须把 K/V 写进 `pos` 指向的槽位，且返回**整条**定长 cache。"""
    cache = StaticKVCache(num_slots=2, num_kv_heads=2, head_dim=4, max_len=8)
    cache.pos.fill_(3)
    key = torch.arange(1 * 2 * 1 * 4, dtype=torch.float16, device="cuda").view(1, 2, 1, 4)
    value = key + 100
    k_out, v_out = cache.update(key, value, layer_idx=1)

    assert k_out.shape == (1, 2, 8, 4) and v_out.shape == (1, 2, 8, 4)
    assert torch.equal(k_out[:, :, 3:4, :], key)
    assert torch.equal(v_out[:, :, 3:4, :], value)
    # 其它槽位必须保持 0（靠 attention mask 屏蔽，而不是靠内容）
    assert k_out[:, :, :3, :].abs().sum().item() == 0
    assert k_out[:, :, 4:, :].abs().sum().item() == 0
    # 槽位 0 不能被写
    assert cache.key_cache[0].abs().sum().item() == 0


def test_prefill_does_not_duplicate_last_token(bundle, prompt_ids):
    """**锁死开发中踩到的 bug。**

    `prefill()` 已把整个 prompt 写进 cache（位置 0..n-1），`pos` 指向 n。
    若此时把"最后一个 prompt token"填进输入缓冲，图的第一步会在位置 n 上
    **把它再算一遍** —— 结果看着像模像样，但序列是错的。
    """
    nova, _, _ = bundle
    dec = GraphDecoder(nova, max_len=MAX_LEN)
    logits = dec.prefill(prompt_ids)

    expected = logits[:, -1].argmax(-1).view(1, 1)
    assert torch.equal(dec.input_ids, expected), "prefill 后输入缓冲应是第一个生成 token"
    assert not torch.equal(dec.input_ids, prompt_ids[:, -1:]), "输入缓冲仍是最后一个 prompt token"
    assert int(dec.cache.pos.item()) == prompt_ids.shape[1]


def test_graph_decode_matches_eager(bundle, prompt_ids):
    """图解码（CUDA Graph + 静态 cache + 加性 mask）必须与 eager 解码 token 逐个相同。

    ⚠️ 必须在 `inference_mode` 内 replay —— 图内对输入缓冲做了原地写入。
    """
    nova, _, _ = bundle
    n = N_NEW

    ref_ids = []
    cache = DynamicCache()
    with torch.inference_mode():
        logits = nova(input_ids=prompt_ids, past_key_values=cache, cross_mode="off")
        cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        ref_ids.append(cur.clone())
        for _ in range(n - 1):
            logits = nova(input_ids=cur, past_key_values=cache, cross_mode="off")
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
            ref_ids.append(cur.clone())
    ref = torch.cat(ref_ids, dim=1)

    dec = GraphDecoder(nova, max_len=MAX_LEN)
    dec.prefill(prompt_ids)
    dec.capture()
    got = dec.generate(prompt_ids, n)

    assert got.shape == ref.shape
    n_same = int((got == ref).sum().item())
    assert n_same == n, f"图解码与 eager 只有 {n_same}/{n} 个 token 相同"


def test_graph_memory_budget(bundle, prompt_ids):
    """D17：显存预算 < 7GB。"""
    nova, _, _ = bundle
    torch.cuda.reset_peak_memory_stats()
    dec = GraphDecoder(nova, max_len=MAX_LEN)
    dec.prefill(prompt_ids)
    dec.capture()
    for _ in range(4):
        dec.step()
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    assert peak_gib < 7.0, f"峰值显存 {peak_gib:.2f} GiB 超出 D17 预算"
