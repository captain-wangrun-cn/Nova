"""P0 · 掩码表拆除 + 容量分桶的验收测试。

对应 [reports/decode-mask-bucket.md](../reports/decode-mask-bucket.md)：

| 测试 | 判据 |
|------|------|
| `test_bucket_for` | 分桶边界正确，超最大桶时按需分配 |
| `test_mask_row_matches_table_bitwise` | 即时掩码行与旧 `_build_mask_table[pos]` **逐位一致** |
| `test_no_mask_table_resident` | 解码器不再常驻 O(max_len²) 的掩码整表 |
| `test_grow_copies_kv_prefix` | 跨桶 `grow()` 保住已写入的 KV 前缀与 `pos` |
| `test_bucketed_decode_matches_wide_decode` | 窄桶解码与宽 max_len 解码 **token 逐个相同** |
| `test_grow_preserves_decode` | 搬家之后继续解码，与不搬家 **token 逐个相同** |

跑法：`& .\\.venv\\Scripts\\python.exe -m pytest tests -q`
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from nova.decode import GraphDecoder, bucket_for

MAX_LEN = 128
N_NEW = 8


def test_bucket_for():
    """桶边界：不小于 n 的最小桶；超过最大的桶就按 n 精确分配。"""
    assert bucket_for(1) == 2070
    assert bucket_for(2070) == 2070
    assert bucket_for(2071) == 4096
    assert bucket_for(8192) == 8192
    assert bucket_for(16384 + 64) == 32768
    assert bucket_for(65536) == 65536
    assert bucket_for(70000) == 70000


def test_mask_row_matches_table_bitwise():
    """`_mask_row(pos)` 与 `_build_mask_table(max_len)[pos]` **逐位一致**（含 -inf 位）。

    这是 P0 的核心判据：省掉 O(max_len²) 的常驻，但一位都不能变。
    不加载模型 —— 这两个方法只用到 `max_len` / `_arange` / `_zero` / `_ninf`。
    """
    max_len = 64
    table = GraphDecoder._build_mask_table(max_len, torch.float16, "cpu")
    fake = SimpleNamespace(
        max_len=max_len,
        _arange=torch.arange(max_len, dtype=torch.long),
        _zero=torch.zeros((), dtype=torch.float16),
        _ninf=torch.full((), torch.finfo(torch.float16).min, dtype=torch.float16),
    )
    for pos in (0, 1, 7, 63):
        row = GraphDecoder._mask_row(fake, torch.tensor([pos]))
        assert row.shape == (1, 1, 1, max_len)
        assert torch.equal(row.reshape(max_len), table[pos]), f"pos={pos} 与旧表不逐位一致"


def test_no_mask_table_resident(bundle):
    """解码器里不能再有 `(max_len, max_len)` 的常驻张量。"""
    nova, _, _ = bundle
    dec = GraphDecoder(nova, max_len=MAX_LEN)
    assert not hasattr(dec, "mask_table"), "O(max_len²) 的掩码整表又回来了"
    assert dec._arange.numel() == MAX_LEN


def _greedy(nova, prompt_ids, max_len, n_new, grow: tuple[int, int] | None = None) -> torch.Tensor:
    """贪心生成 `n_new` 个 token；`grow=(第几步, 新桶)` 时中途跨桶搬家。"""
    dec = GraphDecoder(nova, max_len=max_len)
    dec.prefill(prompt_ids)
    dec.capture()
    out = [dec.input_ids.clone()]
    for i in range(n_new - 1):
        if grow is not None and i == grow[0]:
            dec.grow(grow[1])
            dec.capture()
        dec.step()
        out.append(dec.input_ids.clone())
    return torch.cat(out, dim=1)


def test_grow_copies_kv_prefix(bundle, prompt_ids):
    """`grow()` 必须把 `[0, used)` 的 K/V 原样搬过去，并保住 `pos`；之后必须重捕图。"""
    nova, _, _ = bundle
    dec = GraphDecoder(nova, max_len=MAX_LEN)
    dec.prefill(prompt_ids)
    used = int(dec.cache.pos.item())
    before_k = [t[:, :, :used, :].clone() for t in dec.cache.key_cache]
    before_v = [t[:, :, :used, :].clone() for t in dec.cache.value_cache]

    dec.grow(MAX_LEN * 2)

    assert dec.max_len == MAX_LEN * 2
    assert int(dec.cache.pos.item()) == used
    assert dec.graph is None and not dec._captured, "形状变了却没有要求重捕"
    for old, new in zip(before_k, dec.cache.key_cache):
        assert torch.equal(old, new[:, :, :used, :])
    for old, new in zip(before_v, dec.cache.value_cache):
        assert torch.equal(old, new[:, :, :used, :])
    # 更大的桶里，used 之后仍是 0（靠掩码屏蔽，不靠内容）
    assert dec.cache.key_cache[0][:, :, used:, :].abs().sum().item() == 0


def test_bucketed_decode_matches_wide_decode(bundle, prompt_ids):
    """按需选桶（`for_length`）只是少读零槽位，**不能改变结果**。"""
    nova, _, _ = bundle
    n = prompt_ids.shape[1]
    ref = _greedy(nova, prompt_ids, max_len=MAX_LEN, n_new=N_NEW)
    dec = GraphDecoder.for_length(nova, n, reserve=8)
    assert dec.max_len == bucket_for(n + 8) > MAX_LEN
    got = _greedy(nova, prompt_ids, max_len=dec.max_len, n_new=N_NEW)
    assert torch.equal(got, ref), "窄桶解码与宽 max_len 解码结果不同"


def test_grow_preserves_decode(bundle, prompt_ids):
    """跨桶搬家之后继续解码，必须与"一开始就用大桶" **token 逐个相同**。"""
    nova, _, _ = bundle
    ref = _greedy(nova, prompt_ids, max_len=MAX_LEN, n_new=N_NEW)
    moved = _greedy(nova, prompt_ids, max_len=MAX_LEN, n_new=N_NEW, grow=(3, MAX_LEN * 2))
    assert torch.equal(moved, ref), "搬家改变了后续解码"
