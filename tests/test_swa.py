"""E2 · 滑动窗口（SWA）的验收测试。

对应 [reports/swa-window.md](../reports/swa-window.md)：

| 测试 | 判据 |
|------|------|
| `test_windowed_cache_ring_wraps` | ring 写完一圈后，每个槽的绝对位置与内容一致 |
| `test_window_mask_selects_last_w` | 掩码恰好选中"最后 W 个位置" |
| `test_swa_off_matches_baseline` | `swa_window=0` 与不装 SWA 完全一致 |
| `test_window_larger_than_context_matches_full` | 窗口盖得住全上下文时，结果与全注意力**逐 token 相同** |
| `test_chunk_size_does_not_change_result` | 换 prefill 分块大小不改变结果 |

跑法：`& .\\.venv\\Scripts\\python.exe -m pytest tests -q`
"""

from __future__ import annotations

import torch

from nova.cache import WindowedKVCache
from nova.decode import GraphDecoder

N_NEW = 6


def _ring(cap=8, heads=2, dim=4):
    return WindowedKVCache(
        capacities=[cap], num_kv_heads=heads, head_dim=dim, max_len=64,
        batch=1, dtype=torch.float16, device="cuda",
    )


def test_windowed_cache_ring_wraps():
    """逐 token 写 3 圈：每个槽的**绝对位置**必须与它装的内容对得上。

    `slot_positions` 的公式一旦错，掩码就会把错的位置当成有效 —— 这是 ring 最危险的地方。
    """
    cap, heads, dim = 8, 2, 4
    cache = _ring(cap, heads, dim)
    for pos in range(3 * cap):
        key = torch.full((1, heads, 1, dim), float(pos), dtype=torch.float16, device="cuda")
        cache.pos.fill_(pos)
        cache.update(key, key + 0.5, layer_idx=0)
    end = torch.tensor(3 * cap - 1, device="cuda")
    positions = cache.slot_positions(0, end)
    assert positions.tolist() == list(range(3 * cap - cap, 3 * cap)), "槽位绝对位置不对"
    for slot in range(cap):
        p = int(positions[slot].item())
        got = cache.key_cache[0][:, :, slot, :].float()
        assert torch.allclose(got, torch.full_like(got, float(p))), f"槽 {slot} 声称位置 {p}，内容却是 {got.flatten()[0].item()}"
        assert torch.allclose(cache.value_cache[0][:, :, slot, :].float(), torch.full_like(got, p + 0.5))


def test_window_mask_selects_last_w():
    """掩码必须**恰好**选中最后 W 个位置（写过、≤ 当前、距离 < W）。"""
    dec = _SwaStub(window=4, slots=8, first=0)
    mask = dec._window_mask(torch.tensor(9, device="cuda"), 1).view(-1)
    keep = (mask == 0).nonzero().flatten().tolist()
    assert len(keep) == 4, f"窗口里应恰好有 4 个位置，实际 {len(keep)}"
    end = torch.tensor(9, device="cuda")
    p = dec.cache.slot_positions(0, end)
    assert sorted(p[keep].tolist()) == [6, 7, 8, 9]

    # 序列开头（每圈没写满）时，没写过的槽必须被屏蔽
    mask0 = dec._window_mask(torch.tensor(1, device="cuda"), 1).view(-1)
    assert int((mask0 == 0).sum()) == 2, "序列开头只该有 2 个有效位置（0 和 1）"


class _SwaStub(GraphDecoder):
    """只借用 `_window_mask` / `cache`，不走 `__init__`（避免加载模型）。"""

    def __init__(self, window: int, slots: int, first: int) -> None:
        self.swa_window = int(window)
        self.window_slots = int(slots)
        self.max_len = 64
        self._win_arange = torch.arange(slots, dtype=torch.long, device="cuda")
        self._zero = torch.zeros((), dtype=torch.float16, device="cuda")
        self._ninf = torch.full((), torch.finfo(torch.float16).min, dtype=torch.float16, device="cuda")
        self.cache = _ring(slots)
        self.cache.first.fill_(first)


def _greedy(nova, prompt_ids, n_new: int, max_len: int) -> torch.Tensor:
    dec = GraphDecoder(nova, max_len=max_len)
    dec.prefill(prompt_ids)
    dec.capture()
    out = [dec.input_ids.clone()]
    for _ in range(n_new - 1):
        dec.step()
        out.append(dec.input_ids.clone())
    return torch.cat(out, dim=1)


def test_swa_off_matches_baseline(bundle, prompt_ids):
    """`swa_window=0` 时 `layer_windows` 全是 0，cache 还是 `StaticKVCache`。"""
    nova, _, _ = bundle
    dec = GraphDecoder(nova, max_len=128)
    assert not dec.swa_window
    assert set(dec.layer_windows) == {0}
    assert not isinstance(dec.cache, WindowedKVCache)
    assert _greedy(nova, prompt_ids, N_NEW, 128).shape[1] == N_NEW


def test_window_larger_than_context_matches_full(prompt_ids, nova_swa):
    """窗口盖得住全上下文 ⇒ 必须与**全注意力逐 token 相同**（ring + 掩码 + 位置公式全测到）。

    两边都用 `nova_swa(...)` 造（`num_paths=1`）—— 必须同架构，否则比的不是窗口。
    """
    full = _greedy(nova_swa(0), prompt_ids, N_NEW, 256)
    swa = _greedy(nova_swa(256), prompt_ids, N_NEW, 256)
    assert torch.equal(swa, full), "窗口盖得住时结果却不一样 ⇒ ring/掩码有问题"


def test_chunk_size_does_not_change_result(prompt_ids, nova_swa):
    """换个 prefill 分块大小（跨块边界的写法不同）结果必须一致。"""
    # max_len=256 让局部层容量（2W=128）**小于** max_len ⇒ 真的走 ring；
    # 窗口 64 盖得住整段（15+6 token）⇒ 三者必须都与全注意力一致。
    a = _greedy(nova_swa(64, chunk=64), prompt_ids, N_NEW, 256)
    b = _greedy(nova_swa(64, chunk=32), prompt_ids, N_NEW, 256)
    full = _greedy(nova_swa(0), prompt_ids, N_NEW, 256)
    assert torch.equal(a, b), "分块边界改变了结果"
    assert torch.equal(a, full), "ring 路径与全注意力不一致"


def test_second_prefill_keeps_window(nova_swa, prompt_ids, bundle):
    """**分两次 prefill**（第二次 offset 非 0）必须与一次 prefill 等价。

    这是 S4 记忆注入的路径（先写前缀，再从 offset 继续）。实测踩过的坑：
    `prefill` 把 ring 的 `first` 覆盖成 offset 之后，局部层会把 `[0, offset)` 全判成
    "没写过" —— 表现是**所有问题都答不出来**（0/4），而且只生成几个 token 的测试抓不到它。
    """
    _, _, tok = bundle
    head = prompt_ids
    tail = tok("潮汐是月球引力造成的。", return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    nova = nova_swa(256)

    one = GraphDecoder(nova, max_len=512)
    one.prefill(torch.cat([head, tail], dim=1))
    one.capture()

    two = GraphDecoder(nova, max_len=512)
    two.prefill(head)
    two.prefill(tail, offset=head.shape[1], reset=False)
    two.capture()

    assert torch.equal(one.input_ids, two.input_ids), "第二次 prefill 之后的首个 token 不同"
    for _ in range(N_NEW - 1):
        one.step()
        two.step()
        assert torch.equal(one.input_ids, two.input_ids), "后续 token 分叉"
