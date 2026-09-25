r"""② · 记忆段走 `SegmentPrefetcher` 的验收测试（[reports/memory-prefetch-load.md](../reports/memory-prefetch-load.md)）。

| 测试 | 判据 |
|------|------|
| `test_layout_matches_safe_open` | safetensors 布局解析出的偏移，逐张量与 `safe_open` **逐位一致** |
| `test_prefetched_load_is_bit_identical` | `load_prefetched()` 与 `load()` 读出的每个张量**逐位相同** |
| `test_prefetched_load_survives_segment_boundaries` | 段长**不能整除**文件大小时（含最后一段不满）也必须一致 |
| `test_prefetched_load_checks_schema` | D09 校验在预取路径上**同样生效**（不匹配就拒绝，不静默读） |
| `test_prefetched_load_allows_foreign` | `allow_foreign=True` 时放行，且 `store.foreign=True` |
| `test_prefetched_views_are_independent` | 多个记忆项的视图互不覆盖（一块缓冲切出来的，偏移必须算对） |
| `test_prefetch_range` | `SegmentPrefetcher(offset, length)` 的区间语义正确（不改动别处字节） |

**为什么"逐位一致"是这条路径的正确判据**：预取路径没有做任何数值计算，只是换了一种
把字节搬上来的方式。任何一位不同都说明**偏移算错 / 段边界错**，而不是精度问题。
"""

from __future__ import annotations

import json

import pytest
import torch

from nova.memory import (
    MemorySchema,
    MemorySchemaMismatch,
    MemoryStore,
    read_safetensors_layout,
)
from nova.prefetch import SegmentPrefetcher


def _store(seed: int = 0, items: tuple[int, ...] = (5, 7, 3)) -> MemoryStore:
    """造一个与真实 schema 同形的记忆库（形状与 Qwen3-VL-4B 文本塔一致）。"""
    schema = MemorySchema(hidden_size=32, num_kv_heads=2, head_dim=8,
                          num_cache_layers=4, num_paths=2)
    store = MemoryStore(schema, fingerprint="fp-test")
    torch.manual_seed(seed)
    for i, n in enumerate(items):
        store.add(
            torch.randn(4, 2, n, 8, dtype=torch.float16),
            torch.randn(4, 2, n, 8, dtype=torch.float16),
            label=f"item{i}",
        )
    return store


def test_layout_matches_safe_open(tmp_path):
    """头部解析出的偏移能正确还原每个张量。"""
    from safetensors import safe_open

    path = _store().save(tmp_path / "m.safetensors")
    meta, data_start, layout = read_safetensors_layout(path)
    assert meta["format"] == "nova-memory"
    assert data_start > 8, "数据区起点必须在头之后"
    assert file_size_of(path) == data_start + max(t[3] for t in layout.values()), "数据区末尾必须就是文件末尾"
    with open(path, "rb") as fh:
        fh.seek(data_start)
        blob = fh.read(max(t[3] for t in layout.values()))
    buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name, (dt, shape, s, e) in layout.items():
            got = buf[s:e].view(dt).view(shape)
            assert torch.equal(got, f.get_tensor(name)), f"{name} 偏移还原不一致"


def file_size_of(path) -> int:
    import os

    return os.path.getsize(path)


def test_prefetched_load_is_bit_identical(tmp_path):
    """两条路读出来的每个张量必须逐位相同。"""
    path = _store().save(tmp_path / "m.safetensors")
    want = MemoryStore.load(path, device="cuda")
    got = MemoryStore.load_prefetched(path, device="cuda", verify=True)
    assert len(got) == len(want) == 3
    assert [it.label for it in got.items] == [it.label for it in want.items]
    for a, b in zip(got.items, want.items):
        assert torch.equal(a.k, b.k), "k 不一致"
        assert torch.equal(a.v, b.v), "v 不一致"
    assert got.schema.digest == want.schema.digest
    assert got.fingerprint == want.fingerprint


@pytest.mark.parametrize("seg_bytes", [64, 1000, 4096, 1 << 20])
def test_prefetched_load_survives_segment_boundaries(tmp_path, seg_bytes: int):
    """段长各种取值（含"最后一段不满"与"段长不整除张量"）都必须一致。"""
    path = _store(seed=7).save(tmp_path / "m.safetensors")
    want = MemoryStore.load(path, device="cuda")
    got = MemoryStore.load_prefetched(path, device="cuda", seg_bytes=seg_bytes, ring=2)
    for a, b in zip(got.items, want.items):
        assert torch.equal(a.k, b.k) and torch.equal(a.v, b.v), f"seg_bytes={seg_bytes} 不一致"


def test_prefetched_load_checks_schema(tmp_path):
    """D09 校验在预取路径上同样生效 —— 不匹配必须拒绝，不静默读。"""
    path = _store().save(tmp_path / "m.safetensors")
    other = MemorySchema(hidden_size=64, num_kv_heads=2, head_dim=8,
                         num_cache_layers=4, num_paths=2)
    with pytest.raises(MemorySchemaMismatch):
        MemoryStore.load_prefetched(path, schema=other, device="cuda")
    # 指纹不匹配同理
    with pytest.raises(MemorySchemaMismatch):
        MemoryStore.load_prefetched(path, fingerprint="fp-别的", device="cuda")


def test_prefetched_load_allows_foreign(tmp_path):
    path = _store().save(tmp_path / "m.safetensors")
    got = MemoryStore.load_prefetched(path, fingerprint="fp-别的", device="cuda", allow_foreign=True)
    assert got.foreign is True
    assert len(got) == 3


def test_prefetched_views_are_independent(tmp_path):
    """多个记忆项从同一块缓冲切出来 —— 偏移算错就会互相覆盖，这里专卡这个。"""
    path = _store(items=(3, 3, 3)).save(tmp_path / "m.safetensors")
    got = MemoryStore.load_prefetched(path, device="cuda")
    ks = [it.k for it in got.items]
    assert not torch.equal(ks[0], ks[1]) and not torch.equal(ks[1], ks[2])
    want = MemoryStore.load(path, device="cuda")
    for a, b in zip(got.items, want.items):
        assert torch.equal(a.k, b.k)


def test_prefetch_range(tmp_path):
    """`offset`/`length` 的区间语义：只搬指定字节，一个字节都不多不少。"""
    path = tmp_path / "blob.bin"
    data = bytes(range(256)) * 4
    path.write_bytes(data)
    for offset, length in ((0, 0), (0, 100), (7, 100), (1000, 24), (1000, 0)):
        pf = SegmentPrefetcher(path, seg_bytes=64, ring=2, offset=offset, length=length)
        # 缓冲比区间大 1 字节：多出来那字节必须保持哨兵值（证明没有越界写）
        dst = torch.full((length + 1,), 0xAA, dtype=torch.uint8, device="cuda")
        moved = pf.stream_into(dst)
        torch.cuda.synchronize()
        assert moved == length
        if length:
            assert torch.equal(dst[:length].cpu(), torch.tensor(list(data[offset:offset + length]), dtype=torch.uint8))
        assert int(dst[length]) == 0xAA, "越界写了"


def test_prefetch_range_matches_stream(tmp_path):
    """`stream_into`（预分配）与 `stream`（逐段产出）必须给出同样的字节。"""
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(200)) * 3)
    for offset, length in ((0, 600), (13, 400)):
        pf = SegmentPrefetcher(path, seg_bytes=97, ring=3, offset=offset, length=length)
        joined = torch.cat([seg.cpu() for seg in pf.stream(device="cuda")])
        pf2 = SegmentPrefetcher(path, seg_bytes=97, ring=3, offset=offset, length=length)
        dst = torch.zeros(length, dtype=torch.uint8, device="cuda")
        pf2.stream_into(dst)
        torch.cuda.synchronize()
        assert torch.equal(joined, dst.cpu())
        assert torch.equal(joined, torch.tensor(list(path.read_bytes()[offset:offset + length]), dtype=torch.uint8))
