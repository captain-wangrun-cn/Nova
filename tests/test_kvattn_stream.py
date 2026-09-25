r"""int8 流式 KV cache · 验收测试（①.5，对应 [reports/kv-int8-fused-attn.md](../reports/kv-int8-fused-attn.md)）。

| 测试 | 判据 |
|------|------|
| `test_matches_dequant_then_sdpa` | **主判据**：流式 cache 的一步输出 = "int8 前缀还原 + fp16 尾部"再算 SDPA，**≤4 ULP 或 ≤1e-3 绝对** |
| `test_prefix_grows_with_pos` | `_qlen` 每 64 个 token 涨一格；没涨的那一步**不许**把未定尺子的组算进去 |
| `test_ring_mask_excludes_prefix` | 环里落在 int8 前缀内的槽位必须被掩掉（否则那批 token 被加权两次） |
| `test_prefill_multi_block_equals_single` | 分块 prefill（块边界不对齐 64）与一次 prefill 结果**逐位一致** |
| `test_finish_prefill_matches_quantizer` | 量化缓冲与 `kvquant.quantize_int8(..., "token")` **逐位同源** |
| `test_slots_are_independent` | 两个槽位各存一份，互不污染 |
| `test_short_context_is_all_fp16` | 上下文 < 64 时前缀为空，输出必须是有限值（不许 NaN） |
| `test_device_scalars_match_python_ints` | 显存标量版核（图内用）与 Python 参数版**逐位一致** |
| `test_device_scalars_allow_partial_split` | `n_splits` 按桶长固定、`used` 可变时，多余 split 必须被安全忽略 |
| `test_cuda_graph_capture` | 预分配缓冲后能进 CUDA Graph，重放与 eager **逐位一致** |

## 为什么主判据是"≤4 ULP 或 ≤1e-3 绝对"（而不是融合核那 2 ULP）

融合核（`tests/test_kvattn.py`，2 ULP）比的是**同一条**路径的两个实现（核 vs `dequantize_int8`
→ SDPA）。流式 cache 多了一次**结构性**的舍入：前缀走 int8 核、尾部单独走一次 fp16 SDPA，
两者按 `logsumexp` 合并后再舍到 fp16。于是参考（一次性 SDPA 后舍一次）与它（合并后再舍）
不可能逐位对齐。实测合成数据上最大 **4 ULP**（fp16 在 0.1 量级上 1 ULP = 6.1e-5）。
判据取 4 ULP / 1e-3 绝对，是"与结构差异匹配"的门槛。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nova.kernels import triton_available
from nova.kvattn import GROUP_T, KVInt8Cache, int8_attn_decode, pack_int8_kv
from nova.kvquant import dequantize_int8, quantize_int8

pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton 不可用")

HQ, H_KV, D = 32, 8, 128
FP16_TINY = torch.finfo(torch.float16).tiny
MAX_ULP = 4.0
ABS_TOL = 1e-3


def _ulp(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    up = (torch.nextafter(want, torch.full_like(want, float("inf"))).float() - want.float()).abs()
    up = up.clamp_min(FP16_TINY * 2 ** -10)
    return (got.float() - want.float()).abs() / up


def _assert_close(got: torch.Tensor, want: torch.Tensor, tag: str) -> None:
    ulp = _ulp(got, want)
    absdiff = (got.float() - want.float()).abs()
    ok = (ulp <= MAX_ULP) | (absdiff <= ABS_TOL)
    assert (~ok).sum().item() == 0, (
        f"{tag}: {(~ok).sum().item()} 个元素超判据"
        f"（最大 {ulp.max().item():.1f} ULP / {absdiff.max().item():.3e} 绝对）"
    )


def _make(used: int, seed: int = 0):
    torch.manual_seed(seed)
    k = torch.randn(1, H_KV, used, D, device="cuda", dtype=torch.float16) * 0.7
    v = torch.randn(1, H_KV, used, D, device="cuda", dtype=torch.float16) * 0.7
    q = torch.randn(1, HQ, 1, D, device="cuda", dtype=torch.float16) * 0.7
    return k, v, q


def _reference(k, v, q, used: int, prefix: int):
    """**先还原再算**：前 `prefix` 个 token 走 int8t64 往返，其余保持 fp16 → SDPA(math)。"""
    if prefix:
        kq, kmn, kst = quantize_int8(k[:, :, :prefix], GROUP_T, "token")
        vq, vmn, vst = quantize_int8(v[:, :, :prefix], GROUP_T, "token")
        kk = dequantize_int8(kq, kmn, kst, GROUP_T, "token")
        vv = dequantize_int8(vq, vmn, vst, GROUP_T, "token")
        kk = torch.cat([kk, k[:, :, prefix:used]], dim=2)
        vv = torch.cat([vv, v[:, :, prefix:used]], dim=2)
    else:
        kk, vv = k[:, :, :used], v[:, :, :used]
    gq = HQ // H_KV
    kk = kk[:, :, None, :, :].expand(1, H_KV, gq, used, D).reshape(1, HQ, used, D).contiguous()
    vv = vv[:, :, None, :, :].expand(1, H_KV, gq, used, D).reshape(1, HQ, used, D).contiguous()
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, kk, vv, scale=D ** -0.5)


def _cache(slots: int = 1, max_len: int = 512) -> KVInt8Cache:
    return KVInt8Cache(num_slots=slots, num_kv_heads=H_KV, head_dim=D, max_len=max_len,
                       num_q_heads=HQ)


def _prefill_then_last(cache, k, v, q, layer: int = 0, chunk: int = 0):
    """前 L-1 个 token 走 prefill，第 L 个当**解码一步**（与真实解码流程一致）。"""
    L = k.shape[2]
    n0 = L - 1
    cache.begin_prefill(n0)
    if chunk:
        for c0 in range(0, n0, chunk):
            nc = min(chunk, n0 - c0)
            cache.set_prefill_pos(c0)
            cache.update(k[:, :, c0 : c0 + nc], v[:, :, c0 : c0 + nc], layer)
    else:
        cache.set_prefill_pos(0)
        cache.update(k[:, :, :n0], v[:, :, :n0], layer)
    cache.finish_prefill(n0)
    cache.pos.fill_(L - 1)
    cache.update(k[:, :, L - 1 :], v[:, :, L - 1 :], layer)
    return cache.attend(q, layer)


# ---------------------------------------------------------------- 主判据


@pytest.mark.parametrize("used", [40, 64, 100, 128, 192, 320])
def test_matches_dequant_then_sdpa(used: int):
    k, v, q = _make(used)
    cache = _cache()
    got = _prefill_then_last(cache, k, v, q)
    want = _reference(k, v, q, used, used // GROUP_T * GROUP_T)
    _assert_close(got, want, f"used={used}")
    assert int(cache._qlen.item()) == used // GROUP_T * GROUP_T, "前缀长度必须落在整组边界上"


def test_prefix_grows_with_pos():
    """`_qlen` 只在跨过 64 的整数倍时涨 —— 这是"没定尺子的组不许算进前缀"的守卫。"""
    k, v, _ = _make(128)
    cache = _cache()
    cache.begin_prefill(63)
    cache.set_prefill_pos(0)
    cache.update(k[:, :, :63], v[:, :, :63], 0)
    cache.finish_prefill(63)
    assert int(cache._qlen.item()) == 0
    for p in range(63, 128):
        cache.pos.fill_(p)
        cache.update(k[:, :, p : p + 1], v[:, :, p : p + 1], 0)
        expect = (p + 1) // GROUP_T * GROUP_T
        assert int(cache._qlen.item()) == expect, f"pos={p} 的 qlen 应为 {expect}"
        assert int(cache._written.item()) == p + 1


def test_ring_mask_excludes_prefix():
    """环里落在 int8 前缀内的槽位必须掩掉 —— 否则那批 token 会被加权两次。"""
    k, v, _ = _make(200)
    cache = _cache()
    cache.begin_prefill(199)
    cache.set_prefill_pos(0)
    cache.update(k[:, :, :199], v[:, :, :199], 0)
    cache.finish_prefill(199)
    cache.pos.fill_(199)
    cache.update(k[:, :, 199:], v[:, :, 199:], 0)
    mask = cache.ring_mask().view(-1)
    qlen = int(cache._qlen.item())
    written = int(cache._written.item())
    idx = torch.arange(GROUP_T, device="cuda")
    abs_pos = (written - 1) - torch.remainder((written - 1) - idx, GROUP_T)
    for i in range(GROUP_T):
        if bool(mask[i]):
            assert qlen <= int(abs_pos[i]) < written, f"槽 {i}（位置 {int(abs_pos[i])}）不该有效"
    assert bool(mask.any()), "掩码全空"
    assert int(mask.sum()) == written - qlen, "有效槽位数必须等于尾部 token 数"


@pytest.mark.parametrize("chunk", [64, 96, 128])
def test_prefill_multi_block_equals_single(chunk: int):
    """分块 prefill（块边界可以不对齐 64）必须与一次 prefill 完全一致。"""
    used = 300
    k, v, q = _make(used)
    one = _prefill_then_last(_cache(), k, v, q)
    chunked = _prefill_then_last(_cache(), k, v, q, chunk=chunk)
    assert torch.equal(one, chunked), f"chunk={chunk} 与单块结果不一致"


def test_finish_prefill_matches_quantizer():
    """量化缓冲必须与 `kvquant.quantize_int8(..., "token")` 逐位同源。"""
    used = 192
    k, v, _ = _make(used)
    cache = _cache()
    cache.begin_prefill(used)
    cache.set_prefill_pos(0)
    cache.update(k, v, 0)
    cache.finish_prefill(used)
    kq, kmn, kst = quantize_int8(k, GROUP_T, "token")
    vq, vmn, vst = quantize_int8(v, GROUP_T, "token")
    assert torch.equal(cache.kq[0][:, :, :used], kq)
    assert torch.equal(cache.vq[0][:, :, :used], vq)
    n = used // GROUP_T
    assert torch.equal(cache.kmn[0][:, :, :n], kmn)
    assert torch.equal(cache.kst[0][:, :, :n], kst)
    assert torch.equal(cache.vmn[0][:, :, :n], vmn)
    assert torch.equal(cache.vst[0][:, :, :n], vst)
    assert cache.staging is None, "finish_prefill 必须释放 fp16 暂存区（这才是真省显存）"


def test_slots_are_independent():
    used = 200
    k0, v0, q = _make(used)
    torch.manual_seed(99)
    k1 = torch.randn_like(k0)
    v1 = torch.randn_like(v0)
    got0 = _prefill_then_last(_cache(slots=1), k0, v0, q)
    got1 = _prefill_then_last(_cache(slots=1), k1, v1, q)
    _assert_close(got0, _reference(k0, v0, q, used, used // GROUP_T * GROUP_T), "slot 0")
    _assert_close(got1, _reference(k1, v1, q, used, used // GROUP_T * GROUP_T), "slot 1")
    assert not torch.equal(got0, got1)


def test_short_context_is_all_fp16():
    """上下文 < 64：前缀为空，输出必须有限（核在 prefix=0 时不许吐 NaN）。"""
    used = 33
    k, v, q = _make(used)
    cache = _cache()
    got = _prefill_then_last(cache, k, v, q)
    assert int(cache._qlen.item()) == 0
    assert torch.isfinite(got).all(), "前缀为空时输出出现了非有限值"
    _assert_close(got, _reference(k, v, q, used, 0), "prefix=0")


def test_device_scalars_match_python_ints():
    """显存标量版（图内姿势）与 Python 参数版（eager 姿势）必须逐位一致。"""
    used = 256
    torch.manual_seed(0)
    k = torch.randn(1, H_KV, used, D, device="cuda", dtype=torch.float16) * 0.7
    v = torch.randn_like(k) * 0.7
    q = torch.randn(1, HQ, 1, D, device="cuda", dtype=torch.float16) * 0.7
    kq, kmn, kst, vq, vmn, vst = pack_int8_kv(k, v, GROUP_T)
    a = int8_attn_decode(q, kq, kmn, kst, vq, vmn, vst, used=used, prefix_groups=used // GROUP_T)
    used_t = torch.tensor([used], dtype=torch.int64, device="cuda")
    pg_t = torch.tensor([used // GROUP_T], dtype=torch.int64, device="cuda")
    b = int8_attn_decode(q, kq, kmn, kst, vq, vmn, vst, used=None, used_t=used_t,
                         prefix_groups_t=pg_t, n_splits=1)
    assert torch.equal(a, b), "显存标量版与 Python 参数版输出不同"


def test_device_scalars_allow_partial_split():
    """`n_splits` 按桶长固定、`used` 可变时，尾部 split 必须被安全忽略（不出现 0/0）。"""
    used = 320
    k, v, q = _make(used)
    kq, kmn, kst, vq, vmn, vst = pack_int8_kv(k, v, GROUP_T)
    n_splits = 4
    shape = (H_KV, n_splits, 16)
    pm = torch.zeros(shape, dtype=torch.float32, device="cuda")
    pl = torch.zeros_like(pm)
    pa = torch.zeros((*shape, D), dtype=torch.float32, device="cuda")
    out = torch.zeros(1, HQ, 1, D, dtype=torch.float16, device="cuda")
    lse = torch.zeros(1, HQ, dtype=torch.float32, device="cuda")
    want = int8_attn_decode(q, kq, kmn, kst, vq, vmn, vst, used=used,
                            prefix_groups=used // GROUP_T)
    used_t = torch.tensor([used], dtype=torch.int64, device="cuda")
    pg_t = torch.tensor([used // GROUP_T], dtype=torch.int64, device="cuda")
    got = int8_attn_decode(q, kq, kmn, kst, vq, vmn, vst, used=None, used_t=used_t,
                           prefix_groups_t=pg_t, n_splits=n_splits,
                           scratch=(pm, pl, pa), out=out, lse_out=lse)
    assert torch.isfinite(got).all(), "多余 split 引入了非有限值"
    _assert_close(got, want, "partial split")


# ---------------------------------------------------------------- CUDA Graph


def test_cuda_graph_capture():
    """整条解码注意力（核 + 环 SDPA + 合并）能进图，重放与 eager 逐位一致。"""
    used = 200
    k, v, q = _make(used)
    cache = _cache(max_len=256)
    eager = _prefill_then_last(cache, k, v, q).clone()

    cache.pos.fill_(used - 1)
    cache.update(k[:, :, used - 1 :], v[:, :, used - 1 :], 0)
    cache.refresh_scalars()
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(g):
            cache.attend(q, 0)
    cache.merged.zero_()
    with torch.inference_mode():
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache.merged, eager), "图内重放与 eager 不一致"

