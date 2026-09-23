r"""int4 KV cache · 数值正确性测试。

对应 [src/nova/kvquant.py](../src/nova/kvquant.py) 与 [reports/kv-int4.md](../reports/kv-int4.md)。
**先证明量化器本身没错，再去解释 needle 掉分** —— 否则分不清"精度损失"和"实现 bug"。

| 测试 | 判据 |
|------|------|
| `test_dequant_matches_formula` | 反量化结果与按公式手算的 `q*step+min` **逐位一致** |
| `test_roundtrip_error_within_half_step` | 每个元素误差 ≤ 本组 `step/2` + fp16 余量 |
| `test_group_independent` | 同一条向量里一组量程放大 100x，**不影响**另一组（分组量化的全部意义） |
| `test_constant_tensor_exact` | 常量张量往返无损（`max==min` 的退化情形） |
| `test_shapes_and_groups` | group 16/32/128/整条 —— packed/mins/还原形状全对 |
| `test_bytes_accounting` | Qwen3-4B 配置下 int4 = 42 KiB/token、3.46x |
| `test_cache_newest_token_not_stale` | 回归：`used` 少算一个会让最新 token 读到 scratch 的 0 |
| `test_cache_prefill_appends_quantized` | prefill 路径写入范围与量化范围一致，尾部仍是 0 |

跑法：`& .\.venv\Scripts\python.exe -m pytest tests -q`
"""

from __future__ import annotations

import pytest
import torch

from nova.kvquant import (
    GROUP_K,
    bytes_per_token,
    dequantize_int8,
    quantize_int8,
    roundtrip_fp8,
    roundtrip_int8,
    NIBBLE_MAX,
    bytes_per_token,
    dequantize_int4,
    quantize_int4,
    roundtrip_int4,
)

HEAD_DIM = 128
KV_HEADS = 4
ROWS = 37


def _x(seed: int = 0, scale: float = 3.0) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(1, KV_HEADS, ROWS, HEAD_DIM) * scale).to(torch.float16)


def _groups(x: torch.Tensor, group: int) -> torch.Tensor:
    g = group or HEAD_DIM
    return x.float().reshape(1, KV_HEADS, ROWS, HEAD_DIM // g, g)


def test_dequant_matches_formula():
    """反量化必须是 `q*step+min` 的忠实逆运算（打包 -> 解包不能错位）。"""
    x = _x()
    packed, mins, steps = quantize_int4(x, GROUP_K)
    got = dequantize_int4(packed, mins, steps, GROUP_K)

    g = _groups(x, GROUP_K)
    xmin = g.amin(dim=-1)
    step = ((g.amax(dim=-1) - xmin) / NIBBLE_MAX).clamp_min(1e-8)
    q = ((g - xmin.unsqueeze(-1)) / step.unsqueeze(-1)).round().clamp_(0, NIBBLE_MAX)
    assert torch.equal(mins, xmin.to(torch.float16))
    assert torch.equal(steps, step.to(torch.float16))

    # ⚠️ 反量化用的是**存下来的 fp16 尺子**，不是 fp32 的原尺子 —— 真实方案同样如此
    # （scale/min 只有 fp16 精度）。参考值必须用 fp16 舍入后的 min/step 算才能逐位一致。
    xmin16, step16 = xmin.to(torch.float16).float(), step.to(torch.float16).float()
    want = (q * step16.unsqueeze(-1) + xmin16.unsqueeze(-1)).reshape(1, KV_HEADS, ROWS, HEAD_DIM)
    assert torch.equal(got, want.to(torch.float16))

    # fp16 尺子 vs fp32 尺子：差 ≤ 1% 个 step，可忽略（换 fp32 元数据时的判据）
    want32 = q * step.unsqueeze(-1) + xmin.unsqueeze(-1)
    assert torch.all((got.float().reshape_as(g) - want32).abs() <= step.unsqueeze(-1) * 0.01)


def test_roundtrip_error_within_half_step():
    """误差上界就是本组 step/2 —— 超过它说明分组/打包错了。"""
    x = _x()
    step = ((_groups(x, GROUP_K).amax(-1) - _groups(x, GROUP_K).amin(-1)) / NIBBLE_MAX).clamp_min(1e-8)
    err = (roundtrip_int4(x, GROUP_K).float() - x.float()).reshape_as(_groups(x, GROUP_K)).abs()
    bound = step.unsqueeze(-1) / 2 + 0.01  # 0.01 = 值域 ~10 时的 fp16 舍入余量
    assert torch.all(err <= bound)
    assert err.max().item() > 0.0  # 排除"根本没量化"的假通过


def test_group_independent():
    """一组的量程放大 100x，另一组的量化结果必须**逐位不变**。"""
    torch.manual_seed(0)
    x = torch.cat([torch.randn(1, 1, 8, 32) * 0.1, torch.randn(1, 1, 8, 32) * 10.0], dim=-1)
    x2 = torch.cat([x[..., :32], x[..., 32:] * 100.0], dim=-1)
    x, x2 = x.to(torch.float16), x2.to(torch.float16)

    # 按 32 分组：第一组不被第二组的离群值污染
    assert torch.equal(roundtrip_int4(x, 32)[..., :32], roundtrip_int4(x2, 32)[..., :32])
    # 不分组（整条 64 一组）：第二组的量程把尺子撑大 100 倍，第一组误差跟着爆 —— 这就是要分组的理由
    want = x[..., :32].float()
    grouped_err = (roundtrip_int4(x, 32)[..., :32].float() - want).abs().max().item()
    flat_err = (roundtrip_int4(x, 64)[..., :32].float() - want).abs().max().item()
    assert flat_err > 10 * grouped_err


def test_constant_tensor_exact():
    """`max == min`：step 被 clamp 到 1e-8，q 全 0，往返无损。"""
    x = torch.full((1, 2, 5, HEAD_DIM), 0.5, dtype=torch.float16)
    assert torch.equal(roundtrip_int4(x, GROUP_K), x)
    z = torch.zeros(1, 2, 5, HEAD_DIM, dtype=torch.float16)
    assert torch.equal(roundtrip_int4(z, GROUP_K), z)


@pytest.mark.parametrize("group", [16, 32, 128, 0])
def test_shapes_and_groups(group: int):
    x = _x()
    packed, mins, steps = quantize_int4(x, group)
    g = group or HEAD_DIM
    groups = HEAD_DIM // g
    assert packed.shape == (1, KV_HEADS, ROWS, groups, g // 2)
    assert mins.shape == steps.shape == (1, KV_HEADS, ROWS, groups)
    assert packed.dtype == torch.uint8
    assert mins.dtype == steps.dtype == torch.float16
    assert (packed & 0x0F).max().item() <= NIBBLE_MAX  # 两个半字节都只用了 4 位
    assert (packed >> 4).max().item() <= NIBBLE_MAX

    out = dequantize_int4(packed, mins, steps, group)
    assert out.shape == x.shape
    assert out.dtype == torch.float16


def test_bytes_accounting():
    """Qwen3-4B：36 层 / 8 KV 头 / head_dim 128。"""
    c = bytes_per_token(num_slots=36, num_kv_heads=8, head_dim=128)
    assert c["fp16_kv"] == 36 * 8 * 128 * 2 * 2 == 147456
    assert abs(c["int4_kv"] / 1024 - 42) < 1.0
    assert abs(c["ratio"] - 3.46) < 0.01


# ---- 缓存层（需要 CUDA）----


def _caches(max_len: int = 16):
    from nova.cache import StaticKVCache
    from nova.kvquant import QuantRoundTripCache

    kw = dict(num_slots=1, num_kv_heads=2, head_dim=HEAD_DIM, max_len=max_len)
    return StaticKVCache(**kw, device="cpu"), QuantRoundTripCache(**kw, device="cpu")


def _kv(n: int):
    torch.manual_seed(0)
    shape = (1, 2, n, HEAD_DIM)
    return torch.randn(*shape).to(torch.float16), torch.randn(*shape).to(torch.float16)


def test_cache_newest_token_not_stale():
    """回归：`StaticKVCache.update` 不推进 pos，所以已写入范围是 `pos + 1`。

    写成 `pos` 会让最新 token 从 scratch 读回 0 —— 实测直接把 needle 打成 0/4。
    """
    base, quant = _caches()
    k, v = _kv(1)
    bk, bv = base.update(k, v, 0)
    qk, qv = quant.update(k, v, 0)

    assert torch.equal(qk, roundtrip_int4(bk, GROUP_K))
    assert torch.equal(qv, roundtrip_int4(bv, quant.group_v))
    assert qk[0, :, 0, :].abs().max().item() > 0.0  # 不是 scratch 的 0
    assert int(quant.pos.item()) == int(base.pos.item()) == 0  # pos 语义与 StaticKVCache 一致
    assert torch.equal(qk[0, :, 1:, :], torch.zeros_like(qk[0, :, 1:, :]))


def test_cache_prefill_appends_quantized():
    base, quant = _caches()
    k, v = _kv(5)
    bk, bv = base.append_prefill(k, v, 0)
    qk, qv = quant.append_prefill(k, v, 0)

    assert torch.equal(qk, roundtrip_int4(bk, GROUP_K))
    assert torch.equal(qv, roundtrip_int4(bv, quant.group_v))
    assert torch.equal(qk[0, :, 5:, :], torch.zeros_like(qk[0, :, 5:, :]))


def test_update_multi_token_covers_all_written_positions():
    """回归：prefill 走的是 `update`（多 token），已写入范围是 `pos + n`，不是 `pos + 1`。

    写成 `pos + 1` 只会把第 0 个位置拷进 scratch，1..n-1 留在 scratch 的 0 上 ——
    注意力看到一片零，15 token 的 prompt 就足以让输出变胡言乱语（实测 needle 0/4）。
    """
    base, quant = _caches()
    k, v = _kv(5)
    bk, bv = base.update(k, v, 0)  # 模型 prefill 的真实调用路径（多 token）
    qk, qv = quant.update(k, v, 0)

    assert torch.equal(qk, roundtrip_int4(bk, GROUP_K))
    assert torch.equal(qv, roundtrip_int4(bv, quant.group_v))
    assert qk[0, :, 1:5, :].abs().max().item() > 0.0  # 不是 scratch 的 0
    assert qv[0, :, 1:5, :].abs().max().item() > 0.0
    assert torch.equal(qk[0, :, 5:, :], torch.zeros_like(qk[0, :, 5:, :]))


def test_cache_residual_window_keeps_tail_fp16():
    """`residual=2` 时最近 2 个位置保持 fp16（KIVI 的残留窗）。"""
    from nova.kvquant import QuantRoundTripCache

    quant = QuantRoundTripCache(
        num_slots=1, num_kv_heads=2, head_dim=HEAD_DIM, max_len=16, residual=2, device="cpu"
    )
    k, v = _kv(5)
    qk, qv = quant.append_prefill(k, v, 0)
    assert torch.equal(qk[0, :, 3:5, :], k[0, :, 3:5, :])  # 最近 2 个原样
    assert not torch.equal(qk[0, :, :3, :], k[0, :, :3, :])  # 前面那些被量化过
    assert torch.equal(qk[0, :, 5:, :], torch.zeros_like(qk[0, :, 5:, :]))  # 后面还没写


# ---------------------------------------------------------------- 8 位（E3）


def _outlier_kv(n_tokens: int = 256, dim: int = HEAD_DIM):
    """带**通道级离群**的 K/V（模拟实测里第 0 层 65x 那种）：第 7 通道整体放大 40x。"""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 2, n_tokens, dim, generator=g, dtype=torch.float16)
    x[:, :, :, 7] *= 40
    return x


def test_int8_error_far_below_int4():
    """判据：int8 的相对误差要**远小于** int4（文献说 8 位几乎无损）。"""
    x = _outlier_kv()
    e4 = (roundtrip_int4(x, GROUP_K).float() - x.float()).norm() / x.float().norm()
    e8 = (roundtrip_int8(x, GROUP_K, "channel").float() - x.float()).norm() / x.float().norm()
    assert float(e8) < float(e4) / 5, f"int8 {float(e8):.4f} 没比 int4 {float(e4):.4f} 好 5 倍"
    assert float(e8) < 0.02, f"int8 相对 L2 误差 {float(e8):.4f} 偏大"


def test_int8_token_axis_handles_channel_outlier():
    """**按 token 维分组**（每通道一条跨 token 的尺子）应当比按通道分组更能扛通道级离群。"""
    x = _outlier_kv()
    e_chan = (roundtrip_int8(x, GROUP_K, "channel").float() - x.float()).norm() / x.float().norm()
    e_tok = (roundtrip_int8(x, 64, "token").float() - x.float()).norm() / x.float().norm()
    assert float(e_tok) <= float(e_chan), f"token 维 {float(e_tok):.4f} 不比 channel 维 {float(e_chan):.4f} 好"


def test_int8_constant_exact_and_shapes():
    """常量张量无损；两个方向的形状都回到原样。"""
    const = torch.full((1, 2, 8, HEAD_DIM), 3.5, dtype=torch.float16)
    for axis, gs in (("channel", GROUP_K), ("token", 4)):
        q, mins, steps = quantize_int8(const, gs, axis)
        assert q.shape == const.shape, f"{axis} 方向形状不对：{tuple(q.shape)}"
        assert torch.equal(dequantize_int8(q, mins, steps, gs, axis), const)


def test_fp8_per_token_beats_per_tensor_with_outlier():
    """有通道级离群时，per-token scale 必须明显好于 per-tensor（离群会拉低整体精度）。"""
    x = _outlier_kv()
    e_tok = (roundtrip_fp8(x, "token").float() - x.float()).norm() / x.float().norm()
    e_ten = (roundtrip_fp8(x, "tensor").float() - x.float()).norm() / x.float().norm()
    assert float(e_tok) < float(e_ten), f"per-token {float(e_tok):.4f} 没比 per-tensor {float(e_ten):.4f} 好"
    assert float(e_tok) < 0.03


def test_bytes_accounting_8bit():
    """记账：int8 约 1.86x（判据 1.85–1.95），fp8 约 1.97x，且都远好于 int4 的 3.46x。"""
    c8 = bytes_per_token(36, 8, 128, bits=8)
    c8t = bytes_per_token(36, 8, 128, bits=8, k_axis="token", token_group=64)
    cf8 = bytes_per_token(36, 8, 128, bits="fp8")
    assert 1.85 <= c8["ratio"] <= 1.95, f"int8 ratio {c8['ratio']:.3f} 不在判据区间"
    assert 1.85 <= c8t["ratio"] <= 1.95
    assert cf8["ratio"] >= 1.95
    assert abs(c8["int8_kv"] / 1024 - 77.6) < 1.0, "int8 应约 77.6 KiB/token"
    assert c8["int8_kv"] < bytes_per_token(36, 8, 128)["int4_kv"] * 2.0


def test_cache_supports_int8_kind():
    """`QuantRoundTripCache(kind="int8")` 走的是 int8 往返，且尾部仍是 0。"""
    from nova.kvquant import QuantRoundTripCache

    quant = QuantRoundTripCache(
        num_slots=1, num_kv_heads=2, head_dim=HEAD_DIM, max_len=16, kind="int8", device="cpu"
    )
    k, v = _kv(5)
    qk, qv = quant.append_prefill(k, v, 0)
    # 只比**写过的那一段**：缓存里 5..15 槽是 0（还没写），拿它们去量化会得到非 0 的小数
    assert torch.equal(qk[:, :, :5, :], roundtrip_int8(k[:, :, :5, :], GROUP_K, "channel"))
    assert torch.equal(qk[0, :, 5:, :], torch.zeros_like(qk[0, :, 5:, :]))
