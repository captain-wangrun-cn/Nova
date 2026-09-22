"""Triton 融合 kernel。

背景（见 [reports/s2-speed-diagnosis.md](../../reports/s2-speed-diagnosis.md) 第八节）：
本机 torch 2.6.0 的 `F.rms_norm` **并没有融合实现** —— 它被拆成
`pow / mean / add / rsqrt / mul / _to_copy` 共 **18 个 kernel**，比 HF 的 eager 版（16 个）还多。
所以必须自己写。

约束：
- 只在 CUDA + 半精度下启用；其余情况调用方回退到 eager。
- 张量必须**最后一维连续**且等于归一化长度（本项目的 2560 / 128 都满足）。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _ensure_triton_cache() -> None:
    """Triton 默认把缓存写到 `C:\\Users\\<user>\\.triton`，本项目**禁止写 C 盘**。

    这里在 import triton 之前把缓存目录改到项目内的 `.tmp/triton-cache`。
    调用方显式设过 `TRITON_CACHE_DIR` 时不覆盖。
    """
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    project_root = Path(__file__).resolve().parents[2]
    target = project_root / ".tmp" / "triton-cache"
    try:
        target.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(target)
    except OSError:
        pass


_ensure_triton_cache()

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - 无 Triton 环境
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rms_norm_fwd(X, W, Y, N, eps, BLOCK: tl.constexpr):
        """一行一个 program。

        刻意对齐 HF `Qwen3VLTextRMSNorm` 的**算子顺序**：
            h32 = x.to(fp32); v = h32.pow(2).mean(-1); h32 = h32 * rsqrt(v + eps)
            return weight * h32.to(dtype)
        即"先转回 fp16 再乘 weight"，而不是在 fp32 里连乘 —— 这样能把与
        `impl="exact"` 的差异压到 fp16 舍入级别。
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        h = (x * rstd).to(Y.dtype.element_ty)
        tl.store(Y + row * N + cols, w.to(Y.dtype.element_ty) * h, mask=mask)


_DRIVER_OK: bool | None = None


def triton_available() -> bool:
    """Triton 的驱动初始化是**惰性**的：import 成功不代表能用。

    第一次调用时真正探一次（编译并跑一个最小 kernel），结果缓存。
    """
    global _DRIVER_OK
    if _DRIVER_OK is not None:
        return _DRIVER_OK
    if not (HAS_TRITON and torch.cuda.is_available()):
        _DRIVER_OK = False
        return False
    try:
        probe = torch.ones(64, device="cuda", dtype=torch.float16)
        weight = torch.ones(64, device="cuda", dtype=torch.float16)
        _rms_norm_fwd[(1,)](probe.view(1, 64), weight, torch.empty_like(probe).view(1, 64), 64, 1e-6, BLOCK=64)
        torch.cuda.synchronize()
        _DRIVER_OK = True
    except Exception:
        _DRIVER_OK = False
    return _DRIVER_OK


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """单 kernel 的 RMSNorm。最后一维 = 归一化维度。"""
    if not triton_available():
        raise RuntimeError("Triton 不可用")
    n = weight.shape[0]
    if x.shape[-1] != n:
        raise ValueError(f"最后一维 {x.shape[-1]} 与 weight 长度 {n} 不符")
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    orig_shape = x.shape
    rows = x.numel() // n
    x2 = x.view(rows, n)
    y = torch.empty_like(x2)
    block = max(triton.next_power_of_2(n), 16)
    _rms_norm_fwd[(rows,)](x2, weight, y, n, float(eps), BLOCK=block, num_warps=min(block // 64, 8))
    return y.view(orig_shape)



# ---------------------------------------------------------------------------
# NF4 4-bit GEMV：自己解包 + 自己算，替掉 bnb 的 `Linear4bit`
# ---------------------------------------------------------------------------
#
# **为什么要自己写**（见 reports/s3-graph-decode.md 第六节，已核查）：
# bnb 在 M=1 时不走 packed 4-bit GEMV —— 它把权重 dequant 成 fp16 workspace
# 再交给 cublas，于是既多搬 2 倍数据，又丢掉 4-bit 的全部带宽优势。
#
# **存储布局**（转换期一次性做好，见 quant.py）：
#   packed  [N, K//2]  uint8   与 bnb 原始布局一致，**沿 K 连续**
#   absmax  [K//64, N] fp32    double quant 已在转换期还原成 fp32
#   lut     [256]      int32   低 16 位 = fp16(code[a>>4])，高 16 位 = fp16(code[a&15])
#
# 为什么是 `[N, K//2]`（而不是转置成 `[K//2, N]`）：GEMV 的归约方向是 K，
# 把 K 放在**最后一维**就能用 `tl.sum(axis=1)` 在寄存器内归约。
#
# **码本查表是唯一的硬成本**（实测，见 reports/speed-path1-nf4-gemv.md）：
# 一次 `tl.load(LUT + idx)` 这种发散访存约 3.5 SM-cycle/warp，而一个 ALU 算子只要 1。
# 所以这里把**一个字节的两次查表压成一次**（表里直接存 fp16x2），
# 把查表次数从「每元素一次」降到「每字节一次」。
#
# 一个 absmax 块 = 64 个元素 = 32 个字节；字节内：高半字节 = 偶数元素，低半字节 = 奇数元素
# （已逐位复现，见 src/diagnostics/probe_nf4_format.py，max|diff| = 0.000e+00）。

NF4_BLOCK = 64   # 每个 absmax 覆盖的权重元素数
NF4_BYTES = 32   # 每个 absmax 覆盖的字节数（一字节装 2 个元素）

_NF4_DEFAULTS: dict[str, int] = {"block_n": 64, "block_m": 1, "num_warps": 2, "num_stages": 2}
_nf4_launch: dict[str, int] = dict(_NF4_DEFAULTS)

# 允许用环境变量覆盖（调参脚本用），例如 NOVA_NF4_BLOCK_N=32
for _k in list(_NF4_DEFAULTS):
    _v = os.environ.get(f"NOVA_NF4_{_k.upper()}")
    if _v:
        _nf4_launch[_k] = int(_v)


def nf4_launch_config() -> dict[str, int]:
    """当前 NF4 kernel 的 launch 参数。"""
    return dict(_nf4_launch)


def set_nf4_launch_config(**kwargs: int) -> dict[str, int]:
    """运行时调整 launch 参数（调参脚本用）。改完必须重新 capture CUDA Graph。"""
    for key, value in kwargs.items():
        if key not in _NF4_DEFAULTS:
            raise KeyError(f"未知的 launch 参数: {key}")
        _nf4_launch[key] = int(value)
    return dict(_nf4_launch)


def build_nf4_lut(code: torch.Tensor) -> torch.Tensor:
    """把 16 项 NF4 码本摊成 256 项 `int32` 表：低 16 位 = code[a>>4]，高 16 位 = code[a&15]。

    两项都是 **fp16 的位模式**，kernel 里用 bitcast 取出来 —— 这样一次发散访存
    就能拿到一个字节里两个元素的值。
    """
    c16 = code.detach().to(torch.float16).cpu()
    idx = torch.arange(256, dtype=torch.int64)
    even_bits = (c16[idx >> 4].view(torch.int16).to(torch.int64) & 0xFFFF)   # 高半字节 = 偶数元素
    odd_bits = (c16[idx & 15].view(torch.int16).to(torch.int64) & 0xFFFF)    # 低半字节 = 奇数元素
    val = (even_bits | (odd_bits << 16)) & 0xFFFFFFFF
    val = torch.where(val >= 2 ** 31, val - 2 ** 32, val)
    return val.to(torch.int32)


if HAS_TRITON:

    @triton.jit
    def _nf4_gemv_kernel(
        X, P, L, A, B, Out,
        M, N, K2, NB,
        stride_xm, stride_om,
        EVEN_N: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        """`out[m, n] = sum_k x[m, k] * code[idx] * absmax[k // 64, n] + bias[n]`。

        - `P` 是 `[N, K//2]` uint8；`L` 是 `[256]` int32（fp16x2 码本）；`A` 是 `[K//64, N]` fp32
        - 网格 `(cdiv(N, BLOCK_N), cdiv(M, BLOCK_M))`
        - 每个 program 处理 `BLOCK_N` 个输出列、`BLOCK_M` 个 token，沿 K 走完全程
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_b = tl.arange(0, 32)   # 一个 absmax 块内的 32 个字节
        offs_k = tl.arange(0, 64)   # 一个 absmax 块内的 64 个元素

        for mi in tl.static_range(BLOCK_M):
            m = pid_m * BLOCK_M + mi
            if m < M:
                acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                x_row = X + m * stride_xm
                for blk in tl.range(0, NB):
                    if EVEN_N:
                        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
                        am = tl.load(A + blk * N + offs_n)
                    else:
                        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :],
                                    mask=mask_n[:, None], other=0)
                        am = tl.load(A + blk * N + offs_n, mask=mask_n, other=0.0)
                    xs = tl.load(x_row + blk * 64 + offs_k)
                    # 偶数元素 = 高半字节，奇数元素 = 低半字节
                    x_even, x_odd = tl.split(tl.reshape(xs, (32, 2)))
                    # 一次发散访存拿到一个字节里两个元素的 fp16 位模式（低 16 = 偶数元素）
                    t = tl.load(L + p.to(tl.int32))
                    v_even = (t & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                    v_odd = (t >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                    # absmax 在 64 元素块内是常数 -> 先归约、最后再乘，省掉一整轮逐元素乘法
                    part = (tl.sum(v_even * x_even[None, :], axis=1)
                            + tl.sum(v_odd * x_odd[None, :], axis=1))
                    acc += part * am
                if HAS_BIAS:
                    if EVEN_N:
                        acc += tl.load(B + offs_n)
                    else:
                        acc += tl.load(B + offs_n, mask=mask_n, other=0.0)
                if EVEN_N:
                    tl.store(Out + m * stride_om + offs_n, acc.to(Out.dtype.element_ty))
                else:
                    tl.store(Out + m * stride_om + offs_n, acc.to(Out.dtype.element_ty), mask=mask_n)


def fused_nf4_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    absmax: torch.Tensor,
    lut: torch.Tensor,
    out_features: int,
    bias: torch.Tensor | None = None,
    **launch_overrides: int,
) -> torch.Tensor:
    """NF4 权重线性层：`x @ dequant(packed, absmax, code).T + bias`。

    `x` 必须是 CUDA 上的 fp16（本项目 compute_dtype 固定 fp16），最后一维 = K。
    """
    if not triton_available():
        raise RuntimeError("Triton 不可用")
    k = x.shape[-1]
    k2 = packed.shape[-1]
    n_blocks = absmax.shape[0]
    if k2 * 2 != k:
        raise ValueError(f"packed 最后一维 {k2} 与 K={k} 不匹配（应为 K//2）")
    if n_blocks * 64 != k:
        raise ValueError(f"absmax 行数 {n_blocks} 与 K={k} 不匹配（应为 K//64）")
    if packed.shape[0] != out_features or absmax.shape[1] != out_features:
        raise ValueError(f"权重形状与 out_features={out_features} 不匹配")

    n = out_features
    orig_shape = x.shape
    x2 = x.reshape(-1, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    m = x2.shape[0]

    cfg = dict(_nf4_launch)
    cfg.update(launch_overrides)
    block_n, block_m = cfg["block_n"], cfg["block_m"]
    even_n = (n % block_n) == 0

    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    _nf4_gemv_kernel[grid](
        x2, packed, lut, absmax, bias if bias is not None else x2, out,
        m, n, k2, n_blocks, x2.stride(0), out.stride(0),
        EVEN_N=even_n, HAS_BIAS=bias is not None,
        BLOCK_N=block_n, BLOCK_M=block_m,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out.view(*orig_shape[:-1], n)
