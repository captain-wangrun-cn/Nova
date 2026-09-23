r"""int8 KV **融合注意力核**（解码专用）。

## 这个模块解决什么

`kvquant.QuantRoundTripCache` 是**模拟版**：KV 仍以 fp16 常驻，只是"喂给 SDPA 之前先做一次
int8 往返"。它证明得了精度，**证明不了收益** —— 显存与带宽一分没省。

本模块把「读 int8 → 反量化 → 打分 → softmax → 加权 V」融合进 kernel，
于是 KV 的 **1.88x 记账**（`int8t64`）才真的兑现成带宽/显存。

靶子由 **D38**（选型：int8 + K 按 token 维分组，每 64 token 一条尺子）与
**D40**（可行性：int8 解包+反量化能打到纯读上限的 86.3%）定死。

## 验收判据

**与"先还原再算"的输出 ULP 级一致**：

1. 逐元素偏差 **≤ 2 ULP(fp16)**，**或** 绝对偏差 **≤ 1e-5**；
2. 参考统计量：逐位相同的元素占比（随张量元素数下降，不作为判据）。

> **判据改过两次，都记录在 [reports/kv-int8-fused-attn.md](../../reports/kv-int8-fused-attn.md) 与 D42**：
> 1. 原稿还写了"≥99% 逐位相同"，实测后**降级为统计量** —— 这个比例随元素数单调下降
>    （元素越多越有机会落在 fp16 舍入边界上），16K 下必然被击穿，测的是运气不是正确性。
> 2. 真数据复核发现"有些输出元素是大数相消出来的"，此时 fp32 归约顺序差异会被放大成几十 ULP，
>    于是补上**绝对兜底 1e-5**（= fp32 归约噪声的量级，实测全层最大 3.9e-6，比该层自身 fp16
>    舍入噪声低一个数量级）。用小元素的 ULP 去卡一个绝对噪声，等于要求两种求和顺序逐位相同。

参考 = `kvquant.dequantize_int8(kind="int8t64")` → `SDPA(MATH 后端)`，
即现有代码路径真正会看到的那套数值（`LeanAttention` 也是把 fp16 交给 SDPA）。

## 数值上必须对齐的三件事（否则 ULP 判据过不了）

| # | 参考实现怎么做 | 本核怎么做 |
|---|---|---|
| 1 | 反量化用**存下来的 fp16 尺子**：`q*fp32(min/step)` → **舍到 fp16** | 同样：fp32 乘加 → `.to(fp16)` |
| 2 | SDPA math 后端把 fp16 的 Q/K/V **上转 fp32** 再算 | 打分走 **fp16 张量核 + fp32 累加**（乘积精确，见下）；PV 走 **拆成 hi+lo 两半的 fp16 张量核** |
| 3 | `softmax` 在 fp32 里做（减行最大值 → exp → 归一化） | 在线 softmax，同样在 fp32 里 |

**为什么敢用 fp16 张量核**（这一条是实测逼出来的，不是想当然）：

| 写法 | 16K 单层耗时 | 最大 ULP |
|---|---:|---:|
| 两个点积都 `input_precision="ieee"`（fp32 FMA） | 1.34 ms | 1.00 |
| 打分 fp16 张量核 + PV `ieee` | 0.57 ms | 1.00 |
| 打分 fp16 张量核 + **PV 把 p 直接舍成 fp16** | 0.30 ms | **70.0 ✗** |
| 打分 fp16 张量核 + **PV 拆成 hi+lo 两个 fp16**（采用） | **0.29 ms** | **1.00 ✅** |
| 打分 fp16 张量核 + PV `tf32` | 0.46 ms | 81.0 ✗ |
| 打分 fp16 张量核 + PV `tf32x3` | 0.62 ms | 1.00 |

> ⚠️ 表里带 `ieee` 的两行是 **`num_warps=4`** 下测的：`tl.dot(..., input_precision="ieee")`
> 在本核里 **warps=8 实测输出异常**（未归因，最小复现没复现出来）。默认路径不依赖它 ——
> 但以后要在别处用 ieee dot，先按 warps 验一遍。

1. **打分**：q/k 本来就是 fp16 存下来的，`fp16×fp16` 的乘积有 22 位有效位，fp32 的 24 位装得下 ⇒
   乘积**精确**，只有累加顺序与参考不同（量级 1e-7）。所以换张量核不损失精度，白拿 2.4x。
2. **PV**：p 是 fp32 的概率值，**直接舍成 fp16 会掉 70 ULP**（相对误差 5e-4 直接进输出）。
   拆成 `p_hi + p_lo`（各 11 位，合起来 22 位）再各做一次张量核点积，误差回到 fp32 舍入量级 ⇒
   仍 ≤1 ULP，且比 `ieee` 快 2x。`tf32`（10 位尾数）同理不行。
3. 代价：PV 做两次点积。**这是实测选出来的最优点**，不是猜的。

差异只剩**归约顺序**（在线 softmax 的分段 rescale、fp32 求和的次序），量级 1e-7 相对，
落到 fp16 输出上就是"个别元素差 1 ULP"。

## 结构

```
grid = (kv_heads, n_splits)
  └─ _int8_attn_partial_kernel   每个 program 扫 CHUNK 个 token，产出在线 softmax 的
                                 (m, l, acc) 部分和 —— 每个 program 一次处理**共享同一
                                 KV 头的那 GQ 个 Q 头**，K/V 只读一遍（GQA 不放大流量）
  └─ _int8_attn_combine_kernel   把 n_splits 份部分和合并、归一化、写回 fp16
```

**GQA 为什么要这么切**：Qwen3-VL-4B 是 32 Q 头 / 8 KV 头。若按 Q 头切 32 个 program，
同一个 KV 头会被 4 个 program 各读一遍 —— 流量 ×4。按 KV 头切则每个 program 摊到 4 个 Q 头，
K/V 每个字节只读一次。代价是 `tl.dot` 的最小 M=16，4 个 Q 头要**补零到 16 行**（算力浪费可忽略，
访存才是瓶颈）。

## 还没做的（写清楚，别当已经做了）

- **没接进解码路径**：`GraphDecoder` 还是 fp16 cache + SDPA。接进去要连带解决
  "流式写入时分组尺子还没定"（KIVI 式 fp16 residual 窗口）与 CUDA Graph 的形状恒定，
  是下一步的事。
- **没做 prefill**：本核只处理 `q_len == 1`（解码一步）。prefill 是 `q_len > 1` + 因果掩码，
  另一套写法。
"""

from __future__ import annotations

import torch

from .kernels import HAS_TRITON, triton_available

if HAS_TRITON:
    import triton
    import triton.language as tl


GROUP_T = 64  # token 维分组：每 64 个 token 一条尺子（= kvquant 的 int8t64）
CHUNK = 512  # 每个 program 负责的 token 区间（split-K 的粒度）；见报告第四节的扫描
PAD_M = 16  # tl.dot 要求 M/N/K ≥ 16；GQ=4 个 Q 头要补到 16 行


if HAS_TRITON:

    @triton.jit
    def _int8_attn_partial_kernel(
        Q, KQ, KMN, KST, VQ, VMN, VST, PM, PL, PA,
        used,
        s_qh, s_kh, s_kt, s_mh, s_mg, s_ph, s_ah,
        GQ: tl.constexpr, D: tl.constexpr, GT: tl.constexpr,
        PM_ROWS: tl.constexpr, CHUNK: tl.constexpr, SCALE: tl.constexpr,
    ):
        """一个 KV 头 × 一个 token 区间 → 在线 softmax 的 (m, l, acc) 部分和。

        - `KQ`/`VQ`：[H_kv, L, D] uint8；`KMN`/`KST`：[H_kv, ceil(L/GT), D] fp16
        - `PM`/`PL`：[H_kv, n_splits, PM_ROWS] fp32；`PA`：[H_kv, n_splits, PM_ROWS, D] fp32
        - 循环步长 = `GT`，且 `start` 是 `GT` 的整数倍 ⇒ **每个 t0 正好是一个分组的起点**，
          尺子按 `[D]` 向量读一次（不是每 token 读一遍，省 64 倍的元数据流量）。
        """
        pid_h = tl.program_id(0)  # KV 头
        pid_s = tl.program_id(1)  # token 区间

        offs_m = tl.arange(0, PM_ROWS)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < GQ
        qh = pid_h * GQ + offs_m
        # q 本来就是 fp16 存的 —— 直接以 fp16 进张量核，乘积精确（见模块头第 2 条）
        q16 = tl.load(Q + qh[:, None] * s_qh + offs_d[None, :], mask=m_ok[:, None], other=0.0)

        m_i = tl.full((PM_ROWS,), float("-inf"), tl.float32)
        l_i = tl.zeros((PM_ROWS,), tl.float32)
        acc = tl.zeros((PM_ROWS, D), tl.float32)

        start = pid_s * CHUNK
        stop = tl.minimum(start + CHUNK, used)

        kq_h = KQ + pid_h * s_kh
        vq_h = VQ + pid_h * s_kh
        kmn_h = KMN + pid_h * s_mh
        kst_h = KST + pid_h * s_mh
        vmn_h = VMN + pid_h * s_mh
        vst_h = VST + pid_h * s_mh

        for t0 in tl.range(start, stop, GT):
            offs_t = t0 + tl.arange(0, GT)
            tm = offs_t < stop
            g = t0 // GT
            md = g * s_mg + offs_d

            kst = tl.load(kst_h + md).to(tl.float32)
            kmn = tl.load(kmn_h + md).to(tl.float32)
            kq = tl.load(kq_h + offs_t[:, None] * s_kt + offs_d[None, :],
                         mask=tm[:, None], other=0)
            # 与 `dequantize_int8` 逐位对齐：fp32 乘加，再舍到 fp16
            k16 = (kq.to(tl.float32) * kst[None, :] + kmn[None, :]).to(tl.float16)

            s = tl.dot(q16, tl.trans(k16), out_dtype=tl.float32) * SCALE
            s = tl.where(tm[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            m_i = m_new

            vst = tl.load(vst_h + md).to(tl.float32)
            vmn = tl.load(vmn_h + md).to(tl.float32)
            vq = tl.load(vq_h + offs_t[:, None] * s_kt + offs_d[None, :],
                         mask=tm[:, None], other=0)
            v16 = (vq.to(tl.float32) * vst[None, :] + vmn[None, :]).to(tl.float16)
            # p 拆成 hi+lo（各 11 位 ⇒ 合起来 22 位）再走张量核：比 ieee 快 2x，ULP 仍 ≤1
            p_hi = p.to(tl.float16)
            p_lo = (p - p_hi.to(tl.float32)).to(tl.float16)
            acc += tl.dot(p_hi, v16, out_dtype=tl.float32)
            acc += tl.dot(p_lo, v16, out_dtype=tl.float32)

        tl.store(PM + pid_h * s_ph + pid_s * PM_ROWS + offs_m, m_i)
        tl.store(PL + pid_h * s_ph + pid_s * PM_ROWS + offs_m, l_i)
        tl.store(PA + pid_h * s_ah + pid_s * PM_ROWS * D + offs_m[:, None] * D + offs_d[None, :], acc)


    @triton.jit
    def _int8_attn_combine_kernel(
        PM, PL, PA, OUT,
        n_splits,
        s_ph, s_ah, s_oh,
        GQ: tl.constexpr, D: tl.constexpr, PM_ROWS: tl.constexpr,
    ):
        """合并 n_splits 份 (m, l, acc)：`out = Σ acc_i·e^(m_i−m) / Σ l_i·e^(m_i−m)`。

        逐 split 串行累加（而不是把 [n_splits, 16, D] 一次性读进来），寄存器占用与 split 数无关。
        """
        pid_h = tl.program_id(0)
        offs_m = tl.arange(0, PM_ROWS)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < GQ

        m_run = tl.full((PM_ROWS,), float("-inf"), tl.float32)
        l_run = tl.zeros((PM_ROWS,), tl.float32)
        acc = tl.zeros((PM_ROWS, D), tl.float32)
        base_m = PM + pid_h * s_ph
        base_l = PL + pid_h * s_ph
        base_a = PA + pid_h * s_ah
        for i in tl.range(0, n_splits):
            m_i = tl.load(base_m + i * PM_ROWS + offs_m)
            l_i = tl.load(base_l + i * PM_ROWS + offs_m)
            a_i = tl.load(base_a + i * PM_ROWS * D + offs_m[:, None] * D + offs_d[None, :])
            m_new = tl.maximum(m_run, m_i)
            w_run = tl.exp(m_run - m_new)
            w_i = tl.exp(m_i - m_new)
            l_run = l_run * w_run + l_i * w_i
            acc = acc * w_run[:, None] + a_i * w_i[:, None]
            m_run = m_new

        # ⚠️ 一个 KV 头对应 GQ 个 Q 头：base = pid_h * GQ（不是 pid_h）
        tl.store(OUT + (pid_h * GQ + offs_m)[:, None] * s_oh + offs_d[None, :],
                 (acc / l_run[:, None]).to(tl.float16), mask=m_ok[:, None])


def _check(x: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} 必须在 CUDA 上")
    if x.dtype != dtype:
        raise ValueError(f"{name} 必须是 {dtype}，收到 {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"{name} 必须连续")


def int8_attn_decode(
    q: torch.Tensor,
    kq: torch.Tensor,
    kmn: torch.Tensor,
    kst: torch.Tensor,
    vq: torch.Tensor,
    vmn: torch.Tensor,
    vst: torch.Tensor,
    *,
    used: int | None = None,
    group: int = GROUP_T,
    chunk: int = CHUNK,
    num_warps: int = 8,
    num_stages: int = 2,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """解码一步的 int8 融合注意力。

    - `q`：fp16 `[1, H_q, 1, D]`（已过 RoPE）
    - `kq`/`vq`：uint8 `[1, H_kv, L, D]`；`kmn`/`kst`/`vmn`/`vst`：fp16 `[1, H_kv, ceil(L/group), D]`
    - `used`：本步真正有效的 KV 长度（默认 `L`）；只读 `[0, used)`
    - `scratch` / `out`：可选的**预分配**缓冲区。**CUDA Graph 捕获时必须给**
      （图内不许分配）；形状要按 `n_splits = ceil(used/chunk)` 备好，`out` 为 `[1, H_q, 1, D]`。

    返回 fp16 `[1, H_q, 1, D]`，**与"先还原再算"（`dequantize_int8` → SDPA math）ULP 级一致**。
    """
    if not triton_available():
        raise RuntimeError("Triton 不可用")

    if q.dim() != 4 or q.shape[2] != 1:
        raise ValueError(f"本核只处理解码一步（q 形状 [1, H_q, 1, D]），收到 {tuple(q.shape)}")
    if q.shape[0] != 1:
        raise ValueError("batch 必须为 1")
    _check(q, "q", torch.float16)
    for name, t, dt in (("kq", kq, torch.uint8), ("vq", vq, torch.uint8),
                        ("kmn", kmn, torch.float16), ("kst", kst, torch.float16),
                        ("vmn", vmn, torch.float16), ("vst", vst, torch.float16)):
        _check(t, name, dt)

    b, hq, _, d = q.shape
    h_kv, length = kq.shape[1], kq.shape[2]
    if kq.shape != vq.shape:
        raise ValueError(f"kq {tuple(kq.shape)} 与 vq {tuple(vq.shape)} 形状不一致")
    if kmn.shape != kst.shape or kmn.shape != vmn.shape or kmn.shape != vst.shape:
        raise ValueError("四张尺子的形状必须一致")
    if kmn.shape[0] != 1 or kmn.shape[1] != h_kv or kmn.shape[3] != d:
        raise ValueError(f"尺子形状 {tuple(kmn.shape)} 与 KV {tuple(kq.shape)} 不符")
    if hq % h_kv:
        raise ValueError(f"Q 头 {hq} 不能被 KV 头 {h_kv} 整除")
    gq = hq // h_kv
    if gq > PAD_M:
        raise ValueError(f"每个 KV 头摊到 {gq} 个 Q 头，超过补零上限 {PAD_M}")
    if d & (d - 1):
        raise ValueError(f"head_dim {d} 必须是 2 的幂（tl.arange 要求）")
    if group & (group - 1):
        raise ValueError(f"group {group} 必须是 2 的幂（tl.arange 要求）")
    if chunk % group:
        raise ValueError(f"chunk {chunk} 必须是 group {group} 的整数倍（分块与分组必须对齐）")
    if kmn.shape[2] != -(-length // group):
        raise ValueError(f"尺子组数 {kmn.shape[2]} != ceil(L/group) = {-(-length // group)}")

    used = length if used is None else int(used)
    if not 0 <= used <= length:
        raise ValueError(f"used={used} 超出 [0, {length}]")
    if out is None:
        out = torch.empty((1, hq, 1, d), dtype=torch.float16, device=q.device)
    elif out.shape != (1, hq, 1, d) or out.dtype != torch.float16 or not out.is_cuda:
        raise ValueError(f"out 形状/类型不对：{tuple(out.shape)} {out.dtype}")
    if used == 0:
        out.zero_()
        return out

    n_splits = triton.cdiv(used, chunk)
    if scratch is None:
        pm = torch.empty((h_kv, n_splits, PAD_M), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pa = torch.empty((h_kv, n_splits, PAD_M, d), dtype=torch.float32, device=q.device)
    else:
        pm, pl, pa = scratch
        want = (h_kv, n_splits, PAD_M)
        if tuple(pm.shape) != want or tuple(pl.shape) != want or tuple(pa.shape) != (*want, d):
            raise ValueError(f"scratch 形状不对：要 {want} / {(*want, d)}，"
                             f"收到 {tuple(pm.shape)} / {tuple(pa.shape)}")

    # SDPA 的 scale：与 `LeanAttention.scaling` 同源（head_dim ** -0.5）
    scale = float(d) ** -0.5
    _int8_attn_partial_kernel[(h_kv, n_splits)](
        q, kq, kmn, kst, vq, vmn, vst, pm, pl, pa,
        used,
        q.stride(1), kq.stride(1), kq.stride(2), kmn.stride(1), kmn.stride(2),
        pm.stride(0), pa.stride(0),
        GQ=gq, D=d, GT=group, PM_ROWS=PAD_M, CHUNK=chunk, SCALE=scale,
        num_warps=num_warps, num_stages=num_stages,
    )
    _int8_attn_combine_kernel[(h_kv,)](
        pm, pl, pa, out,
        n_splits,
        # ⚠️ out 的形状是 [1, H_q, 1, D]：**头维是 dim 1**，不是 dim 0。
        # 传 stride(0) 会让 pid_h≥1 的 store 越界（写进别的张量的显存，且不报错）。
        pm.stride(0), pa.stride(0), out.stride(1),
        GQ=gq, D=d, PM_ROWS=PAD_M,
        num_warps=1, num_stages=1,
    )
    return out


def pack_int8_kv(k: torch.Tensor, v: torch.Tensor, group: int = GROUP_T):
    """把 fp16 的 K/V 打成融合核要的布局 —— 复用 `kvquant.quantize_int8(..., "token")`。

    **不重写量化器**：融合核的输入必须与 E3/E5 量过精度的那个量化器逐位同源，
    否则"精度已核查"这条结论就接不上了。
    """
    from .kvquant import quantize_int8

    kq, kmn, kst = quantize_int8(k, group, "token")
    vq, vmn, vst = quantize_int8(v, group, "token")
    return (kq.contiguous(), kmn.contiguous(), kst.contiguous(),
            vq.contiguous(), vmn.contiguous(), vst.contiguous())
