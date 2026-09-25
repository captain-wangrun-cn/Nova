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

from typing import Any

import torch
import torch.nn.functional as F

from .kernels import HAS_TRITON, triton_available
from .kvquant import dequantize_int8, quantize_int8

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
        USED, PG,
        s_qh, s_kh, s_kt, s_mh, s_mg, s_ph, s_ah,
        GQ: tl.constexpr, D: tl.constexpr, GT: tl.constexpr,
        PM_ROWS: tl.constexpr, CHUNK: tl.constexpr, SCALE: tl.constexpr,
    ):
        """一个 KV 头 × 一个 token 区间 → 在线 softmax 的 (m, l, acc) 部分和。

        - `KQ`/`VQ`：[H_kv, L, D] uint8；`KMN`/`KST`：[H_kv, ceil(L/GT), D] fp16
        - `PM`/`PL`：[H_kv, n_splits, PM_ROWS] fp32；`PA`：[H_kv, n_splits, PM_ROWS, D] fp32
        - 循环步长 = `GT`，且 `start` 是 `GT` 的整数倍 ⇒ **每个 t0 正好是一个分组的起点**，
          尺子按 `[D]` 向量读一次（不是每 token 读一遍，省 64 倍的元数据流量）。

        ## 为什么 `used` / `prefix_groups` 走**显存标量**而不是 Python 参数

        流式 cache 的"已量化前缀"每 64 个 token 涨一次。若把它做成编译期常量，CUDA Graph
        每 64 步就得重捕 —— 而且重捕之间图**读到的是过期前缀，会静默算错**。
        改成 `tl.load` 一个 1 元素张量后，前缀长度是**图内可变的运行期值**：
        形状不变（grid 按桶长固定），数值每步都能涨。实测 `tl.load` 读标量可用
        （`src/diagnostics/probe_kvattn_scalar.py`）。

        - `USED`：本步真正有效的 KV 长度（`[1]` int64）
        - `PG`：尺子数组里**真正有效**的组数（`[1]` int64）。流式 cache 会预分配 `ceil(L/GT)`
          条，当下只写了 `PG` 条 —— 越过它的组索引必须**夹住**，否则读到未初始化的尺子。
        """
        pid_h = tl.program_id(0)  # KV 头
        pid_s = tl.program_id(1)  # token 区间

        used = tl.load(USED)
        prefix_groups = tl.load(PG)

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
        # `prefix_groups`：尺子数组里真正有效的组数。**扫到那里就停**，别让超出的 token 去
        # 读未初始化的尺子（实测：不设上限直接 NaN / 垃圾，输出偏差 3.4）。
        stop = tl.minimum(stop, prefix_groups * GT)

        kq_h = KQ + pid_h * s_kh
        vq_h = VQ + pid_h * s_kh
        kmn_h = KMN + pid_h * s_mh
        kst_h = KST + pid_h * s_mh
        vmn_h = VMN + pid_h * s_mh
        vst_h = VST + pid_h * s_mh

        for t0 in tl.range(start, stop, GT):
            offs_t = t0 + tl.arange(0, GT)
            tm = offs_t < stop
            g = tl.minimum(t0 // GT, prefix_groups - 1)
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
        PM, PL, PA, OUT, LSE,
        n_splits,
        s_ph, s_ah, s_oh, s_lh,
        GQ: tl.constexpr, D: tl.constexpr, PM_ROWS: tl.constexpr,
    ):
        """合并 n_splits 份 (m, l, acc)：`out = Σ acc_i·e^(m_i−m) / Σ l_i·e^(m_i−m)`。

        逐 split 串行累加（而不是把 [n_splits, 16, D] 一次性读进来），寄存器占用与 split 数无关。

        `LSE`（可选，形状 `[H_q]`）写出 `log Σ exp(s)` = `m + log(l)` —— **int8 前缀与 fp16
        尾部合并**要靠它（两段各算一次注意力，再用 log-sum-exp 加权）。`LSE is None` 时跳过，
        老路径（不带尾部）逐位不变。
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
        # ⚠️ `l_run == 0` 表示**这一整批 split 都没有有效 token**（流式 cache 里前缀不足一组时
        # 就会这样：grid 按桶长固定，而 `used` 是运行期值）。此时不许做 0/0 —— 输出 0、
        # lse 给 -inf，调用方按 logsumexp 合并时权重自然是 0（前缀为空时结果 = 纯尾部）。
        l_safe = tl.maximum(l_run, 1e-30)
        tl.store(OUT + (pid_h * GQ + offs_m)[:, None] * s_oh + offs_d[None, :],
                 tl.where(l_run[:, None] > 0.0, acc / l_safe[:, None], 0.0).to(tl.float16),
                 mask=m_ok[:, None])
        if LSE is not None:
            tl.store(LSE + pid_h * GQ + offs_m,
                     tl.where(l_run > 0.0, m_run + tl.log(l_safe), float("-inf")), mask=m_ok)


def _check(x: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} 必须在 CUDA 上")
    if x.dtype != dtype:
        raise ValueError(f"{name} 必须是 {dtype}，收到 {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"{name} 必须连续")


# 设备标量的**常驻缓冲**（每个 device 一对）。图内不许分配张量 —— 若每次调用都新建
# `torch.tensor([used])`，CUDA Graph 捕获会直接报 "operation not permitted when stream
# is capturing"（实测）。所以复用一个常驻张量、只做原地 `fill_`。
_SCALAR_BUFFERS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _scalar_buffers(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    buf = _SCALAR_BUFFERS.get(key)
    if buf is None:
        buf = (torch.zeros(1, dtype=torch.int64, device=device),
               torch.zeros(1, dtype=torch.int64, device=device))
        _SCALAR_BUFFERS[key] = buf
    return buf


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
    lse_out: torch.Tensor | None = None,
    prefix_groups: int | torch.Tensor = -1,
    used_t: torch.Tensor | None = None,
    prefix_groups_t: torch.Tensor | None = None,
    n_splits: int | None = None,
) -> torch.Tensor:
    """解码一步的 int8 融合注意力。

    - `q`：fp16 `[1, H_q, 1, D]`（已过 RoPE）
    - `kq`/`vq`：uint8 `[1, H_kv, L, D]`；`kmn`/`kst`/`vmn`/`vst`：fp16 `[1, H_kv, ceil(L/group), D]`
    - `used`：本步真正有效的 KV 长度（默认 `L`）；只读 `[0, used)`
    - `scratch` / `out`：可选的**预分配**缓冲区。**CUDA Graph 捕获时必须给**
      （图内不许分配）；形状要按 `n_splits = ceil(used/chunk)` 备好，`out` 为 `[1, H_q, 1, D]`。
    - `lse_out`：可选的 fp32 `[1, H_q]`，写出 `log Σ exp(s)`。**只用于把 fp16 尾部并进来**
      （见 `KVInt8Cache.attend`）；不传时行为与以前完全一致。
    - `prefix_groups`：尺子数组里真正有效的组数（流式 cache 用；`-1` = 全部有效）。
    - `used_t` / `prefix_groups_t`：**显存标量**版本（各 1 元素，int64）。给了它们就**忽略**
      `used` / `prefix_groups`，由核内 `tl.load` 读取 —— 这是 CUDA Graph 能用的唯一姿势
      （流式 cache 的前缀每 64 个 token 涨一次，编译期常量会失效）。见 `KVInt8Cache`。
    - `n_splits`：覆盖 split 数（图内 split 数必须**恒定** = `ceil(L/chunk)`，
      而 `used` 是可变的运行期值）。

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
    if not isinstance(prefix_groups, torch.Tensor):
        prefix_groups = int(prefix_groups)
        if prefix_groups < 0:
            prefix_groups = -(-used // group)
        elif prefix_groups > kmn.shape[2]:
            raise ValueError(f"prefix_groups={prefix_groups} 超过尺子组数 {kmn.shape[2]}")

    # 走了设备标量：长度由核内 `tl.load` 决定，形状必须在调用前就定好（图内不能提前返回）
    used_is_device = used_t is not None
    if used_is_device:
        if tuple(used_t.shape) != (1,) or used_t.dtype != torch.int64 or not used_t.is_cuda:
            raise ValueError(f"used_t 必须是 CUDA int64 [1]，收到 {tuple(used_t.shape)} {used_t.dtype}")
        for name, t in (("prefix_groups_t", prefix_groups_t),):
            if t is None or tuple(t.shape) != (1,) or t.dtype != torch.int64 or not t.is_cuda:
                raise ValueError(f"{name} 必须是 CUDA int64 [1]")
    else:
        # 非图路径：把 Python 标量写进**常驻**缓冲（图内不许分配新张量）
        used_t, prefix_groups_t = _scalar_buffers(q.device)
        used_t.fill_(used)
        prefix_groups_t.fill_(int(prefix_groups))
    if out is None:
        out = torch.empty((1, hq, 1, d), dtype=torch.float16, device=q.device)
    elif out.shape != (1, hq, 1, d) or out.dtype != torch.float16 or not out.is_cuda:
        raise ValueError(f"out 形状/类型不对：{tuple(out.shape)} {out.dtype}")
    if lse_out is not None and (
        tuple(lse_out.shape) != (1, hq) or lse_out.dtype != torch.float32 or not lse_out.is_cuda
    ):
        raise ValueError(f"lse_out 形状/类型不对：{tuple(lse_out.shape)} {lse_out.dtype}")
    if not used_is_device and used == 0:
        out.zero_()
        if lse_out is not None:
            lse_out.fill_(float("-inf"))
        return out

    n_splits = triton.cdiv(used, chunk) if n_splits is None else int(n_splits)
    if n_splits < 1:
        raise ValueError(f"n_splits={n_splits} 必须 ≥ 1")
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
        used_t, prefix_groups_t,
        q.stride(1), kq.stride(1), kq.stride(2), kmn.stride(1), kmn.stride(2),
        pm.stride(0), pa.stride(0),
        GQ=gq, D=d, GT=group, PM_ROWS=PAD_M, CHUNK=chunk, SCALE=scale,
        num_warps=num_warps, num_stages=num_stages,
    )
    _int8_attn_combine_kernel[(h_kv,)](
        pm, pl, pa, out, lse_out,
        n_splits,
        # ⚠️ out 的形状是 [1, H_q, 1, D]：**头维是 dim 1**，不是 dim 0。
        # 传 stride(0) 会让 pid_h≥1 的 store 越界（写进别的张量的显存，且不报错）。
        pm.stride(0), pa.stride(0), out.stride(1),
        lse_out.stride(0) if lse_out is not None else 0,
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


# ---------------------------------------------------------------- fp16 尾部


def tail_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tail_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp16 **残差尾部**（分组尺子还没定死的那 ≤64 个 token）的注意力 + 它的 `log Σ exp(s)`。

    与 `LeanAttention` 同一条 SDPA 路径（同一后端、同一 `scale`），所以这一段的数值
    **逐位等于现状代码**；它与 int8 前缀的输出靠 `logsumexp` 加权合并（见 `KVInt8Cache.attend`）。

    - `q`：fp16 `[1, H_q, 1, D]`
    - `k`/`v`：fp16 `[1, H_kv, R, D]`，R 是尾部槽位数（固定）
    - `tail_mask`：**bool** `[1, 1, 1, R]`，True = 这一槽有效

    返回 `(out [1, H_q, 1, D], lse [1, H_q])`，都是 fp32 参与后续合并。
    """
    hq = q.shape[1]
    h_kv = k.shape[1]
    if hq % h_kv:
        raise ValueError(f"Q 头 {hq} 不能被 KV 头 {h_kv} 整除")
    gq = hq // h_kv
    k = k[:, :, None, :, :].expand(1, h_kv, gq, k.shape[2], k.shape[3]).reshape(1, hq, k.shape[2], k.shape[3])
    v = v[:, :, None, :, :].expand(1, h_kv, gq, v.shape[2], v.shape[3]).reshape(1, hq, v.shape[2], v.shape[3])
    scale = float(k.shape[-1]) ** -0.5
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=tail_mask, dropout_p=0.0, scale=scale)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~tail_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)[:, :, 0]
    # 掩码全空（前缀已经覆盖整条环）时 SDPA 会给垃圾值 —— 显式归零 / -inf，
    # 这样 `logsumexp` 合并里这一段的权重正好是 0。
    # ⚠️ **不许**写成 `if tail_mask.any()`：那是图内 CPU 同步，CUDA Graph 会捕获失败。
    any_valid = tail_mask.any()
    out = torch.where(any_valid, out, torch.zeros((), dtype=out.dtype, device=out.device))
    lse = torch.where(any_valid, lse, torch.full_like(lse, float("-inf")))
    return out, lse


# ---------------------------------------------------------------- 流式 int8 cache


class KVInt8Cache:
    r"""**流式 int8 KV cache**（①.5）：常驻只有 int8 + 一把尺子 + 一个 64 槽 fp16 尾部环。

    这是 ①.5 的载体（见 [reports/kv-int8-fused-attn.md](../../reports/kv-int8-fused-attn.md)、
    决策 D42）。靶子由 D38+D40 定死，**没有重新选型**：

    | # | 约束 | 落点 |
    |---|------|------|
    | 1 | int8、K 按 **token 维**每 64 个一组（D38） | 组的尺子必须等第 64 个 token 到齐才定死 |
    | 2 | CUDA Graph **形状恒定** | 全部张量预分配；长度靠**显存标量**在核内 `tl.load`（见下） |
    | 3 | 与"先还原再算"ULP 一致 | 前缀走融合核、尾部走 fp16 SDPA，两段用 `logsumexp` 合并 |

    ## 三段结构

    ```
    [0, Q)      int8：Q = 64·(U//64)，尺子已定死的整组（U = 已写入总数）
    [Q, U)      fp16 尾部环：最近 ≤64 个 token，槽位 = 绝对位置 mod 64
    ```

    ## 为什么 `used` 必须走显存标量（这是本模块的关键设计，别再退回编译期常量）

    `Q` 每 64 个 token 涨一次。若把 `Q` 当编译期常量：

    - 想把 `Q` 冻结进图 → 图**每 64 步就过期**，而且过期不是"变慢"而是**静默算错**
      （把刚写满的组仍当 fp16 尾部，而环槽已被更新的 token 覆盖 ⇒ 那段历史凭空消失）；
    - 想每 64 步重捕 → 每 64 个 token 一次 capture，代价远超收益。

    所以 `used` / `prefix_groups` 改成**显存里的 1 元素张量**，核内用 `tl.load` 读。
    形状（grid）按**桶长**固定，数值每步都能涨 —— 这才是"形状恒定"的正确解法。
    可行性已核查：`tl.load` 读 1 元素张量可取到更新后的值。

    ## 解码步里做了什么（每层、每步，全部图内安全）

    1. `update()`：把 K/V 写进环槽 `pos % 64`；
    2. **无条件**把整条环（64 个 token）量化成 int8，写进第 `pos // 64` 组 —— 部分填充时
       这一组虽然尺子不对，但**不会被读到**（`Q` 只到整组边界，见第 3 步）；
       写满那一刻（`pos % 64 == 63`）同一层内先写后量化 ⇒ 尺子正好定死；
    3. 用 `pos` 推出两个设备标量：`_written = pos+1`、`_qlen = ((pos+1)//64)*64`、
       `_pg = _qlen//64`；
    4. `attend()`：int8 前缀（融合核，`used=_qlen`）+ 环尾部（SDPA，mask 由设备标量算）
       两段 `logsumexp` 合并。

    ## 与"先还原再算"的差别（写清楚，别误读）

    参考实现把 `[0, U)` **整条**量化；本 cache 让 `[Q, U)` 那 ≤63 个 token **保持 fp16**。
    差别只在最后那一组的"尺子取多宽"，且方向是**更准**（少一次量化）。

    ## prefill 为什么走 fp16 暂存区

    融合核只处理 `q_len == 1`；prefill 是变长 + 因果掩码，另一套写法（模块头"还没做的"）。
    所以 prefill 期间用一块**按实际 prompt 长度分配**的 fp16 暂存区（`begin_prefill` 建、
    `finish_prefill` 释放）。不按 `max_len` 分配，是 8GB 显存下的硬要求：
    `max_len=8192` 的整段暂存区就是 2.7 GiB。

    ## 已知不兼容

    - **S4 记忆注入**：`memory.MemoryStore.inject` 直接写 `cache.key_cache[slot]`，
      而本 cache 的常驻是 int8 + 环 —— 两条路不通用，`quant="int8"` 下不要开记忆注入。
    - **跨桶 `GraphDecoder.grow()`**：int8 缓冲、尺子、环、标量都要重建，暂不支持。
    """

    def __init__(
        self,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        num_q_heads: int,
        batch: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        group: int = GROUP_T,
        chunk: int = CHUNK,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> None:
        if head_dim & (head_dim - 1):
            raise ValueError(f"head_dim {head_dim} 必须是 2 的幂（tl.arange 要求）")
        if num_q_heads % num_kv_heads:
            raise ValueError(f"Q 头 {num_q_heads} 不能被 KV 头 {num_kv_heads} 整除")
        if num_q_heads // num_kv_heads > PAD_M:
            raise ValueError(f"每个 KV 头摊到 {num_q_heads // num_kv_heads} 个 Q 头，"
                             f"超过补零上限 {PAD_M}")
        if group & (group - 1):
            raise ValueError(f"group {group} 必须是 2 的幂")
        if chunk % group:
            raise ValueError(f"chunk {chunk} 必须是 group {group} 的整数倍")

        self.num_slots = int(num_slots)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_len = int(max_len)
        self.num_q_heads = int(num_q_heads)
        self.batch = int(batch)
        self.dtype = dtype
        self.device = device
        self.group = int(group)
        self.chunk = int(chunk)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

        n_groups = -(-self.max_len // self.group)
        dshape = (batch, self.num_kv_heads, self.max_len, self.head_dim)
        sshape = (batch, self.num_kv_heads, n_groups, self.head_dim)
        tshape = (batch, self.num_kv_heads, self.group, self.head_dim)
        # **每层一份** —— 36 个槽位共用一份会互相覆盖
        self.kq = [torch.zeros(dshape, dtype=torch.uint8, device=device) for _ in range(self.num_slots)]
        self.vq = [torch.zeros(dshape, dtype=torch.uint8, device=device) for _ in range(self.num_slots)]
        self.kmn = [torch.zeros(sshape, dtype=dtype, device=device) for _ in range(self.num_slots)]
        self.kst = [torch.zeros(sshape, dtype=dtype, device=device) for _ in range(self.num_slots)]
        self.vmn = [torch.zeros(sshape, dtype=dtype, device=device) for _ in range(self.num_slots)]
        self.vst = [torch.zeros(sshape, dtype=dtype, device=device) for _ in range(self.num_slots)]
        self.tail_k = [torch.zeros(tshape, dtype=dtype, device=device) for _ in range(self.num_slots)]
        self.tail_v = [torch.zeros(tshape, dtype=dtype, device=device) for _ in range(self.num_slots)]

        # `pos`：写入游标（与 `StaticKVCache` 同语义 —— prefill 由解码器按块设，解码由 `_body` 收尾 `+= 1`）
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.first = torch.zeros(1, dtype=torch.long, device=device)
        # **设备标量**（解码步的全部长度信息；图内零 `.item()`）
        self._written = torch.zeros(1, dtype=torch.long, device=device)  # 已写入总数 U
        self._qlen = torch.zeros(1, dtype=torch.long, device=device)      # int8 前缀长度 Q
        self._pg = torch.zeros(1, dtype=torch.long, device=device)        # 有效组数 Q//group

        self._idx64 = torch.arange(self.group, dtype=torch.long, device=device)
        self._arange = torch.arange(self.max_len, dtype=torch.long, device=device)
        self._mask_true = torch.ones((1, 1, 1, self.group), dtype=torch.bool, device=device)
        self._zero = torch.zeros((), dtype=dtype, device=device)
        self._ninf = torch.full((), float("-inf"), dtype=dtype, device=device)

        # split 数按**桶长**固定（图内 split 数必须恒定）
        self.n_splits = -(-self.max_len // self.chunk)
        sshape2 = (self.num_kv_heads, self.n_splits, PAD_M)
        self.pm = torch.zeros(sshape2, dtype=torch.float32, device=device)
        self.pl = torch.zeros_like(self.pm)
        self.pa = torch.zeros((*sshape2, self.head_dim), dtype=torch.float32, device=device)
        self.out = torch.zeros((batch, self.num_q_heads, 1, self.head_dim), dtype=dtype, device=device)
        self.lse = torch.zeros((batch, self.num_q_heads), dtype=torch.float32, device=device)
        # 合并后的最终输出（写进预分配缓冲，图内零分配；测试也靠它比对重放结果）
        self.merged = torch.zeros((batch, self.num_q_heads, 1, self.head_dim), dtype=dtype, device=device)

        # prefill 用的 fp16 暂存区（`begin_prefill` 建、`finish_prefill` 释放）
        self.staging: Any = None

    # ---- 与 StaticKVCache 对齐的接口 ----

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.pos.item())

    def reset(self) -> None:
        self.pos.zero_()
        self.first.zero_()
        self._written.zero_()
        self._qlen.zero_()
        self._pg.zero_()
        if self.staging is not None:
            self.staging.reset()

    # ---- prefill：fp16 暂存区 ----

    def begin_prefill(self, total_len: int) -> None:
        """prefill 开始前：按**实际会被写入的 token 数**建 fp16 暂存区。

        不按 `max_len` 分配 —— `max_len=8192` 的整段暂存区 = 2.7 GiB，8GB 卡上不能这么花。
        """
        from .cache import StaticKVCache

        total_len = int(total_len)
        if total_len <= 0 or total_len > self.max_len:
            raise ValueError(f"prefill 长度 {total_len} 超出 (0, {self.max_len}]")
        self.staging = StaticKVCache(
            num_slots=self.num_slots, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            max_len=total_len, batch=self.batch, dtype=self.dtype, device=self.device,
        )

    def set_prefill_pos(self, off: int) -> None:
        """prefill 每块开始前：把 `pos` 与暂存区游标都设到该块起点。"""
        self.pos.fill_(int(off))
        self._ensure_staging()
        self.staging.pos.fill_(int(off))

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        """写入。**返回 `(None, None)`** —— 注意力改走 `attend()` / `attend_prefill()`。

        两条路按 `staging` 是否存在切分：

        - **有暂存区（prefill）**：只写 fp16 暂存区，段末 `finish_prefill` 一次性量化；
        - **无暂存区（解码）**：写环槽 + 无条件量化整条环 + 刷设备标量。
        """
        if self.staging is not None:
            if key.shape[2] != 1:
                self.staging.append_prefill(key, value, layer_idx)
            else:
                self.staging.update(key, value, layer_idx)
            return None, None
        if key.shape[2] != 1:
            raise RuntimeError("没有暂存区时只支持解码一步（n == 1）；prefill 请先 `begin_prefill()`")
        slot = self.pos % self.group
        self.tail_k[layer_idx].index_copy_(2, slot, key)
        self.tail_v[layer_idx].index_copy_(2, slot, value)
        # 无条件量化整条环 → 第 `pos // group` 组。写满那一层里"先写后量化"，尺子正好定死；
        # 没写满时这一组不会被读到（`_qlen` 只到整组边界）。
        self._quantize_ring(layer_idx, self.pos // self.group)
        self._sync_scalars()
        return None, None

    append_prefill = update

    def _ensure_staging(self) -> None:
        if self.staging is None:
            raise RuntimeError("fp16 暂存区不存在：prefill 前先 `begin_prefill(total_len)`")

    def _sync_scalars(self) -> None:
        """把三个设备标量刷成"本步"的值。**全部是张量运算**（图内安全，不许 `.item()`）。

        `_written` = pos + 1（本步 token 也算已写入）；`_qlen` = 整组边界；`_pg` = 组数。
        """
        self._written.copy_(self.pos)
        self._written.add_(1)
        self._qlen.copy_(self._written)
        self._qlen.div_(self.group, rounding_mode="floor")
        self._qlen.mul_(self.group)
        self._pg.copy_(self._qlen)
        self._pg.div_(self.group, rounding_mode="floor")

    def refresh_scalars(self) -> None:
        """按 `pos` 重算设备标量。**只能图外调用**（warmup 之后复位用）。"""
        self._sync_scalars()

    def _quantize_ring(self, layer_idx: int, g: torch.Tensor) -> None:
        """把整条环（64 个 token）量化进第 `g` 组 —— 与 `kvquant.quantize_int8(..., "token")` 同源。

        环槽 `i` 装的是绝对位置 `p`（`p % 64 == i`）的 KV，所以**槽序 == 组内位置序**
        （组起点是 64 的整数倍）⇒ 直接按槽序量化即可，不需要重排。
        """
        idx = g * self.group + self._idx64          # 该组在缓冲里的绝对下标
        mn = self.kmn[layer_idx]
        st = self.kst[layer_idx]
        vmn, vst = self.vmn[layer_idx], self.vst[layer_idx]

        kf = self.tail_k[layer_idx].float()
        kmin = kf.amin(dim=2)
        kmax = kf.amax(dim=2)
        kstep = ((kmax - kmin) / 255.0).clamp_min(1e-8)
        kq = ((kf - kmin[:, :, None, :]) / kstep[:, :, None, :]).round().clamp_(0, 255).to(torch.uint8)
        self.kq[layer_idx].index_copy_(2, idx, kq)
        mn.index_copy_(2, g, kmin.unsqueeze(2).to(self.dtype))
        st.index_copy_(2, g, kstep.unsqueeze(2).to(self.dtype))

        vf = self.tail_v[layer_idx].float()
        vmin = vf.amin(dim=2)
        vmax = vf.amax(dim=2)
        vstep = ((vmax - vmin) / 255.0).clamp_min(1e-8)
        vq = ((vf - vmin[:, :, None, :]) / vstep[:, :, None, :]).round().clamp_(0, 255).to(torch.uint8)
        self.vq[layer_idx].index_copy_(2, idx, vq)
        vmn.index_copy_(2, g, vmin.unsqueeze(2).to(self.dtype))
        vst.index_copy_(2, g, vstep.unsqueeze(2).to(self.dtype))

    def finish_prefill(self, total: int) -> None:
        """prefill 结束：量化整组 → 填尾部环 → 刷设备标量 → **释放暂存区**（真省显存的一步）。

        只量化 `[0, Q)` 的整组；`[Q, total)` 那 ≤63 个 token 留在环里保持 fp16。
        """
        self._ensure_staging()
        from .kvquant import quantize_int8

        total = int(total)
        q = total // self.group * self.group
        n_groups = q // self.group
        for layer in range(self.num_slots):
            src_k = self.staging.key_cache[layer]
            src_v = self.staging.value_cache[layer]
            # ⚠️ **一次量化整段**，不要按组循环：按组是 36 层 × 64 组 × 2 张量 ≈ 4600 次
            # 小 kernel 调用，实测把 4096 token 的 prefill 拖到 18s（本机单次启动 ~13.5us）。
            if n_groups:
                kq, kmn, kst = quantize_int8(src_k[:, :, :q], self.group, "token")
                vq, vmn, vst = quantize_int8(src_v[:, :, :q], self.group, "token")
                self.kq[layer][:, :, :q] = kq
                self.vq[layer][:, :, :q] = vq
                self.kmn[layer][:, :, :n_groups] = kmn
                self.kst[layer][:, :, :n_groups] = kst
                self.vmn[layer][:, :, :n_groups] = vmn
                self.vst[layer][:, :, :n_groups] = vst
            # 尾部环：最近 ≤64 个 token，槽位 = 绝对位置 mod 64
            take = min(total, self.group)
            if take:
                slots = torch.remainder(
                    torch.arange(total - take, total, device=self.pos.device), self.group)
                self.tail_k[layer].index_copy_(2, slots, src_k[:, :, total - take : total])
                self.tail_v[layer].index_copy_(2, slots, src_v[:, :, total - take : total])

        self.pos.fill_(total)
        self.first.fill_(0)
        self._written.fill_(total)
        self._qlen.fill_(q)
        self._pg.fill_(n_groups)
        self.staging = None  # 释放 fp16 暂存区

    def release_staging(self) -> None:
        """丢掉 fp16 暂存区（`finish_prefill` 已做；这里留个显式入口给调试）。"""
        self.staging = None

    # ---- prefill 的注意力（fp16 + 因果掩码；融合核只做解码）----

    def attend_prefill(self, query: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """prefill 一块的注意力：`[0, off+n)` 的 fp16 暂存区 + 因果掩码。

        ⚠️ `LeanAttention.forward` 先 `update` 再调这里，所以暂存区里**已经**有本块的 K/V。
        """
        self._ensure_staging()
        n = query.shape[2]
        off = int(self.staging.pos.item())
        hist = off + n
        key = self.staging.key_cache[layer_idx][:, :, :hist]
        value = self.staging.value_cache[layer_idx][:, :, :hist]
        gq = self.num_q_heads // self.num_kv_heads
        key = key[:, :, None, :, :].expand(1, self.num_kv_heads, gq, hist, self.head_dim)
        key = key.reshape(1, self.num_q_heads, hist, self.head_dim).contiguous()
        value = value[:, :, None, :, :].expand(1, self.num_kv_heads, gq, hist, self.head_dim)
        value = value.reshape(1, self.num_q_heads, hist, self.head_dim).contiguous()
        rows = off + torch.arange(n, device=query.device)
        keep = self._arange[:hist].view(1, 1, 1, -1) <= rows.view(1, 1, n, 1)
        # ⚠️ 掩码**不要**按头展开：`[1, H_q, n, hist]` 在 4096 token 上是 1 GiB
        # （实测峰值因此冲到 7.62 GiB，越过 D17 的 7GB）。`[1,1,n,hist]` 让 SDPA 自己广播。
        mask = torch.where(keep, self._zero, self._ninf)
        return F.scaled_dot_product_attention(
            query.contiguous(), key, value, attn_mask=mask, dropout_p=0.0,
            scale=self.head_dim ** -0.5,
        )

    # ---- 解码：环掩码 / 合并 ----

    def ring_mask(self) -> torch.Tensor:
        """尾部环的有效性掩码 `[1,1,1,64]`（bool）。**全张量运算**，图内安全。

        环槽 `i` 的绝对位置 = `u - ((u - i) % 64)`，`u = _written - 1`（最后一个已写位置）。
        有效 = `_qlen ≤ p ≤ u`（且 `p ≥ first`）—— 下界是 `_qlen`，正是"已经在前缀里、
        不能再算一遍"的那条线。
        """
        u = self._written - 1
        p = u.view(1, 1) - torch.remainder(u.view(1, 1) - self._idx64.view(1, -1), self.group)
        valid = (
            (p >= self._qlen.view(1, 1))
            & (p <= u.view(1, 1))
            & (p >= self.first.view(1, 1))
        )
        return valid.view(1, 1, 1, self.group)

    def attend(self, query: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """**解码一步**的注意力：int8 前缀（融合核）+ fp16 尾部环（SDPA），`logsumexp` 合并。

        图内零 `.item()` —— 长度全部来自设备标量 `_qlen` / `_pg`。
        """
        if query.shape[2] != 1:
            raise ValueError("attend() 只处理解码一步（q_len == 1）；prefill 走 attend_prefill()")
        tail_out, tail_lse = tail_attend(
            query, self.tail_k[layer_idx], self.tail_v[layer_idx], self.ring_mask())
        int8_attn_decode(
            query, self.kq[layer_idx], self.kmn[layer_idx], self.kst[layer_idx],
            self.vq[layer_idx], self.vmn[layer_idx], self.vst[layer_idx],
            group=self.group, chunk=self.chunk,
            num_warps=self.num_warps, num_stages=self.num_stages,
            scratch=(self.pm, self.pl, self.pa), out=self.out, lse_out=self.lse,
            used_t=self._qlen, prefix_groups_t=self._pg, n_splits=self.n_splits,
        )
        # 两段合并：out = (out_p·e^{lse_p} + out_t·e^{lse_t}) / (e^{lse_p} + e^{lse_t})
        # ⚠️ 权重必须是 **4 维 [1,H,1,1]**：写成 [1,H,1] 会与 [1,H,1,D] 广播成 [1,H,H,D]，
        # 每个元素被别的头污染（实测偏差 4.8e-2，远超判据）。
        lse_p = self.lse.view(1, -1, 1, 1)
        lse_t = tail_lse.view(1, -1, 1, 1)
        m = torch.maximum(lse_p, lse_t)
        w_p = torch.exp(lse_p - m)
        w_t = torch.exp(lse_t - m)
        merged = (self.out.float() * w_p + tail_out.float() * w_t) / (w_p + w_t)
        self.merged.copy_(merged.to(self.dtype))
        return self.merged

    # ---- 显存 ----

    def nbytes(self) -> int:
        """KV 常驻字节数（含环与尺子，不含 scratch / 暂存区）。"""
        q = sum(t.numel() for t in self.kq) + sum(t.numel() for t in self.vq)
        scales = sum(t.numel() * t.element_size()
                     for t in self.kmn + self.kst + self.vmn + self.vst)
        tail = sum(t.numel() * t.element_size() for t in self.tail_k + self.tail_v)
        return q + scales + tail

    def fp16_nbytes(self) -> int:
        """同样槽位数下 fp16 cache 的字节数 —— 用来算"真省了多少"。"""
        return 2 * self.batch * self.num_slots * self.num_kv_heads * self.max_len * self.head_dim * 2

    def extra_repr(self) -> str:
        return (f"max_len={self.max_len}, slots={self.num_slots}, "
                f"kv={self.nbytes()/1024**2:.0f}MiB（fp16 同长度 {self.fp16_nbytes()/1024**2:.0f}MiB）")
