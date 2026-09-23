r"""int4 KV cache：量化方案 + 「往返模拟」缓存 + 显存记账。

## 这个模块**不省显存**，别误会

PyTorch 的 `F.scaled_dot_product_attention` 必须吃 fp16 的 K/V，所以"先反量化再算注意力"
这条路上，你必须把整段历史反量化成一块 fp16 缓冲 —— **那块缓冲跟原来的 fp16 cache 一样大**。
省显存只有两条路：把 dequant 融进 attention kernel，或做"分块 dequant + 在线 softmax"。
两者都还没做（见 [reports/kv-int4.md](../../reports/kv-int4.md) 第六节）。

它模拟的就是**最朴素的那条实现**：每个 decode 步、每一层，把 `[0, used)` 从 fp16 原值
反量化到一块 fp16 工作区再喂给 SDPA。所以：

- **精度**：注意力看到的 K/V 就是 int4 往返后的值 —— 精度损失是**真的**。
- **开销**：每层每步 O(used) 的拷贝 + 反量化 —— 这就是"先还原再算"的真实代价（实测很糟）。
- **显存**：缓冲**共用**一块（层是串行处理的），所以只多 `2 × max_len × kv_heads × head_dim`
  字节（18432 长度下 76 MiB），不按层数放大。收益仍只靠公式记账。

本模块提供的是**精度与开销的精确测量**：把 fp16 存下来，但在注意力看到它之前做一次
"量化 -> 反量化"往返。于是：

- **精度影响**：注意力看到的 K/V 就是 int4 往返后的值 —— 测出来的精度损失是**真的**。
- **dequant 开销**：往返的算术量就是真方案里 dequant 的量，能测出它值不值。
- **显存收益**：单独按公式算（`bytes_per_token`），不靠模拟。

## 量化方案（K 与 V 区别对待）

| | 分组方式 | 理由 |
|---|---|---|
| **K** | 沿 `head_dim` 每 **32 个通道**一组 | K 的离群值是**通道级**的（某些通道长期偏大）；小分组让每组有自己的尺子 |
| **V** | 每个 `(token, head)` 的**整条 128 维向量**一组 | V 的离群值是**token 级**的；文献上 V 可以更狠 |

两组都用**非对称**量化（存 `min` 与 `step`，各 fp16）：

```
q = clamp(round((x - min) / step), 0, 15)      step = (max - min) / 15
x' = q * step + min
```

两个 int4 值打进一个 `uint8`（低半字节在前）。

## ⚠️ 与 S4 记忆模块的冲突（D34 第 5 条）

S4 的记忆**就是 K/V 张量**，D09 要求表示空间冻结。所以规矩必须先定死：
**记忆一律按 fp16 存**（它是"原话"，L0 层承诺无损）；int4 只是**运行时 cache 的存储格式**，
注入记忆时按当时的 cache 格式量化。这样"记忆精度"不被 cache 存储格式绑架。
"""

from __future__ import annotations

import torch

GROUP_K = 32
GROUP_V = 0  # 0 = 整条 head_dim 一组
NIBBLE_MAX = 15
BYTE_MAX = 255
FP8_MAX = 448.0  # float8_e4m3fn 的最大可表示值


def _as_groups(x: torch.Tensor, group_size: int) -> torch.Tensor:
    *lead, n, dim = x.shape
    if not group_size:  # 0 / None = 整条向量一组（V 的默认）
        group_size = dim
    if dim % group_size:
        raise ValueError(f"head_dim {dim} 不能被 group_size {group_size} 整除")
    return x.reshape(*lead, n, dim // group_size, group_size)


def quantize_int4(x: torch.Tensor, group_size: int = GROUP_K):
    """非对称 int4 量化。`x` 形状 `[..., n, dim]`（fp16）-> `(packed uint8, mins, steps)`。"""
    if not group_size:
        group_size = x.shape[-1]
    g = _as_groups(x.float(), group_size)
    xmin = g.amin(dim=-1)
    xmax = g.amax(dim=-1)
    step = ((xmax - xmin) / NIBBLE_MAX).clamp_min(1e-8)
    q = ((g - xmin.unsqueeze(-1)) / step.unsqueeze(-1)).round().clamp_(0, NIBBLE_MAX).to(torch.uint8)
    lo, hi = q[..., 0::2], q[..., 1::2]
    packed = (lo | (hi << 4)).contiguous()
    return packed, xmin.to(torch.float16), step.to(torch.float16)


def dequantize_int4(
    packed: torch.Tensor,
    mins: torch.Tensor,
    steps: torch.Tensor,
    group_size: int = GROUP_K,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """`quantize_int4` 的逆运算。"""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    x = q.float() * steps.unsqueeze(-1).float() + mins.unsqueeze(-1).float()
    return x.reshape(*x.shape[:-2], -1).to(dtype)


def roundtrip_int4(x: torch.Tensor, group_size: int = GROUP_K, dtype: torch.dtype = torch.float16):
    """量化 -> 反量化，返回与 `x` 同形的张量（注意力实际会看到的值）。"""
    packed, mins, steps = quantize_int4(x, group_size)
    return dequantize_int4(packed, mins, steps, group_size, dtype)


# ---------------------------------------------------------------- 8 位（E3）
#
# 判据（reports/kv-quant-8bit.md）：误差应当**远小于** int4（int4 的 16 档 vs int8 的 256 档），
# 而记账只比 int4 差一点（41.6 KiB/token -> 约 74 KiB/token，约 1.86x）。
# 三个变体分别对应文献里的三种做法：
#
# | 变体 | 分组方向 | 对应文献 |
# |---|---|---|
# | `int8` + `axis="channel"` | 每 token、沿 `head_dim` 每 32 通道一组 | 与现有 int4 的 K 同构 |
# | `int8` + `axis="token"` | 每个通道**跨 token** 分组（G 个 token 一条尺子） | llama.cpp `q4_0/q8_0` 的做法 |
# | `fp8` | 每 token 一个 scale（或每张量一个） | vLLM / TRT-LLM 的 FP8 KV |


def quantize_int8(x: torch.Tensor, group_size: int = GROUP_K, axis: str = "channel"):
    """非对称 int8 量化。`x` 形状 `[..., n, dim]` -> `(uint8, mins, steps)`。

    - `axis="channel"`：沿最后一维（`head_dim`）每 `group_size` 个通道一组，**每 token 独立尺子**；
    - `axis="token"`：沿**倒数第二维**（token）每 `group_size` 个 token 一组，**每个通道一条跨 token 的尺子**
      —— 这就是文献里"K 按 token 维分组"的做法，用来修第 0 层那种**通道级**离群。
    """
    if axis not in ("channel", "token"):
        raise ValueError(f"axis 必须是 'channel' / 'token'，收到 {axis!r}")
    xf = x.float()
    if axis == "channel":
        g = _as_groups(xf, group_size or xf.shape[-1])
        xmin, xmax = g.amin(dim=-1), g.amax(dim=-1)
        step = ((xmax - xmin) / BYTE_MAX).clamp_min(1e-8)
        q = ((g - xmin.unsqueeze(-1)) / step.unsqueeze(-1)).round().clamp_(0, BYTE_MAX)
        # ⚠️ 与 int4 不同：int8 不打包，必须 reshape 回**原始形状**（int4 是故意留成分组形状再打包的）
        return q.to(torch.uint8).reshape(*xf.shape), xmin.to(torch.float16), step.to(torch.float16)

    *lead, n, dim = xf.shape
    gs = int(group_size or n)
    pad = (-n) % gs
    if pad:
        # 末尾补 `pad` 个 token（复制最后一个）凑成整组；`dequantize_int8` 会把它们切掉。
        # 不补的话 1894 这种长度直接报错，而真实序列长度本来就是任意的。
        idx = torch.arange(n + pad, device=xf.device).clamp_(max=n - 1)
        xf = xf.index_select(-2, idx)
    g = xf.reshape(*lead, (n + pad) // gs, gs, dim)
    xmin, xmax = g.amin(dim=-2), g.amax(dim=-2)  # [..., n//gs, dim]
    step = ((xmax - xmin) / BYTE_MAX).clamp_min(1e-8)
    q = ((g - xmin.unsqueeze(-2)) / step.unsqueeze(-2)).round().clamp_(0, BYTE_MAX)
    return (q.to(torch.uint8).reshape(*lead, n + pad, dim)[..., :n, :],
            xmin.to(torch.float16), step.to(torch.float16))


def dequantize_int8(
    q: torch.Tensor,
    mins: torch.Tensor,
    steps: torch.Tensor,
    group_size: int = GROUP_K,
    axis: str = "channel",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """`quantize_int8` 的逆运算。"""
    if axis == "channel":
        gs = group_size or q.shape[-1]
        g = q.float().reshape(*q.shape[:-1], q.shape[-1] // gs, gs)
        x = g * steps.unsqueeze(-1).float() + mins.unsqueeze(-1).float()
        return x.reshape(*q.shape).to(dtype)

    *lead, n, dim = q.shape
    gs = int(group_size or n)
    pad = (-n) % gs
    qf = q.float()
    if pad:
        idx = torch.arange(n + pad, device=q.device).clamp_(max=n - 1)
        qf = qf.index_select(-2, idx)
    g = qf.reshape(*lead, (n + pad) // gs, gs, dim)
    x = g * steps.unsqueeze(-2).float() + mins.unsqueeze(-2).float()
    return x.reshape(*lead, n + pad, dim)[..., :n, :].to(dtype)


def roundtrip_int8(
    x: torch.Tensor,
    group_size: int = GROUP_K,
    axis: str = "channel",
    dtype: torch.dtype = torch.float16,
):
    q, mins, steps = quantize_int8(x, group_size, axis)
    return dequantize_int8(q, mins, steps, group_size, axis, dtype)


def roundtrip_fp8(
    x: torch.Tensor,
    per: str = "token",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """`float8_e4m3fn` 往返：`scale = amax / 448`，量化 `x/scale`，再乘回来。

    `per="token"`：每个 `(token, head)` 一个 scale（文献里 FP8 KV 的常见做法）；
    `per="tensor"`：整张张量一个 scale（scale 开销可忽略，但离群会拉低整体精度）。
    """
    if per not in ("token", "tensor"):
        raise ValueError(f"per 必须是 'token' / 'tensor'，收到 {per!r}")
    xf = x.float()
    if per == "tensor":
        amax = xf.abs().amax().clamp_min(1e-8)
        scale = (amax / FP8_MAX).clamp_min(1e-8)
        q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return (q.float() * scale).to(dtype)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (amax / FP8_MAX).clamp_min(1e-8)
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(dtype)


def roundtrip(x: torch.Tensor, kind: str, group_size: int = GROUP_K, dtype: torch.dtype = torch.float16):
    """按变体名往返一次。名字与 `probe_kvquant_bits.py` / `exp_needle.py --kv` 一致。"""
    if kind == "fp16":
        return x
    if kind == "int4":
        return roundtrip_int4(x, group_size, dtype)
    if kind == "int8":
        return roundtrip_int8(x, group_size, "channel", dtype)
    if kind == "int8t":
        return roundtrip_int8(x, 0, "token", dtype)  # 0 = 整条序列一组（跨全部 token）
    if kind == "int8t64":
        return roundtrip_int8(x, 64, "token", dtype)
    if kind == "fp8":
        return roundtrip_fp8(x, "token", dtype)
    if kind == "fp8t":
        return roundtrip_fp8(x, "tensor", dtype)
    raise ValueError(f"未知的量化变体 {kind!r}")


def bytes_per_token(
    num_slots: int,
    num_kv_heads: int,
    head_dim: int,
    group_k: int = GROUP_K,
    group_v: int = GROUP_V,
    bits: int = 4,
    k_axis: str = "channel",
    token_group: int = 64,
    fp8_per: str = "token",
) -> dict[str, float]:
    """每 token 的 KV 存储字节数：fp16 基线 vs 量化方案（含元数据）。

    - `bits=4`：非对称 int4（数据 0.5 B/元素 + 每组 `min`/`step` 各 fp16）；
    - `bits=8`：非对称 int8（数据 1 B/元素 + 每组元数据），`k_axis` 选 K 的分组方向；
    - `bits="fp8"`：`float8_e4m3fn`（数据 1 B/元素），`fp8_per="token"` 时每个 `(token,head)` 一个 fp16 scale。

    元数据是记账里最容易漏掉的一块：int4 的 K 每头 4 组 ⇒ **+25%**（实测已由
    `test_bytes_accounting` 钉住）。
    """
    n = num_kv_heads * head_dim

    def one(dim: int, group: int) -> float:
        g = dim if group in (0, None) else group
        groups = dim // g
        return dim / 2 + 2 * 2 * groups  # int4 数据 + (min, step) 各 fp16

    kv16 = 2 * head_dim * 2
    if bits == "fp8":
        scale_bytes = 2 if fp8_per == "token" else 2 / head_dim  # 每 token 一个 fp16 scale / 每张量一个
        per = head_dim * 1 + scale_bytes
        return {
            "fp16_kv": kv16 * num_slots * num_kv_heads,
            "int8_kv": 2 * per * num_slots * num_kv_heads,
            "ratio": kv16 / (2 * per),
        }
    if int(bits) == 8:
        def one8(dim: int, group: int) -> float:
            g = dim if group in (0, None) else group
            return dim * 1 + 2 * 2 * (dim // g)  # int8 数据 + (min, step) 各 fp16

        k8 = one8(head_dim, token_group if k_axis == "token" else group_k)
        v8 = one8(head_dim, group_v)
        return {
            "fp16_kv": kv16 * num_slots * num_kv_heads,
            "int8_kv": (k8 + v8) * num_slots * num_kv_heads,
            "ratio": kv16 / (k8 + v8),
        }

    k4 = one(head_dim, group_k)
    v4 = one(head_dim, group_v)
    return {
        "fp16_kv": kv16 * num_slots * num_kv_heads,
        "int4_kv": (k4 + v4) * num_slots * num_kv_heads,
        "k_int4_per_token": k4 * num_slots * num_kv_heads,
        "v_int4_per_token": v4 * num_slots * num_kv_heads,
        "ratio": kv16 / (k4 + v4),
    }


class QuantRoundTripCache:
    """`StaticKVCache` 的替代品：写入照旧用 fp16，**读出来之前做一次 int4 往返**。

    ⚠️ 它**不省显存**（fp16 存储仍在），用途是精确测量 int4 的精度影响与 dequant 开销。
    `residual>0` 时最近 `residual` 个位置保持 fp16 不量化（KIVI 的 residual window 做法）。

    工作区（scratch）**只有一块**，36 层共用：层是串行处理的，每层在算自己的注意力之前把它
    重建成 `[0, used)`。代价是每层每步 O(used)，好处是显存不按层数放大（早期按层各留一块
    的写法在 1894 token 就把峰值顶到 11.8 GiB，Windows 直接换页、prefill 从 1.2s 变 19.6s）。
    """

    def __init__(
        self,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        batch: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        quant_k: bool = True,
        quant_v: bool = True,
        group_k: int = GROUP_K,
        group_v: int = GROUP_V,
        residual: int = 0,
        kind: str = "int4",
        base=None,
    ) -> None:
        from .cache import StaticKVCache

        # `base`：**复用**调用方已有的 cache，而不是再分配一份。
        # 不传的话会新建 —— 而 `dec.cache = QuantRoundTripCache(...)` 这种写法在赋值完成前
        # 旧 cache 仍然活着，于是两份 cache 同时在显存里（实测 18432 槽位下 2×2.7 GiB + 权重 = 8.3 GiB，
        # 直接 OOM）。传 `base=dec.cache` 就只占一份。
        self._base = base if base is not None else StaticKVCache(
            num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim,
            max_len=max_len, batch=batch, dtype=dtype, device=device,
        )
        self.num_slots = num_slots
        self.max_len = max_len
        self.batch = batch
        self.dtype = dtype
        self.device = device
        self.quant_k = quant_k
        self.quant_v = quant_v
        self.group_k = group_k
        self.group_v = group_v
        self.residual = int(residual)
        # `kind` 决定往返用哪套量化（int4 / int8 / int8t / fp8 / fp8t ...），见 `roundtrip`
        self.kind = kind
        shape = (batch, num_kv_heads, max_len, head_dim)
        # 预分配**共用** scratch，避免每步每层都新分配（否则是 allocator churn，不是真实开销）
        self._k_scratch = torch.zeros(shape, dtype=dtype, device=device)
        self._v_scratch = torch.zeros(shape, dtype=dtype, device=device)

    # ---- 与 StaticKVCache 对齐的接口 ----

    @property
    def pos(self):
        return self._base.pos

    @property
    def key_cache(self):
        return self._base.key_cache

    @property
    def value_cache(self):
        return self._base.value_cache

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._base.get_seq_length(layer_idx)

    def reset(self) -> None:
        self._base.reset()

    def nbytes(self) -> int:
        return self._base.nbytes()  # 模拟版不省，见报告

    def simulate(self, x: torch.Tensor, scratch: torch.Tensor, used: int, group: int) -> torch.Tensor:
        """把 `[0, used-residual)` 位置换成 int4 往返的结果，尾部保持 fp16。

        `used` 由调用方按 `start + n` 算好 —— 它同时管着写入范围（`StaticKVCache` 的语义）
        和这里的重建范围。**不能**在内部按 `pos` 重算，见 `update` 里的说明。
        """
        keep = max(0, used - self.residual)
        scratch[:, :, :used] = x[:, :, :used]
        if keep > 0:
            scratch[:, :, :keep] = roundtrip(x[:, :, :keep], self.kind, group, self.dtype)
        return scratch

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        # ⚠️ 已写入范围是 `pos + n`，**不是** `pos + 1`：
        # - decode（n=1）：`pos` 是要写的那一槽，所以是 `pos + 1` ✓
        # - prefill（n=多）：`StaticKVCache.update` 内部转调 `append_prefill` 一次写 n 个位置，
        #   但 `pos` **不推进**（推进在 `GraphDecoder.prefill` 收尾处）。这时候写 "pos+1" 只会
        #   把**第 0 个位置**拷进 scratch，1..n-1 位置留下 scratch 里的 0 -> 注意力看到一片零。
        #   实测：15 token 的 prompt 输出直接变胡言乱语（`residual` 再大也没用，因为根本没拷）。
        n = key.shape[2]
        start = int(self._base.pos.item())
        k, v = self._base.update(key, value, layer_idx)
        used = start + n
        if self.quant_k:
            k = self.simulate(k, self._k_scratch, used, self.group_k)
        if self.quant_v:
            v = self.simulate(v, self._v_scratch, used, self.group_v)
        return k, v

    def scratch_bytes(self) -> int:
        """工作区额外占用（本模拟版独有，真实方案里由融合核消掉）。"""
        return self._k_scratch.numel() * self._k_scratch.element_size() * 2

    def append_prefill(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        """与 `update` 同义（`StaticKVCache.update` 内部就是转调它），只保留一个实现。"""
        return self.update(key, value, layer_idx)
