"""精简解码器层：结构与 HF `Qwen3VLTextDecoderLayer` 对齐，但算子更少、组件可插拔。

与 HF 的差异（都是"减法"，不改变语义）：
1. `q_norm` / `k_norm` / 两个 layernorm 换成 `LeanRMSNorm`（可切 `exact` / `triton`）
2. 直接调 `F.scaled_dot_product_attention`，不经过 transformers 的 attention 派发层
3. 线性层由外部注入 —— 可以是 `nn.Linear`，也可以是 `bnb.nn.Linear4bit`

**数值目标：`norm_impl="exact"` 时必须与 HF 逐位一致**（S3 的"门控关闭 ≈ 基线"判据依赖这一点）。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from .config import NovaConfig
from .norm import LeanRMSNorm


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """与 HF 同序：`(q * cos) + (rotate_half(q) * sin)`。"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把 KV 头复制到与 Q 头同数。仅在 SDPA 不支持 GQA 时才走这条路。"""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LeanAttention(nn.Module):
    def __init__(self, hf_attn: nn.Module, config: NovaConfig, layer_idx: int, norm_impl: str = "exact") -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.head_dim ** -0.5
        self.is_causal = True
        # ⚠️ **默认 False（先 repeat_kv 展平成 32 头再进 SDPA）。这是实测逼出来的，不是随手写的。**
        #
        # 本机 torch 2.6.0+cu124 **没有编译 flash attention**，而 mem-efficient / cuDNN 两个融合内核
        # 都要求 Q/K/V 头数相同。GQA（32 Q / 8 KV）加上 `enable_gqa=True` 会让 SDPA **退回 math 后端**，
        # 实体化 O(n²) 的 **fp32** 分数矩阵：
        #
        # | 4096 token 前向 | 峰值显存 | 耗时 |
        # |------|:---:|:---:|
        # | `enable_gqa=True`（math 回退） | 6.76 GiB | 5293 ms |
        # | `False`（展平 → EFFICIENT_ATTENTION） | **3.26 GiB** | **1299 ms** |
        #
        # 展平后长度翻 4 倍（1749 → 7146）显存只从 3.15 涨到 3.53 GiB；不展平时 7146 token 直接 OOM。
        # 代价是**与 HF 的逐位一致没有了**（32/32 个贪心 token 仍然相同，见
        # `test_fused_attention_agrees_on_tokens`）；架构保真度由 `sdpa_kernel(MATH)` 下的
        # 逐位一致测试单独保证。
        # 证据：`reports/long-context-attention.md`、`src/diagnostics/probe_attention_kernel.py`。
        self.gqa_in_sdpa = False

        self.q_proj = hf_attn.q_proj
        self.k_proj = hf_attn.k_proj
        self.v_proj = hf_attn.v_proj
        self.o_proj = hf_attn.o_proj
        self.q_norm = LeanRMSNorm.from_hf(hf_attn.q_norm, norm_impl)
        self.k_norm = LeanRMSNorm.from_hf(hf_attn.k_norm, norm_impl)

        # ---- 记忆写入用的捕获开关（S4，见 src/nova/memory.py）----
        # `capture_slice` 非 None 时，把该 token 区间的 **RoPE 之前** 的 Q / K / V 存进 `captured`。
        # 抓 RoPE 之前的值是为了让 K 与位置解耦 —— 注入时按**新位置**重新旋转即可。
        # Q 用于"取回"时的寻址（注意力打分 = Q·K，见 memory.py）。
        # ⚠️ 图解码路径必须保持 None：捕获会 clone 张量，不能进 CUDA Graph。
        self.capture_slice: tuple[int, int] | None = None
        self.captured: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.capture_slice is not None:
            s0, s1 = self.capture_slice
            if not 0 <= s0 < s1 <= key.shape[2]:
                raise ValueError(f"capture_slice {self.capture_slice} 超出本层序列长度 {key.shape[2]}")
            self.captured = (
                query[:, :, s0:s1, :].detach().clone(),
                key[:, :, s0:s1, :].detach().clone(),
                value[:, :, s0:s1, :].detach().clone(),
            )

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        sdpa_kwargs: dict = {}
        if self.num_key_value_groups > 1:
            # 与 HF `use_gqa_in_sdpa` 同一判据
            if self.gqa_in_sdpa and key.shape[-1] == value.shape[-1] <= 256:
                sdpa_kwargs["enable_gqa"] = True
            else:
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)

        q_len, kv_len = query.shape[2], key.shape[2]
        is_causal = q_len > 1 and attention_mask is None and self.is_causal
        if is_causal and kv_len > q_len:
            key = key[:, :, :q_len, :]
            value = value[:, :, :q_len, :]

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, scale=self.scaling,
            is_causal=is_causal, **sdpa_kwargs,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class LeanMLP(nn.Module):
    def __init__(self, hf_mlp: nn.Module, config: NovaConfig) -> None:
        super().__init__()
        self.gate_proj = hf_mlp.gate_proj
        self.up_proj = hf_mlp.up_proj
        self.down_proj = hf_mlp.down_proj
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LeanDecoderLayer(nn.Module):
    """单条通路里的一层。"""

    def __init__(self, hf_layer: nn.Module, config: NovaConfig, layer_idx: int, norm_impl: str = "exact") -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = LeanAttention(hf_layer.self_attn, config, layer_idx, norm_impl)
        self.mlp = LeanMLP(hf_layer.mlp, config)
        self.input_layernorm = LeanRMSNorm.from_hf(hf_layer.input_layernorm, norm_impl)
        self.post_attention_layernorm = LeanRMSNorm.from_hf(hf_layer.post_attention_layernorm, norm_impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
