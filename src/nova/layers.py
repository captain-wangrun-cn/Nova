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
        self.gqa_in_sdpa = True

        self.q_proj = hf_attn.q_proj
        self.k_proj = hf_attn.k_proj
        self.v_proj = hf_attn.v_proj
        self.o_proj = hf_attn.o_proj
        self.q_norm = LeanRMSNorm.from_hf(hf_attn.q_norm, norm_impl)
        self.k_norm = LeanRMSNorm.from_hf(hf_attn.k_norm, norm_impl)

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
