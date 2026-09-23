"""Nova 双通路文本模型骨架（S3）。

结构（见 [docs/02-architecture.md](../../docs/02-architecture.md) 第二节）：

```
tokens → embed → [共享前段 6 层] → 分叉
                                     ├─ 通路 0（层 6-29 的副本 A）─┐
                                     └─ 通路 1（层 6-29 的副本 B）─┤ 每 4 层一处交叉
                                                                  ↓
                                                        融合（mean）→ [共享后段 6 层] → norm → logits
```

**KV cache 槽位**：两条通路各有自己的层，必须用**不同的 cache 槽位**，否则会互相污染。
映射见 `cache_slot_*` 三个方法；总槽位数 = `num_cache_layers`。
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NovaConfig
from .cross import CrossPathBlock
from .layers import LeanDecoderLayer
from .norm import LeanRMSNorm


class NovaTextModel(nn.Module):
    def __init__(self, hf_text_model: nn.Module, config: NovaConfig, norm_impl: str = "exact") -> None:
        super().__init__()
        self.config = config
        self.norm_impl = norm_impl

        self.embed_tokens = hf_text_model.embed_tokens
        self.rotary_emb = hf_text_model.rotary_emb
        self.norm = LeanRMSNorm.from_hf(hf_text_model.norm, norm_impl)

        hf_layers = hf_text_model.layers
        pfx, npath, nlay = config.num_prefix_layers, config.num_paths, config.num_path_layers

        self.prefix_layers = nn.ModuleList(
            LeanDecoderLayer(hf_layers[i], config, self.cache_slot_prefix(i), norm_impl,
                             window=config.window_for_layer(i))
            for i in config.prefix_range
        )

        # 通路 0 复用 HF 原层（同一份权重）；通路 1..N-1 深拷贝 —— 这才是双通路的真实显存代价
        path_layers = []
        for p in range(npath):
            per_path = []
            for i in range(nlay):
                src = hf_layers[pfx + i]
                hf_layer = src if p == 0 else copy.deepcopy(src)
                per_path.append(LeanDecoderLayer(hf_layer, config, self.cache_slot_path(p, i), norm_impl,
                                                 window=config.window_for_layer(pfx + i)))
            path_layers.append(nn.ModuleList(per_path))
        self.path_layers = nn.ModuleList(path_layers)

        self.suffix_layers = nn.ModuleList(
            LeanDecoderLayer(hf_layers[j], config, self.cache_slot_suffix(j - config.suffix_range.start), norm_impl,
                             window=config.window_for_layer(j))
            for j in config.suffix_range
        )

        self.cross_blocks = nn.ModuleDict(
            {str(i): CrossPathBlock(config) for i in config.path_layer_indices()}
        )
        # 交叉模块是**新建**参数，不会跟着 HF 权重一起在 GPU 上；必须显式对齐 device / dtype
        ref = self.embed_tokens.weight
        self.cross_blocks = self.cross_blocks.to(device=ref.device, dtype=ref.dtype)
        self.cross_indices = set(config.path_layer_indices())

    # ---- cache 槽位映射（各通路必须互不重叠）----

    def cache_slot_prefix(self, i: int) -> int:
        return i

    def cache_slot_path(self, path: int, i: int) -> int:
        return self.config.num_prefix_layers + path * self.config.num_path_layers + i

    def cache_slot_suffix(self, j: int) -> int:
        return (
            self.config.num_prefix_layers
            + self.config.num_paths * self.config.num_path_layers
            + j
        )

    @property
    def num_cache_layers(self) -> int:
        return self.config.num_prefix_layers + self.config.num_paths * self.config.num_path_layers + self.config.num_suffix_layers

    @property
    def layer_windows(self) -> list[int]:
        """每个 cache 槽位的注意力窗口（0 = 全局层）。给 `WindowedKVCache` 定容量用。

        层号用**变换器层号**：两条通路的同一个变换器层拿到同一个窗口 —— 否则两条路的表示会分叉。
        """
        out = [0] * self.num_cache_layers
        for i in self.config.prefix_range:
            out[self.cache_slot_prefix(i)] = self.config.window_for_layer(i)
        for p in range(self.config.num_paths):
            for i in range(self.config.num_path_layers):
                out[self.cache_slot_path(p, i)] = self.config.window_for_layer(self.config.num_prefix_layers + i)
        for j in self.config.suffix_range:
            out[self.cache_slot_suffix(j - self.config.suffix_range.start)] = self.config.window_for_layer(j)
        return out

    # ---- 前向 ----

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        window_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        cross_mode: str = "off",
        return_paths: bool = False,
        **kwargs: Any,
    ) -> Any:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("必须且只能给 input_ids / inputs_embeds 之一")

        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        b, t = hidden.shape[:2]

        if position_ids is None:
            # ⚠️ 解码时必须从"已有 cache 长度"开始编号，不能从 0 开始 ——
            # 否则第 8 步会拿到位置 0 的 RoPE，prefill 对而 decode 错。
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(past_len, past_len + t, device=hidden.device)
                .view(1, 1, -1)
                .expand(3, b, -1)
                .contiguous()
            )
        position_embeddings = self.rotary_emb(hidden, position_ids)

        for layer in self.prefix_layers:
            hidden = layer(hidden, position_embeddings, attention_mask, window_mask, past_key_values)

        paths = [hidden] * self.config.num_paths if self.config.num_paths == 1 else [hidden, hidden.clone()]

        for i in range(self.config.num_path_layers):
            paths = [
                self.path_layers[p][i](
                    paths[p], position_embeddings, attention_mask, window_mask, past_key_values
                )
                for p in range(self.config.num_paths)
            ]
            if i in self.cross_indices:
                paths = self.cross_blocks[str(i)](paths, mode=cross_mode)

        path_states = paths
        hidden = paths[0] if len(paths) == 1 else torch.stack(paths, dim=0).mean(dim=0)

        for layer in self.suffix_layers:
            hidden = layer(hidden, position_embeddings, attention_mask, window_mask, past_key_values)

        hidden = self.norm(hidden)
        return (hidden, path_states) if return_paths else hidden


class NovaForCausalLM(nn.Module):
    """`lm_head` **不单独建参数** —— 基座 `tie_word_embeddings=True` 且 checkpoint 里没有 lm_head。

    直接复用 `embed_tokens.weight` 做投影，避免多出 389M 参数的副本（约 0.78 GB）。
    """

    def __init__(self, text_model: NovaTextModel) -> None:
        super().__init__()
        self.model = text_model
        self.config = text_model.config
        # 可选的 4-bit 输出投影（见 loader.enable_lm_head_4bit）。
        # None = 复用 fp16 的 embed_tokens.weight（每 token 要读 778MB -> 3.11ms）
        self.lm_head4: nn.Module | None = None

    def get_output_embeddings(self) -> nn.Parameter:
        return self.model.embed_tokens.weight

    def lm_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """输出投影。装了 4-bit 副本就用它，否则复用 fp16 的 `embed_tokens.weight`。"""
        if self.lm_head4 is not None:
            return self.lm_head4(hidden)
        return F.linear(hidden, self.model.embed_tokens.weight)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, **kwargs)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        return self.lm_head_forward(hidden)
