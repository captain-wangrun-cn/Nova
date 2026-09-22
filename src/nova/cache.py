"""固定长度 KV cache —— 所有写入都是**原地**的，因此可以被 CUDA Graph 捕获。

与 `transformers.DynamicCache` 的区别：`DynamicCache.update()` 用 `torch.cat` 增长张量，
形状每步都变，无法进图。这里预分配 `[batch, kv_heads, max_len, head_dim]`，每步只写一个槽位。

见 [reports/s3-graph-decode.md](../../reports/s3-graph-decode.md)。
"""

from __future__ import annotations

import torch


class StaticKVCache:
    """`update()` 与 HF 的 `Cache.update` 同签名，可直接替换。"""

    def __init__(
        self,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        batch: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ) -> None:
        self.num_slots = num_slots
        self.max_len = max_len
        self.batch = batch
        self.dtype = dtype
        self.device = device
        shape = (batch, num_kv_heads, max_len, head_dim)
        self.key_cache = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_slots)]
        self.value_cache = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_slots)]
        # 当前要写入的槽位下标。**必须放在 GPU 上**，否则图内 `index_copy_` 会触发 CPU 同步。
        self.pos = torch.zeros(1, dtype=torch.long, device=device)

    # ---- 供 NovaTextModel 使用（与 HF Cache 接口对齐）----

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.pos.item())

    def reset(self) -> None:
        self.pos.zero_()

    # ---- 写入 ----

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """单 token 原地写入。**图内安全**：不读 CPU、不分配、形状恒定。

        返回**整条** cache（长度恒为 `max_len`）；未写入的槽位是 0，
        调用方必须用 attention mask 把它们屏蔽掉。
        """
        if key.shape[2] != 1:
            # prefill（多 token）：走 eager 路径
            return self.append_prefill(key, value, layer_idx)
        self.key_cache[layer_idx].index_copy_(2, self.pos, key)
        self.value_cache[layer_idx].index_copy_(2, self.pos, value)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def append_prefill(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """prefill 用：一次写入 n 个位置。**只用于 eager 路径**（会读 CPU 的 pos）。"""
        n = key.shape[2]
        start = int(self.pos.item())
        self.key_cache[layer_idx][:, :, start : start + n] = key
        self.value_cache[layer_idx][:, :, start : start + n] = value
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # ---- 显存 ----

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.key_cache + self.value_cache)
