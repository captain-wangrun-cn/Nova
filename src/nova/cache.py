"""固定长度 KV cache —— 所有写入都是**原地**的，因此可以被 CUDA Graph 捕获。

与 `transformers.DynamicCache` 的区别：`DynamicCache.update()` 用 `torch.cat` 增长张量，
形状每步都变，无法进图。这里预分配 `[batch, kv_heads, max_len, head_dim]`，每步只写一个槽位。

见 [reports/s3-graph-decode.md](../../reports/s3-graph-decode.md)。
"""

from __future__ import annotations

from typing import Sequence

import torch

class WindowedKVCache:
    """**逐槽位容量**的 KV cache：滑动窗口层用定长 ring，全局层用全长。

    E2 的载体（见 [reports/swa-window.md](../../reports/swa-window.md)）。与 `StaticKVCache`
    同签名（`update` / `append_prefill` / `get_seq_length` / `reset` / `nbytes`），可以直接换进
    `GraphDecoder`。

    ## 两个约定

    1. **容量**：`capacities[slot] >= max_len` 表示全长（= 现在的行为）；更小就是 ring。
       滑动窗口层的容量取 **2W**（W = 窗口）：一个 chunk 的查询最多要看它前面 W 个 token，
       而 chunk 本身 ≤ W —— 所以"上一块 + 本块" = 2W 必须同时在 ring 里。
    2. **槽位绝对位置**：ring 里第 i 槽装的是"最近一次写到 i 的那个 token"，
       所以它的绝对位置是 `p_i = end - ((end - i) % cap)`（`end` = 最后一个写入位置的绝对值）。
       `p_i < first` 的槽位**从没写过** ⇒ 必须靠掩码屏蔽（内容里是 0）。

    `first` 是本次会话最早写入的位置（一般是 0；记忆注入时会 > 0），
    由调用方在 `reset` / `prefill` 时设好 —— 它决定了"哪些槽已经有效"。
    """

    def __init__(
        self,
        capacities: Sequence[int],
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        batch: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ) -> None:
        self.capacities = [int(c) for c in capacities]
        self.max_len = int(max_len)
        self.num_slots = len(self.capacities)
        self.batch = batch
        self.dtype = dtype
        self.device = device
        self.key_cache = [
            torch.zeros(batch, num_kv_heads, c, head_dim, dtype=dtype, device=device) for c in self.capacities
        ]
        self.value_cache = [
            torch.zeros(batch, num_kv_heads, c, head_dim, dtype=dtype, device=device) for c in self.capacities
        ]
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.first = torch.zeros(1, dtype=torch.long, device=device)

    # ---- 供 NovaTextModel 使用（与 HF Cache 接口对齐）----

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.pos.item())

    def reset(self) -> None:
        self.pos.zero_()
        self.first.zero_()

    # ---- 写入 ----

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        """单 token 原地写入。**图内安全**：不读 CPU、不分配、形状恒定。"""
        if key.shape[2] != 1:
            return self.append_prefill(key, value, layer_idx)
        cap = self.capacities[layer_idx]
        slot = self.pos if cap >= self.max_len else torch.remainder(self.pos, cap)
        self.key_cache[layer_idx].index_copy_(2, slot, key)
        self.value_cache[layer_idx].index_copy_(2, slot, value)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def append_prefill(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        """prefill 用：一次写 n 个位置（**只用于 eager 路径**，会读 CPU 的 pos）。

        n 必须 ≤ 该槽位容量，否则 ring 里会有两个位置落到同一个槽（`index_copy_` 行为未定义）。
        """
        n = key.shape[2]
        start = int(self.pos.item())
        cap = self.capacities[layer_idx]
        if n > cap:
            raise ValueError(f"一次写 {n} 个 token 超过槽位 {layer_idx} 的容量 {cap}（ring 会覆盖）")
        if cap >= self.max_len:
            self.key_cache[layer_idx][:, :, start : start + n] = key
            self.value_cache[layer_idx][:, :, start : start + n] = value
        else:
            idx = torch.remainder(torch.arange(start, start + n, device=self.pos.device), cap)
            self.key_cache[layer_idx].index_copy_(2, idx, key)
            self.value_cache[layer_idx].index_copy_(2, idx, value)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # ---- 掩码要用到的位置 ----

    def slot_positions(self, layer_idx: int, end: torch.Tensor) -> torch.Tensor:
        """`end`（**最后一个写入位置**的绝对值）对应的各槽位绝对位置。

        全长槽位就是 `0..cap-1`；ring 用 `end - ((end - i) % cap)`。
        """
        cap = self.capacities[layer_idx]
        i = torch.arange(cap, device=self.pos.device)
        if cap >= self.max_len:
            return i
        return end - torch.remainder(end - i, cap)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.key_cache + self.value_cache)


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
