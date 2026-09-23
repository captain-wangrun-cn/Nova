r"""E6 · 主机内存 / PCIe 分层：**双缓冲预取加载器**。

侧会话实测（可直接引用）：盘顺序读 **5.09 GiB/s**、裸 Python `read()` 只有 **3.1 GiB/s**、
pin 内存 H2D **12.2 GiB/s**、磁盘→显存双缓冲流水 **4.8 GiB/s**（PCIe 被藏住，只掉 6%）。

**结论就是这一条工程约束**：加载记忆段只有一种正确写法 ——
**pin 内存 + `readinto` + 双缓冲 + `non_blocking`**。本模块把它做成可复用的类。

用法::

    pf = SegmentPrefetcher(path, seg_bytes=1 << 20, ring=2)
    for i, tensor in pf.stream(device="cuda"):
        ...  # tensor 已经在显存里，PCIe 搬运与下一次盘读重叠
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


class SegmentPrefetcher:
    """把一个大文件切成定长段，**双缓冲**地读进 pin 内存并异步搬到显存。

    - `ring`：环里放几个 pin 缓冲（2 就够双缓冲；再多只是多占内存）。
    - 每次 `next()`：先在**当前段**上发起 H2D（`non_blocking=True`），
      再在**下一段**上做磁盘读 —— 两者在两条不同的路上，天然重叠。
    - `pin_memory=True` 是**必须**的：可分页内存的 H2D 会掉到 10.4 GiB/s 且同步。
    """

    def __init__(self, path: str | Path, seg_bytes: int = 1 << 20, ring: int = 2) -> None:
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.seg_bytes = int(seg_bytes)
        self.ring = max(2, int(ring))
        self.buffers = [torch.empty(self.seg_bytes, dtype=torch.uint8, pin_memory=True)
                        for _ in range(self.ring)]
        self._fh = None

    def __len__(self) -> int:
        return (self.size + self.seg_bytes - 1) // self.seg_bytes

    def stream(self, device: str | torch.device = "cuda", dtype: torch.dtype = torch.uint8):
        """逐段产出**显存里的**张量（最后一段可能不足 `seg_bytes`）。"""
        # `readinto` 要的是**可写 buffer**，torch 张量不是；用它的 numpy 视图包一层 memoryview
        views = [memoryview(b.numpy()) for b in self.buffers]
        fh = open(self.path, "rb")
        try:
            n = len(self)
            buf = self.buffers[0]
            n_read = fh.readinto(views[0][: min(self.seg_bytes, self.size)])
            for i in range(n):
                cur = buf
                cur_n = n_read
                # 先发起 H2D（异步），再去读下一段 —— 顺序不能反
                gpu = cur[:cur_n].to(device, non_blocking=True).view(dtype) if dtype != torch.uint8 \
                    else cur[:cur_n].to(device, non_blocking=True)
                nxt = self.buffers[(i + 1) % self.ring]
                start = (i + 1) * self.seg_bytes
                n_read = (fh.readinto(views[(i + 1) % self.ring][: min(self.seg_bytes, self.size - start)])
                          if start < self.size else 0)
                yield gpu
                buf = nxt
        finally:
            fh.close()

    def read_all_sync(self, device: str | torch.device = "cuda") -> torch.Tensor:
        """对照用的**朴素**写法：一次性 `read()` 再一次性 H2D（会退化成串行）。"""
        data = self.path.read_bytes()
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
