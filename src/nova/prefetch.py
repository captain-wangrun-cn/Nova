r"""E6 · 主机内存 / PCIe 分层：**双缓冲预取加载器**。

侧会话实测（可直接引用）：盘顺序读 **5.09 GiB/s**、裸 Python `read()` 只有 **3.1 GiB/s**、
pin 内存 H2D **12.2 GiB/s**、磁盘→显存双缓冲流水 **4.8 GiB/s**（PCIe 被藏住，只掉 6%）。

**结论就是这一条工程约束**：加载记忆段只有一种正确写法 ——
**pin 内存 + `readinto` + 双缓冲 + `non_blocking`**。本模块把它做成可复用的类。

用法::

    pf = SegmentPrefetcher(path, seg_bytes=1 << 20, ring=2)
    for i, tensor in pf.stream(device="cuda"):
        ...  # tensor 已经在显存里，PCIe 搬运与下一次盘读重叠

## `offset` / `length`：只搬文件里的一段（记忆段接进来必须用到）

`.safetensors` 不是"从第 0 字节就是张量数据" —— 前面有 8 字节头长 + JSON 头。所以"搬数据区"
必须能从**任意偏移**开始流式读。见 `MemoryStore.load_prefetched`（D07：记忆用 safetensors，
**不另存一份裸 blob**，直接按数据区偏移搬原始字节）。

## `stream_into()`：写进**预分配**显存缓冲

`stream()` 每段 `to(device)` 会**新分配**一个显存张量；搬 GB 级记忆时这是分配器抖动。
`stream_into(dst)` 改成写进调用方给的缓冲（只做一次 H2D，不多一次拷贝）。
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

    def __init__(
        self,
        path: str | Path,
        seg_bytes: int = 1 << 20,
        ring: int = 2,
        offset: int = 0,
        length: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.file_size = self.path.stat().st_size
        self.offset = int(offset)
        if length is None:
            length = self.file_size - self.offset
        self.size = int(length)
        if self.offset < 0 or self.size < 0 or self.offset + self.size > self.file_size:
            raise ValueError(
                f"区间 [{self.offset}, {self.offset + self.size}) 超出文件大小 {self.file_size}"
            )
        self.seg_bytes = int(seg_bytes)
        if self.seg_bytes <= 0:
            raise ValueError(f"seg_bytes 必须 > 0，收到 {seg_bytes}")
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
            fh.seek(self.offset)
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

    def stream_into(self, dst: torch.Tensor) -> int:
        """把本区间**原地**搬进预分配的显存 `dst`（uint8，1 维，长度 ≥ `size`）。

        顺序与 `stream()` 一致：**先发起当前段的 H2D，再读下一段** —— 顺序反了 PCIe 就藏不住。
        返回实际搬运的字节数。调用方负责 `torch.cuda.synchronize()`（或依赖同 stream 顺序）。
        """
        if dst.dim() != 1 or dst.dtype != torch.uint8:
            raise ValueError(f"dst 必须是一维 uint8，收到 {tuple(dst.shape)} {dst.dtype}")
        if dst.numel() < self.size:
            raise ValueError(f"dst 只有 {dst.numel()} 字节，放不下 {self.size} 字节")
        views = [memoryview(b.numpy()) for b in self.buffers]
        fh = open(self.path, "rb")
        try:
            fh.seek(self.offset)
            n = len(self)
            buf = self.buffers[0]
            n_read = fh.readinto(views[0][: min(self.seg_bytes, self.size)])
            moved = 0
            for i in range(n):
                cur, cur_n = buf, n_read
                if cur_n:
                    # 预分配缓冲的切片：一次 H2D，不新分配
                    dst.narrow(0, moved, cur_n).copy_(cur[:cur_n], non_blocking=True)
                    moved += cur_n
                nxt = self.buffers[(i + 1) % self.ring]
                start = (i + 1) * self.seg_bytes
                n_read = (fh.readinto(views[(i + 1) % self.ring][: min(self.seg_bytes, self.size - start)])
                          if start < self.size else 0)
                buf = nxt
            return moved
        finally:
            fh.close()

    def read_all_sync(self, device: str | torch.device = "cuda") -> torch.Tensor:
        """对照用的**朴素**写法：一次性 `read()` 再一次性 H2D（会退化成串行）。"""
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read(self.size)
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
