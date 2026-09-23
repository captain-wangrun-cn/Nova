"""CUDA Graph 捕获的贪心解码器。

**为什么需要它**（实测数据见 [reports/s3-graph-decode.md](../../reports/s3-graph-decode.md)）：

- 本机单次 CUDA kernel 启动约 **13.5us CPU**（Windows/WDDM；`x * 2.0` 都要 16us）
- Nova 单通路每 token 有 **~6500 次**启动 -> 光启动就 ~62ms，GPU 全程挨饿
- 把**整步**捕获成一张 CUDA Graph 后，每 token 只剩 **1 次 replay**（~28us CPU）

结构：整步（embed -> 36/60 层 -> lm_head -> argmax -> 写回 input_ids -> pos+1）
全部在图内，**CPU 每 token 只发一次 replay**。

**前提**：KV cache 必须是定长原地写入（`StaticKVCache`），
且图内**不能有任何 `.item()` / CPU 同步 / 动态形状**。

**P0 之后的两条改动**（2026-09-22，见 [reports/decode-mask-bucket.md](../../reports/decode-mask-bucket.md)）：

1. **加性掩码不再预分配 O(max_len²) 的表** —— 65536 时是 **8.0 GiB**，比 KV cache 更早爆，
   它是"上下文上限"的第一堵墙。改成**图内即时构造一行**，常驻从 `max_len²×2` 降到 `max_len×2`。
2. **容量按桶分配**（`BUCKETS`）：decode 的 KV 读量与掩码长度都跟"分配的槽位数"走，不跟真实长度走。
   预留 18432 而只写 2048 时，**89% 的 KV 读发生在没写过的零槽位上**。
   跨桶用 `grow()` 搬家 + 重捕（**实测重捕不泄漏**；共享 graph pool 反而会踩 PyTorch 的裸 assert，
   所以默认不开 —— 证据见 `src/diagnostics/probe_graph_recapture.py`）。
"""

from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F

from .cache import StaticKVCache, WindowedKVCache

# ---- 容量分桶 ----

# 桶边界（token）。选这些数是因为它们覆盖了本项目实际会用的档位：
# 2048 档留 22 个 token 余量（chat 模板 + 生成），再往上每翻一倍一档。
BUCKETS: tuple[int, ...] = (2070, 4096, 8192, 16384, 32768, 65536)


def bucket_for(n: int, buckets: tuple[int, ...] = BUCKETS) -> int:
    """不小于 `n` 的最小桶。超过最大的桶就按 `n` 精确分配（不做无限分桶）。"""
    n = int(n)
    for b in buckets:
        if n <= b:
            return int(b)
    return n


_GRAPH_POOL = None
_POOL_USERS: list = []  # 用过当前池的 `CUDAGraph` 的弱引用


def shared_graph_pool(fresh_if_idle: bool = True):
    """取一个可用的 CUDA Graph 内存池。

    **背景**：旧报告写过"每题重捕，重捕 9 次就把显存顶到 7905 / 8188 MiB 然后崩"，
    据此引入共享池。但 P0 的实测（`probe_graph_recapture.py`）**推翻了这条**：
    每轮显式 `del` + `gc` + `empty_cache` 之后，逐轮 allocated 基本是平的，
    三种策略（不共享 / 共享 / 自适应）都不涨。图上一次没被正确释放才是真凶，
    不是"没共享池"。

    ⚠️ **实测的 PyTorch 坑**：把一个**已经销毁**的图用过的池再拿去捕获新图，会在
    `CUDACachingAllocator.cpp:2225` 抛裸 assert
    `it->second->use_count > 0 INTERNAL ASSERT FAILED`（没有消息，很难查）。
    而且它只在**特定分配器状态**下触发：`tests/test_decode_mask_bucket.py` 全量跑会中，
    单跑那一题不会 —— 脚本里复现不出来，所以只能绕开。

    `fresh_if_idle=True`（`pool="auto"`）时跟踪"池的活用户"：上一个用它的图还活着就复用，
    否则换新池；`False`（`pool="shared"`）就是死用一个池，专门用来复现上面那个 assert。
    两条都不是默认路径（默认 `pool="off"`）。
    """
    global _GRAPH_POOL
    alive = [w for w in _POOL_USERS if w() is not None]
    _POOL_USERS[:] = alive
    if _GRAPH_POOL is None or (fresh_if_idle and not alive):
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


def register_pool_user(graph: torch.cuda.CUDAGraph) -> None:
    """登记"这张图用了当前池"。弱引用，图被回收后自动退出统计。"""
    _POOL_USERS.append(weakref.ref(graph))


class GraphDecoder:
    """把 `NovaForCausalLM` 的单步解码包成一张 CUDA Graph。

    用法::

        dec = GraphDecoder(nova, max_len=256)
        dec.prefill(prompt_ids)          # eager
        dec.capture()                    # 捕获
        for _ in range(n):
            dec.step()                   # 一次 replay = 一个 token
    """

    def __init__(
        self,
        model,
        max_len: int = 256,
        batch: int = 1,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.text = model.model
        self.cfg = model.config
        self.max_len = max_len
        self.batch = batch
        self.device = device
        self.dtype = dtype

        # 滑动窗口（E2）：`swa_window > 0` 时局部层只用 **2W** 个槽（ring），全局层仍按 max_len。
        self.swa_window = int(getattr(self.cfg, "swa_window", 0) or 0)
        self.swa_chunk = int(getattr(self.cfg, "swa_chunk", 0) or 0) or self.swa_window
        if self.swa_window and self.swa_chunk > self.swa_window:
            raise ValueError(f"prefill 分块 {self.swa_chunk} 不能大于窗口 {self.swa_window}（ring 装不下）")
        self.layer_windows = list(self.text.layer_windows)
        self.window_slots = 2 * self.swa_window if self.swa_window else 0
        if self.swa_window:
            self.cache = WindowedKVCache(
                capacities=[2 * w if w else max_len for w in self.layer_windows],
                num_kv_heads=self.cfg.num_key_value_heads,
                head_dim=self.cfg.head_dim,
                max_len=max_len,
                batch=batch,
                dtype=dtype,
                device=device,
            )
            self._win_arange = torch.arange(self.window_slots, dtype=torch.long, device=device)
        else:
            self.cache = StaticKVCache(
                num_slots=self.text.num_cache_layers,
                num_kv_heads=self.cfg.num_key_value_heads,
                head_dim=self.cfg.head_dim,
                max_len=max_len,
                batch=batch,
                dtype=dtype,
                device=device,
            )
            self._win_arange = None
        self.input_ids = torch.zeros(batch, 1, dtype=torch.long, device=device)
        # ⚠️ 不再分配 `(max_len, max_len)` 的掩码表。只留一份 arange 与两个 0 维标量，
        # 掩码在 `_mask_row()` 里即时构造 —— 形状恒定，图内安全，与旧表逐位一致。
        self._arange = torch.arange(max_len, dtype=torch.long, device=device)
        self._zero = torch.zeros((), dtype=dtype, device=device)
        self._ninf = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None
        self._captured = False

    # ---- 构造 ----

    @staticmethod
    def _build_mask_table(max_len: int, dtype: torch.dtype, device) -> torch.Tensor:
        """`table[pos]` = 加性 attention mask：前 `pos+1` 位为 0，其余为 -inf。

        ⚠️ **解码路径已经不用它了**（改成 `_mask_row` 图内即时构造）。留着只为两件事：
        与旧实现**逐位对照**（`tests/test_graph_decode.py`）、以及核算"旧方案要多少显存"。
        **别在生产路径里调**：`max_len=65536` 时要 8.0 GiB。
        """
        neg = torch.finfo(dtype).min
        idx = torch.arange(max_len, device=device)
        keep = idx.view(1, -1) <= idx.view(-1, 1)
        zero = torch.zeros((), dtype=dtype, device=device)
        ninf = torch.full((), neg, dtype=dtype, device=device)
        return torch.where(keep, zero, ninf)

    @classmethod
    def for_length(cls, model, n_tokens: int, reserve: int = 64, **kwargs) -> "GraphDecoder":
        """按"要用多少 token"选桶 —— 而不是随手给一个巨大的 `max_len`。

        预设 18432 却只用到 2048 时，decode 要读满 18432 个槽位（实测 139.3 ms/token，
        `clocks.sm` 2475）；按桶给 2070 就是 60.4 ms/token。`reserve` 留给生成的新 token。
        """
        return cls(model, max_len=bucket_for(int(n_tokens) + int(reserve)), **kwargs)

    def _mask_row(self, pos: torch.Tensor) -> torch.Tensor:
        """`pos` 那一行的加性掩码（前 `pos+1` 位 0，其余 -inf），形状恒为 `(1,1,1,max_len)`。

        与 `_build_mask_table(max_len)[pos]` **逐位一致**，但不占 `max_len²` 的常驻。
        """
        return torch.where(self._arange <= pos, self._zero, self._ninf).view(1, 1, 1, self.max_len)

    def _window_mask(self, q_start, n: int) -> torch.Tensor:
        """**局部层**的加性掩码，形状 `(1,1,n,2W)`，列按 ring 的**存储序**排列。

        槽位 `i` 的绝对位置 `p_i = end - ((end - i) % cap)`（`end` = 最后一个写入位置）。
        有效 = 写过（`p_i >= first`）且 `p_i <= q` 且 `q - p_i < W`。

        存储序不需要重排：注意力是对 key 集合求和，集合对了顺序无所谓。
        """
        end = q_start + (n - 1)
        p = end - torch.remainder(end - self._win_arange, self.window_slots)  # [2W]
        q = torch.arange(n, device=p.device, dtype=torch.long) + q_start       # [n]
        pr, qr = p.view(1, -1), q.view(-1, 1)
        valid = (pr >= self.cache.first) & (pr <= qr) & ((qr - pr) < self.swa_window)
        return torch.where(valid, self._zero, self._ninf).view(1, 1, n, self.window_slots)

    # ---- eager prefill ----

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        offset: int | None = None,
        reset: bool = True,
        prefix_writer=None,
    ) -> torch.Tensor:
        """用静态 cache 做一次 eager prefill（长度可变，不进图）。

        - `offset`：本段文本的起始位置。默认 0（`reset=True`）或 cache 当前长度（`reset=False`，
          用于"先把历史写进 cache，再插记忆，再写当前轮"这种分段 prefill）。
        - `prefix_writer(cache)`：复位之后调用，负责把前缀写进 `[offset, offset+m)` 并返回**新的起点**。
        - 非 0 起点时必须显式给 attention mask —— **不能走 `is_causal` 那条路**
          （那条路会把 kv 截断到前 n 个槽位，也就是前缀）。
        """
        if reset:
            self.cache.reset()
        if prefix_writer is not None:
            offset = int(prefix_writer(self.cache))
        if offset is None:
            offset = int(self.cache.pos.item())
        offset = int(offset)
        n = input_ids.shape[1]
        if n + offset > self.max_len:
            raise ValueError(f"prompt 长度 {n} + 起点 {offset} 超过 max_len {self.max_len}")
        self.cache.pos.fill_(offset)
        if self.swa_window:
            # ring 的掩码靠 `first` 判断"哪些槽还没写过"；全长 cache 不需要。
            # ⚠️ **取 min，不能直接覆盖**：S4 的记忆注入是"先写前缀 -> 再从 offset 继续 prefill"，
            # 若这里把 first 覆盖成 offset，第二次 prefill 之后局部层会把 `[0, offset)` 全判成
            # "没写过"，窗口里只剩当前这一小段 —— 实测表现是**所有问题都答不出来**（0/4）。
            self.cache.first.fill_(min(int(self.cache.first.item()), offset))
        # 滑动窗口开启时**必须分块**：局部层的 ring 只有 2W 个槽，一次写 n > 2W 会互相覆盖。
        # 分块大小取 W（== 窗口），这样"上一块的尾巴 + 本块" ≤ 2W，局部层看得见它需要的一切。
        chunk = self.swa_chunk if self.swa_window else n
        out = None
        for c0 in range(0, n, chunk):
            nc = min(chunk, n - c0)
            off = offset + c0
            self.cache.pos.fill_(off)
            # 非 0 起点、或开了滑窗（第 2 块起必然非 0 起点）时都要显式掩码：
            # 不能走 `is_causal` 那条路（它把 kv 截断到前 n 个槽位 = 前缀）。
            if off or self.swa_window:
                rows = torch.arange(off, off + nc, device=self.cache.pos.device)
                keep = self._arange.view(1, 1, 1, -1) <= rows.view(1, 1, nc, 1)
                mask = torch.where(keep, self._zero, self._ninf)
            else:
                mask = None
            wmask = self._window_mask(off, nc) if self.swa_window else None
            out = self.model(
                input_ids=input_ids[:, c0 : c0 + nc],
                past_key_values=self.cache,
                attention_mask=mask,
                window_mask=wmask,
                cross_mode="off",
                logits_to_keep=1,
            )
        # ⚠️ 必须填**第一个生成 token**，不能填最后一个 prompt token：
        # prefill 已把整个 prompt 写进 cache（位置 offset..offset+n-1），pos 指向 offset+n。
        # 若填最后一个 prompt token，图的第一步会把它在位置 offset+n 上**再算一遍**。
        self.input_ids.copy_(out[:, -1].argmax(-1).view(self.batch, 1))
        self.cache.pos.fill_(offset + n)
        return out

    # ---- 图内的单步 ----

    def _body(self) -> torch.Tensor:
        """一个完整的解码步。**图内不得出现 CPU 同步**。"""
        t = self.text
        pos = self.cache.pos
        position_ids = pos.view(1, 1, 1).expand(3, self.batch, 1)
        mask = self._mask_row(pos)
        wmask = self._window_mask(pos, 1) if self.swa_window else None
        hidden = t(
            input_ids=self.input_ids,
            past_key_values=self.cache,
            position_ids=position_ids,
            attention_mask=mask,
            window_mask=wmask,
            cross_mode="off",
        )
        logits = self.model.lm_head_forward(hidden)
        # 贪心选下一个 token 并**原地写回**输入缓冲 —— 下一次 replay 直接读它
        self.input_ids.copy_(logits[:, -1].argmax(dim=-1, keepdim=True).view(self.batch, 1))
        pos.add_(1)
        return logits

    # ---- 捕获 / 回放 ----

    def _save_ring_slots(self, start: int, warmup: int):
        """把 warmup 会覆盖到的 ring 槽位存下来（只对**局部层**）。

        warmup 写的位置是 `start .. start+warmup-1`；其中 `start` 那一槽本来就还没写过
        （下一次真实 replay 会写它），只有 `start+1 ..` 才是真的污染。
        """
        if not self.swa_window:
            return []
        out = []
        for slot, cap in enumerate(self.cache.capacities):
            if cap >= self.max_len:
                continue
            idx = [int((start + j) % cap) for j in range(1, warmup)]
            if not idx:
                continue
            t = torch.tensor(idx, device=self.cache.pos.device)
            out.append((slot, idx,
                        self.cache.key_cache[slot][:, :, t, :].clone(),
                        self.cache.value_cache[slot][:, :, t, :].clone()))
        return out

    def _restore_ring_slots(self, saved) -> None:
        for slot, idx, kk, vv in saved:
            t = torch.tensor(idx, device=self.cache.pos.device)
            self.cache.key_cache[slot][:, :, t, :] = kk
            self.cache.value_cache[slot][:, :, t, :] = vv

    def capture(self, warmup: int = 3, pool: str | object = "off") -> None:
        """捕获。调用前必须先 `prefill()`（`pos` 决定图的起始位置）。

        `pool` 四档：

        | 取值 | 行为 |
        |---|---|
        | `"off"`（**默认**） | 不传池，每张图用自己的 —— **实测不泄漏**，见下面的实测 |
        | `"auto"` | 上一个用池的图还活着就复用，否则换新池 |
        | `"shared"` | 死用一个池。会踩 PyTorch 的裸 assert，只用于复现 |
        | 其它 | 直接当 pool handle 传给 `torch.cuda.graph` |

        ⚠️ **为什么默认不是共享池**（实测，`probe_graph_recapture.py`）：
        ① "重捕就爆显存"的说法**不成立** —— 每轮 `del` + `gc` + `empty_cache` 之后
        allocated 逐轮稳定（4 轮基本平的），起点/终点都不涨，三种策略都一样；
        ② 共享池反而会踩 `CUDACachingAllocator.cpp:2225` 的裸 assert
        （`it->second->use_count > 0`），而且只在**特定分配器状态**下触发、脚本里复现不出来
        （`tests/test_decode_mask_bucket.py` 全量跑会中，单跑不会）。
        所以默认走最朴素、没有额外机制的一档；要共享得自己传 pool 并承担上面那个坑。
        """
        self.model.eval()
        start = int(self.cache.pos.item())
        saved_ids = self.input_ids.clone()
        # ⚠️ warmup 会**原地写** `warmup` 个位置。全长槽位无所谓（它们在 `pos` 之后，被掩码挡掉），
        # 但 ring 槽位会**覆盖掉窗口内的老 token**（槽 i 装的是"最近一次写到 i 的 token"，
        # 而掩码只认位置、不认内容）⇒ 必须先把要被动的那几个槽存下来，warmup 后写回。
        saved_ring = self._save_ring_slots(start, warmup)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            with torch.inference_mode():
                for _ in range(warmup):
                    self._body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # warmup 把 pos 推进了 warmup 步，复位到起点
        self.cache.pos.fill_(start)
        self.input_ids.copy_(saved_ids)
        self._restore_ring_slots(saved_ring)

        g = torch.cuda.CUDAGraph()
        if pool == "off":
            handle = None
        elif pool in ("auto", "shared"):
            handle = shared_graph_pool(fresh_if_idle=(pool == "auto"))
        else:
            handle = pool
        with torch.inference_mode():
            with torch.cuda.graph(g, pool=handle):
                self.logits = self._body()
        self.graph = g
        self._captured = True
        if handle is not None:
            register_pool_user(g)

    def grow(self, new_max_len: int) -> None:
        """把已写入的 KV 搬进更大的 cache（**跨桶**时用）。形状变了 ⇒ 必须重捕图。

        返回时 `_captured` 已被清成 False，调用者负责重新 `capture()`。

        ⚠️ **故意不加 `@torch.inference_mode()`**：这里的 `torch.zeros` 造出来的新 cache
        会变成 **inference tensor**，而 `capture()` 在 inference_mode **之外**调
        （它自己内部才开），届时 `cache.pos.fill_(...)` 会报
        "Inplace update to inference tensor outside InferenceMode is not allowed"（实测踩过）。
        """
        new_max_len = int(new_max_len)
        if new_max_len <= self.max_len:
            return
        used = int(self.cache.pos.item())
        first = int(self.cache.first.item()) if hasattr(self.cache, "first") else 0
        old = self.cache
        if self.swa_window:
            # 局部层的容量（2W）**不随 max_len 变**，所以它们的 ring 布局原样搬过去；
            # 只有全局层的容量变。`pos` / `first` 必须一起带过去，否则 ring 的位置公式会错。
            new = WindowedKVCache(
                capacities=[2 * w if w else new_max_len for w in self.layer_windows],
                num_kv_heads=self.cfg.num_key_value_heads,
                head_dim=self.cfg.head_dim,
                max_len=new_max_len,
                batch=self.batch,
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(new.num_slots):
                n = min(used, new.capacities[i])
                new.key_cache[i][:, :, :n] = old.key_cache[i][:, :, :n]
                new.value_cache[i][:, :, :n] = old.value_cache[i][:, :, :n]
            new.first.fill_(first)
        else:
            new = StaticKVCache(
                num_slots=old.num_slots,
                num_kv_heads=self.cfg.num_key_value_heads,
                head_dim=self.cfg.head_dim,
                max_len=new_max_len,
                batch=self.batch,
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(new.num_slots):
                new.key_cache[i][:, :, :used] = old.key_cache[i][:, :, :used]
                new.value_cache[i][:, :, :used] = old.value_cache[i][:, :, :used]
        new.pos.fill_(used)
        self.cache = new
        self.max_len = new_max_len
        self._arange = torch.arange(new_max_len, dtype=torch.long, device=self.device)
        self.graph = None
        self._captured = False
        del old

    @torch.inference_mode()
    def step(self) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("还没 capture()")
        # ⚠️ 必须在 inference_mode 内 replay：图内对 `input_ids` 做了原地写入，
        # 若在 inference_mode 之外回放，torch 会报 "Inplace update to inference tensor"。
        self.graph.replay()
        return self.logits

    # ---- 便捷接口 ----

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, n_new_tokens: int) -> torch.Tensor:
        self.prefill(input_ids)
        if not self._captured:
            self.capture()
        out = [self.input_ids.clone()]
        for _ in range(n_new_tokens - 1):
            self.step()
            out.append(self.input_ids.clone())
        return torch.cat(out, dim=1)

    def extra_repr(self) -> str:
        return f"max_len={self.max_len}, slots={self.text.num_cache_layers}, kv={self.cache.nbytes()/1024**2:.0f}MiB"
