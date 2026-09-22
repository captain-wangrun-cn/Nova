"""CUDA Graph 捕获的贪心解码器。

**为什么需要它**（实测数据见 [reports/s3-graph-decode.md](../../reports/s3-graph-decode.md)）：

- 本机单次 CUDA kernel 启动约 **13.5us CPU**（Windows/WDDM；`x * 2.0` 都要 16us）
- Nova 单通路每 token 有 **~6500 次**启动 -> 光启动就 ~62ms，GPU 全程挨饿
- 把**整步**捕获成一张 CUDA Graph 后，每 token 只剩 **1 次 replay**（~28us CPU）

结构：整步（embed -> 36/60 层 -> lm_head -> argmax -> 写回 input_ids -> pos+1）
全部在图内，**CPU 每 token 只发一次 replay**。

**前提**：KV cache 必须是定长原地写入（`StaticKVCache`），
且图内**不能有任何 `.item()` / CPU 同步 / 动态形状**。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cache import StaticKVCache


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

        self.cache = StaticKVCache(
            num_slots=self.text.num_cache_layers,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
            max_len=max_len,
            batch=batch,
            dtype=dtype,
            device=device,
        )
        self.input_ids = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.mask_table = self._build_mask_table(max_len, dtype, device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None
        self._captured = False

    # ---- 构造 ----

    @staticmethod
    def _build_mask_table(max_len: int, dtype: torch.dtype, device) -> torch.Tensor:
        """`table[pos]` = 加性 attention mask：前 `pos+1` 位为 0，其余为 -inf。

        预先算好整张表，图内只需一次 `index_select` 就能取到当前行 —— 形状恒定。
        """
        neg = torch.finfo(dtype).min
        idx = torch.arange(max_len, device=device)
        keep = idx.view(1, -1) <= idx.view(-1, 1)
        zero = torch.zeros((), dtype=dtype, device=device)
        ninf = torch.full((), neg, dtype=dtype, device=device)
        return torch.where(keep, zero, ninf)

    # ---- eager prefill ----

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """用静态 cache 做一次 eager prefill（长度可变，不进图）。"""
        n = input_ids.shape[1]
        if n > self.max_len:
            raise ValueError(f"prompt 长度 {n} 超过 max_len {self.max_len}")
        self.cache.reset()
        out = self.model(
            input_ids=input_ids, past_key_values=self.cache, cross_mode="off", logits_to_keep=1
        )
        # ⚠️ 必须填**第一个生成 token**，不能填最后一个 prompt token：
        # prefill 已把整个 prompt 写进 cache（位置 0..n-1），pos 指向 n。
        # 若填最后一个 prompt token，图的第一步会把它在位置 n 上**再算一遍**。
        self.input_ids.copy_(out[:, -1].argmax(-1).view(self.batch, 1))
        self.cache.pos.fill_(n)
        return out

    # ---- 图内的单步 ----

    def _body(self) -> torch.Tensor:
        """一个完整的解码步。**图内不得出现 CPU 同步**。"""
        t = self.text
        pos = self.cache.pos
        position_ids = pos.view(1, 1, 1).expand(3, self.batch, 1)
        mask = self.mask_table.index_select(0, pos).view(1, 1, 1, self.max_len)
        hidden = t(
            input_ids=self.input_ids,
            past_key_values=self.cache,
            position_ids=position_ids,
            attention_mask=mask,
            cross_mode="off",
        )
        logits = self.model.lm_head_forward(hidden)
        # 贪心选下一个 token 并**原地写回**输入缓冲 —— 下一次 replay 直接读它
        self.input_ids.copy_(logits[:, -1].argmax(dim=-1, keepdim=True).view(self.batch, 1))
        pos.add_(1)
        return logits

    # ---- 捕获 / 回放 ----

    def capture(self, warmup: int = 3) -> None:
        """捕获。调用前必须先 `prefill()`（`pos` 决定图的起始位置）。"""
        self.model.eval()
        start = int(self.cache.pos.item())
        saved_ids = self.input_ids.clone()

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

        g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(g):
                self.logits = self._body()
        self.graph = g
        self._captured = True

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
