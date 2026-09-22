"""交叉通路模块：交叉注意力 + 预测编码门控。

对应 [docs/02-architecture.md](../../docs/02-architecture.md) 第三、四节。

**S3 的核心验收判据在 `mode="off"` 上：** 此时交叉模块**完全不执行**，
两条通路的输出各自等于基线 → 融合后必须与单通路基线一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NovaConfig
from .norm import LeanRMSNorm

CROSS_MODES = ("off", "on", "predictive")


class CrossPathAttention(nn.Module):
    """本通路（Q）去看另一条通路（K/V）。

    **必须是因果的**：位置 t 只能看另一条通路位置 ≤ t 的状态，否则会从未来偷信息。
    """

    def __init__(self, config: NovaConfig) -> None:
        super().__init__()
        self.num_heads = config.cross_num_heads
        self.head_dim = config.hidden_size // config.cross_num_heads
        inner = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = LeanRMSNorm(config.hidden_size)
        self.kv_norm = LeanRMSNorm(config.hidden_size)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        b, t, _ = target.shape
        shape = (b, t, self.num_heads, self.head_dim)

        q = self.q_proj(self.q_norm(target)).view(shape).transpose(1, 2)
        src = self.kv_norm(source)
        k = self.k_proj(src).view(shape).transpose(1, 2)
        v = self.v_proj(src).view(shape).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class Predictor(nn.Module):
    """预测另一条通路的当前状态（预测编码的"预测器"）。"""

    def __init__(self, config: NovaConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.gate_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(config.gate_hidden, config.hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossPathBlock(nn.Module):
    """一处交叉点：双向交叉注意力 + 门控。"""

    def __init__(self, config: NovaConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = nn.ModuleList([CrossPathAttention(config) for _ in range(config.num_paths)])
        self.predictor = nn.ModuleList([Predictor(config) for _ in range(config.num_paths)])
        # 通信强度。初始 0 → sigmoid = 0.5；真正"关闭"靠 mode="off"，不靠这个值。
        self.gate = nn.Parameter(torch.zeros(config.num_paths))

    def forward(self, paths: list[torch.Tensor], mode: str = "off") -> list[torch.Tensor]:
        if mode not in CROSS_MODES:
            raise ValueError(f"mode 必须是 {CROSS_MODES} 之一，收到 {mode!r}")
        if mode == "off":
            return paths
        if self.config.num_paths != 2:
            raise NotImplementedError("S3 骨架只实现 2 条通路的配对交叉")

        out: list[torch.Tensor] = []
        for i, h in enumerate(paths):
            other = paths[1 - i]
            delta = self.attn[i](h, other)
            g = torch.sigmoid(self.gate[i])
            if mode == "predictive":
                # 预测偏差越大 → 越该通信（偏差小则压低门控）
                err = (self.predictor[i](h) - other).pow(2).mean(dim=-1, keepdim=True)
                g = g * torch.sigmoid(err)
            out.append(h + g * delta)
        return out
