"""RMSNorm 的两种实现，由 `impl` 开关切换。

| impl | kernel 数/次 | 数值 | 用途 |
|------|:---:|------|------|
| `"exact"`  | ~16 | 与 HF **逐位一致** | S3 正确性验收（"门控关闭 ≈ 基线"要能归零） |
| `"triton"` | **1** | 有极小差异（归约顺序不同） | 速度路径 |

HF 原版把 hidden 转 fp32 算 variance 再转回来；`"exact"` 保持同样的算子顺序，
所以能做到 `max|diff| == 0`。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import kernels


class LeanRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        impl: str = "exact",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
        self.eps = float(eps)
        self.impl = impl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl == "triton" and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            return kernels.fused_rms_norm(x, self.weight, self.eps)
        # ---- 与 HF Qwen3VLTextRMSNorm 完全同序 ----
        dtype = x.dtype
        h = x.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.eps)
        return self.weight * h.to(dtype)

    @classmethod
    def from_hf(cls, hf_norm, impl: str = "exact") -> "LeanRMSNorm":
        """从 HF 的 `Qwen3VLTextRMSNorm` 复制权重。

        ⚠️ **必须连 dtype / device 一起继承。** 早期版本用 `torch.ones(dim)` 建参数，
        结果是 fp32；而 HF 的权重是 fp16。`weight(fp32) * h(fp16)` 会被**提升到 fp32**，
        于是 `impl="exact"` 反而与 HF 不一致（实测 max|diff| ≈ 4.6e-3）。
        """
        src = hf_norm.weight.detach()
        dim = int(src.shape[0])
        eps = float(getattr(hf_norm, "variance_epsilon", getattr(hf_norm, "eps", 1e-6)))
        new = cls(dim, eps=eps, impl=impl, dtype=src.dtype, device=src.device)
        with torch.no_grad():
            new.weight.copy_(src)
        return new

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}, impl={self.impl}"
