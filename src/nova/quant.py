r"""NF4 线性层：自写 Triton dequant+GEMV，替掉 bitsandbytes 的 `Linear4bit`。

**为什么**（见 [reports/s3-graph-decode.md](../../reports/s3-graph-decode.md) 第六节，已核查）：

- 图解码后单通路 16.2 ms/token，其中 **36 层占 12.90 ms（75.9%）**
- 36 层的 4-bit 权重总量 1.23 GB，带宽下界 **5.3 ms** -> **2.4x 缺口**
- 原因：**bnb 在 M=1 时没走 packed 4-bit GEMV**。实测单算子 GPU 耗时
  bnb `Linear4bit` **29.2us** vs fp16 `nn.Linear` **27.9us**，几乎相同 ——
  它在 dequant 到 fp16 workspace 再走 cublas，既多搬 2 倍数据，又丢掉 4-bit 的全部带宽优势。

**NF4 存储格式**（已在 [src/diagnostics/probe_nf4_format.py](../../src/diagnostics/probe_nf4_format.py) 逐位复现，
`max|diff| = 0.000e+00`）：

```
一个字节装 2 个 4-bit 索引：元素 2j = 高半字节，元素 2j+1 = 低半字节
每 64 个元素共享一个 absmax
double quant 时：absmax = state2.code[q_absmax] * state2.absmax[b // 256] + qs.offset
权重 = nf4_code[索引] * absmax[元素下标 // 64]
```

**本模块的布局选择：** 保持 bnb 的原始 `packed [N, K//2]`（**沿 K 连续**），
把 `absmax` 转置成 `[K//64, N]`（沿 N 连续）。

这样 kernel 里加载的 tile 是 `[BLOCK_N, 32]`：最后一维（K 方向）连续，
归约走 `tl.sum(axis=1)` —— **纯寄存器内规约，不需要跨线程/共享内存**。
代价是 absmax 一次性转置（0.41 GB/36 层，可忽略），packed 完全不用动。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .kernels import build_nf4_lut, fused_nf4_linear, triton_available

NF4_BLOCK = 64  # 与 kernels.NF4_BLOCK 一致（这里用普通 int，host 侧断言用）


def extract_nf4(weight, quant_state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """从 bnb 的 `Params4bit` 里取出 `(packed, absmax, code, (N, K))`。

    - `packed`：`[N, K//2]` uint8（沿 K 连续，与 bnb 原始布局一致）
    - `absmax`：`[K//64, N]` fp32（已转置，沿 N 连续；double quant 已还原）
    - `code`：`[16]` fp32（NF4 码本）
    """
    qs = quant_state
    n, k = int(qs.shape[0]), int(qs.shape[1])
    assert k % NF4_BLOCK == 0, f"K={k} 不是 {NF4_BLOCK} 的整数倍"
    assert k % 2 == 0

    packed = weight.reshape(-1).view(torch.uint8)
    assert packed.numel() == n * k // 2, (packed.numel(), n * k // 2)

    n_blocks = n * k // NF4_BLOCK
    if qs.nested:
        s2 = qs.state2
        qa = qs.absmax.reshape(-1).long()
        assert qa.numel() == n_blocks, (qa.numel(), n_blocks)
        a2 = s2.absmax.reshape(-1).repeat_interleave(int(s2.blocksize))[:n_blocks]
        absmax = s2.code.reshape(-1)[qa] * a2 + qs.offset
    else:
        absmax = qs.absmax.reshape(-1).float()
        assert absmax.numel() == n_blocks

    packed = packed.reshape(n, k // 2).contiguous()                # [N, K//2]，沿 K 连续
    absmax = absmax.reshape(n, k // NF4_BLOCK).t().contiguous()    # [K//64, N]
    code = qs.code.reshape(-1).float().contiguous()
    assert code.numel() == 16, code.numel()
    return packed, absmax, code, (n, k)


class NF4Linear(nn.Module):
    """与 `nn.Linear` 同接口，权重是 bnb 的 NF4 打包格式，前向走自写 Triton kernel。"""

    def __init__(
        self,
        packed: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        shape: tuple[int, int],
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = int(shape[0]), int(shape[1])
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("absmax", absmax, persistent=False)
        self.register_buffer("code", code, persistent=False)
        # 摊成 256 项的 fp16x2 码本表：kernel 里一个字节只查一次表
        self.register_buffer("lut", build_nf4_lut(code).to(packed.device), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_bnb(cls, lin: nn.Module) -> "NF4Linear":
        packed, absmax, code, shape = extract_nf4(lin.weight.data, lin.weight.quant_state)
        bias = lin.bias.data if getattr(lin, "bias", None) is not None else None
        return cls(packed, absmax, code, shape, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_nf4_linear(x, self.packed, self.absmax, self.lut, self.out_features, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, nf4=triton")


def convert_to_nf4(model: nn.Module, verbose: bool = False, skip: set[str] | None = None) -> int:
    """把模型里所有 bnb `Linear4bit` 就地换成 `NF4Linear`。返回替换数量。

    ⚠️ 只替换传进来的这棵模块树 —— HF 基座与 Nova 共用子模块对象，
    但在各自树上 `__setattr__` 是独立的，所以不会互相影响。

    ⚠️ 默认 **跳过 `lm_head4`**：实测在 lm_head 形状（N=151936, K=2560）上
    自写 kernel 1367us vs bnb 1022us（**慢 1.34x**）—— 那个形状的 N 太大，
    `BLOCK_N` 覆盖不住，反而把 bnb 已经不错的实现换掉。
    """
    if not triton_available():
        raise RuntimeError("Triton 不可用，无法启用 NF4Linear")

    import bitsandbytes as bnb

    skip = {"lm_head4"} if skip is None else skip
    n_done = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in skip:
                continue
            if isinstance(child, bnb.nn.Linear4bit):
                setattr(module, child_name, NF4Linear.from_bnb(child))
                n_done += 1
                if verbose:
                    print(f"  nf4: {name}.{child_name} -> {tuple(child.weight.quant_state.shape)}")
    return n_done
