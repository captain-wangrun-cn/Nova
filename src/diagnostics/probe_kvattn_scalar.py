r"""①.5 前置核查 · **Triton 能不能在核内 `tl.load` 一个显存标量**。

## 为什么必须核查这一条

流式 int8 cache 的"已量化前缀长度" `Q` 每 64 个 token 涨一次。若把它当**编译期常量**
（Python int 参数），只有两条路，两条都不可接受：

1. 冻结进图 → 图每 64 步**静默算错**（刚写满的组仍被当 fp16 尾部，而环槽已被更新的 token
   覆盖 ⇒ 那段历史凭空消失）；
2. 每 64 步重捕 → capture 代价远超收益（本机单次 0.3–1s）。

所以要把长度放进**显存里的 1 元素张量**、由核内 `tl.load` 读。这一步在 Triton 里能不能用、
读到的是不是**更新后**的值，必须先单独验一遍 —— 就是本脚本。

## 结论（已核查）

能。`tl.load` 读 1 元素张量可用，且**每次启动都取当前值**（写完再启核，读到的是新值）。
注意：这与"把标量当 kernel 参数"在**启动开销**上不同 —— 参数走 `packed_metadata`，
而 `tl.load` 多一次全局读（1 次 8 字节，可忽略）。

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_kvattn_scalar.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))

import torch  # noqa: E402

from nova.kernels import HAS_TRITON  # noqa: E402

if HAS_TRITON:
    import triton
    import triton.language as tl


def main() -> int:
    if not HAS_TRITON:
        print("Triton 不可用 —— 本探针跳过")
        return 1

    @triton.jit
    def _probe(OUT, SCAL, N: tl.constexpr):
        v = tl.load(SCAL)
        offs = tl.arange(0, N)
        tl.store(OUT + offs, v + offs.to(tl.float32))

    out = torch.zeros(8, device="cuda", dtype=torch.float32)
    scal = torch.tensor([100.0], device="cuda", dtype=torch.int64)
    _probe[(1,)](out, scal, N=8)
    torch.cuda.synchronize()
    first = out[:3].tolist()
    print(f"第一次：{first}")
    assert first == [100.0, 101.0, 102.0], f"读到的不是 100：{first}"

    scal.fill_(7)
    _probe[(1,)](out, scal, N=8)
    torch.cuda.synchronize()
    second = out[:3].tolist()
    print(f"改标量后：{second}")
    assert second == [7.0, 8.0, 9.0], f"读到的是旧值（说明被当常量了）：{second}"

    print("已核查：核内 tl.load 能读到**更新后**的显存标量 —— 流式前缀长度可以走这条路")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
