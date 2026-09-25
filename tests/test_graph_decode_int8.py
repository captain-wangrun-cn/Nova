"""int8 流式 cache 接进 `GraphDecoder` 的端到端验收（①.5）。

| 测试 | 判据 |
|------|------|
| `test_int8_decode_runs_and_agrees_on_tokens` | 图解码跑通；与 fp16 路径比，**至少 3/4 个贪心 token 相同** |
| `test_capture_replay_matches_eager` | 图内 replay 与 eager 前向**逐位一致**（证明捕获没改变数值） |
| `test_int8_cache_is_smaller` | 常驻字节数比 fp16 少（`max_len=2048` 下 < 0.6） |

**为什么 token 判据不是逐位一致**：int8 量化本身就是有损的（E3 已量过精度：8 位 KV 对
perplexity 影响可忽略）。decode 是**自回归**的 —— 一步的微小差异可能让某个 token 分叉，
然后整条序列越走越远。所以这里卡"大部分 token 一致"；位级一致性由
`tests/test_kvattn_stream.py` 对着"先还原再算"单独保证。

**`test_capture_replay_matches_eager` 的坑（踩过）**：`_body()` 会把 argmax 写回
`input_ids`。若不在 eager 之后把 `input_ids` 复原，`capture()` 就会拿**下一个 token** 去捕获，
两次算的根本不是同一步 —— 表现为 logits 差一大截（实测 4.9 vs 9.2），看起来像"图有问题"。
"""

from __future__ import annotations

import pytest
import torch

from nova.decode import GraphDecoder
from nova.kernels import triton_available

pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton 不可用")

MAX_LEN = 256


def _decoder(nova, quant: str, max_len: int = MAX_LEN) -> GraphDecoder:
    return GraphDecoder(nova, max_len=max_len, quant=quant)


@pytest.mark.parametrize("quant", ["off", "int8"])
def test_capture_replay_matches_eager(bundle, prompt_ids, quant: str):
    """图内 replay 与 eager 的下一步 logits 逐位一致。"""
    nova, _, _ = bundle
    dec = _decoder(nova, quant)
    dec.prefill(prompt_ids)
    pos0 = int(dec.cache.pos.item())
    ids0 = dec.input_ids.clone()          # ⚠️ `_body()` 会写回 argmax，必须复原
    with torch.inference_mode():
        eager = dec._body().clone()
    dec.cache.pos.fill_(pos0)
    dec.input_ids.copy_(ids0)
    if quant == "int8":
        dec.cache.refresh_scalars()
    dec.capture(warmup=2)
    dec.step()
    torch.cuda.synchronize()
    assert torch.equal(dec.logits, eager), f"quant={quant} 图内 replay 与 eager 不一致"


def test_int8_decode_runs_and_agrees_on_tokens(bundle, prompt_ids):
    """两条路各生成 4 个 token：int8 至少要与 fp16 有 3 个相同。"""
    nova, _, _ = bundle
    outs = {}
    for quant in ("off", "int8"):
        dec = _decoder(nova, quant)
        dec.prefill(prompt_ids)
        dec.capture(warmup=2)
        toks = [int(dec.input_ids.item())]
        for _ in range(3):
            dec.step()
            toks.append(int(dec.input_ids.item()))
        outs[quant] = toks
    agree = sum(a == b for a, b in zip(outs["off"], outs["int8"]))
    assert agree >= 3, f"int8 与 fp16 只有 {agree}/4 个 token 相同：{outs}"


def test_int8_cache_is_smaller(bundle):
    """int8 cache 的常驻必须明显小于 fp16（真省显存，不只是省带宽）。

    用 `max_len=2048` 比：尾部环是**固定 64 槽**，桶太小时它占比虚高
    （`max_len=256` 下比值 0.78，2048 下 0.56 —— 长上下文才是这套设计的目标场景）。
    """
    nova, _, _ = bundle
    off = _decoder(nova, "off", max_len=2048)
    i8 = _decoder(nova, "int8", max_len=2048)
    ratio = i8.cache.nbytes() / off.cache.nbytes()
    # 理论比值 = (1 + 尺子 0.0625) / 2 = 0.53（+ 环 3%）⇒ 实测 0.56
    assert ratio < 0.6, f"int8/fp16 = {ratio:.3f}，没有明显省显存"
    assert ratio > 0.4, f"int8/fp16 = {ratio:.3f}，与设计不符"
