r"""展平 KV 的代价与收益：能用哪个内核 + 生成层面的 token 一致率。

已知（`probe_kernel_mapping.py` 已核查）：强制两边都走 math 时 `repeat_kv` 与 `enable_gqa`
**逐位一致**，且两者对手写 fp32 参考的相对误差相同（4.07e-04）—— 映射正确，差异来自内核精度。

本脚本回答两件事：
1. 展平头数后，哪个融合内核真的能用？（逐个头数下的后端强制测试）
2. **生成层面**一致率：同一 prompt 贪心 32 token，两条路有多少个 token 相同？
   （项目既有标准：lm_head 4-bit 是 24/24、32/32 逐 token 一致）

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_attention_tradeoff.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))

import torch  # noqa: E402

from nova.decode import GraphDecoder  # noqa: E402
from s4_memory_demo import FILLER, load_bundle  # noqa: E402


def attn_modules(nova):
    t = nova.model
    for layer in list(t.prefix_layers) + [l for p in t.path_layers for l in p] + list(t.suffix_layers):
        yield layer.self_attn


def set_gqa(nova, flag: bool) -> None:
    for attn in attn_modules(nova):
        attn.gqa_in_sdpa = bool(flag)


def build_ids(tok, n_tokens: int, tail: str | None = None):
    n_rounds = max(4, int(round(n_tokens / 36)))
    msgs = []
    for i in range(n_rounds):
        u, a = FILLER[i % len(FILLER)]
        msgs.append({"role": "user", "content": f"{u}（第 {i + 2} 轮）"})
        msgs.append({"role": "assistant", "content": a})
    if tail:
        msgs.append({"role": "user", "content": tail})
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
               return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")


def greedy(nova, ids, n_new: int) -> list[int]:
    dec = GraphDecoder(nova, max_len=ids.shape[1] + n_new + 8)
    dec.prefill(ids)
    out = []
    for _ in range(n_new):
        t = int(dec.input_ids.item())
        if t in (151645, 151643):
            break
        out.append(t)
        dec._body()
    del dec
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from torch.nn.attention import SDPBackend, sdpa_kernel

    nova, tok = load_bundle(1, "triton")
    q = "简单说说你最喜欢哪种天气，两句话。"
    ids = build_ids(tok, 1024, tail=q)

    print("=== 1. 展平头数后各后端的可用性（4096 token 前向）===")
    big = build_ids(tok, 4096, tail=q)
    set_gqa(nova, False)
    for name in ("MATH", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "FLASH_ATTENTION"):
        backend = getattr(SDPBackend, name, None)
        if backend is None:
            continue
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.inference_mode(), sdpa_kernel(backend):
                nova.model(input_ids=big, cross_mode="off")
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  {name:22s} 可用 · 峰值 {peak:.2f} GiB · {dt * 1000:.0f}ms")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:22s} 不可用：{type(exc).__name__}: {str(exc)[:70]}")

    print("\n=== 2. 生成层面一致率（1024 token + 问句，贪心 32 token）===")
    outs = {}
    for flag in (True, False):
        set_gqa(nova, flag)
        outs[flag] = greedy(nova, ids, 32)
        print(f"  gqa_in_sdpa={flag!s:5s} -> {tok.decode(outs[flag], skip_special_tokens=True)[:70]!r}")
    n = min(len(outs[True]), len(outs[False]))
    same = sum(1 for a, b in zip(outs[True][:n], outs[False][:n]) if a == b)
    print(f"  逐 token 一致：{same}/{n}")
    if same < n:
        for i, (a, b) in enumerate(zip(outs[True][:n], outs[False][:n])):
            if a != b:
                print(f"    第 {i + 1} 个 token 起分歧：{tok.decode([a])!r} vs {tok.decode([b])!r}")
                break

    print("\n注：两边唯一的差别是 SDPA 走 math（fp32）还是融合内核（fp16 累加）。")


if __name__ == "__main__":
    main()
