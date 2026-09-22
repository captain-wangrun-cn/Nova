r"""`gqa_in_sdpa=False`（repeat_kv 展平）的数值等价性与长上下文天花板。

两件事：
1. **数值等价**：同一个 prompt，GQA 留在 SDPA（math 后端）vs 展平成 32 头（融合内核），
   比 logits 的最大绝对差与 argmax 是否一致。**改注意力路径必须先过这一关。**
2. **新天花板**：把 KV cache 按长度精确分配（max_len = 长度 + 256），逐档试到 OOM。

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_kernel_equivalence.py
"""

from __future__ import annotations

import argparse
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


def build_ids(tok, n_tokens: int):
    n_rounds = max(4, int(round(n_tokens / 36)))
    msgs = []
    for i in range(n_rounds):
        u, a = FILLER[i % len(FILLER)]
        msgs.append({"role": "user", "content": f"{u}（第 {i + 2} 轮）"})
        msgs.append({"role": "assistant", "content": a})
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False),
               return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[8192, 16384, 24576, 32768])
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")

    print("=== 1. 数值等价（1024 token prompt）===")
    ids = build_ids(tok, 1024)
    outs = {}
    for flag in (True, False):
        set_gqa(nova, flag)
        with torch.inference_mode():
            outs[flag] = nova.model(input_ids=ids, cross_mode="off").float()
    d = (outs[True] - outs[False]).abs()
    same = (outs[True].argmax(-1) == outs[False].argmax(-1)).all().item()
    print(f"  max|diff| = {d.max().item():.3e}   mean|diff| = {d.mean().item():.3e}   argmax 全同 = {same}")
    print(f"  logits 量级 max|v| = {outs[True].abs().max().item():.2f}")
    del outs, d
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== 2. 展平后的长上下文天花板（1 通路 · KV cache 按长度分配）===")
    print(f"{'token':>7s} {'峰值':>9s} {'prefill':>10s}   KV cache")
    for target in args.lens:
        ids = build_ids(tok, target)
        n = ids.shape[1]
        max_len = n + 256
        set_gqa(nova, False)
        try:
            gc.collect()
            torch.cuda.empty_cache()
            dec = GraphDecoder(nova, max_len=max_len)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            dec.prefill(ids)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"{n:>7d} {peak:>8.2f}G {dt * 1000:>8.0f}ms   {dec.cache.nbytes() / 1024 ** 3:.2f} GiB")
            del dec
        except Exception as exc:  # noqa: BLE001
            print(f"{n:>7d}   —— 失败：{type(exc).__name__}: {str(exc)[:80]}")
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("\n注：峰值含权重（约 3.2 GiB）。D17 预算是 < 7GB。")


if __name__ == "__main__":
    main()
