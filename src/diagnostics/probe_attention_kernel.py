r"""注意力内核：GQA 在 SDPA 里 vs 展平 KV vs 强制 cuDNN。

背景（`probe_long_context_vram.py` 已核查）：本机 torch 2.6.0+cu124 **没有编译 flash attention**，
且 mem-efficient / cuDNN 两个融合内核都**要求 Q/K/V 头数相同** —— 而我们是 GQA（32 Q / 8 KV）。
于是 SDPA 退回 **math 后端，实体化 O(n^2) 的 fp32 分数矩阵**：
3520 token 的 prefill 峰值 8.30 GiB（其中 ~3.5 GiB 是分数矩阵），8192 token 直接崩。

本脚本比三种做法的峰值显存与耗时：
  (a) 现状：`enable_gqa=True`（GQA 留在 SDPA 里）
  (b) `gqa_in_sdpa=False`：先 `repeat_kv` 把 KV 展平成 32 头再进 SDPA —— 内存换内核
  (c) 强制 cuDNN 后端

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_attention_kernel.py --lens 2048 3520
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

from s4_memory_demo import FILLER, load_bundle  # noqa: E402


def attn_modules(nova):
    t = nova.model
    for layer in list(t.prefix_layers) + [l for p in t.path_layers for l in p] + list(t.suffix_layers):
        yield layer.self_attn


def set_gqa(nova, flag: bool) -> None:
    for attn in attn_modules(nova):
        attn.gqa_in_sdpa = bool(flag)


def run(nova, ids, use_cudnn: bool):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if use_cudnn:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            nova.model(input_ids=ids, cross_mode="off")
    else:
        nova.model(input_ids=ids, cross_mode="off")
    torch.cuda.synchronize()
    return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 1024 ** 3


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[2048, 3520, 8192])
    ap.add_argument("--max-len", type=int, default=10240)
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    base = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"权重占用 {base:.2f} GiB · 只算前向峰值（不含 KV cache）\n")

    modes = [
        ("(a) GQA 留在 SDPA（现状）", True, False),
        ("(b) repeat_kv 展平 32 头", False, False),
        ("(c) 强制 cuDNN", None, True),
    ]

    print(f"{'长度':>7s}  " + "  ".join(f"{m[0][:22]:>26s}" for m in modes))
    for n in args.lens:
        n_rounds = max(4, int(round(n / 36)))
        msgs = []
        for i in range(n_rounds):
            u, a = FILLER[i % len(FILLER)]
            msgs.append({"role": "user", "content": f"{u}（第 {i + 2} 轮）"})
            msgs.append({"role": "assistant", "content": a})
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False),
                  return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")

        cells = []
        for _label, gqa, cudnn in modes:
            if gqa is not None:
                set_gqa(nova, gqa)
            try:
                dt, peak = run(nova, ids, cudnn)
                cells.append(f"{peak:6.2f}G {dt * 1000:6.0f}ms")
            except Exception as exc:  # noqa: BLE001
                cells.append(f"失败 {type(exc).__name__}")
        print(f"{ids.shape[1]:>7d}  " + "  ".join(f"{c:>26s}" for c in cells))

    print("\n若 (b) 明显更低 —— 说明 GQA 是 O(n^2) 实体化的原因，改一个开关即可解锁长上下文。")


if __name__ == "__main__":
    main()
