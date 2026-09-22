"""解码速度基准：HF 基线 vs Nova 精简前向 vs Nova 双通路。

用法：
    & .\\.venv\\Scripts\\python.exe src\\bench_nova.py --mode hf
    & .\\.venv\\Scripts\\python.exe src\\bench_nova.py --mode nova --paths 1 --norm exact
    & .\\.venv\\Scripts\\python.exe src\\bench_nova.py --mode nova --paths 2 --norm triton

为什么要有 `--paths 1`：Nova 单通路 = **同样的 36 层**，是"精简前向到底省了多少"的
唯一干净对照（双通路是 60 次层计算，天然比基线多 1.67 倍工作量）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

PROMPT = "用两句话介绍一下你自己。"


def timed_decode(step_fn, ids, n=40, warm=6):
    """step_fn(input_ids, cache) -> logits[:, -1]；返回 (tok/s, ms/token)。"""
    cache = DynamicCache()
    with torch.inference_mode():
        logits = step_fn(ids, cache)
        cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(warm):
            logits = step_fn(cur, cache)
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits = step_fn(cur, cache)
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return n / dt, dt / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "nova"], required=True)
    ap.add_argument("--paths", type=int, default=2)
    ap.add_argument("--norm", choices=["exact", "triton"], default="exact")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from nova.loader import load_hf_base
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    hf = load_hf_base()
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")

    if args.mode == "hf":
        step = lambda i, c: hf(input_ids=i, past_key_values=c, use_cache=True).logits  # noqa: E731
        tag = "hf-baseline(36层)"
    else:
        from nova.loader import build_nova

        nova = build_nova(hf, norm_impl=args.norm, num_paths=args.paths)
        nlay = nova.model.num_cache_layers
        step = lambda i, c: nova(input_ids=i, past_key_values=c, cross_mode="off")  # noqa: E731
        tag = f"nova-paths{args.paths}-{args.norm}({nlay}层)"

    torch.cuda.reset_peak_memory_stats()
    tps, ms = timed_decode(step, ids)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"RESULT tag={tag}{(' ' + args.tag) if args.tag else ''} tok/s={tps:.2f} ms/token={ms:.1f} peak_gib={peak:.2f}")


if __name__ == "__main__":
    main()
