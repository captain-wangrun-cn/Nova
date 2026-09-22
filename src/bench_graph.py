r"""CUDA Graph 解码基准：HF 基线 vs Nova eager vs Nova + CUDA Graph。

用法：
    & .\.venv\Scripts\python.exe src\bench_graph.py --paths 1
    & .\.venv\Scripts\python.exe src\bench_graph.py --paths 2

三条路径用的是**同一份权重**，所以 tok/s 可以直接对比。
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
    ap.add_argument("--paths", type=int, default=1)
    ap.add_argument("--norm", choices=["exact", "triton"], default="triton")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--no-hf", action="store_true", help="跳过 HF 基线（省时间）")
    ap.add_argument("--quant", choices=["bnb", "nf4"], default="bnb",
                    help="Nova 的线性层用 bnb Linear4bit 还是自写 Triton NF4 GEMV")
    ap.add_argument("--lm-head4", action="store_true",
                    help="给 lm_head 额外装一份 4-bit 副本（省 ~2ms/token，+0.19 GiB）")
    args = ap.parse_args()

    from nova.decode import GraphDecoder
    from nova.loader import build_nova, load_hf_base
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    hf = load_hf_base()
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    print(f"prompt tokens = {ids.shape[1]}")

    if not args.no_hf:
        step = lambda i, c: hf(input_ids=i, past_key_values=c, use_cache=True).logits  # noqa: E731
        torch.cuda.reset_peak_memory_stats()
        tps, ms = timed_decode(step, ids, n=args.n)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"RESULT hf-baseline(36层)  tok/s={tps:6.2f} ms/token={ms:7.1f} peak_gib={peak:.2f}")

    nova = build_nova(hf, norm_impl=args.norm, num_paths=args.paths)
    nlay = nova.model.num_cache_layers
    if args.lm_head4:
        from nova.loader import enable_lm_head_4bit

        _t0 = time.perf_counter()
        enable_lm_head_4bit(nova)
        torch.cuda.synchronize()
        print(f"  enable_lm_head_4bit: 耗时 {time.perf_counter() - _t0:.2f}s，"
              f"allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    if args.quant == "nf4":
        from nova.quant import convert_to_nf4

        _t0 = time.perf_counter()
        _n_done = convert_to_nf4(nova)
        torch.cuda.synchronize()
        print(f"  convert_to_nf4: {_n_done} 个 Linear4bit -> NF4Linear，耗时 {time.perf_counter() - _t0:.1f}s")
    step = lambda i, c: nova(input_ids=i, past_key_values=c, cross_mode="off")  # noqa: E731
    torch.cuda.reset_peak_memory_stats()
    tps, ms = timed_decode(step, ids, n=args.n)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    tag = f"nova-{args.quant}{'-lh4' if args.lm_head4 else ''}-paths{args.paths}"
    print(f"RESULT {tag}-eager({nlay}层) tok/s={tps:6.2f} ms/token={ms:7.1f} peak_gib={peak:.2f}")

    # ---- CUDA Graph ----
    dec = GraphDecoder(nova, max_len=args.max_len)
    dec.prefill(ids)
    t0 = time.perf_counter()
    dec.capture()
    torch.cuda.synchronize()
    print(f"  capture 耗时 {time.perf_counter()-t0:.2f}s")

    # 正确性：图解码的前 n 个 token 应与 eager 一致
    ref_ids = []
    cache = DynamicCache()
    with torch.inference_mode():
        logits = nova(input_ids=ids, past_key_values=cache, cross_mode="off")
        cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        ref_ids.append(cur.clone())
        for _ in range(args.n - 1):
            logits = nova(input_ids=cur, past_key_values=cache, cross_mode="off")
            cur = logits[:, -1].argmax(dim=-1, keepdim=True)
            ref_ids.append(cur.clone())
    ref = torch.cat(ref_ids, dim=1)

    graph_tokens = dec.generate(ids, args.n)
    same = int((graph_tokens == ref).sum().item())
    print(f"  正确性: 图解码 vs eager Nova -> {same}/{args.n} 个 token 相同")

    # 速度：纯 replay
    for _ in range(6):
        dec.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.n):
        dec.step()
    t_cpu = (time.perf_counter() - t0) / args.n * 1000
    torch.cuda.synchronize()
    t_wall = (time.perf_counter() - t0) / args.n * 1000
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"RESULT {tag}-graph({nlay}层) tok/s={1000/t_wall:6.2f} ms/token={t_wall:7.1f} "
          f"cpu={t_cpu:6.1f} peak_gib={peak:.2f}")


if __name__ == "__main__":
    main()
