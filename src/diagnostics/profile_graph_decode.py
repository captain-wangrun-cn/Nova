r"""CUDA Graph 回放的 GPU 时间分解（用**局部图**量，CUPTI 看不到图内 kernel）。

图回放把 CPU 发射清零后，replay 的 wall 就是纯 GPU 时间。于是可以这样拆：
    A = 整图 wall
    B = 只跑 36 层（不含 lm_head）的图 wall
    C = 只跑 lm_head（真实形状 151936x2560）的图 wall
=> lm_head 成本 ~= C，层成本 ~= B，其它（embed / argmax / mask）~= A - B - C

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\profile_graph_decode.py --paths 1
"""

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

import torch
import torch.nn.functional as F

PROMPT = "用两句话介绍一下你自己。"


def time_replay(g, n=24, warm=8):
    # 必须在 inference_mode 内回放：图内对 input_ids 做了原地写入
    with torch.inference_mode():
        for _ in range(warm):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            g.replay()
        t_cpu = (time.perf_counter() - t0) / n * 1000
        torch.cuda.synchronize()
        t_wall = (time.perf_counter() - t0) / n * 1000
    return t_cpu, t_wall


def capture(fn, warm=3):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.inference_mode():
            for _ in range(warm):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(g):
            fn()
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=256)
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
    nova = build_nova(hf, norm_impl="triton", num_paths=args.paths)
    dec = GraphDecoder(nova, max_len=args.max_len)
    dec.prefill(ids)
    dec.capture()
    nlay = nova.model.num_cache_layers

    cpu_a, wall_a = time_replay(dec.graph)
    print(f"A 整图（{nlay} 层 + lm_head）   : cpu={cpu_a:6.2f}  wall={wall_a:7.2f} ms/token")
    del dec.graph
    gc.collect()
    torch.cuda.empty_cache()

    # ---- B: 只跑层，不含 lm_head ----
    t = dec.text

    def body_layers_only():
        pos = dec.cache.pos
        position_ids = pos.view(1, 1, 1).expand(3, 1, 1)
        mask = dec._mask_row(pos)
        hidden = t(
            input_ids=dec.input_ids,
            past_key_values=dec.cache,
            position_ids=position_ids,
            attention_mask=mask,
            cross_mode="off",
        )
        dec.input_ids.zero_()
        pos.add_(1)
        return hidden

    dec.cache.pos.fill_(int(ids.shape[1]))
    g_b = capture(body_layers_only)
    cpu_b, wall_b = time_replay(g_b)
    print(f"B 只有 {nlay} 层（无 lm_head）  : cpu={cpu_b:6.2f}  wall={wall_b:7.2f} ms/token")
    del g_b
    gc.collect()
    torch.cuda.empty_cache()

    # ---- C: 只有 lm_head ----
    emb_w = t.embed_tokens.weight
    h = torch.randn(1, 1, emb_w.shape[1], device="cuda", dtype=emb_w.dtype)

    def body_head_only():
        return F.linear(h, emb_w)

    g_c = capture(body_head_only)
    cpu_c, wall_c = time_replay(g_c)
    print(f"C 只有 lm_head ({emb_w.shape[0]}x{emb_w.shape[1]}): cpu={cpu_c:6.2f}  wall={wall_c:7.2f} ms/token")
    del g_c
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n=> lm_head 占整图 {wall_c/wall_a*100:4.1f}% ；层占 {wall_b/wall_a*100:4.1f}% ；"
          f"其它 {max(wall_a-wall_b-wall_c,0)/wall_a*100:4.1f}%")


if __name__ == "__main__":
    main()
