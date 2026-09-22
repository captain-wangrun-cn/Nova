r"""图解码 vs eager 解码：逐步 logits 对照 + token 序列 dump。

两个问题要回答：
  1. `prefill()` 之后 `input_ids` 指向最后一个 prompt token，图的第一步会不会**重复处理**它？
  2. 双通路只有 17/24 token 相同 —— 数值噪声还是真 bug？

做法：两边**强制吃同一个 token**，逐步比 logits。

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\check_graph_exactness.py --paths 1
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers.cache_utils import DynamicCache

PROMPT = "用两句话介绍一下你自己。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
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
    g0 = dec.prefill(ids).clone()          # 图侧的 prefill logits（pos = n-1）
    print(f"prefill logits 对照: max|diff| = {g0.abs().max().item():.3e} (绝对值)")
    dec.capture()

    cache = DynamicCache()
    with torch.inference_mode():
        e0 = nova(input_ids=ids, past_key_values=cache, cross_mode="off")[:, -1:]
    print(f"prefill logits 差异   : max|diff| = {(g0 - e0).abs().max().item():.3e}")

    # 两边都从"eager prefill 选出的下一个 token"开始，强制吃同一个 token
    cur = e0[:, -1].argmax(-1, keepdim=True)      # [1,1]
    print(f"\n{'step':>4} {'pos':>4} {'max|diff|':>12} {'eager_arg':>10} {'graph_arg':>10} {'同?':>5} {'margin':>9}")
    n_same = 0
    for i in range(1, args.n + 1):
        with torch.inference_mode():
            e_logits = nova(input_ids=cur, past_key_values=cache, cross_mode="off")[:, -1:]
        dec.input_ids.copy_(cur)
        g_logits = dec.step().clone()
        d = (g_logits - e_logits).abs().max().item()
        a_e = int(e_logits.argmax(-1))
        a_g = int(g_logits.argmax(-1))
        top2 = e_logits[0, 0].float().topk(2).values
        same = a_e == a_g
        n_same += same
        print(f"{i:>4} {i + ids.shape[1] - 1:>4} {d:>12.3e} {a_e:>10d} {a_g:>10d} {str(same):>5} "
              f"{float(top2[0] - top2[1]):>9.3f}")
        cur = e_logits[:, -1].argmax(-1, keepdim=True)

    print(f"\n同一输入下 {n_same}/{args.n} 步 argmax 相同")


if __name__ == "__main__":
    main()
