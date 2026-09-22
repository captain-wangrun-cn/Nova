r"""Nova 交互式 CLI —— 用 CUDA Graph 解码边聊边看。

用法：

```powershell
$env:HF_HOME='H:\Nova\.hf-cache'; $env:TMP='H:\Nova\.tmp'; $env:TEMP=$env:TMP; $env:HF_HUB_OFFLINE='1'
$env:TRITON_CACHE_DIR='H:\Nova\.tmp\triton-cache'; $env:TORCHINDUCTOR_CACHE_DIR='H:\Nova\.tmp\inductor-cache'

& .\.venv\Scripts\python.exe src\chat.py                 # 单通路（最快，~70 tok/s）
& .\.venv\Scripts\python.exe src\chat.py --paths 2       # 双通路（Nova 本体架构，~44 tok/s）
& .\.venv\Scripts\python.exe src\chat.py --no-lm-head4   # 关掉 4-bit 输出投影（对照用，慢 2ms/token）
```

交互命令：`/exit` 退出 · `/reset` 清空对话 · `/help` 帮助

**它是什么：** 还没训练过的 Nova 骨架 —— 双通路 + CUDA Graph 解码 + 4-bit lm_head，
权重直接复用 Qwen3-VL-4B-Instruct。**S4 记忆实现之前，它就是一个跑得更快的 Qwen3-VL-4B。**

**解码怎么做的：** 图内是 argmax（快），采样在图**外**做 —— replay 之后覆盖
`input_ids`，下一步 replay 直接读采样结果。所以采样参数不改变图，也不影响速度。
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
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))

import torch  # noqa: E402


# ---- 采样（图外，不改变图）----


def sample_next(logits: torch.Tensor, args, seen_mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """从 `logits [1, vocab]` 采一个 token，返回 `[1, 1]` long。

    ⚠️ **只在 top-k 的 K 个候选上做 top-p / 采样**，不对 15 万词的完整分布排序 ——
    `torch.topk` 已经是降序，省掉一次 `sort`，也省掉 `zeros_like` + `scatter` 两次全词表写。
    """
    if args.temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float()
    if args.repetition_penalty != 1.0:
        pen = torch.where(logits < 0, logits * args.repetition_penalty, logits / args.repetition_penalty)
        logits = torch.where(seen_mask, pen, logits)

    k = args.top_k if args.top_k > 0 else 64
    vals, idx = torch.topk(logits, min(k, logits.shape[-1]), dim=-1)  # 降序

    probs = torch.softmax(vals / args.temperature, dim=-1)
    if args.top_p < 1.0:
        drop = (probs.cumsum(dim=-1) - probs) > args.top_p
        probs = probs.masked_fill(drop, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    pick = torch.multinomial(probs, num_samples=1, generator=gen)
    return idx.gather(-1, pick)


# ---- 对话历史 ----


def build_ids(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")


def fit_history(tok, messages, max_len: int, reserve: int, verbose: bool = True):
    """保证 `prompt + reserve <= max_len`；超了就从最老的对话轮开始丢。"""
    head = [m for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]

    while True:
        ids = build_ids(tok, head + turns)
        if ids.shape[1] + reserve <= max_len:
            return ids, head + turns
        if len(turns) <= 1:
            # 单轮也放不下：从左边截断（保留末尾的 `<|im_start|>assistant` 生成提示）
            keep = max(1, max_len - reserve)
            if verbose:
                print(f"  [警告] 单轮 prompt 就超过 max_len，从左截断到 {keep} 个 token")
            return ids[:, -keep:], head + turns
        if verbose:
            print(f"  [上下文] {ids.shape[1]} token 超预算，丢掉最老的一轮")
        turns = turns[2:] if len(turns) >= 2 else turns[1:]


# ---- 主循环 ----


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Nova 交互式 CLI")
    ap.add_argument("--paths", type=int, default=1, choices=(1, 2),
                    help="1 = 单通路（最快）· 2 = 双通路（Nova 本体架构）")
    ap.add_argument("--norm", choices=["exact", "triton"], default="triton")
    ap.add_argument("--max-len", type=int, default=1024,
                    help="静态 KV cache 的槽位数（也是 prompt + 生成长度的上限）")
    ap.add_argument("--max-new", type=int, default=512, help="单轮最多生成多少个 token")
    ap.add_argument("--system", type=str, default=None, help="可选的 system prompt")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = 贪心解码")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-lm-head4", action="store_true",
                    help="不装 4-bit 输出投影（慢 2ms/token，仅用于对照）")
    args = ap.parse_args()

    from nova.decode import GraphDecoder
    from nova.loader import DEFAULT_REPO, build_nova, enable_lm_head_4bit, load_hf_base
    from transformers import AutoTokenizer

    print(f"加载 tokenizer（{DEFAULT_REPO}）...")
    tok = AutoTokenizer.from_pretrained(DEFAULT_REPO)
    print("加载基座（4-bit，约 10 秒）...")
    hf = load_hf_base()
    print(f"构建 Nova（{args.paths} 条通路）...")
    nova = build_nova(hf, norm_impl=args.norm, num_paths=args.paths)
    if not args.no_lm_head4:
        enable_lm_head_4bit(nova)

    n_layers = nova.model.num_cache_layers
    dec = GraphDecoder(nova, max_len=args.max_len)

    stop_ids = {tok.eos_token_id}
    for extra in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(extra)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)

    gen = torch.Generator(device="cuda")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        gen.manual_seed(args.seed)

    base = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)
    captured = False

    print()
    print("=" * 62)
    print(f"  Nova CLI · {n_layers} 层 · 通路 {args.paths} · max_len {args.max_len}")
    print(f"  采样 temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}"
          f" rep={args.repetition_penalty}")
    print(f"  lm_head: {'4-bit' if not args.no_lm_head4 else 'fp16'}"
          f" · KV cache {dec.cache.nbytes() / 1024**2:.0f} MiB")
    print("  /exit 退出 · /reset 清空对话 · /help 帮助")
    print("=" * 62)

    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user in ("/exit", "/quit", "/q"):
            break
        if user == "/help":
            print("  /exit  退出        /reset 清空对话与 KV cache")
            print("  直接输入内容即可对话。上下文超过 max_len 时会自动丢弃最老的对话轮。")
            continue
        if user == "/reset":
            messages = list(base)
            print("  [已清空对话历史]")
            continue

        messages.append({"role": "user", "content": user})
        ids, messages = fit_history(tok, messages, args.max_len, args.max_new)

        t0 = time.perf_counter()
        dec.prefill(ids)
        if not captured:
            dec.capture()
            captured = True
        t_prefill = time.perf_counter() - t0

        print("\nNova > ", end="", flush=True)
        vocab = int(nova.model.embed_tokens.weight.shape[0])
        seen_mask = torch.zeros(vocab, dtype=torch.bool, device="cuda")
        seen_mask.index_fill_(0, ids[0], True)
        gen_ids: list[int] = []
        text_printed = 0
        n_pos = ids.shape[1]
        n_gen = 0
        t_gen0 = time.perf_counter()

        with torch.inference_mode():
            while n_gen < args.max_new and n_pos < args.max_len:
                logits = dec.step()
                nxt = sample_next(logits[:, -1, :], args, seen_mask, gen)
                tok_id = int(nxt.item())
                if tok_id in stop_ids:
                    break
                dec.input_ids.copy_(nxt)
                seen_mask[tok_id] = True
                gen_ids.append(tok_id)
                n_gen += 1
                n_pos += 1
                # 整段重解码、只打印新增后缀 —— 逐 token 解码会踩到半个 UTF-8 字节
                text = tok.decode(gen_ids, skip_special_tokens=True)
                if len(text) > text_printed:
                    print(text[text_printed:], end="", flush=True)
                    text_printed = len(text)

        t_gen = time.perf_counter() - t_gen0
        print()
        if n_gen:
            print(f"  [prefill {ids.shape[1]} token / {t_prefill * 1000:.0f} ms"
                  f" · 生成 {n_gen} token / {t_gen * 1000:.0f} ms"
                  f" · {n_gen / t_gen:.1f} tok/s · 显存 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB]")
        else:
            print("  [没有生成任何 token]")

        messages.append({"role": "assistant", "content": tok.decode(gen_ids, skip_special_tokens=True)})

    print("再见。")


if __name__ == "__main__":
    main()
