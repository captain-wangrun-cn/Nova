"""诊断：4-bit bnb 基座 + Nova 双通路，反向传播能否走通。

只跑一次 forward/backward，不建优化器、不写权重。
用途：D46 的 10 步训练 spike 之前，先隔离“梯度到底能不能穿过 bnb 4-bit 基座”。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("HF_HOME", str(REPO / ".hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMP", str(REPO / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])

import chatfmt  # noqa: E402
from nova.loader import load_nova  # noqa: E402


def clock_sm() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip().splitlines()[0]
        return f"{out} MHz"
    except Exception as exc:
        return f"? ({exc.__class__.__name__})"


def encode_sft(tok, messages: list[dict], seq_len: int) -> tuple[list[int], list[int]]:
    full_text = chatfmt.render(tok, messages, add_generation_prompt=False)
    prefix_text = chatfmt.render(tok, messages[:-1], add_generation_prompt=True)
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("assistant 前缀对不上；不能安全构造 labels")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
    if seq_len:
        full_ids = full_ids[:seq_len]
        labels = labels[:seq_len]
    return full_ids, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova 4-bit 反向传播诊断")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0, help="0 = 不截断")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--cross-mode", choices=("on", "predictive"), default="predictive")
    args = parser.parse_args()

    data_path = REPO / "data/s5-samples/gemini-3.1-flash-lite.jsonl"
    samples = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample = samples[args.sample_index]
    tok = chatfmt.load_tokenizer()
    input_ids, labels = encode_sft(tok, sample["messages"], args.seq_len)
    print(
        f"sample={sample['id']} seq={len(input_ids)} "
        f"cross_mode={args.cross_mode} checkpointing={args.checkpointing} "
        f"clocks.sm={clock_sm()}"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    nova, hf_model, _ = load_nova(quant="4bit", norm_impl="exact")
    visual = getattr(getattr(hf_model, "model", None), "visual", None)
    if visual is not None:
        try:
            visual.to("cpu")
            torch.cuda.empty_cache()
            print("visual_tower=cpu")
        except Exception as exc:  # pragma: no cover - 仅诊断
            print(f"visual_tower=keep ({exc.__class__.__name__}: {exc})")
    load_s = time.perf_counter() - t0

    nova.train()
    for p in nova.parameters():
        p.requires_grad_(False)
    for p in nova.model.cross_blocks.parameters():
        p.requires_grad_(True)
    trainable = [p for p in nova.parameters() if p.requires_grad]
    print(
        f"trainable_params={sum(p.numel() for p in trainable)} "
        f"load_s={load_s:.1f} alloc={torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
    )

    ids = torch.tensor([input_ids], device="cuda")
    targets = torch.tensor([labels], device="cuda")
    t1 = time.perf_counter()
    logits = nova(
        input_ids=ids,
        cross_mode=args.cross_mode,
        gradient_checkpointing=args.checkpointing,
    )
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    fwd_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    loss.backward()
    bwd_s = time.perf_counter() - t2

    grads = [p.grad for p in trainable]
    grad_none = sum(g is None for g in grads)
    grad_norm = torch.sqrt(
        sum((g.detach().float().pow(2).sum() for g in grads if g is not None))
    ).item()
    base_grads = sum(p.grad is not None for p in nova.parameters() if not p.requires_grad)
    print(
        f"loss={loss.item():.6f} grad_none={grad_none}/{len(grads)} "
        f"grad_norm={grad_norm:.6f} base_grads={base_grads}"
    )
    print(f"fwd_s={fwd_s:.2f} bwd_s={bwd_s:.2f}")
    print(
        f"peak_alloc={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB "
        f"peak_reserved={torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB"
    )
    print(f"clocks.sm_end={clock_sm()}")


if __name__ == "__main__":
    main()
