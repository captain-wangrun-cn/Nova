"""S5 训练 spike：用 5 条 API 教师英文样本，在本地 4060 上跑 10 步。

目的不是质量，是验证：
- 自写 Nova 双通路前向能被反向传播（bnb 4-bit 基座冻结）；
- 只训 `cross_blocks`（交叉注意力 + 门控 + 预测器）；
- 峰值显存 / step 时间 / `clocks.sm`。

用法：
    & .\\.venv\\Scripts\\python.exe src\\s5_train_spike.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("HF_HOME", str(REPO / ".hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMP", str(REPO / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])

import chatfmt  # noqa: E402
from nova.loader import load_nova  # noqa: E402

DEFAULT_DATA = REPO / "data/s5-samples/gemini-3.1-flash-lite.jsonl"
DEFAULT_RESULTS = REPO / "reports/s5-train-spike-results.json"
DEFAULT_CKPT = REPO / ".tmp/s5-train-spike/cross_blocks.pt"


def clock_sm() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def encode_sft(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """整段对话编码；system + user 位置 label = -100，只训 assistant 段。"""
    full_text = chatfmt.render(tok, messages, add_generation_prompt=False)
    prefix_text = chatfmt.render(tok, messages[:-1], add_generation_prompt=True)
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("assistant 前缀对不上；不能安全构造 labels")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
    if max_len:
        full_ids = full_ids[:max_len]
        labels = labels[:max_len]
    return full_ids, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova 本地 10 步训练 spike")
    parser.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=0, help="0 = 不截断")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optim", choices=("paged_adam8bit", "adam8bit"), default="paged_adam8bit")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ckpt", type=pathlib.Path, default=DEFAULT_CKPT)
    args = parser.parse_args()

    samples = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise SystemExit(f"没有样本：{args.data}")
    tok = chatfmt.load_tokenizer()
    encoded = [encode_sft(tok, s["messages"], args.max_len) for s in samples]
    print("samples: " + ", ".join(
        f"{s['id']}={len(ids)}" for s, (ids, _) in zip(samples, encoded)
    ))

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
    print(
        f"load_s={time.perf_counter() - t0:.1f} "
        f"alloc={torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
    )

    nova.train()
    for p in nova.parameters():
        p.requires_grad_(False)
    for p in nova.model.cross_blocks.parameters():
        p.requires_grad_(True)
    trainable = [p for p in nova.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"trainable_params={n_trainable}")

    import bitsandbytes as bnb

    if args.optim == "paged_adam8bit":
        optimizer = bnb.optim.PagedAdam8bit(trainable, lr=args.lr)
    else:
        optimizer = bnb.optim.Adam8bit(trainable, lr=args.lr)

    def run_step(step_idx: int) -> dict:
        input_ids, labels = encoded[step_idx % len(encoded)]
        ids = torch.tensor([input_ids], device="cuda")
        targets = torch.tensor([labels], device="cuda")
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = nova(
            input_ids=ids,
            cross_mode="predictive",
            gradient_checkpointing=True,
        )
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        step_s = time.perf_counter() - t0
        clock = clock_sm()
        base_grads = sum(p.grad is not None for p in nova.parameters() if not p.requires_grad)
        return {
            "step": step_idx,
            "sample": samples[step_idx % len(samples)]["id"],
            "seq_len": len(input_ids),
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "step_s": step_s,
            "clock_sm": clock,
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "grad_none": sum(p.grad is None for p in trainable),
            "base_grads": base_grads,
        }

    warmup_records: list[dict] = []
    for i in range(args.warmup):
        rec = run_step(i)
        warmup_records.append(rec)
        print(
            f"[warmup {i + 1}/{args.warmup}] {rec['sample']} seq={rec['seq_len']} "
            f"loss={rec['loss']:.4f} step={rec['step_s']:.2f}s clock={rec['clock_sm']}MHz"
        )

    records: list[dict] = []
    for i in range(args.steps):
        rec = run_step(args.warmup + i)
        records.append(rec)
        print(
            f"[step {i + 1}/{args.steps}] {rec['sample']} seq={rec['seq_len']} "
            f"loss={rec['loss']:.4f} grad={rec['grad_norm']:.3f} "
            f"step={rec['step_s']:.2f}s clock={rec['clock_sm']}MHz "
            f"peak={rec['peak_alloc_gib']:.2f}GiB"
        )

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cross_blocks": nova.model.cross_blocks.state_dict(),
            "step": args.steps,
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "config": {
                "cross_mode": "predictive",
                "gradient_checkpointing": True,
                "lr": args.lr,
            },
        },
        args.ckpt,
    )
    print(f"checkpoint={args.ckpt}")

    step_times = [r["step_s"] for r in records]
    clocks = [r["clock_sm"] for r in records if r["clock_sm"] is not None]
    result = {
        "created": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "samples": [s["id"] for s in samples],
        "config": {
            "steps": args.steps,
            "warmup": args.warmup,
            "lr": args.lr,
            "max_len": args.max_len,
            "grad_clip": args.grad_clip,
            "optim": args.optim,
            "cross_mode": "predictive",
            "gradient_checkpointing": True,
            "quant": "4bit",
            "norm_impl": "exact",
        },
        "trainable_params": n_trainable,
        "warmup": warmup_records,
        "records": records,
        "summary": {
            "loss_first": records[0]["loss"],
            "loss_last": records[-1]["loss"],
            "step_s_median": statistics.median(step_times),
            "step_s_min": min(step_times),
            "step_s_max": max(step_times),
            "clock_sm_min": min(clocks) if clocks else None,
            "clock_sm_max": max(clocks) if clocks else None,
            "peak_alloc_gib": max(r["peak_alloc_gib"] for r in records),
            "peak_reserved_gib": max(r["peak_reserved_gib"] for r in records),
            "grad_none_last": records[-1]["grad_none"],
            "base_grads_last": records[-1]["base_grads"],
        },
        "checkpoint": str(args.ckpt),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={args.out}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
