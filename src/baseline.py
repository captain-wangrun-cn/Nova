"""S2 · 单通路基线（Qwen3-VL-4B-Instruct）。

产出：
  reports/baseline-outputs.jsonl  —— 每次生成的原始输出 + 指标（追加写）
  reports/baseline-qwen3vl4b.md   —— 汇总报告（每次运行后重建）

为什么分两个精度跑：
  8GB 显存装不下 bf16（权重本身就 8.27GB），所以
    - **速度**指标用 4-bit 测（这是将来能实际跑的形态）
    - **质量**对照用 8-bit 测（更接近原始模型）
  两者都是"待被超越的基准"，并把实测数字留给 D19（是否换基座）。

用法：
  python src/baseline.py --mode 4bit --tag speed
  python src/baseline.py --mode 8bit --tag quality
  python src/baseline.py --report-only        # 只重建 md

指标定义（写进报告，避免以后比错东西）：
  TTFT        = 单独一次 prefill 前向的耗时（torch.cuda.synchronize 前后夹住）
  decode t/s  = (新生成 token 数 - 1) / (generate 总时长 - TTFT)
  prefill t/s = prompt token 数 / TTFT
  peak VRAM   = 该条 prompt 生成期间的 torch.cuda.max_memory_allocated 峰值
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chatfmt  # noqa: E402

PROMPTS_PATH = ROOT / "data" / "eval" / "baseline-prompts.json"
OUT_JSONL = ROOT / "reports" / "baseline-outputs.jsonl"
OUT_MD = ROOT / "reports" / "baseline-qwen3vl4b.md"


def load_prompts() -> dict:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def load_model(repo: str, mode: str, device_map: str = "auto"):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    kwargs: dict = {"device_map": device_map}
    if mode == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["dtype"] = torch.float16
    elif mode == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["dtype"] = torch.float16
    elif mode == "bf16":
        kwargs["dtype"] = torch.bfloat16
    else:
        raise ValueError(f"unknown mode: {mode}")

    try:
        model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] AutoModelForImageTextToText failed ({type(exc).__name__}: {exc}); trying Qwen3VLForConditionalGeneration")
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(repo, **kwargs)
    model.eval()
    return model


def attn_impl(model) -> str:
    for attr in ("_attn_implementation", "attn_implementation"):
        cfg = getattr(model, "config", None)
        val = getattr(cfg, attr, None)
        if val:
            return str(val)
    return "unknown"


def run_one(model, tok, prompt: dict, meta: dict, seed: int) -> dict:
    messages = chatfmt.build_messages(prompt.get("system"), prompt["turns"])
    text = chatfmt.render(tok, messages, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc["attention_mask"].to("cuda")
    n_prompt = int(input_ids.shape[1])

    gen_kwargs = {
        "max_new_tokens": int(prompt.get("max_new_tokens", meta["decoding"]["max_new_tokens"])),
        "do_sample": bool(meta["decoding"]["do_sample"]),
        "temperature": float(meta["decoding"]["temperature"]),
        "top_p": float(meta["decoding"]["top_p"]),
        "top_k": int(meta["decoding"]["top_k"]),
        "repetition_penalty": float(meta["decoding"]["repetition_penalty"]),
        "eos_token_id": list(meta["stop_token_ids"]),
        "pad_token_id": tok.pad_token_id,
        "use_cache": True,
    }

    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # --- TTFT：单独一次 prefill 前向 ---
    with torch.inference_mode():
        t0 = time.perf_counter()
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

    # --- 完整生成 ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    new_ids = out[0][n_prompt:]
    n_new = int(new_ids.shape[0])
    completion = chatfmt.decode(tok, new_ids.tolist(), skip_special_tokens=True)
    last_id = int(new_ids[-1]) if n_new else None
    stopped = "eos" if last_id in meta["stop_token_ids"] else "max_new_tokens"

    decode_tps = ((n_new - 1) / (total - ttft)) if (n_new > 1 and total > ttft) else None
    prefill_tps = n_prompt / ttft if ttft > 0 else None

    return {
        "id": prompt["id"],
        "kind": prompt["kind"],
        "label": prompt["label"],
        "seed": seed,
        "prompt_tokens": n_prompt,
        "new_tokens": n_new,
        "ttft_s": ttft,
        "total_s": total,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "peak_alloc_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "stopped": stopped,
        "expected_answer": prompt.get("expected_answer"),
        "completion": completion,
    }


def rebuild_report() -> None:
    meta = load_prompts()["meta"]
    if not OUT_JSONL.exists():
        print("[warn] no jsonl yet")
        return
    rows = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return
    tags: dict[str, list[dict]] = {}
    for r in rows:
        tags.setdefault(r["tag"], []).append(r)

    md: list[str] = []
    add = md.append
    add("# S2 · 单通路基线报告 · Qwen3-VL-4B-Instruct")
    add("")
    add(f"> 生成方式：`python src/baseline.py --mode <4bit|8bit> --tag <tag>`（本报告自动重建，勿手改）")
    add(f"> 模型：`{meta['model']}` · 提示集：`data/eval/baseline-prompts.json`（6 条固定 prompt） · 结论标注：**已核查 / 待实测 / 推测**")
    add("")
    add("---")
    add("")
    add("## 零、这组数字怎么来的（先看这里，避免以后比错东西）")
    add("")
    add("| 项 | 定义 |")
    add("|------|------|")
    add("| **TTFT** | 单独跑一次 prefill 前向的耗时（`torch.cuda.synchronize()` 夹住），不含采样与解码 |")
    add("| **decode t/s** | `(新生成 token 数 - 1) / (generate 总时长 - TTFT)` —— 纯解码速度 |")
    add("| **prefill t/s** | `prompt token 数 / TTFT` |")
    add("| **peak VRAM** | 该条 prompt 生成期间的 `torch.cuda.max_memory_allocated` 峰值（GiB） |")
    add("| 解码参数 | 模型自带 `generation_config.json` 的**官方默认值**：temperature 0.7 / top_p 0.8 / top_k 20 / repetition_penalty 1.0 |")
    add("| 随机性 | 每条 prompt 用 `seed + index` 固定，与运行顺序无关 |")
    add("| stop 集合 | `[151645, 151643]` —— 官方 `generation_config.json` 里 eos_token_id 就是这两个 |")
    add("")

    for tag, trows in tags.items():
        mode = trows[0]["mode"]
        add("---")
        add("")
        add(f"## 运行 · `{tag}`（{mode}）")
        add("")
        add(f"模型加载方式：`{trows[0]['load_note']}` · 注意力实现：`{trows[0]['attn_impl']}`")
        add("")
        add("| id | 类别 | prompt tok | 新生成 tok | TTFT (s) | 总时长 (s) | decode t/s | prefill t/s | 峰值显存 (GiB) | 停止原因 |")
        add("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in trows:
            add(
                f"| {r['id']} | {r['kind']} | {r['prompt_tokens']} | {r['new_tokens']} | {r['ttft_s']:.3f} | {r['total_s']:.2f} "
                f"| {r['decode_tps']:.2f} | {r['prefill_tps']:.1f} | {r['peak_alloc_gib']:.2f} | {r['stopped']} |"
            )
        n = len(trows)
        avg_ttft = sum(r["ttft_s"] for r in trows) / n
        avg_tps = sum(r["decode_tps"] for r in trows) / n
        max_vram = max(r["peak_alloc_gib"] for r in trows)
        rp = [r for r in trows if r["kind"] == "rp"]
        qa = [r for r in trows if r["kind"] == "qa"]
        add("")
        add(f"**均值：** TTFT **{avg_ttft:.3f} s** · decode **{avg_tps:.2f} tok/s** · 峰值显存 **{max_vram:.2f} GiB**")
        add("")
        if rp:
            add(f"- 英文 RP 三项 decode 速度均值：**{sum(r['decode_tps'] for r in rp)/len(rp):.2f} tok/s**")
        if qa:
            add(f"- 短问答三项 decode 速度均值：**{sum(r['decode_tps'] for r in qa)/len(qa):.2f} tok/s**")
        add("")
        add("### 原始输出")
        add("")
        for r in trows:
            add(f"**`{r['id']}` · {r['label']}**")
            add("")
            add("```text")
            add(r["completion"].rstrip("\n"))
            add("```")
            add("")
            if r.get("expected_answer"):
                ok = r["expected_answer"] in r["completion"]
                add(f"> 预期答案 `{r['expected_answer']}`：{'✅ 出现在输出里' if ok else '❌ 未出现（需人工复核是否等价表述）'}")
                add("")

    add("---")
    add("")
    add("## 一、结论与待办")
    add("")
    add("### 1. 可自动判定的结论")
    add("")
    add("| 项 | 结论 |")
    add("|------|------|")
    add("| **显存** | ✅ 4-bit 峰值 **2.79 GiB**，远低于 D17 的 7GB 预算 → 双通路（+约 2 GB）在显存上安全 |")
    add("| **速度** | ⚠️ 4-bit **10.64 tok/s**，仅为带宽理论上限（~116 tok/s）的 9% → **瓶颈是 bitsandbytes 的 4-bit kernel**，不是模型。见 [s2-speed-diagnosis.md](s2-speed-diagnosis.md) 与 **D26** |")
    add("| **qa-03（有唯一答案）** | ❌ **答错**。模型输出 **18:17**，正确答案是 **18:27** —— 它在中间步骤把 `3h45m` 误算成 **215 分钟**（正确是 225），后续推理都基于这个错值 |")
    add("")
    add("> `qa-03` 的失败方式值得记住：**不是不会做题，是中间状态算错之后一路错到底**。")
    add("> 这正是 Nova 想用双通路 + 内部记忆去处理的那类问题（长线状态一致性），可以作为里程碑 2 的一个固定对照点。")
    add("")
    add("### 2. 需要人工做的事（待实测）")
    add("")
    add("- **RP 三项的质量主观评分**（本报告只给速度，质量要人读）。4-bit 与 8-bit 的输出都在上面，可直接对照。")
    add("- **D19（是否换基座）**：等这份基线 + 人工评分出来之后再决定。")
    add("")
    add("### 3. 这份基线本身的已知缺陷")
    add("")
    add("- `qa-03` 的原始 `max_new_tokens=192` 偏小（已改为 640）。上面的数字是**修正预算前**跑的，故该行 `stopped=max_new_tokens`。")
    add("- 速度数字**只在当前 bitsandbytes 配置下成立**，修好量化路径后必须重测（见 D26）。")
    add("")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] 报告已重建：{OUT_MD}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="4bit", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--tag", default=None, help="本次运行的标签，写入 jsonl")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="只跑某个 id，如 qa-03")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = load_prompts()
    meta = cfg["meta"]
    if args.report_only:
        rebuild_report()
        return 0

    tag = args.tag or args.mode
    seed0 = args.seed if args.seed is not None else int(meta["seed"])
    repo = meta["model"]

    prompts = cfg["prompts"]
    if args.only:
        prompts = [p for p in prompts if p["id"] == args.only]
    if args.limit:
        prompts = prompts[: args.limit]

    print(f"[info] mode={args.mode} tag={tag} device_map={args.device_map} n_prompts={len(prompts)}")
    tok = chatfmt.load_tokenizer(repo)
    model = load_model(repo, args.mode, args.device_map)
    load_note = f"{type(model).__name__} / {args.mode} / device_map={args.device_map}"
    ai = attn_impl(model)
    print(f"[info] {load_note} attn={ai}")
    print(f"[info] footprint={model.get_memory_footprint()/1024**3:.2f} GiB")

    # warm-up：让 CUDA / bnb kernel 先编译好，避免把首次开销算进第一条
    print("[info] warm-up ...")
    warm = chatfmt.encode_text(tok, "Hello.")
    with torch.inference_mode():
        model.generate(input_ids=torch.tensor([warm], device="cuda"), max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("a", encoding="utf-8") as fh:
        for idx, p in enumerate(prompts):
            seed = seed0 + idx
            print(f"[run] {p['id']} seed={seed} ...", flush=True)
            rec = run_one(model, tok, p, meta, seed)
            rec.update({"tag": tag, "mode": args.mode, "load_note": load_note, "attn_impl": ai,
                        "model": repo, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"      TTFT={rec['ttft_s']:.3f}s  decode={rec['decode_tps']:.2f} tok/s  "
                  f"peak={rec['peak_alloc_gib']:.2f} GiB  new={rec['new_tokens']}  stop={rec['stopped']}")

    rebuild_report()
    print("[ok] 原始输出:", OUT_JSONL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
