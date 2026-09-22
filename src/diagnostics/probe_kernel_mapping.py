r"""`repeat_kv` 是"映射写错"还是"内核精度差异"？顺便查清 (b) 用的是哪个内核。

三件事：
1. **强制两边都走 math 后端**比一次：若一致 -> head 映射正确，之前 1.0 的差是内核精度。
2. 与**手写 fp32 参考实现**比：确认 repeat_kv 的 GQA 分组语义正确。
3. profiler 打出 (b) 实际用的 attention 内核名。

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_kernel_mapping.py
"""

from __future__ import annotations

import os
import sys
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

from nova.layers import LeanAttention, repeat_kv  # noqa: E402
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

    from torch.nn.attention import SDPBackend, sdpa_kernel

    nova, tok = load_bundle(1, "triton")
    ids = build_ids(tok, 1024)

    print("=== 1. 两边都强制 math 后端（隔离内核差异）===")
    outs = {}
    for flag in (True, False):
        set_gqa(nova, flag)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            outs[flag] = nova.model(input_ids=ids, cross_mode="off").float()
    d = (outs[True] - outs[False]).abs()
    same = (outs[True].argmax(-1) == outs[False].argmax(-1)).all().item()
    print(f"  max|diff| = {d.max().item():.3e}   argmax 全同 = {same}")
    print("  -> 若 ~0：repeat_kv 的 head 映射正确，默认路径的差异纯粹来自内核精度")
    del outs, d
    torch.cuda.empty_cache()

    print("\n=== 2. 与手写 fp32 参考实现比（一层，短序列）===")
    layer = nova.model.prefix_layers[0]
    attn = layer.self_attn
    x = torch.randn(1, 24, nova.model.config.hidden_size, device="cuda", dtype=torch.float16)
    pos = torch.arange(24, device="cuda")
    with torch.inference_mode():
        # 复刻 attention 前向到 SDPA 之前
        hidden_shape = (*x.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(x).view(hidden_shape).transpose(1, 2)
        # 参考：自己展平 KV（repeat_interleave 按 GQA 分组），再手算 fp32 因果注意力
        kg = repeat_kv(k, attn.num_key_value_groups)
        vg = repeat_kv(v, attn.num_key_value_groups)
        qf, kf, vf = q.float(), kg.float(), vg.float()
        scores = (qf @ kf.transpose(-1, -2)) * attn.scaling
        causal = torch.ones(24, 24, device="cuda", dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ref = probs @ vf
        # SDPA：GQA 路径
        got_gqa = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=attn.scaling, is_causal=True, enable_gqa=True).float()
        # SDPA：展平路径
        got_flat = torch.nn.functional.scaled_dot_product_attention(
            q, kg, vg, scale=attn.scaling, is_causal=True).float()
    for name, got in (("GQA 留 SDPA", got_gqa), ("展平 repeat_kv", got_flat)):
        dd = (got - ref).abs()
        print(f"  {name:16s} vs 手写 fp32: max|diff| = {dd.max().item():.3e}  "
              f"相对 {dd.max().item() / ref.abs().max().item():.2e}")

    print("\n=== 3. 展平路径实际用的内核（profiler）===")
    set_gqa(nova, False)
    with torch.inference_mode():
        nova.model(input_ids=ids, cross_mode="off")  # warmup
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            nova.model(input_ids=ids, cross_mode="off")
            torch.cuda.synchronize()
    keys = ("flash", "fmha", "attention", "cutlass", "mem_eff", "efficient", "sdpa", "softmax")
    hits = {}
    for evt in prof.key_averages():
        n = evt.key.lower()
        if any(k in n for k in keys):
            hits[evt.key] = max(hits.get(evt.key, 0), evt.count)
    for k, v in sorted(hits.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v:>5d} x  {k[:100]}")
    if not hits:
        print("  （没匹配到 attention 关键字，看下面前 15 个热点）")
        for evt in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:15]:
            print(f"  {evt.self_device_time_total / 1000:9.2f} ms  {evt.key[:90]}")


if __name__ == "__main__":
    main()
