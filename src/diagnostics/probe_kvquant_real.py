r"""int4 KV 量化 · 用**真实**的 K/V 量误差（不是随机张量）。

回答一个问题：`--kv int4` 在 needle 上 0/4，到底是**精度真损失**还是**接入有 bug**？

- 用**随机张量**测：`tests/test_kvquant.py`（已 12/12 通过，证明量化器本身没错）
- 用**真实张量**测（本脚本）：把真实 prefill 出来的 K/V 拿出来量化，看误差有多大、
  以及在**短上下文**下 greedy 输出会不会跟着崩

判据：
- 短上下文（~30 token）如果也胡言乱语 -> 是**接入 bug**，不是精度
- 短上下文正常、只有长上下文崩 -> 才是**误差累积**（用户文献第 2 条，也是我们最怕的那条）

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_kvquant_real.py --len 1024
"""

from __future__ import annotations

import argparse
import os
import subprocess
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

from chatfmt import render  # noqa: E402
from exp_needle import build  # noqa: E402
from nova.cache import StaticKVCache  # noqa: E402
from nova.decode import GraphDecoder  # noqa: E402
from nova.kvquant import GROUP_K, QuantRoundTripCache, roundtrip_int4  # noqa: E402
from s4_memory_demo import load_bundle  # noqa: E402


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def make_decoder(nova, kind: str, max_len: int) -> GraphDecoder:
    """`fp16` 就是原生解码器；其余按 VARIANTS 换成 int4 往返版。"""
    dec = GraphDecoder(nova, max_len=max_len)
    if kind == "fp16":
        return dec
    qk, qv, res = VARIANTS[kind]
    cfg = dec.cfg
    dec.cache = QuantRoundTripCache(
        num_slots=dec.text.num_cache_layers, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_len=max_len, quant_k=qk, quant_v=qv, residual=res,
    )
    return dec


def greedy(dec: GraphDecoder, ids: torch.Tensor, n: int) -> list[int]:
    """eager 解码 n 步（不进图，避免反复 capture 吃显存）。"""
    dec.prefill(ids)
    out = []
    for _ in range(n):
        out.append(int(dec.input_ids.item()))
        dec._body()
    return out


def time_run(dec: GraphDecoder, ids: torch.Tensor, reps: int) -> tuple[float, float]:
    """返回 `(prefill 秒, 每 token 毫秒)`。eager 路径，fp16 与 int4 用同一套测法才可比。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dec.prefill(ids)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(reps):
        dec._body()
    torch.cuda.synchronize()
    return prefill_s, (time.perf_counter() - t0) / reps * 1000.0


def error_stats(x: torch.Tensor, group: int) -> dict:
    """往返误差：相对 L2 / 相对本组 step / 相对通道量级。`x` 形状 `[1, heads, n, dim]`。"""
    x32 = x.float()
    xq = roundtrip_int4(x, group).float()
    gs = group or x32.shape[-1]
    g = x32.reshape(*x32.shape[:-1], x32.shape[-1] // gs, gs)
    err = (xq - x32).abs()
    step = ((g.amax(-1) - g.amin(-1)) / 15).clamp_min(1e-8)
    chan = x32.abs().amax(dim=(0, 1, 2))       # 每个通道的整体量级（跨 token/head 取 max）
    chan_err = err.amax(dim=(0, 1, 2))
    return {
        "rel_l2": ((xq - x32).norm() / x32.norm()).item(),
        "step_rel": (err.reshape_as(g).amax(-1) / step).mean().item(),
        "chan_rel": (chan_err / chan.clamp_min(1e-6)).mean().item(),
        "x_peaked": (chan.max() / chan.median()).item(),  # 通道离群程度（越大越难量化）
    }


VARIANTS = {
    "int4": (True, True, 0),
    "int4k": (True, False, 0),
    "int4v": (False, True, 0),
    "int4res": (True, True, 128),
    "nop": (False, False, 0),            # 完全不碰 simulate：只测包装层是否等价
    "copy": (True, True, 1 << 30),       # simulate 只做拷贝不量化：测 scratch 通路
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="int4 KV 在真实 K/V 上的误差")
    ap.add_argument("--len", type=int, default=1024, help="干草堆长度")
    ap.add_argument("--short", type=int, default=20, help="短上下文 greedy 步数")
    ap.add_argument("--kinds", nargs="+", default=["fp16", "int4", "int4k", "int4v", "int4res"])
    ap.add_argument("--reps", type=int, default=6, help="[C] 每个变体测几步 decode")
    ap.add_argument("--max-len", type=int, default=0, help="[C] 显式指定 cache 长度（0 = n+32）")
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    cfg = nova.model.config
    print(f"模型：Qwen3-VL-4B-Instruct · {nova.model.num_cache_layers} 层 · "
          f"{cfg.num_key_value_heads} KV 头 · head_dim {cfg.head_dim} · clocks.sm {clock_sm()}")

    # ---- A. 短上下文：fp16 vs int4 的 greedy 输出是否一致 -----------------------------
    # 这一条是"bug 还是精度"的分水岭：30 token 的上下文里误差累积几乎不存在。
    short_ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": "用一句话说明什么是潮汐。"}],
        tokenize=False, add_generation_prompt=True,
    ), return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")

    print(f"\n[A] 短上下文（prompt {short_ids.shape[1]} token，greedy {args.short} 步）"
          f"—— 「精度损失」还是「实现 bug」的分水岭")
    ref: list[int] | None = None
    for kind in args.kinds:
        dec = make_decoder(nova, kind, 256)
        t0 = time.perf_counter()
        out = greedy(dec, short_ids, args.short)
        dt = time.perf_counter() - t0
        hit = args.short if ref is None else sum(a == b for a, b in zip(ref, out))
        ref = out if ref is None else ref
        text = tok.decode(out, skip_special_tokens=True).strip().replace("\n", " ")[:58]
        flag = "" if hit >= args.short * 0.8 else "  <- 崩"
        print(f"    {kind:>7s} 与 fp16 一致 {hit:>2d}/{args.short} [{dt:>4.1f}s] {text!r}{flag}")
        if hit == 0 and kind != "fp16":
            print(f"            原始 token：{out[:10]}")
    print(f"    clocks.sm {clock_sm()}")

    # ---- B. 长上下文：真实 prefill 出来的 K/V 量化误差有多大 --------------------------
    n_rounds = max(8, int(round(args.len / 36)))
    msgs, _at = build(n_rounds)
    ids = tok(render(tok, msgs, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([ids[: args.len]], device="cuda")
    n = ids.shape[1]

    dec2 = GraphDecoder(nova, max_len=n + 32)
    t0 = time.perf_counter()
    dec2.prefill(ids)
    prefill_s = time.perf_counter() - t0
    print(f"\n[B] 长上下文（{n} token，prefill {prefill_s:.1f}s）：把真实 K/V 拿出来量化")
    print(f"    {'层':>4s} {'K rel_L2':>9s} {'K 误差/step':>11s} {'K 噪声/激活':>11s} {'K 通道离群':>10s}"
          f" {'V rel_L2':>9s} {'V 误差/step':>11s} {'V 噪声/激活':>11s} {'V 通道离群':>10s}")

    base_cache: StaticKVCache = dec2.cache
    ks, vs = [], []
    for slot in (0, 6, 17, 29, 35):
        k = base_cache.key_cache[slot][:, :, :n, :]
        v = base_cache.value_cache[slot][:, :, :n, :]
        ek, ev = error_stats(k, GROUP_K), error_stats(v, 0)
        print(f"    {slot:>4d} {ek['rel_l2']:>9.4f} {ek['step_rel']:>11.3f} {ek['chan_rel']:>11.3f}"
              f" {ek['x_peaked']:>10.1f} {ev['rel_l2']:>9.4f} {ev['step_rel']:>11.3f}"
              f" {ev['chan_rel']:>11.3f} {ev['x_peaked']:>10.1f}")
        ks.append(ek)
        vs.append(ev)

    kk = max(e["rel_l2"] for e in ks)
    vv = max(e["rel_l2"] for e in vs)
    print(f"\n    K 最大 rel_L2 {kk:.4f} · V 最大 rel_L2 {vv:.4f}")
    print(f"\n    误差/step 均值峰值：K {max(e['step_rel'] for e in ks):.3f} · "
          f"V {max(e['step_rel'] for e in vs):.3f}（0.5 = 理论中位，1.0 = 顶满半个 step）")
    print(f"    噪声/激活 峰值：K {max(e['chan_rel'] for e in ks):.3f} · V {max(e['chan_rel'] for e in vs):.3f}"
          f"（量化噪声相对该通道最大激活的比例，越小越无感）")
    print(f"    通道离群度 峰值：K {max(e['x_peaked'] for e in ks):.1f} · V {max(e['x_peaked'] for e in vs):.1f}"
          f"（最大通道 / 中位通道的激活比，>10 说明离群严重、量化难）")

    # ---- C. 开销拆分：prefill（算力受限）与 decode -----------------------------------
    # 文献上说 prefill 是算力受限的，dequant 在那里是纯亏；好处主要在 decode。
    # 同一轮里交替两遍取小值（AGENTS.md 第七节第 4 条：跨时间点的数字不能比）。
    if "fp16" in args.kinds and "int4" in args.kinds:
        print(f"\n[C] 开销拆分（{n} token · 各 {args.reps} 步 · 同轮交替取小值）")
        best: dict[str, tuple[float, float]] = {}
        ml = args.max_len or (n + 32)
        print(f"    cache max_len = {ml}")
        for _round in range(2):
            for kind in ("fp16", "int4"):
                d3 = make_decoder(nova, kind, ml)
                pf, dc = time_run(d3, ids, args.reps)
                cur = best.get(kind)
                best[kind] = (pf, dc) if cur is None else (min(cur[0], pf), min(cur[1], dc))
                del d3
        (pf16, dc16), (pf4, dc4) = best["fp16"], best["int4"]
        print(f"    prefill：fp16 {pf16:.2f}s -> int4 {pf4:.2f}s = {pf4 / pf16:.1f}x")
        print(f"    decode ：fp16 {dc16:.1f} ms/token -> int4 {dc4:.1f} ms/token = {dc4 / dc16:.1f}x")
        print("    注：模拟版每层每步重建整段 [0, used) 的 fp16 工作区，两条都随 n 线性涨；"
              "真正要省只能把 dequant 融进 attention kernel（见报告第六节）。")
    print(f"\n    clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
