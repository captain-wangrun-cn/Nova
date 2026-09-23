r"""信息过载下的注意力选择性（needle-in-haystack + 形近干扰项）。

问题：上下文很长、且里面塞了**多个长得差不多的密码**时，模型问哪一个能挑对哪一个？

设计：
- 干草堆 = 重复的无关闲聊（每轮带唯一编号），长度由 `--lens` 控制
- 埋 **4 条形近事实**（都是"某处的密码是 XX-XX-XX"），分别埋在不同深度
- 对 4 条**各问一次**，看它答出的号码是不是**被问的那一条**

判分：把答案与正确密码都去掉非数字后比子串。答成别的密码 = 挑错了（比答不出更值得警惕）。

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\exp_needle.py --lens 2048 4096 8192 16384

显存注意：fp16 KV 是 144 KiB/token（单通路），16K token 约占 2.4 GB。

⚠️ **不要每题都 `capture()`**：`GraphDecoder.capture` 每次新建一张 CUDA Graph 并分配新内存池，
位置不同就得重捕。实测重捕 9 次后显存 7905/8188 MiB 崩溃。本脚本改用 eager `_body()`。
"""

from __future__ import annotations

import argparse
import os
import re
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

from chatfmt import EOS_ID, EOS_ID_ALT, render  # noqa: E402
from nova.decode import GraphDecoder  # noqa: E402
from s4_memory_demo import FILLER, load_bundle  # noqa: E402

STOP = {EOS_ID, EOS_ID_ALT}


def clock_sm() -> str:
    """速度数字必须带 clocks.sm（[AGENTS.md](../../AGENTS.md) 第七节第 4 条）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"

# 4 条形近事实：格式完全一样，只有"地点"和"号码"不同 —— 这才是"信息过多"的干扰源
NEEDLES = [
    ("健身房储物柜", "73-91-26", 0.10),
    ("办公室门禁", "52-14-88", 0.35),
    ("家里保险箱", "19-73-40", 0.62),
    ("车后备箱", "26-58-31", 0.88),
]

# E5：干扰项数扫描用的地点词表（8 × 8 = 64 种组合，足够铺到 64 条）
_LOC_A = ["健身房", "办公室", "家里", "车里", "学校", "医院", "酒店", "公司"]
_LOC_B = ["储物柜", "门禁", "保险箱", "后备箱", "抽屉", "邮箱", "柜子", "工位"]


def make_needles(n: int) -> list[tuple[str, str, float]]:
    """生成 `n` 条**同形**事实（地点 + 号码），深度均匀铺开。

    `n == 4` 时返回原来那 4 条（保持与既有报告可比）；更多条时按词表组合出唯一地点，
    号码用 `(11+i)-(41+i)-(71+i)` —— 互不为子串，保证"挑错"能被识别出来。
    """
    n = int(n)
    if n == 4:
        return list(NEEDLES)
    out = []
    for i in range(n):
        suffix = f"{i:02d}" if n > 64 else ""
        name = _LOC_A[i % 8] + _LOC_B[(i // 8) % 8] + suffix
        out.append((name, f"{11 + i}-{41 + i}-{71 + i}", (i + 0.5) / n))
    return out


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def make_decoder(nova, kv: str, max_len: int) -> GraphDecoder:
    """按 `kv` 造解码器；非 fp16 时把 cache 换成 int4 往返版（`QuantRoundTripCache`）。

    ⚠️ 模拟版**不省显存**，它测的是精度与 dequant 开销；显存收益只按公式记账。
    """
    dec = GraphDecoder(nova, max_len=max_len)
    if kv == "fp16":
        return dec
    from nova.kvquant import QuantRoundTripCache

    cfg = dec.cfg
    # 8 位那几档（E3）：kind 直接传给 roundtrip；int4 那几档仍用 quant_k/quant_v 开关
    kind = kv if kv in ("int8", "int8t", "int8t64", "fp8", "fp8t") else "int4"
    dec.cache = QuantRoundTripCache(
        num_slots=dec.text.num_cache_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_len=max_len,
        quant_k=kv != "int4v",
        quant_v=kv != "int4k",
        residual=128 if kv == "int4res" else 0,
        kind=kind,
        base=dec.cache,  # 复用解码器自己那份 cache，否则同时存在两份（18432 槽位下直接 OOM）
    )
    return dec


def build(n_rounds: int, needles=None):
    """干草堆 + 按深度插入事实；返回 `(msgs, {事实名: 在第几轮})`。

    `needles` 默认用那 4 条；给 `make_needles(16/64)` 就能做 E5 的干扰项扫描。
    """
    needles = list(NEEDLES if needles is None else needles)
    spots = {int(round(n_rounds * d)): (name, code) for name, code, d in needles}
    msgs, at = [], {}
    for i in range(n_rounds):
        if i in spots:
            name, code = spots[i]
            at[name] = len(msgs)
            msgs.append({"role": "user", "content": f"顺便记一下，我的{name}密码是 {code}。"})
            msgs.append({"role": "assistant", "content": "好的，我记下了。"})
        u, a = FILLER[i % len(FILLER)]
        msgs.append({"role": "user", "content": f"{u}（第 {i + 2} 轮）"})
        msgs.append({"role": "assistant", "content": a})
    return msgs, at


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="信息过载下的注意力选择性")
    ap.add_argument("--lens", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--max-len", type=int, default=18432)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument(
        "--kv",
        nargs="+",
        default=["fp16"],
        choices=("fp16", "int4", "int4res", "int4k", "int4v", "int8", "int8t", "int8t64", "fp8", "fp8t"),
        help="K/V 存储精度，**可给多个**（同一轮里轮着跑，跨时间点的速度不可比）："
             "fp16 基线；int4=K与V都压；int4res=再加最近 128 位置的 fp16 残留窗；int4k/int4v=只压一个；"
             "int8=按通道分组的 int8；int8t/int8t64=按 token 维分组的 int8（E3）；fp8/fp8t=float8_e4m3fn（per-token / per-tensor）",
    )
    ap.add_argument("--n-interfere", type=int, nargs="+", default=[4],
                    help="干扰事实条数（E5 扫描：4 → 16 → 64）；可给多个")
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    from nova.kvquant import bytes_per_token

    cfg = nova.model.config
    cost = bytes_per_token(nova.model.num_cache_layers, cfg.num_key_value_heads, cfg.head_dim)
    cost8 = bytes_per_token(nova.model.num_cache_layers, cfg.num_key_value_heads, cfg.head_dim, bits=8)
    costf8 = bytes_per_token(nova.model.num_cache_layers, cfg.num_key_value_heads, cfg.head_dim, bits="fp8")
    print(f"模型：Qwen3-VL-4B-Instruct · 单通路 · max_len {args.max_len}")
    print(f"KV 记账：int4 {cost['int4_kv'] / 1024:.0f} KiB/token vs fp16 {cost['fp16_kv'] / 1024:.0f} KiB/token"
          f" = {cost['ratio']:.2f}x；int8 {cost8['int8_kv'] / 1024:.0f} = {cost8['ratio']:.2f}x；"
          f"fp8 {costf8['int8_kv'] / 1024:.0f} = {costf8['ratio']:.2f}x")
    print(f"（模拟版不省显存，测的是精度与 dequant 开销）")
    print(f"干草堆：重复闲聊（每轮带唯一编号）+ 形近事实，埋在不同深度\n")

    scores: dict[tuple[str, int, int], tuple[int, int, int]] = {}
    for kv in args.kv:
        dec = make_decoder(nova, kv, args.max_len)
        print(f"\n【{kv}】 干扰项 {'/'.join(str(n) for n in args.n_interfere)} 条")
        print(f"{'长度':>8s} {'实际token':>9s}  {'干扰':>4s}  答对 / 挑错 / 没答"
              f"{'':>16s}{'prefill':>9s} {'峰值':>9s} {'clocks.sm':>10s}")
        for n_int in args.n_interfere:
            needles = make_needles(n_int)
            for target in args.lens:
                n_rounds = max(8, int(round(target / 36)))
                msgs, _at = build(n_rounds, needles)
                hist_text = render(tok, msgs, add_generation_prompt=False)
                hist_ids = tok(hist_text, add_special_tokens=False)["input_ids"]
                if len(hist_ids) + 64 > args.max_len:
                    print(f"{target:>8d}  —— 超过 max_len，跳过（实际 {len(hist_ids)}）")
                    continue
                hist_t = torch.tensor([hist_ids], device="cuda")

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                dec.prefill(hist_t)
                torch.cuda.synchronize()
                prefill_s = time.perf_counter() - t0
                h = len(hist_ids)
                clk = clock_sm()

                right = wrong = none = 0
                detail = []
                for name, code, _d in needles:
                    qtext = f"我的{name}密码是多少？只回答那串号码。"
                    full_text = render(tok, msgs + [{"role": "user", "content": qtext}], add_generation_prompt=True)
                    q_ids = tok(full_text, add_special_tokens=False)["input_ids"][h:]
                    q_t = torch.tensor([q_ids], device="cuda")
                    dec.prefill(q_t, offset=h, reset=False)
                    out = []
                    for _ in range(args.max_new):
                        t = int(dec.input_ids.item())
                        if t in STOP:
                            break
                        out.append(t)
                        dec._body()
                    text = tok.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
                    got = digits(text)
                    if digits(code) in got:
                        right += 1
                        detail.append("O")
                    elif any(digits(c) in got for _n, c, _dd in needles if c != code):
                        wrong += 1
                        detail.append("!")
                    else:
                        none += 1
                        detail.append("X")
                    dec.cache.pos.fill_(h)
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                scores[(kv, target, n_int)] = (right, wrong, none)
                print(f"{target:>8d} {h:>9d}  {n_int:>4d}  {right}/{len(needles)} 挑错 {wrong} 没答 {none}"
                      f"   [prefill {prefill_s:>6.1f}s · 峰值 {peak:>5.2f} GiB · clocks.sm {clk}]", flush=True)
            del msgs
            torch.cuda.empty_cache()
        del dec
        torch.cuda.empty_cache()

    print("\nO=答对  !=挑成别的密码（最危险）  X=没答出")
    print("注：4 条事实格式完全相同，只有地点与号码不同。")
    if len(args.kv) > 1 and all((("fp16", t) in scores) for t in args.lens):
        print("\n与 fp16 基线对比（同一轮，可横向比）：")
        for kv in args.kv:
            if kv == "fp16":
                continue
            drops = [f"{t}:{scores[(kv, t)][0] - scores[('fp16', t)][0]:+d}" for t in args.lens]
            print(f"    {kv:>7s} 答对数变化 {' '.join(drops)}")


if __name__ == "__main__":
    main()
