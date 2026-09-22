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
    dec.cache = QuantRoundTripCache(
        num_slots=dec.text.num_cache_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_len=max_len,
        quant_k=kv != "int4v",
        quant_v=kv != "int4k",
        residual=128 if kv == "int4res" else 0,
    )
    return dec


def build(n_rounds: int):
    """干草堆 + 按深度插入 4 条事实；返回 (msgs, 每条事实在第几轮)。"""
    spots = {int(round(n_rounds * d)): (name, code) for name, code, d in NEEDLES}
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
        choices=("fp16", "int4", "int4res", "int4k", "int4v"),
        help="K/V 存储精度，**可给多个**（同一轮里轮着跑，跨时间点的速度不可比）："
             "fp16 基线；int4=K与V都压；int4res=再加最近 128 位置的 fp16 残留窗；int4k/int4v=只压一个",
    )
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    from nova.kvquant import bytes_per_token

    cfg = nova.model.config
    cost = bytes_per_token(nova.model.num_cache_layers, cfg.num_key_value_heads, cfg.head_dim)
    print(f"模型：Qwen3-VL-4B-Instruct · 单通路 · max_len {args.max_len}")
    print(f"KV 记账：int4 {cost['int4_kv'] / 1024:.0f} KiB/token vs fp16 {cost['fp16_kv'] / 1024:.0f} KiB/token"
          f" = {cost['ratio']:.2f}x（模拟版不省显存，测的是精度与 dequant 开销）")
    print(f"干草堆：重复闲聊（每轮带唯一编号）+ 4 条形近事实，埋在不同深度\n")

    scores: dict[tuple[str, int], tuple[int, int, int]] = {}
    for kv in args.kv:
        dec = make_decoder(nova, kv, args.max_len)
        print(f"\n【{kv}】")
        print(f"{'长度':>8s} {'实际token':>9s}  追问 4 条：答对 / 挑错 / 没答"
              f"{'':>16s}{'prefill':>9s} {'峰值':>9s} {'clocks.sm':>10s}")
        for target in args.lens:
            n_rounds = max(8, int(round(target / 36)))
            msgs, _at = build(n_rounds)
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
            for name, code, _d in NEEDLES:
                qtext = f"我的{name}密码是多少？只回答那串号码。"
                full_text = render(tok, msgs + [{"role": "user", "content": qtext}], add_generation_prompt=True)
                q_ids = tok(full_text, add_special_tokens=False)["input_ids"][h:]
                q_t = torch.tensor([q_ids], device="cuda")
                dec.prefill(q_t, offset=h, reset=False)
                # ⚠️ 这里**不能**用 capture()+step()：capture 每次都会新建一张 CUDA Graph
                # 并分配新的内存池，而每个问题的起点位置都不同 -> 每个问题都要重捕一次，
                # 几次之后显存就爆了（实测跑到第 9 次时 7905/8188 MiB 崩溃）。
                # 只生成 ~20 个 token，直接 eager 调 `_body()` 就够，还省掉整块图内存。
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
                elif any(digits(c) in got for _n, c, _dd in NEEDLES if c != code):
                    wrong += 1
                    detail.append("!")
                    print(f"        ↳ 挑错！问「{name}」（应 {code}）答的是：{text[:50]}")
                else:
                    none += 1
                    detail.append("X")
                    print(f"        ↳ {name}（应 {code}）没答出：{text[:50]}")
                dec.cache.pos.fill_(h)  # 回到干草堆末尾，复用这份 prefill
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            scores[(kv, target)] = (right, wrong, none)
            print(f"{target:>8d} {h:>9d}  {right}/4 挑错 {wrong} 没答 {none}   {' '.join(detail):<18s}"
                  f" [prefill {prefill_s:>6.1f}s · 峰值 {peak:>5.2f} GiB · clocks.sm {clk}]")
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
