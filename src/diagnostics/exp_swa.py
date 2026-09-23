r"""E2 · 滑动窗口：质量退化拐点扫描。

**问题**：把一部分层换成"只看最近 W 个 token"（局部层），其余层仍看全上下文（全局层），
质量会从哪一档开始掉？这个数字决定后面所有"训练版压缩器"值不值得做。

**结构**（`NovaConfig.swa_window` / `swa_global_every`，见 `src/nova/cache.py` 的 `WindowedKVCache`）：
局部层用 **2W** 个槽的 ring（定长、与 CUDA Graph 兼容），全局层仍按全长。
层号取**变换器层号** `t_idx % swa_global_every == 0` 的是全局层（默认每 4 层 1 个）。

**判据**：16K 档（fp16 全注意力还能当基线的最高档）必须 4/4 零挑错、与全量基线同分；
32K/64K 记录"哪一档开始掉"—— 拐点本身就是产物。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\exp_swa.py --lens 16384 --swa 0 1024 2048 4096
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

from chatfmt import EOS_ID, EOS_ID_ALT, render  # noqa: E402
from exp_needle import NEEDLES, build, digits  # noqa: E402
from nova.decode import GraphDecoder  # noqa: E402
from nova.loader import build_nova, enable_lm_head_4bit, load_hf_base  # noqa: E402

STOP = {EOS_ID, EOS_ID_ALT}


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def kv_bytes(dec) -> int:
    return dec.cache.nbytes()


def run_one(nova, tok, msgs, h: int, max_new: int, max_len: int):
    """一次：prefill 干草堆 -> 逐题 greedy 追问。返回 (答对, 挑错, 没答, prefill 秒, 峰值 GiB, 明细)。"""
    dec = GraphDecoder.for_length(nova, h, reserve=max_new) if not max_len else GraphDecoder(nova, max_len=max_len)
    hist_text = render(tok, msgs, add_generation_prompt=False)
    hist_ids = tok(hist_text, add_special_tokens=False)["input_ids"]
    hist_t = torch.tensor([hist_ids], device="cuda")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    dec.prefill(hist_t)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    right = wrong = none = 0
    detail = []
    for name, code, _d in NEEDLES:
        qtext = f"我的{name}密码是多少？只回答那串号码。"
        full_text = render(tok, msgs + [{"role": "user", "content": qtext}], add_generation_prompt=True)
        q_ids = tok(full_text, add_special_tokens=False)["input_ids"][h:]
        dec.prefill(torch.tensor([q_ids], device="cuda"), offset=h, reset=False)
        out = []
        for _ in range(max_new):
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
        else:
            none += 1
            detail.append("X")
        dec.cache.pos.fill_(h)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return right, wrong, none, prefill_s, peak, detail, kv_bytes(dec)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="滑动窗口质量拐点")
    ap.add_argument("--lens", type=int, nargs="+", default=[16384])
    ap.add_argument("--swa", type=int, nargs="+", default=[0, 1024, 2048, 4096],
                    help="窗口大小列表；0 = 全注意力基线")
    ap.add_argument("--global-every", type=int, default=4, help="每 N 层留 1 个全局层")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=0, help="0 = 按实际长度选桶（推荐）")
    args = ap.parse_args()

    # ⚠️ 必须显式 `num_paths=1` + `enable_lm_head_4bit`：`build_nova` 的默认是 **双通路**，
    # 会深拷贝通路层（多 ~2.4 GiB、cache 槽位从 36 变 60）—— 与基线的唯一差别就只能是窗口。
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    hf = load_hf_base()

    def make(window: int):
        m = build_nova(hf, norm_impl="triton", num_paths=1,
                       swa_window=window, swa_global_every=args.global_every)
        enable_lm_head_4bit(m)
        return m

    nova0 = make(0)
    print(f"Qwen3-VL-4B · 单通路 · 4 条形近事实 · 每 {args.global_every} 层 1 个全局层"
          f" · clocks.sm {clock_sm()}")
    print(f"{'窗口':>6s} {'长度':>7s} {'实际':>7s}  {'答对':>4s} {'挑错':>4s} {'没答':>4s}  {'明细':<8s}"
          f" {'KV':>8s} {'prefill':>8s} {'峰值':>8s}")

    for window in args.swa:
        nova = nova0 if window == 0 else make(window)
        for target in args.lens:
            n_rounds = max(8, int(round(target / 36)))
            msgs, _at = build(n_rounds)
            hist_text = render(tok, msgs, add_generation_prompt=False)
            h = len(tok(hist_text, add_special_tokens=False)["input_ids"])
            right, wrong, none, prefill_s, peak, detail, kv = run_one(
                nova, tok, msgs, h, args.max_new, args.max_len
            )
            print(f"{window:>6d} {target:>7d} {h:>7d}  {right:>3d}/4 {wrong:>4d} {none:>4d}  "
                  f"{' '.join(detail):<8s} {kv / 1024 ** 3:>6.2f}GiB {prefill_s:>7.1f}s {peak:>6.2f}GiB")
            del msgs
            torch.cuda.empty_cache()

    print(f"\n判据：16K 档 4/4 零挑错且与全量基线同分；32K/64K 记录拐点。clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
