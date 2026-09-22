r"""S4 · 记忆开销基准：检索 / 注入 prefill / 显存。

跑法（环境变量见 [s4_memory_demo.py](s4_memory_demo.py) 头部）：

```powershell
& .\.venv\Scripts\python.exe src\bench_memory.py --mem .tmp\s4-memory\demo.safetensors
```

三种条件，都是"608 token 历史 + 当前轮"这一条真实 prompt：

| 条件 | 说明 |
|------|------|
| `off` | 不注入（基线 prefill） |
| `auto` | 检索（`rank`）+ 注入 + 分段 prefill（`place=turn`） |

⚠️ **`clocks.sm` 一并打印** —— 本机 GPU 空闲时会停在 ~780 MHz（上限 3105 MHz），
不记频率的速度数字没法跨时间点对比（[AGENTS.md](../AGENTS.md) 第七节第 4 条）。
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
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

from chatfmt import find_span, render  # noqa: E402
from nova.memory import (  # noqa: E402
    MemorySchema,
    MemorySession,
    MemoryStore,
    RetrievalInfo,
    model_fingerprint,
)
from s4_memory_demo import QUESTIONS, build_filler, load_bundle  # noqa: E402


def clocks() -> tuple[int, int, float, int] | None:
    """当前 `clocks.sm` / `clocks.max.sm` / 功耗 / 温度；取不到返回 None。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) != 4:
            return None
        return int(float(parts[0])), int(float(parts[1])), float(parts[2]), int(float(parts[3]))
    except Exception as exc:  # pragma: no cover - 环境相关
        print(f"（nvidia-smi 不可用：{exc}）")
        return None


def clock_line(tag: str) -> str:
    c = clocks()
    if c is None:
        return f"{tag} clocks.sm=?（取不到）"
    sm, sm_max, power, temp = c
    return f"{tag} clocks.sm={sm} MHz（上限 {sm_max}，{sm / sm_max:.0%}）{power:.1f} W {temp}°C"


def ms(x: float) -> str:
    return f"{x:7.1f}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Nova S4 记忆开销基准")
    ap.add_argument("--mem", default=str(ROOT / ".tmp" / "s4-memory" / "demo.safetensors"))
    ap.add_argument("--turns", type=int, default=20, help="写入之后插入多少轮无关对话")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3, help="预热轮数；本机 GPU 空闲会停在 ~210-780 MHz，先跑热")
    ap.add_argument("--paths", type=int, default=2, choices=(1, 2))
    ap.add_argument("--norm", choices=["exact", "triton"], default="triton")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    print(f"GPU：{torch.cuda.get_device_name(0)}")

    nova, tok = load_bundle(args.paths, args.norm)
    store = MemoryStore.load(
        args.mem,
        schema=MemorySchema.for_model(nova),
        fingerprint=model_fingerprint(nova),
    )
    sizes = [it.n_tokens for it in store.items]
    print(f"记忆：{store}  （{len(sizes)} 条，{min(sizes)}–{max(sizes)} token/条）")
    print(f"      每条记忆的 K/V = 240 KiB/token（60 槽位 × 8 KV 头 × 128 维 × K,V × fp16）")

    filler = build_filler(args.turns)
    hist_ids = tok(render(tok, filler, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    hist_t = torch.tensor([hist_ids], device="cuda")
    qtext = QUESTIONS[0][0]
    full_text = render(tok, filler + [{"role": "user", "content": qtext}], add_generation_prompt=True)
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    assert full_ids[: len(hist_ids)] == hist_ids, "历史不是前缀（模板变了？）"
    cur_t = torch.tensor([full_ids[len(hist_ids) :]], device="cuda")
    qspan = find_span(tok, full_text, qtext)
    print(f"prompt：历史 {len(hist_ids)} token + 当前轮 {cur_t.shape[1]} token\n")

    sess = MemorySession(store, nova, max_len=args.max_len)
    for _ in range(max(1, args.warmup)):
        info = sess.prefill(hist_t, cur_t, query_span=qspan)
    print(f"预热 {max(1, args.warmup)} 轮（不计入），命中 {info.prefix_len} token · {clock_line('预热后')}\n")

    rows = {}
    for cond in ("off", "rank", "auto"):
        tot, que, sco, pre, clk = [], [], [], [], []
        for _ in range(args.reps):
            info = RetrievalInfo()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if cond == "off":
                info = sess.prefill(hist_t, cur_t, use_memory=False)
            elif cond == "rank":
                sess.rank(hist_t, cur_t, qspan, info=info)
            else:
                info = sess.prefill(hist_t, cur_t, query_span=qspan)
            torch.cuda.synchronize()
            tot.append((time.perf_counter() - t0) * 1000)
            clk.append(clocks())
            que.append(info.query_ms)
            sco.append(info.score_ms)
            pre.append(info.prefill_ms)
        rows[cond] = tuple(statistics.median(x) for x in (tot, que, sco, pre))
        sms = [c[0] for c in clk if c]
        rows[cond] += (f"{min(sms)}–{max(sms)} MHz" if sms else "?",)

    # 逐轮交替：时钟漂移（笔记本 GPU 满载会从 2.4GHz 掉到 1GHz）对两边同幅影响，差值才可比
    pairs = []
    for _ in range(args.reps):
        clk = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sess.prefill(hist_t, cur_t, use_memory=False)
        torch.cuda.synchronize()
        t_off = time.perf_counter() - t0
        clk.append(clocks())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        info = sess.prefill(hist_t, cur_t, query_span=qspan)
        torch.cuda.synchronize()
        t_on = time.perf_counter() - t0
        clk.append(clocks())
        pairs.append(((t_on - t_off) * 1000, info.query_ms, info.score_ms, (info.prefill_ms - t_off * 1000), clk))
    d_tot = statistics.median(p[0] for p in pairs)
    d_q = statistics.median(p[1] for p in pairs)
    d_s = statistics.median(p[2] for p in pairs)
    d_pf = statistics.median(p[3] for p in pairs)
    pair_sm = [c[0] for p in pairs for c in p[4] if c]

    print(f"{'阶段':22s} {'中位':>9s}  {'clocks.sm':>16s}   说明")
    labels = {
        "off": "prefill 基线（不注入）",
        "rank": "检索（取 Q + 打分）",
        "auto": "检索 + 注入 + prefill",
    }
    for cond, (t, q, s, p, sm) in rows.items():
        extra = ""
        if cond == "rank":
            extra = f"其中 取 Q {q:.0f} + 打分 {s:.0f}"
        if cond == "auto":
            extra = f"其中 prefill {p:.0f}（含注入）"
        print(f"{labels[cond]:22s} {ms(t)}  {sm:>16s}   {extra}")
    t_off, t_rank, t_on = rows["off"][0], rows["rank"][0], rows["auto"][0]
    if pair_sm:
        print(f"\n逐轮交替（{args.reps} 轮，clocks.sm {min(pair_sm)}–{max(pair_sm)} MHz）：记忆的额外开销")
        print(f"  **{d_tot:.0f}ms = 取 Q {d_q:.0f} + 打分 {d_s:.0f} + 注入 {d_pf:.0f}**")
    print("  取 Q = 一次**完整无 cache 前向**（K/V 依赖因果上下文，必须整段跑）；打分 = 6 条候选的旋转 + einsum")
    print(f"  上面那张表是整段实测（含时钟漂移，跨条件不可比）；单独看「检索」一档：{t_rank:.0f}ms")
    print(f"\n与时钟无关的比值（跨时间点可比）：检索/基线前向 = {t_rank / t_off:.2f}x，"
          f"检索+注入/基线前向 = {t_on / t_off:.2f}x")

    torch.cuda.synchronize()
    print(f"峰值显存：{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB（{args.paths} 通路 · max_len={args.max_len}）")
    print(clock_line("测量后"))


if __name__ == "__main__":
    main()
