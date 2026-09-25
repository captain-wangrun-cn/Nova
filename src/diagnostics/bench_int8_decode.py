r"""①.5 · int8 流式 KV cache 的**端到端**解码延迟（`GraphDecoder` 级）。

比的是同一件事：**整步图解码**（embed → 36/60 层 → lm_head → argmax → 写回），
只换 KV 常驻格式：

| 配置 | KV 常驻 | 注意力 |
|---|---|---|
| `off` | fp16 | SDPA（`repeat_kv` 展平成 32 头） |
| `int8` | int8 + 尺子 + 64 槽 fp16 环 | int8 融合核 + 环 SDPA，`logsumexp` 合并 |

**判据不是"比 fp16 快多少"**，而是三条都要看：

1. **能不能跑**：长上下文下 int8 与 fp16 都要能出 token；
2. **省不省**：常驻字节数比值（长上下文应 ≈ 0.53）；
3. **快不快**：ms/token。注意力只占整步的一部分，`L` 越长它占比越大 —— 所以**短上下文
   看不出收益是正常的**，要看 `L` 从 1K 到 8K 的趋势。

⚠️ 速度数字必须带 `clocks.sm`（本机 GPU 空闲会停在 ~780 MHz，上限 3105，同一条命令实测差 1.9x）。
本脚本按配置**逐个**建、逐个删（8GB 上 int8 与 fp16 的 8K 缓存不能同时驻留）。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\bench_int8_decode.py --lens 1024 2048 4096
"""

from __future__ import annotations

import argparse
import gc
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

from nova.decode import GraphDecoder, bucket_for  # noqa: E402
from nova.loader import load_nova  # noqa: E402


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def measure(nova, quant: str, prompt_len: int, reps: int, warmup: int) -> dict:
    """建一个解码器 → prefill → capture → 量 reps 步。返回延迟与显存。"""
    max_len = bucket_for(prompt_len + reps + 16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dec = GraphDecoder(nova, max_len=max_len, quant=quant)
    ids = torch.randint(0, 1000, (1, prompt_len), device="cuda")
    t0 = time.perf_counter()
    dec.prefill(ids)
    t_prefill = time.perf_counter() - t0
    dec.capture(warmup=3)
    for _ in range(warmup):
        dec.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dec.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    out = {
        "quant": quant,
        "max_len": max_len,
        "ms_per_token": dt / reps * 1000,
        "tok_s": reps / dt,
        "peak_gib": peak,
        "prefill_s": t_prefill,
        "kv_mib": dec.cache.nbytes() / 1024 ** 2,
        "clocks_sm": clock_sm(),
    }
    del dec
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--quants", nargs="+", default=["off", "int8"])
    args = ap.parse_args()

    nova, _, _ = load_nova(norm_impl="exact")
    print(f"clocks.sm {clock_sm()}（每次测量前都会重新采样）\n")
    rows = []
    for prompt_len in args.lens:
        for quant in args.quants:
            try:
                r = measure(nova, quant, prompt_len, args.reps, args.warmup)
            except Exception as exc:  # noqa: BLE001
                print(f"L={prompt_len} {quant}: 失败 —— {type(exc).__name__}: {exc}")
                continue
            r["prompt_len"] = prompt_len
            rows.append(r)
            print(f"L={prompt_len:5d} {quant:5s} max_len={r['max_len']:5d} "
                  f"{r['ms_per_token']:7.3f} ms/token  {r['tok_s']:6.2f} tok/s  "
                  f"峰值 {r['peak_gib']:.2f} GiB  KV {r['kv_mib']:.0f} MiB  "
                  f"prefill {r['prefill_s']:.2f}s  clocks.sm {r['clocks_sm']}")

    print("\n=== 汇总（同长度下 int8 相对 fp16）===")
    for prompt_len in args.lens:
        got = {r["quant"]: r for r in rows if r["prompt_len"] == prompt_len}
        if "off" not in got or "int8" not in got:
            continue
        a, b = got["off"], got["int8"]
        print(f"L={prompt_len:5d}  速度 {a['ms_per_token']/b['ms_per_token']:.2f}x  "
              f"KV 常驻 {b['kv_mib']/a['kv_mib']:.3f}x  "
              f"（{a['clocks_sm']} / {b['clocks_sm']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
