r"""E6 · PCIe / 主机内存分层：**两条路并行**到底能不能兑现。

侧会话的公式：把比例为 B 的 KV 放主机内存，总时间 = `max(A / 显存带宽, B / PCIe带宽)`，
与"全在显存"同速的条件是 `B ≤ T × 12.2 / 显存带宽`（理想 **5.3%**，按现状有效带宽约 **20%**）。

本探针**不写 attention 核**，只量那条并行假设本身：
1. **A 段**：双缓冲预取（`nova/prefetch.py`）从 H: 读一个文件到显存，对比朴素 `read()`；
2. **B 段**：一条"读满显存"的核（用 Triton reduce 模拟 decode 的读量）与一次 H2D 搬运，
   **分别在两条 stream 上同时跑**，看总时间是不是 `max(...)` 而不是 `sum(...)`。

判据：`t_both ≈ max(t_vram, t_h2d)` ⇒ 并行成立，侧会话的 5.3% / 20% 上限可兑现；
若 `t_both ≈ t_vram + t_h2d` ⇒ 分层没有免费额度，E6 直接结案。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_kv_tiering.py --gib 1.0
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
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from nova.prefetch import SegmentPrefetcher  # noqa: E402


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


@triton.jit
def _read_reduce(x_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(x))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="PCIe / 主机内存分层探针")
    ap.add_argument("--gib", type=float, default=1.0, help="测试文件大小（GiB）")
    ap.add_argument("--seg-mib", type=int, default=8, help="预取段大小（MiB）")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    tmp = ROOT / ".tmp"
    tmp.mkdir(exist_ok=True)
    path = tmp / "tiering.bin"
    nbytes = int(args.gib * 1024 ** 3)
    if not path.exists() or path.stat().st_size != nbytes:
        print(f"生成测试文件 {args.gib:.2f} GiB …", flush=True)
        chunk = torch.randn(1 << 24, dtype=torch.float16).numpy().tobytes()
        with open(path, "wb") as fh:
            written = 0
            while written < nbytes:
                fh.write(chunk[: min(len(chunk), nbytes - written)])
                written += len(chunk)
    gib = path.stat().st_size / 1024 ** 3
    print(f"文件 {path} {gib:.2f} GiB · 段 {args.seg_mib} MiB · clocks.sm {clock_sm()}")

    # ---- A 段：朴素读 vs 双缓冲预取 ----
    best_sync = 1e9
    for _ in range(args.reps):
        t0 = time.perf_counter()
        t = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).to("cuda")
        torch.cuda.synchronize()
        best_sync = min(best_sync, time.perf_counter() - t0)
        del t
        torch.cuda.empty_cache()
    print(f"\n[A] 朴素 read() + H2D        : {gib / best_sync:>6.2f} GiB/s  ({best_sync * 1000:.0f} ms)")

    pf = SegmentPrefetcher(path, seg_bytes=args.seg_mib << 20)
    best_pref = 1e9
    for _ in range(args.reps):
        t0 = time.perf_counter()
        for _seg in pf.stream("cuda"):
            pass
        torch.cuda.synchronize()
        best_pref = min(best_pref, time.perf_counter() - t0)
        torch.cuda.empty_cache()
    print(f"[A] pin + 双缓冲 + non_blocking: {gib / best_pref:>6.2f} GiB/s  ({best_pref * 1000:.0f} ms)"
          f"  ⇒ {best_sync / best_pref:.2f}x")

    # ---- B 段：显存读 与 PCIe 搬运 并行 ----
    vram_gib = args.gib
    x = torch.empty(int(vram_gib * 1024 ** 3 // 4), dtype=torch.float32, device="cuda")
    out = torch.zeros(4096, dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(x.numel(), 4096),)

    def vram_read():
        _read_reduce[grid](x, out, x.numel(), BLOCK=4096, num_warps=4)

    host = torch.empty(int(args.gib * 1024 ** 3 // 4), dtype=torch.float32, pin_memory=True)
    gpu_dst = torch.empty_like(host, device="cuda")

    def t_of(fn, reps=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    t_vram = t_of(vram_read)
    t_h2d = t_of(lambda: gpu_dst.copy_(host, non_blocking=True))
    print(f"\n[B] 显存读 {vram_gib:.2f} GiB        : {t_vram * 1000:>6.1f} ms  ({vram_gib / t_vram:>6.1f} GiB/s)")
    print(f"[B] H2D 搬运 {vram_gib:.2f} GiB      : {t_h2d * 1000:>6.1f} ms  ({vram_gib / t_h2d:>6.1f} GiB/s)")

    side = torch.cuda.Stream()
    def both():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            gpu_dst.copy_(host, non_blocking=True)
        vram_read()
        torch.cuda.current_stream().wait_stream(side)
    t_both = t_of(both)
    print(f"[B] 两条 stream 同时跑       : {t_both * 1000:>6.1f} ms"
          f"  （max={max(t_vram, t_h2d) * 1000:.1f} / sum={(t_vram + t_h2d) * 1000:.1f}）")
    ratio = t_both / max(t_vram, t_h2d)
    print(f"    ⇒ 并行效率 {ratio:.2f}x max（1.00 = 完全重叠，2.00 = 完全串行）")

    free_frac = 12.2 / (vram_gib / t_vram)  # B ≤ T × PCIe / 显存带宽
    print(f"\n判据：若接近 max ⇒ 分层成立，可白放主机内存的 KV 比例上限 = PCIe/显存带宽 = {free_frac:.1%}")
    print(f"clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
