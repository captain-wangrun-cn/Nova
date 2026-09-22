r"""NF4 kernel 分阶段实验：逐层加功能，定位时间花在哪。

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\exp_nf4_stages.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))

import torch
import triton
import triton.language as tl


def graph_time(fn, iters=60, reps=3, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / iters * 1e6


# --- 阶段 1：只读 packed + 规约，不做 LUT ---------------------------------
@triton.jit
def _k_loadonly(P, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        acc += tl.sum(p.to(tl.float32), axis=1)
    tl.store(Out + offs_n, acc)


# --- 阶段 2：加上码本 gather（两次）----------------------------------------
@triton.jit
def _k_lut(P, C, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc += tl.sum(hi, axis=1) + tl.sum(lo, axis=1)
    tl.store(Out + offs_n, acc)


# --- 阶段 3：加上 x 的乘 + 两次规约 ----------------------------------------
@triton.jit
def _k_lut_x(P, C, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    offs_k = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        xs = tl.load(X + blk * 64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc += tl.sum(hi * xe[None, :], axis=1) + tl.sum(lo * xo[None, :], axis=1)
    tl.store(Out + offs_n, acc)


# --- 阶段 4：完整（再加 absmax）--------------------------------------------
@triton.jit
def _k_full(P, C, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    offs_k = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        am = tl.load(A + blk * N + offs_n)
        xs = tl.load(X + blk * 64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc += (tl.sum(hi * xe[None, :], axis=1) + tl.sum(lo * xo[None, :], axis=1)) * am
    tl.store(Out + offs_n, acc)


# --- 阶段 4b：单次 interleave + 单次规约 -----------------------------------
@triton.jit
def _k_interleave(P, C, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        am = tl.load(A + blk * N + offs_n)
        xs = tl.load(X + blk * 64 + tl.arange(0, 64))
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        w = tl.interleave(hi, lo)                       # [BLOCK_N, 64]，列 = k
        acc += tl.sum(w * xs[None, :], axis=1) * am
    tl.store(Out + offs_n, acc)


# --- 阶段 4c：不查表，直接用索引当值（隔离 gather 成本）--------------------
@triton.jit
def _k_nogather(P, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    offs_k = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        am = tl.load(A + blk * N + offs_n)
        xs = tl.load(X + blk * 64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        hi = (p >> 4).to(tl.float32)
        lo = (p & 0x0F).to(tl.float32)
        acc += (tl.sum(hi * xe[None, :], axis=1) + tl.sum(lo * xo[None, :], axis=1)) * am
    tl.store(Out + offs_n, acc)


KERNELS = {
    "1 loadonly   ": _k_loadonly,
    "2 +lut       ": _k_lut,
    "3 +lut+x     ": _k_lut_x,
    "4 full       ": _k_full,
    "4b interleave": _k_interleave,
    "4c nogather  ": _k_nogather,
}


def main():
    torch.manual_seed(0)
    for N, K in ((2560, 2560), (9728, 2560)):
        print(f"\n===== N={N} K={K} =====")
        packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
        absmax = torch.rand(N, K // 64, device="cuda", dtype=torch.float32).t().contiguous()
        code = torch.randn(16, device="cuda", dtype=torch.float32)
        x = torch.randn(1, K, device="cuda", dtype=torch.float16)
        out = torch.empty(1, N, device="cuda", dtype=torch.float16)
        nb = K // 64
        nbytes = N * K / 2 + N * (K / 64) * 4
        print(f"{'stage':<14s} " + " ".join(f"{'bn='+str(bn):>10s}" for bn in (32, 64, 128)))
        for name, kern in KERNELS.items():
            row = []
            for bn in (32, 64, 128):
                grid = (triton.cdiv(N, bn),)
                args = {
                    "1 loadonly   ": (packed, out.view(-1), N, K // 2, nb, out.stride(0)),
                    "2 +lut       ": (packed, code, out.view(-1), N, K // 2, nb, out.stride(0)),
                    "3 +lut+x     ": (packed, code, x.view(-1), out.view(-1), N, K // 2, nb, out.stride(0)),
                    "4 full       ": (packed, code, absmax, x.view(-1), out.view(-1), N, K // 2, nb, out.stride(0)),
                    "4b interleave": (packed, code, absmax, x.view(-1), out.view(-1), N, K // 2, nb, out.stride(0)),
                    "4c nogather  ": (packed, absmax, x.view(-1), out.view(-1), N, K // 2, nb, out.stride(0)),
                }[name]
                for nw in (2, 4):
                    try:
                        fn = lambda: kern[grid](*args, BLOCK_N=bn, num_warps=nw, num_stages=2)
                        t = graph_time(fn, iters=40)
                        if nw == 2:
                            row.append(t)
                    except Exception as exc:  # noqa: BLE001
                        if nw == 2:
                            row.append(float("nan"))
            bw = nbytes / (row[1] * 1e-6) / 1e9 if row[1] == row[1] else float("nan")
            print(f"{name:<14s} " + " ".join(f"{v:10.1f}" for v in row) + f"   ({bw:5.0f} GB/s @bn64)")


if __name__ == "__main__":
    main()