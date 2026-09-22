r"""NF4 kernel 结构实验：把规约挪出内层循环。

核心想法（变体 F）：内层循环只做**逐元素**累加，把 `tl.sum` 留到循环结束做一次。
  acc2 [BLOCK_N, 32] += (v_hi*xe + v_lo*xo) * am[:, None]
块与块之间只是把贡献加到不同的槽位，求和顺序变了但结果等价，
好处是**内层循环里没有任何 layout 转换**。

用法：
    & .\.venv\Scripts\python.exe src\diagnostics\exp_nf4_struct.py
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


def graph_time(fn, iters=50, reps=3, warm=2):
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


# ===== A：现状（内层两次 tl.sum）===========================================
@triton.jit
def _kA(P, C, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
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
    tl.store(Out + offs_n, acc.to(Out.dtype.element_ty))


# ===== F：内层纯逐元素，规约只做一次 ======================================
@triton.jit
def _kF(P, C, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    offs_k = tl.arange(0, 64)
    acc2 = tl.zeros((BLOCK_N, 32), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        am = tl.load(A + blk * N + offs_n)
        xs = tl.load(X + blk * 64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc2 += (hi * xe[None, :] + lo * xo[None, :]) * am[:, None]
    tl.store(Out + offs_n, tl.sum(acc2, axis=1).to(Out.dtype.element_ty))


# ===== F16：F 的 fp16 码本版（gather 只搬 2 字节）=========================
@triton.jit
def _kF16(P, C, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    offs_k = tl.arange(0, 64)
    acc2 = tl.zeros((BLOCK_N, 32), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        am = tl.load(A + blk * N + offs_n)
        xs = tl.load(X + blk * 64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        hi = tl.load(C + (p >> 4).to(tl.int32)).to(tl.float32)
        lo = tl.load(C + (p & 0x0F).to(tl.int32)).to(tl.float32)
        acc2 += (hi * xe[None, :] + lo * xo[None, :]) * am[:, None]
    tl.store(Out + offs_n, tl.sum(acc2, axis=1).to(Out.dtype.element_ty))


# ===== FP：F 的「一字节一次 gather」版（表里存 fp16x2）=====================
@triton.jit
def _kFP(P, C2, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    offs_k = tl.arange(0, 64)
    acc2 = tl.zeros((BLOCK_N, 32), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 32 + offs_b)[None, :])
        am = tl.load(A + blk * N + offs_n)
        xs = tl.load(X + blk * 64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        t = tl.load(C2 + p.to(tl.int32))            # [BLOCK_N,32] int32 = (lo16=code[hi], hi16=code[lo])
        hi = (t & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        lo = (t >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        acc2 += (hi * xe[None, :] + lo * xo[None, :]) * am[:, None]
    tl.store(Out + offs_n, tl.sum(acc2, axis=1).to(Out.dtype.element_ty))


# ===== F2：F + 每轮处理 2 个 absmax 块（tile 更宽，加载更连续）=============
@triton.jit
def _kF2(P, C, A, X, Out, N, K2, NB, stride_om, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 64)      # 64 字节 = 2 个 absmax 块
    offs_k = tl.arange(0, 128)
    acc2 = tl.zeros((BLOCK_N, 64), tl.float32)
    for blk in tl.range(0, NB // 2):
        p = tl.load(P + offs_n[:, None] * K2 + (blk * 64 + offs_b)[None, :])
        # absmax 按 32 字节一组重复
        am = tl.load(A + (blk * 2 + offs_b // 32) * N + offs_n[:, None])
        xs = tl.load(X + blk * 128 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (64, 2)))
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc2 += (hi * xe[None, :] + lo * xo[None, :]) * am
    tl.store(Out + offs_n, tl.sum(acc2, axis=1).to(Out.dtype.element_ty))


KERNELS = {
    "A  sum-in-loop": (_kA, 4),
    "F  elemwise   ": (_kF, 4),
    "F16 fp16-lut  ": (_kF16, 4),
    "FP packed-lut ": (_kFP, 4),
    "F2 2blocks    ": (_kF2, 4),
}


def main():
    torch.manual_seed(0)
    for N, K in ((2560, 2560), (9728, 2560)):
        print(f"\n===== N={N} K={K} =====")
        packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
        absmax = torch.rand(N, K // 64, device="cuda", dtype=torch.float32).t().contiguous()
        code = torch.randn(16, device="cuda", dtype=torch.float32)
        code16 = code.to(torch.float16)
        idx = torch.arange(256, dtype=torch.int64)
        lo_bits = (code16[idx & 15].view(torch.int16).to(torch.int64) & 0xFFFF)
        hi_bits = (code16[idx >> 4].view(torch.int16).to(torch.int64) & 0xFFFF)
        val = (lo_bits | (hi_bits << 16)) & 0xFFFFFFFF
        c2 = torch.where(val >= 2**31, val - 2**32, val).to(torch.int32).cuda()
        x = torch.randn(1, K, device="cuda", dtype=torch.float16)
        out = torch.empty(1, N, device="cuda", dtype=torch.float16)
        nb = K // 64
        nbytes = N * K / 2 + N * (K / 64) * 4
        print(f"{'variant':<16s}" + "".join(f"{'bn='+str(b):>11s}" for b in (32, 64, 128, 256)))
        for name, (kern, nw0) in KERNELS.items():
            row = []
            for bn in (32, 64, 128, 256):
                best = float("inf")
                for nw in (1, 2, 4):
                    grid = (triton.cdiv(N, bn),)
                    args = ((packed, code, absmax, x.view(-1), out.view(-1), N, K // 2, nb, out.stride(0))
                            if name != "FP packed-lut " else
                            (packed, c2, absmax, x.view(-1), out.view(-1), N, K // 2, nb, out.stride(0)))
                    try:
                        t = graph_time(lambda: kern[grid](*args, BLOCK_N=bn, num_warps=nw, num_stages=2), iters=40)
                        best = min(best, t)
                    except Exception:
                        pass
                row.append(best)
            print(f"{name:<16s}" + "".join(f"{v:11.1f}" for v in row) +
                  f"   ({nbytes / (min(row) * 1e-6) / 1e9:5.0f} GB/s)")


if __name__ == "__main__":
    main()