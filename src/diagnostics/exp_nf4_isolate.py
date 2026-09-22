r"""隔离实验：到底哪一步在吃时间。
用法：& .\.venv\Scripts\python.exe src\diagnostics\exp_nf4_isolate.py
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
import torch, triton
import triton.language as tl


def gt(fn, iters=100, reps=5, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e6)
    return min(ts)


@triton.jit
def _s1_load(P, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N); offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        acc += tl.sum(p.to(tl.float32), axis=1)
    tl.store(Out + offs_n, acc)


@triton.jit
def _s2_gather(P, C, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N); offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        v = tl.load(C + p.to(tl.int32))
        acc += tl.sum(v, axis=1)
    tl.store(Out + offs_n, acc)


@triton.jit
def _s3_nib(P, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N); offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        acc += tl.sum((p >> 4).to(tl.float32), axis=1) + tl.sum((p & 0x0F).to(tl.float32), axis=1)
    tl.store(Out + offs_n, acc)


@triton.jit
def _s4_gather_nib(P, C, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N); offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc += tl.sum(hi, axis=1) + tl.sum(lo, axis=1)
    tl.store(Out + offs_n, acc)


@triton.jit
def _s5_flatgather(P, C, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    """整块 1-D 加载后再 reshape，看是否改善 layout。"""
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N); offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        hi = tl.load(C + (p >> 4).to(tl.int32))
        lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc += tl.sum(hi + lo, axis=1)
    tl.store(Out + offs_n, acc)


@triton.jit
def _s6_int8(P, C, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    """把 p 直接当 fp16 低位用：p 先转 fp16 再查 256 项表。"""
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N); offs_b = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        v = tl.load(C + p.to(tl.int32))
        acc += tl.sum(v, axis=1)
    tl.store(Out + offs_n, acc)


STAGES = [("1 load+sum     ", _s1_load, False), ("2 +gather      ", _s2_gather, True),
          ("3 +nibble      ", _s3_nib, False), ("4 +gather+nib  ", _s4_gather_nib, True),
          ("5 gather(hi+lo)", _s5_flatgather, True)]


def main():
    torch.manual_seed(0)
    for N, K in ((2560, 2560), (9728, 2560)):
        print(f"\n===== N={N} K={K}  (packed {N*K/2/1e6:.1f} MB) =====")
        packed = torch.randint(0, 256, (N, K//2), device="cuda", dtype=torch.uint8)
        code = torch.randn(16, device="cuda", dtype=torch.float32)
        out = torch.empty(1, N, device="cuda", dtype=torch.float16)
        nb = K // 64
        for name, kern, needs_c in STAGES:
            row = []
            for bn in (32, 64, 128):
                best = float("inf")
                for nw in (1, 2, 4, 8):
                    grid = (triton.cdiv(N, bn),)
                    args = ((packed, code, out.view(-1), N, K//2, nb) if needs_c
                            else (packed, out.view(-1), N, K//2, nb))
                    try:
                        t = gt(lambda: kern[grid](*args, BLOCK_N=bn, num_warps=nw, num_stages=2))
                        best = min(best, t)
                    except Exception:
                        pass
                row.append(best)
            print(f"{name:<16s}" + "".join(f"{v:10.1f}" for v in row))


if __name__ == "__main__":
    main()