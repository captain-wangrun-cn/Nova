r"""最后一轮结构实验：tl.gather / tl.dot / 打包 LUT。
用法：& .\.venv\Scripts\python.exe src\diagnostics\exp_nf4_final.py
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


# ---- A：现状（两次 tl.sum，逐元素查表）------------------------------------
@triton.jit
def _kA(P, C, A, X, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32); offs_k = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        am = tl.load(A + blk*N + offs_n)
        xs = tl.load(X + blk*64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        hi = tl.load(C + (p >> 4).to(tl.int32)); lo = tl.load(C + (p & 0x0F).to(tl.int32))
        acc += (tl.sum(hi * xe[None, :], axis=1) + tl.sum(lo * xo[None, :], axis=1)) * am
    tl.store(Out + offs_n, acc.to(Out.dtype.element_ty))


# ---- FP：一字节一次 gather（表里存 fp16x2）--------------------------------
@triton.jit
def _kFP(P, C2, A, X, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32); offs_k = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        am = tl.load(A + blk*N + offs_n)
        xs = tl.load(X + blk*64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        t = tl.load(C2 + p.to(tl.int32))
        hi = (t & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        lo = (t >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        acc += (tl.sum(hi * xe[None, :], axis=1) + tl.sum(lo * xo[None, :], axis=1)) * am
    tl.store(Out + offs_n, acc.to(Out.dtype.element_ty))


# ---- G：tl.gather（让 Triton 自己选 shuffle / shared 实现）-----------------
@triton.jit
def _kG(P, C, A, X, Out, N, K2, NB, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0); offs_n = pid*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32); offs_k = tl.arange(0, 64)
    tbl = tl.load(C + tl.arange(0, 16))
    tbl2 = tl.broadcast_to(tbl[None, :], (BLOCK_N, 16))
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + offs_n[:, None]*K2 + (blk*32 + offs_b)[None, :])
        am = tl.load(A + blk*N + offs_n)
        xs = tl.load(X + blk*64 + offs_k)
        xe, xo = tl.split(tl.reshape(xs, (32, 2)))
        hi = tl.gather(tbl2, (p >> 4).to(tl.int32), axis=1)
        lo = tl.gather(tbl2, (p & 0x0F).to(tl.int32), axis=1)
        acc += (tl.sum(hi * xe[None, :], axis=1) + tl.sum(lo * xo[None, :], axis=1)) * am
    tl.store(Out + offs_n, acc.to(Out.dtype.element_ty))


# ---- DOT：join/permute/reshape 成 [BLOCK_K, BLOCK_N]，交给 tl.dot ----------
@triton.jit
def _kDOT(P, C, A, X, Out, N, K2, NB, K, stride_om,
          BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32); offs_k = tl.arange(0, 64)
    offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for blk in tl.range(0, NB):
        p = tl.load(P + (blk*32 + offs_b)[:, None]*N + offs_n[None, :])          # [32, BLOCK_N]
        am = tl.load(A + blk*N + offs_n)
        xs = tl.load(X + offs_m[:, None]*K + blk*64 + offs_k[None, :])           # [BLOCK_M, 64]
        vh = tl.load(C + (p >> 4).to(tl.int32))
        vl = tl.load(C + (p & 0x0F).to(tl.int32))
        w = tl.reshape(tl.permute(tl.join(vh, vl), (0, 2, 1)), (64, BLOCK_N))    # [64, BLOCK_N]
        acc += tl.dot(xs, w.to(tl.float16)) * am[None, :]
    tl.store(Out + offs_m[:, None]*stride_om + offs_n[None, :], acc.to(Out.dtype.element_ty))


def main():
    import bitsandbytes as bnb
    from diagnostics.bench_nf4_gemv import make_bnb_linear
    from nova.quant import extract_nf4

    shapes = [("q_proj", 4096, 2560), ("kv", 1024, 2560), ("o_proj", 2560, 4096),
              ("gate/up", 9728, 2560), ("down", 2560, 9728)]
    print(f"{'shape':<9s} {'N':>5s} {'K':>5s} {'bnb':>8s} {'A':>8s} {'FP':>8s} {'G':>8s} {'DOT':>8s}")
    tot = {"bnb": 0.0, "A": 0.0, "FP": 0.0, "G": 0.0, "DOT": 0.0}
    for tag, N, K in shapes:
        lin = make_bnb_linear(N, K)
        packed, absmax, code, _ = extract_nf4(lin.weight.data, lin.weight.quant_state)
        packedT = packed.t().contiguous()
        code16 = code.to(torch.float16)
        idx = torch.arange(256, dtype=torch.int64)
        lo_bits = (code16[idx & 15].view(torch.int16).to(torch.int64) & 0xFFFF)
        hi_bits = (code16[idx >> 4].view(torch.int16).to(torch.int64) & 0xFFFF)
        val = (lo_bits | (hi_bits << 16)) & 0xFFFFFFFF
        c2 = torch.where(val >= 2**31, val - 2**32, val).to(torch.int32).cuda()
        x = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.5
        out = torch.empty(1, N, device="cuda", dtype=torch.float16)
        nb = K // 64
        res = {}
        with torch.inference_mode():
            res["bnb"] = gt(lambda: lin(x))
        for name, kern, extra in (("A", _kA, False), ("FP", _kFP, False), ("G", _kG, False), ("DOT", _kDOT, True)):
            best = float("inf")
            for bn in (32, 64, 128, 256):
                for nw in (2, 4, 8):
                    if name == "DOT":
                        if bn > 128: continue
                        grid = (triton.cdiv(N, bn), 1)
                        args = (packedT, code, absmax, x.view(-1), out.view(-1), N, K//2, nb, K, out.stride(0))
                        call = lambda: kern[grid](*args, BLOCK_N=bn, BLOCK_M=16, num_warps=nw, num_stages=2)
                    else:
                        grid = (triton.cdiv(N, bn),)
                        args = ((packed, c2, absmax, x.view(-1), out.view(-1), N, K//2, nb) if name == "FP"
                                else (packed, code, absmax, x.view(-1), out.view(-1), N, K//2, nb))
                        call = lambda: kern[grid](*args, BLOCK_N=bn, num_warps=nw, num_stages=2)
                    try:
                        best = min(best, gt(call, iters=60, reps=3))
                    except Exception:
                        pass
            res[name] = best
        for k2 in tot: tot[k2] += res[k2]
        print(f"{tag:<9s} {N:>5d} {K:>5d} " + " ".join(f"{res[k2]:8.1f}" for k2 in ("bnb","A","FP","G","DOT")))
    print(f"{'合计':<9s} {'':>5s} {'':>5s} " + " ".join(f"{tot[k2]:8.1f}" for k2 in ("bnb","A","FP","G","DOT")))


if __name__ == "__main__":
    main()