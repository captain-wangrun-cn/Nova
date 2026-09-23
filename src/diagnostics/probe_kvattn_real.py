r"""D42 · **真模型 K/V** 上的 ULP 复核：int8 融合核 vs「先还原再算」。

合成随机 K/V 只能证明"机制上没写错"；证明不了"真数据的动态范围下也成立"。
本脚本抓真模型若干层的 **post-RoPE K/V**（= cache 里实际存的东西）与**同一次前向的
decode 位 Q**，跑一遍与 `tests/test_kvattn.py` 同一条判据。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_kvattn_real.py --len 2048
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
sys.path.insert(0, str(ROOT / "src" / "diagnostics"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from chatfmt import render  # noqa: E402
from exp_needle import build  # noqa: E402
from nova.kvattn import GROUP_T, int8_attn_decode, pack_int8_kv  # noqa: E402
from nova.kvquant import dequantize_int8  # noqa: E402
from nova.memory import _rotate, capture_qkv  # noqa: E402
from s4_memory_demo import load_bundle  # noqa: E402

FP16_TINY = torch.finfo(torch.float16).tiny


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


ABS_TOL = 1e-5  # 与 tests/test_kvattn.py 同一条兜底阈值（fp32 归约顺序噪声的绝对上界）


def ulp_stats(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    up = (torch.nextafter(want, torch.full_like(want, float("inf"))).float() - want.float()).abs()
    up = up.clamp_min(FP16_TINY * 2 ** -10)
    diff = (got.float() - want.float()).abs()
    ulp = diff / up
    return {
        "max_ulp": ulp.max().item(),
        "gt2": int((ulp > 2.0).sum().item()),
        "gt1": int((ulp > 1.0).sum().item()),
        "max_abs": diff.max().item(),
        "exact": (got == want).float().mean().item(),
        "bad": int(((ulp > 2.0) & (diff > ABS_TOL)).sum().item()),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="真模型 K/V 上的 ULP 复核")
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--capture-chunk", type=int, default=1024)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 18, 35])
    ap.add_argument("--dump", type=str, default="", help="把该层的 q/k/v 存到 .tmp 供隔离实验")
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    cfg = nova.model.config
    n_rounds = max(8, int(round(args.len / 36)))
    msgs, _at = build(n_rounds)
    text = render(tok, msgs, add_generation_prompt=False)
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"][: args.len]], device="cuda")
    h = int(ids.shape[1])

    # 分块抓 **pre-RoPE** K/V（一次全抓会把 Q 也 materialize 出来，显存吃不消）
    ks, vs, q_last = [], [], None
    t0 = time.perf_counter()
    for c0 in range(0, h, args.capture_chunk):
        c1 = min(c0 + args.capture_chunk, h)
        q_c, k_c, v_c = capture_qkv(nova.model, ids, (c0, c1))
        ks.append(k_c)
        vs.append(v_c)
        if c1 == h:
            q_last = q_c
        del q_c
        torch.cuda.empty_cache()
    k_pre = torch.cat(ks, dim=2)  # [层, KV头, h, D]（pre-RoPE）
    v_all = torch.cat(vs, dim=2)
    torch.cuda.synchronize()
    print(f"真实 K/V：{h} token · {k_pre.shape[0]} 层 × {k_pre.shape[1]} KV 头 · "
          f"抓取 {time.perf_counter() - t0:.1f}s · clocks.sm {clock_sm()}")

    hq, h_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    print(f"{'层':>4s} {'K通道离群':>10s} {'打分|max|':>10s} {'max_ulp':>8s} {'>2ULP':>6s} "
          f"{'>1ULP':>6s} {'max_abs':>9s} {'逐位相同':>9s} │ {'核vs真值':>9s} {'SDPAvs真值':>11s}")
    for layer in args.layers:
        # 旋转到真实位置 —— 与 `MemoryStore.inject` 用的是同一个函数
        # ⚠️ `_rotate` 要 `[层, 头, t, D]`（4 维）：这里拿单层，所以先补一个前导维
        k_rot = _rotate(k_pre[layer].unsqueeze(0), 0, cfg.hidden_size, nova.model.rotary_emb)
        q_rot = _rotate(q_last[layer][:, -1:, :].unsqueeze(0), h - 1, cfg.hidden_size, nova.model.rotary_emb)
        k = k_rot.contiguous()                       # [1, KV头, h, D]（post-RoPE，= cache 内容）
        v = v_all[layer].unsqueeze(0).contiguous()
        q = q_rot.contiguous()                       # [1, Q头, 1, D]（decode 一位）

        # 离群程度：post-RoPE K 每个通道的整体量级 / 中位数
        chan = k.float().abs().amax(dim=(0, 1, 2))
        outlier = float(chan.max() / chan.median().clamp_min(1e-6))
        # GQA：第 j 个 Q 头对第 j // gq 个 KV 头
        gq_ = hq // h_kv
        k_flat = k[0].repeat_interleave(gq_, dim=0).float()
        scores = torch.bmm(q[0, :, 0].float().unsqueeze(1), k_flat.transpose(1, 2)).squeeze(1) * (d ** -0.5)

        packed = pack_int8_kv(k, v)
        kq, kmn, kst, vq, vmn, vst = packed
        kd = dequantize_int8(kq, kmn, kst, GROUP_T, "token")
        vd = dequantize_int8(vq, vmn, vst, GROUP_T, "token")
        kd = kd[:, :, None, :, :].expand(1, h_kv, gq_, h, d).reshape(1, hq, h, d)
        vd = vd[:, :, None, :, :].expand(1, h_kv, gq_, h, d).reshape(1, hq, h, d)
        with sdpa_kernel([SDPBackend.MATH]):
            want = F.scaled_dot_product_attention(q, kd, vd, scale=d ** -0.5)
        got = int8_attn_decode(q, *packed, used=h)

        if args.dump and layer == args.layers[-1]:
            torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu(),
                        "kd": kd.cpu(), "vd": vd.cpu()}, ROOT / ".tmp" / args.dump)
            print(f"  （已把第 {layer} 层的 q/k/v 存到 .tmp/{args.dump}）")

        # 真值对照：K/V 是 fp16 舍入后的值，fp64 里可精确表示 ⇒ fp64 算的就是**这次比较的真值**
        k64 = kd[0].double()
        v64 = vd[0].double()
        s64 = (q[0, :, 0].double()[:, None, :] @ k64.transpose(1, 2)).squeeze(1) * (d ** -0.5)
        p64 = torch.softmax(s64, dim=-1)
        st = ulp_stats(got, want)
        exact = (torch.bmm(p64.unsqueeze(1), v64).squeeze(1)).to(torch.float16).unsqueeze(0).unsqueeze(2)
        st_exact_k = ulp_stats(got, exact)
        st_exact_s = ulp_stats(want, exact)

        print(f"{layer:>4d} {outlier:>9.2f}x {scores.abs().max().item():>10.2f} "
              f"{st['max_ulp']:>8.2f} {st['gt2']:>6d} {st['gt1']:>6d} "
              f"{st['max_abs']:>9.2e} {st['exact']:>8.4%} │ "
              f"{st_exact_k['max_ulp']:>9.2f} {st_exact_s['max_ulp']:>11.2f}")
        print(f"     判据：超 2 ULP 且超 {ABS_TOL:g} 绝对阈值的元素 = {st['bad']} 个"
              f"（最大绝对差 {st['max_abs']:.2e}）⇒ {'通过' if st['bad'] == 0 else '不通过'}")

    print("\n核vs参考 = 融合核 vs「先还原再算」(SDPA math)｜核vs真值 / SDPAvs真值 = 各自与 fp64 真值的距离")
    print(f"判据（机制正确性）：核与真值的偏差不应**大于**参考与真值的偏差 · clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
