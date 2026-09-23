r"""E3 · 8 位 KV：用**真实** K/V 量误差（int4 / int8 两个方向 / fp8 两个 scale 口径）。

回答：8 位到底比 int4 好多少，值不值得为它改选型？
判据（见 [reports/kv-quant-8bit.md](../../reports/kv-quant-8bit.md)）：
**K 相对 L2 最坏误差 ≤ int4 的一半**，记账落在 1.85–1.95x。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_kvquant_bits.py --len 2048
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

from chatfmt import render  # noqa: E402
from exp_needle import build  # noqa: E402
from nova.kvquant import GROUP_K, bytes_per_token, roundtrip  # noqa: E402
from s4_memory_demo import load_bundle  # noqa: E402

# 变体 -> (roundtrip 名, group_size)；fp16 只作参考
VARIANTS: dict[str, tuple[str, int]] = {
    "int4": ("int4", GROUP_K),
    "int8": ("int8", GROUP_K),
    "int8t": ("int8t", 0),
    "int8t64": ("int8t64", 0),
    "fp8": ("fp8", 0),
    "fp8t": ("fp8t", 0),
}

ACC_KEY = {"int4": "int4", "int8": "int8", "int8t": "int8t64", "int8t64": "int8t64",
           "fp8": "fp8", "fp8t": "fp8"}


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def err_stats(x: torch.Tensor, kind: str, group: int) -> dict[str, float]:
    """往返误差：相对 L2、每通道最坏相对误差、按层看的最坏相对 L2。"""
    x32 = x.float()
    xq = roundtrip(x, kind, group).float()
    err = (xq - x32).abs()
    chan = x32.abs().amax(dim=(0, 1, 2))            # 每个通道的整体量级
    chan_err = err.amax(dim=(0, 1, 2))
    per_layer = [
        float((xq[i] - x32[i]).norm() / x32[i].norm().clamp_min(1e-6))
        for i in range(x32.shape[0])
    ]
    return {
        "rel_l2": float((xq - x32).norm() / x32.norm()),
        "chan_worst": float((chan_err / chan.clamp_min(1e-6)).max()),
        "layer_worst": max(per_layer),
        "layer0": per_layer[0],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="8 位 KV 的真实误差与记账")
    ap.add_argument("--len", type=int, default=2048, help="抓多少 token 的真实 K/V")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    from nova.memory import capture_kv

    nova, tok = load_bundle(1, "triton")
    cfg = nova.model.config
    n_rounds = max(8, int(round(args.len / 36)))
    msgs, _at = build(n_rounds)
    text = render(tok, msgs, add_generation_prompt=False)
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"][: args.len]], device="cuda")
    h = int(ids.shape[1])

    # 分块抓（一次抓太大会把 Q 也 materialize 出来）
    ks, vs = [], []
    t0 = time.perf_counter()
    for c0 in range(0, h, args.chunk):
        k, v = capture_kv(nova.model, ids, (c0, min(c0 + args.chunk, h)))
        ks.append(k)
        vs.append(v)
        torch.cuda.empty_cache()
    k_all, v_all = torch.cat(ks, dim=2), torch.cat(vs, dim=2)
    torch.cuda.synchronize()
    print(f"Qwen3-VL-4B · 单通路 · 真实 K/V {h} token（{k_all.shape[0]} 层 × {k_all.shape[1]} KV头）"
          f" · 抓取 {time.perf_counter() - t0:.1f}s · clocks.sm {clock_sm()}")

    nslots, kvh, hd = nova.model.num_cache_layers, cfg.num_key_value_heads, cfg.head_dim
    acc = {
        "int4": bytes_per_token(nslots, kvh, hd),
        "int8": bytes_per_token(nslots, kvh, hd, bits=8),
        "int8t64": bytes_per_token(nslots, kvh, hd, bits=8, k_axis="token", token_group=64),
        "fp8": bytes_per_token(nslots, kvh, hd, bits="fp8"),
    }

    print(f"\n{'变体':>8s} {'K relL2':>9s} {'V relL2':>9s} {'K 通道最坏':>11s} {'K 第0层':>9s}"
          f" {'K 层最坏':>9s} {'记账':>7s} {'KiB/token':>10s}")
    rows = {}
    for name in args.variants:
        kind, group = VARIANTS[name]
        ek = err_stats(k_all, kind, group)
        ev = err_stats(v_all, kind, group)
        rows[name] = (ek, ev)
        a = acc[ACC_KEY[name]]
        kib = a.get("int8_kv", a.get("int4_kv")) / 1024
        print(f"{name:>8s} {ek['rel_l2']:>9.4f} {ev['rel_l2']:>9.4f} {ek['chan_worst']:>11.4f}"
              f" {ek['layer0']:>9.4f} {ek['layer_worst']:>9.4f} {a['ratio']:>6.2f}x {kib:>10.1f}")

    if "int4" in rows:
        k4 = rows["int4"][0]["rel_l2"]
        print(f"\n判据（K relL2 ≤ int4 的一半 = {k4 / 2:.4f}）：")
        for name in args.variants:
            if name == "int4":
                continue
            kk = rows[name][0]["rel_l2"]
            print(f"  {name:>8s} {kk:.4f}  {'达标' if kk <= k4 / 2 else '未达'}（int4/{name} = {k4 / kk:.1f}x）")
    print(f"\nclocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
