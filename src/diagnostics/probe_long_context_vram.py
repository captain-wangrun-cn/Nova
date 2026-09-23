r"""长上下文显存拆解：到底是 KV cache、激活，还是注意力内核？

背景：`--lens 2048 4096` 实测峰值 **8.80 GiB**（> 显卡物理 8188 MiB，说明在往系统内存换页），
prefill 从 3.6s 掉到 29.1s；8192 直接崩。而按 KV cache 算（144 KiB/token × max_len 10240）
只该占 1.38 GiB —— 有 3 GiB 以上没归因。本脚本定位它。

三个问题：
1. SDPA 到底有没有 flash / mem-efficient 后端？（没有就会实体化 O(n²) 分数矩阵）
2. 峰值显存随长度怎么长？（翻倍长度配 ~2x = 线性；~4x = O(n²) 实体化）
3. cache / mask_table / 权重各占多少？

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_long_context_vram.py
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
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

from nova.decode import GraphDecoder  # noqa: E402
from s4_memory_demo import FILLER, load_bundle  # noqa: E402


def gib(x: float) -> float:
    return x / 1024 ** 3


def probe_peak(fn) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return gib(torch.cuda.max_memory_allocated())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--max-len", type=int, default=10240)
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    print("=== SDPA 后端可用性 ===")
    print(f"  flash        : {torch.backends.cuda.flash_sdp_enabled()}（可用 {torch.backends.cuda.is_flash_attention_available() if hasattr(torch.backends.cuda, 'is_flash_attention_available') else '?'}）")
    print(f"  mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  math         : {torch.backends.cuda.math_sdp_enabled()}")
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        print(f"  cudnn        : {torch.backends.cuda.cudnn_sdp_enabled()}")
    print(f"  torch {torch.__version__} · {torch.cuda.get_device_name(0)}")

    dec = GraphDecoder(nova, max_len=args.max_len)
    print(f"\n=== 固定占用（与序列长度无关）===")
    print(f"  权重（模型）      : {gib(torch.cuda.memory_allocated()):.2f} GiB")
    print(f"  KV cache ({args.max_len} 槽位): {dec.cache.nbytes() / 1024 ** 3:.2f} GiB"
          f"  = {dec.cache.nbytes() / args.max_len / 1024:.0f} KiB/token")
    # P0 之后掩码不再预分配整表：常驻只有一份 arange(int64)，掩码行(fp16)在图内即时构造
    arange_mib = dec._arange.numel() * dec._arange.element_size() / 1024 ** 2
    row_kib = dec.max_len * 2 / 1024
    old_gib = dec.max_len * dec.max_len * 2 / 1024 ** 3
    print(f"  mask（即时构造）  : 常驻 arange {arange_mib:.3f} MiB(int64)"
          f" + 图内掩码行 {row_kib:.1f} KiB(fp16)  —— 旧方案整表要 {old_gib:.2f} GiB")

    print(f"\n=== prefill 峰值显存随长度的增长 ===")
    print(f"{'token':>7s} {'峰值':>9s} {'比上一档':>9s}   （翻倍长度：2x=线性，4x=O(n^2)）")
    prev = None
    for n in args.lens:
        n_rounds = max(4, int(round(n / 36)))
        msgs = []
        for i in range(n_rounds):
            u, a = FILLER[i % len(FILLER)]
            msgs.append({"role": "user", "content": f"{u}（第 {i + 2} 轮）"})
            msgs.append({"role": "assistant", "content": a})
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False),
                  return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        if ids.shape[1] + 8 > args.max_len:
            print(f"{n:>7d}  —— 超过 max_len，跳过（实际 {ids.shape[1]}）")
            continue
        peak = probe_peak(lambda: dec.prefill(ids))
        ratio = f"{peak / prev:.2f}x" if prev else "—"
        print(f"{ids.shape[1]:>7d} {peak:>8.2f}G {ratio:>9s}")
        prev = peak

    print("\n=== 强制 flash 内核（若报错 = 当前配置用不了 flash）===")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": "你好" * 200}], tokenize=False, add_generation_prompt=False),
        return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    for name, backend in (("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                          ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
                          ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None))):
        if backend is None:
            continue
        try:
            with sdpa_kernel(backend):
                probe_peak(lambda: nova.model(input_ids=ids, cross_mode="off"))
            print(f"  {name:22s} 可用（{ids.shape[1]} token 前向成功）")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:22s} 不可用：{type(exc).__name__}: {str(exc)[:120]}")


if __name__ == "__main__":
    main()
