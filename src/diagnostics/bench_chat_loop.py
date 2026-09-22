r"""CLI 逐 token 循环的开销分解 —— 为什么交互式只有 ~30 tok/s，而基准是 71。

**区别在于同步点**：`bench_graph.py` 把 32 次 replay 连着发出去再同步一次，
CPU 跑在前面、GPU 一直有活干；CLI 每个 token 都要 `.item()` 取回来才能采样和打印，
于是每步都强制一次同步 —— CPU 与 GPU 串行，且中间的 CPU 空档会让 GPU 掉频。

逐级加负载，看每一步吃掉多少：

```
A  replay × n，最后同步一次     <- 基准口径（纯 GPU 吞吐）
B  A + 每步 .item()             <- 加上强制同步
C  B + sample_next()            <- 加上采样（top-k/top-p/multinomial）
D  C + copy_ + decode + print   <- 真实 CLI 循环
```

跑法：
```powershell
$env:HF_HOME='H:\Nova\.hf-cache'; $env:TMP='H:\Nova\.tmp'; $env:TEMP=$env:TMP
$env:HF_HUB_OFFLINE='1'; $env:TRITON_CACHE_DIR='H:\Nova\.tmp\triton-cache'
& .\.venv\Scripts\python.exe src\diagnostics\bench_chat_loop.py
```
"""

from __future__ import annotations

import io
import os
import sys
import time
from contextlib import redirect_stdout
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

PROMPT = "用一句话说明什么是潮汐。"
N = 32


class SampleArgs:
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    repetition_penalty = 1.05


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from nova.decode import GraphDecoder
    from nova.loader import build_nova, enable_lm_head_4bit, load_hf_base
    from transformers import AutoTokenizer

    from chat import sample_next

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    hf = load_hf_base()
    nova = build_nova(hf, norm_impl="triton", num_paths=1)
    enable_lm_head_4bit(nova)

    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")

    dec = GraphDecoder(nova, max_len=1024)
    dec.prefill(ids)
    dec.capture()

    vocab = int(nova.model.embed_tokens.weight.shape[0])
    seen_mask = torch.zeros(vocab, dtype=torch.bool, device="cuda")
    seen_mask.index_fill_(0, ids[0], True)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    args = SampleArgs()
    sink = io.StringIO()
    gen_ids: list[int] = []

    def timed(fn, n=N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(n)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    def stage_a(n):
        for _ in range(n):
            dec.step()

    def stage_b(n):
        for _ in range(n):
            dec.step()
            dec.input_ids.item()

    def stage_c(n):
        for _ in range(n):
            logits = dec.step()
            sample_next(logits[:, -1, :], args, seen_mask, gen).item()

    def stage_d(n):
        for _ in range(n):
            logits = dec.step()
            nxt = sample_next(logits[:, -1, :], args, seen_mask, gen)
            tok_id = int(nxt.item())
            dec.input_ids.copy_(nxt)
            seen_mask[tok_id] = True
            gen_ids.append(tok_id)
            piece = tok.decode(gen_ids, skip_special_tokens=True)
            print(piece, end="", flush=True, file=sink)

    rows = []
    for name, fn, desc in [
        ("A", stage_a, "replay × n，最后同步一次（基准口径）"),
        ("B", stage_b, "A + 每步 .item()（强制同步）"),
        ("C", stage_c, "B + 采样（top-k / top-p / multinomial）"),
        ("D", stage_d, "C + copy_ + decode + print（真实 CLI 循环）"),
    ]:
        with redirect_stdout(sink):
            ms = timed(fn)
        rows.append((name, desc, ms))
        print(f"{name}  {ms:6.2f} ms/token  {1000 / ms:6.1f} tok/s   {desc}")

    base = rows[0][2]
    print()
    print(f"逐级增量（相对 A = {base:.2f} ms）：")
    prev = base
    for name, desc, ms in rows[1:]:
        print(f"  {name}: +{ms - prev:5.2f} ms  ->  {ms:6.2f} ms/token")
        prev = ms
    print(f"\nCLI 循环比基准口径慢 {rows[-1][2] / base:.2f}x"
          f"（{1000 / base:.1f} -> {1000 / rows[-1][2]:.1f} tok/s）")


if __name__ == "__main__":
    main()
