r"""int4 KV · 接线排查：模型**实际读到**的 K/V 是不是我们以为的那份。

`probe_kvquant_real.py` 发现一个反常：`residual=128` 的 `int4res` 在 15 token 的短上下文里
**一个位置都不该被量化**（`keep = used - 128 == 0`），输出却和 fp16 不一样。这说明
问题不在量化本身，而在**包装层/scratch 通路**。本脚本把这条通路拆开逐层比对。

三份 K/V 互相对照（取「模型实际读到的张量」= `update()` 的返回值）：

| 编号 | 内容 |
|---|---|
| ① base | `_base.key_cache[slot][:, :, :used]` —— 真正写进 cache 的原始 K/V |
| ② seen | `update()` 返回值 `[:, :, :used]` —— **模型注意力实际吃到的**（在 update 内当场比对） |
| ③ fp16 | 同一个 prompt 走原生 `StaticKVCache` 时的 ① |

⚠️ ②**必须在 `update()` 内当场比**：工作区是 36 层共用的一块，返回后再看已经被下一层覆盖了。

判据：
- ① == ③：写入没问题（fp16 与包装版的 cache 内容一致）
- ② == ①：scratch 通路没走样（`copy` 模式下必须逐位相等）
- logits 差：①②都对但 logits 还差 -> 问题在**别处**（掩码 / 槽位映射 / 层数）

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_kvquant_wiring.py
"""

from __future__ import annotations

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

from nova.cache import StaticKVCache  # noqa: E402
from nova.decode import GraphDecoder  # noqa: E402
from nova.kvquant import QuantRoundTripCache  # noqa: E402
from s4_memory_demo import load_bundle  # noqa: E402

PROMPT = "用一句话说明什么是潮汐。"


def make_cache(dec: GraphDecoder, kind: str):
    cfg = dec.cfg
    if kind in ("fp16", "fp16b"):
        return StaticKVCache(
            num_slots=dec.text.num_cache_layers, num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_len=dec.max_len,
        )
    return QuantRoundTripCache(
        num_slots=dec.text.num_cache_layers, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_len=dec.max_len,
        quant_k=kind != "int4v", quant_v=kind != "int4k",
        residual=1 << 30 if kind == "copy" else (128 if kind == "int4res" else 0),
    )


def run(dec: GraphDecoder, ids: torch.Tensor, kind: str):
    """prefill + 1 个解码步；返回 (logits, 每槽位「模型实际读到」的 k/v, used)。"""
    dec.cache = make_cache(dec, kind)
    diff = {"k": 0.0, "v": 0.0}  # ② seen vs ① base，当场比
    orig = dec.cache.update

    def rec(key, value, layer_idx):
        k, v = orig(key, value, layer_idx)
        used_now = int(dec.cache._base.pos.item()) + key.shape[2]
        bk = dec.cache.key_cache[layer_idx][:, :, :used_now, :].float()
        bv = dec.cache.value_cache[layer_idx][:, :, :used_now, :].float()
        diff["k"] = max(diff["k"], (k[:, :, :used_now].float() - bk).abs().max().item())
        diff["v"] = max(diff["v"], (v[:, :, :used_now].float() - bv).abs().max().item())
        return k, v

    if kind not in ("fp16", "fp16b"):
        dec.cache.update = rec
    dec.prefill(ids)
    logits = dec._body().float().clone()
    return logits, diff, int(dec.cache.pos.item()), dec.cache, int(dec.input_ids.item())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    nova, tok = load_bundle(1, "triton")
    ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True,
    ), return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    print(f"prompt {ids.shape[1]} token · {nova.model.num_cache_layers} 槽位 · "
          f"head_dim {nova.model.config.head_dim}")

    ref_logits, _, ref_used, ref_cache, ref_tok0 = run(GraphDecoder(nova, max_len=64), ids, "fp16")
    print(f"fp16：pos {ref_used} · input_ids {ref_tok0} · argmax {int(ref_logits.argmax())} · "
          f"logits 范围 [{ref_logits.min():.2f}, {ref_logits.max():.2f}]")

    for kind in ("fp16b", "copy", "int4k", "int4v", "int4"):
        dec = GraphDecoder(nova, max_len=64)
        logits, diff, used, cache, tok0 = run(dec, ids, kind)
        dlogit = (logits - ref_logits).abs().max().item()
        print(f"\n[{kind}] pos {used} · input_ids {tok0} · argmax {int(logits.argmax())} · "
              f"logits 最大差 {dlogit:.4f} · 首个 token 一致 "
              f"{int(logits.argmax()) == int(ref_logits.argmax())}")

        d_base_k = d_base_v = 0.0
        bad: list[int] = []
        for slot in range(len(cache.key_cache)):
            bk = cache.key_cache[slot][:, :, :used, :].float()
            bv = cache.value_cache[slot][:, :, :used, :].float()
            rk = ref_cache.key_cache[slot][:, :, :used, :].float()
            rv = ref_cache.value_cache[slot][:, :, :used, :].float()
            dk, dv = (bk - rk).abs().max().item(), (bv - rv).abs().max().item()
            d_base_k, d_base_v = max(d_base_k, dk), max(d_base_v, dv)
            if dk > 1e-3 or dv > 1e-3:
                bad.append(slot)
        tail = f"  ⚠️ 不一致槽位 {bad[:8]}" if bad else "  ✅ 全部一致"
        print(f"    ①写入 vs fp16：K {d_base_k:.5f} · V {d_base_v:.5f}{tail}")
        print(f"    ②模型读到 vs ①写入：K {diff['k']:.5f} · V {diff['v']:.5f}"
              f"   (copy 模式必须 0.00000；量化模式应 ≈ 半个 step)")
        if bad:
            prof = (cache.key_cache[bad[0]][:, :, :used, :].float()
                    - ref_cache.key_cache[bad[0]][:, :, :used, :].float()).abs()
            per_pos = prof.amax(dim=(0, 1, 3)).tolist()
            print(f"    槽位 {bad[0]} 的逐位置最大差："
                  + " ".join(f"{p:.1f}" for p in per_pos[:used]))


if __name__ == "__main__":
    main()
