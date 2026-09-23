r"""E1 · 记忆路由：段级索引该拿什么当代表？mean-K / top-‖K‖ / 注入帧旋转

## 为什么测这个

侧会话的两级索引构想：记忆不再是"拿 Q 跟全部记忆 token 打分"，而是
**每 256 token 一段、每段留几条代表 K** ⇒ 索引常驻显存、命中后再拉那一段。

- 100K token 记忆 = 391 段；**精确路径要扫 391×256×36×8×128×2 B = 7.2 GB**（显存放不下）。
- 代表 K 索引只有 **27 MB（mean）/ 111 MB（top-4）**，扫一遍是毫秒级。

代表 K 怎么选？侧会话的判断是"**用范数最大的几条，不要用平均**"。
但**这条路在同一个坐标系里才可比**：现有 `MemoryStore.scores()` 是"按**注入帧**旋转 Q/K"之后打分的，
而"pre-RoPE 的 Q·K"是非旋转的。所以本实验把这两件事分开量：

| 方法 | 说明 |
|---|---|
| `mean(pre-RoPE)` | 段内 K 平均，用 pre-RoPE 的 Q 打分（**不需要任何旋转**，最省） |
| `rot(mean-K)` | 段内 K 平均后，**整体按注入帧旋转**，用旋转后的 Q 打分 |
| `mean(rot K)` | 段内每个 K **先按自己的位置在注入帧里旋转**再平均（文献担心的"旋转互相抵消"就是这条） |
| `top4-‖K‖ max/mean` | 段内取 `‖k‖` 最大的 4 条（`‖k‖` 与旋转无关），pre-RoPE 打分 |
| `精确打分(现有)` | `MemoryStore.scores()`，作为**上限与代价基准** |

## 真值怎么定

**装着那条事实（号码）的那一段** —— 不用"精确打分的 argmax"当真值：实测它恒选第 0 段
（`scores()` 的常见分量把整段 softmax 质量拉到序列开头），那衡量的是常分量不是检索。

## 判据

**recall@4 ≥ 95%** 才考虑 adopt；索引 ≤ 150 MiB/100K、扫一遍 ≤ 1 ms。

跑法：
    & .\.venv\Scripts\python.exe -u src\diagnostics\probe_memory_routing.py --tokens 8192
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

from chatfmt import find_span, render  # noqa: E402
from exp_needle import NEEDLES, build  # noqa: E402
from nova.memory import MemoryStore, _rotate, capture_kv, query_vectors  # noqa: E402
from s4_memory_demo import load_bundle  # noqa: E402

SEG = 256


def clock_sm() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() + " MHz"
    except Exception:  # noqa: BLE001
        return "?"


def fact_segment(tok, hay_text: str, code: str, seg: int) -> int:
    """装着那条事实（号码）的是第几段 —— 这就是真值。"""
    return find_span(tok, hay_text, code)[0] // seg


def build_store(nova, hay_ids, seg: int, chunk: int) -> tuple[MemoryStore, int]:
    """分块抓 K/V 并切段写入 store。返回 `(store, 用到的 token 数)`。

    必须分块：一次抓 16K 会把 **Q** 也 materialize 出来（36×32×16384×128×2 = 4.8 GB），直接爆。
    """
    store = MemoryStore.for_model(nova)
    n = (int(hay_ids.shape[1]) // seg) * seg
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        k, v = capture_kv(nova.model, hay_ids, (c0, c1))
        for s in range(c0, c1, seg):
            a = s - c0
            store.add(k[:, :, a : a + seg, :], v[:, :, a : a + seg, :], label=f"seg{s // seg:03d}")
        del k, v
        torch.cuda.empty_cache()
    return store, n


def router_scores(q: torch.Tensor, reps: torch.Tensor, head_dim: int) -> torch.Tensor:
    """一次算完所有段的分数 -> `[S, m]`。

    GQA 必须按 `repeat_kv` 的同一映射分组（Q 头 32 / KV 头 8 = 4），
    用与 `MemoryStore.logits` 同一个 einsum 模式。
    """
    qf = q.float()
    kvh = reps.shape[2]
    groups = qf.shape[1] // kvh
    qg = qf.view(qf.shape[0], kvh, groups, qf.shape[2], qf.shape[3])
    out = torch.einsum("lhgtc,slhmc->slhgtm", qg, reps.float()) * (head_dim ** -0.5)
    return out.mean(dim=(1, 2, 4)).mean(dim=1)  # 层/头/t 平均 -> [S, m]


def rank_of(target: int, scores: torch.Tensor) -> int:
    """`target` 在这个分数里的名次（1 起）。"""
    return torch.argsort(scores, descending=True).tolist().index(target) + 1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="记忆路由：段级代表 K 的选法")
    ap.add_argument("--tokens", type=int, default=8192, help="干草堆目标 token 数")
    ap.add_argument("--chunk", type=int, default=2048, help="每次抓多少 token 的 K/V（太大爆 Q）")
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--seg", type=int, default=SEG)
    args = ap.parse_args()

    nova, tok = load_bundle(1, "triton")
    text = nova.model
    cfg = nova.model.config
    head_dim, hidden = int(cfg.head_dim), int(cfg.hidden_size)
    rotary = text.rotary_emb

    n_rounds = max(8, int(round(args.tokens / 36)))
    msgs, _at = build(n_rounds)
    hay_text = render(tok, msgs, add_generation_prompt=False)
    hay_ids = tok(hay_text, add_special_tokens=False)["input_ids"]
    hay_t = torch.tensor([hay_ids], device="cuda")
    h = int(hay_t.shape[1])

    print(f"Qwen3-VL-4B · 单通路 · 干草堆 {h} token · 段 {args.seg} · clocks.sm {clock_sm()}")
    t0 = time.perf_counter()
    store, used = build_store(nova, hay_t, args.seg, args.chunk)
    torch.cuda.synchronize()
    items = store.items
    n_seg = len(items)
    print(f"抓 K/V 并切段：{used} token → {n_seg} 段 · {time.perf_counter() - t0:.1f}s")

    # ---- 索引（一次性建好，之后每题只扫索引）----
    t0 = time.perf_counter()
    mean_r = torch.stack([it.k.float().mean(dim=2).unsqueeze(-2) for it in items])
    norm_r = torch.stack(
        [it.k.float()[:, :, it.k.float().norm(dim=-1).mean(dim=(0, 1)).topk(args.top_n).indices, :]
         for it in items]
    )
    rot_mean_r = torch.stack(
        [_rotate(it.k.float().mean(dim=2).unsqueeze(2), h, hidden, rotary).squeeze(2).unsqueeze(-2) for it in items]
    )
    mean_rot_r = torch.stack(
        [_rotate(it.k.float(), h, hidden, rotary).mean(dim=2).unsqueeze(-2) for it in items]
    )
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000
    r100 = 100000 // args.seg
    print(f"索引：mean {mean_r.numel() * 2 / 1024 ** 2:.2f} MiB · top{args.top_n} "
          f"{norm_r.numel() * 2 / 1024 ** 2:.2f} MiB ⇒ 外推 100K（{r100} 段）："
          f"{mean_r.numel() * 2 * r100 / n_seg / 1024 ** 2:.0f} / "
          f"{norm_r.numel() * 2 * r100 / n_seg / 1024 ** 2:.0f} MiB · 建索引 {build_ms:.0f} ms（含 4 种旋转）")

    segnorm = torch.tensor([float(it.k.float().norm(dim=-1).mean()) for it in items])
    print(f"\n段级平均 ‖K‖：中位 {float(segnorm.median()):.2f} · 最大/中位 "
          f"{float(segnorm.max()) / float(segnorm.median()):.2f}x · 最小/中位 "
          f"{float(segnorm.min()) / float(segnorm.median()):.2f}x（1.0x 附近 = 没有 sink 离群）")

    cols = ["mean(pre)", "rot(mean)", "mean(rot K)", f"top{args.top_n}-max", f"top{args.top_n}-mean", "精确"]
    print(f"\n{'问题':<12s} {'事实段':>6s} " + " ".join(f"{c:>11s}" for c in cols) + f" {'精确耗时':>9s}")
    rows = {c: [] for c in cols}
    t_exact = t_route = 0.0
    for name, code, _d in NEEDLES:
        qtext = f"我的{name}密码是多少？只回答那串号码。"
        cur_text = render(tok, [{"role": "user", "content": qtext}], add_generation_prompt=True)
        s0, s1 = find_span(tok, cur_text, qtext)
        cur_ids = tok(cur_text, add_special_tokens=False)["input_ids"]
        full = torch.tensor([hay_ids + cur_ids], device="cuda")
        q0 = h + s0

        t1 = time.perf_counter()
        q_pre = query_vectors(text, full, span=(q0, h + s1))
        q_rot = _rotate(q_pre, q0 + args.seg, hidden, rotary)
        item_starts = [h] * n_seg
        exact = store.scores(q_pre, item_starts, [q0 + it.n_tokens for it in items], rotary)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_exact += t2 - t1

        t3 = time.perf_counter()
        vals = {
            "mean(pre)": router_scores(q_pre, mean_r, head_dim).squeeze(-1),
            "rot(mean)": router_scores(q_rot, rot_mean_r, head_dim).squeeze(-1),
            "mean(rot K)": router_scores(q_rot, mean_rot_r, head_dim).squeeze(-1),
            f"top{args.top_n}-max": router_scores(q_pre, norm_r, head_dim).amax(dim=-1),
            f"top{args.top_n}-mean": router_scores(q_pre, norm_r, head_dim).mean(dim=-1),
            "精确": exact,
        }
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        t_route += t4 - t3

        want = fact_segment(tok, hay_text, code, args.seg)
        rk = {c: rank_of(want, vals[c]) for c in cols}
        for c in cols:
            rows[c].append(rk[c])
        print(f"{name:<12s} {want:>6d} " + " ".join(f"{rk[c]:>11d}" for c in cols)
              + f" {(t2 - t1) * 1000:>7.0f}ms")

    print()
    for c in cols:
        rk = rows[c]
        r1 = sum(1 for x in rk if x == 1) / len(rk)
        r4 = sum(1 for x in rk if x <= 4) / len(rk)
        mrr = sum(1.0 / x for x in rk) / len(rk)
        print(f"  {c:<14s} recall@1 {r1:.2f} · recall@4 {r4:.2f} · MRR {mrr:.3f}   名次 {rk}")

    print(f"\n代价：精确（现有路径）{t_exact / len(NEEDLES) * 1000:.0f} ms/题"
          f" vs 六个一起扫 {t_route / len(NEEDLES) * 1000:.2f} ms/题")
    print("判据：recall@4 ≥ 0.95 才 adopt；索引 ≤ 150 MiB/100K、扫一遍 ≤ 1 ms。")
    print(f"clocks.sm {clock_sm()}")


if __name__ == "__main__":
    main()
