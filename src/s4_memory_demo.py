r"""S4 · 记忆最小实现演示：写入 -> 20 轮后取回 -> 存盘重启后取回。

**两个进程跑**（"重启后仍能取回"的证据必须跨进程）：

```powershell
$env:HF_HOME=(Resolve-Path .).Path+'\.hf-cache'; $env:TMP=(Resolve-Path .).Path+'\.tmp'; $env:TEMP=$env:TMP
$env:HF_HUB_OFFLINE='1'; $env:TRITON_CACHE_DIR=$env:TMP+'\triton-cache'; $env:TORCHINDUCTOR_CACHE_DIR=$env:TMP+'\inductor-cache'

& .\.venv\Scripts\python.exe src\s4_memory_demo.py --phase write --mem .tmp\s4-memory\demo.safetensors
& .\.venv\Scripts\python.exe src\s4_memory_demo.py --phase ask   --mem .tmp\s4-memory\demo.safetensors
```

`--phase all` 在一个进程里全跑（只看流程时用）；报告里的"重启"证据用的是上面两条命令。

**流程**：第 1 轮用户说了 6 件事 -> 抓成 6 条记忆（K/V 张量）存盘 -> **第 1 轮从可见历史里去掉**
（= 压缩进记忆，不再占 KV cache）-> 20 轮无关闲聊 -> 第 22 轮提问 -> 看答案。

每个问题跑三种条件：

| 条件 | 说明 |
|------|------|
| `auto` | 自动检索 top-1 并注入 |
| `off`  | 不注入（对照：没有记忆就该答不上来） |
| `swap` | 故意注入**另一条**记忆（证明答案跟着注入内容走，不是模型自己猜的） |
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))

import torch  # noqa: E402

from chatfmt import EOS_ID, EOS_ID_ALT, find_span, render  # noqa: E402

# ---- 场景：第 1 轮说的 6 件事（写入这些，然后把这一轮从历史里去掉）----
FACTS = [
    ("红裙子", "我今天换了一条红裙子，是上周在巴黎买的。"),
    ("橘猫", "我养的猫叫雷纳德，是一只橘猫。"),
    ("花瓶", "我把阳台上的蓝色花瓶打碎了。"),
    ("咖啡", "我最近改喝无糖黑咖啡了。"),
    ("出差", "我下周三要去成都出差三天。"),
    ("吉他", "我小时候学过两年古典吉他。"),
]

# ---- 20 轮无关闲聊（脚本给定：条件之间唯一的差别就是"注入了哪条记忆"）----
FILLER = [
    ("今天天气怎么样？", "今天多云转晴，风不大。"),
    ("三加五等于几？", "三加五等于八。"),
    ("你会下棋吗？", "会一点，但不擅长。"),
    ("推荐一部电影吧。", "可以看《星际穿越》。"),
    ("现在几点了？", "我没有时钟，看不到时间。"),
]

# ---- 第 22 轮的提问（含 3 个改写问法，不给词面线索）----
# 判分：答案里必须**同时**出现全部关键词才算对（单看"咖啡"会被"喝得更频繁"这种泛泛而谈蒙对）
QUESTIONS = [
    ("我今天穿的裙子是什么颜色的？", "红裙子", ("红", "裙子")),
    ("我上周在哪里买的衣服？", "红裙子", ("巴黎",)),
    ("我养的猫叫什么名字？", "橘猫", ("雷纳德",)),
    ("我家的宠物是什么品种？", "橘猫", ("橘猫",)),
    ("我把什么东西打碎了？", "花瓶", ("花瓶",)),
    ("我最近喝咖啡有什么变化？", "咖啡", ("无糖",)),
    ("我下周要去哪里出差？", "出差", ("成都",)),
    ("我小时候学过什么乐器？", "吉他", ("古典",)),
]
STOP_IDS = {EOS_ID, EOS_ID_ALT}


class _Tee:
    """把 stdout 同时写进日志文件（Windows 下用 PowerShell 管道会乱编码，所以自己写）。"""

    def __init__(self, path: str) -> None:
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> int:
        sys.__stdout__.write(text)
        self.fh.write(text)
        return len(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.fh.flush()


def build_filler(turns: int) -> list[dict]:
    msgs = []
    for i in range(turns):
        user, assistant = FILLER[i % len(FILLER)]
        msgs.append({"role": "user", "content": f"{user}（第 {i + 2} 轮）"})
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def load_bundle(paths: int, norm: str):
    from nova.loader import build_nova, enable_lm_head_4bit, load_hf_base
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    hf = load_hf_base()
    nova = build_nova(hf, norm_impl=norm, num_paths=paths)
    enable_lm_head_4bit(nova)
    return nova, tok


def cmd_write(args) -> None:
    from nova.memory import MemorySession, MemoryStore

    nova, tok = load_bundle(args.paths, args.norm)
    store = MemoryStore.for_model(nova)
    sess = MemorySession(store, nova, max_len=args.max_len)

    text = render(tok, [{"role": "user", "content": " ".join(f[1] for f in FACTS)}], add_generation_prompt=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    print(f"[写入] 第 1 轮 prompt {ids.shape[1]} token（{len(FACTS)} 条事实）")
    for label, sentence in FACTS:
        span = find_span(tok, text, sentence)
        sess.write(ids, span, label=label)
        print(f"       [{len(store) - 1}] {label:5s} token {span} -> {store.items[-1].n_tokens} 个位置")
    path = store.save(args.mem)
    print(f"[写入] {store}")
    print(f"[写入] 存盘 -> {path}（{path.stat().st_size / 1024 ** 2:.2f} MiB）")
    print(f"[写入] 表示空间 schema={store.schema.digest[:12]} 模型指纹={store.fingerprint[:12]}")


def cmd_ask(args) -> None:
    from nova.memory import MemorySchema, MemorySession, MemoryStore, model_fingerprint

    nova, tok = load_bundle(args.paths, args.norm)
    store = MemoryStore.load(
        args.mem,
        schema=MemorySchema.for_model(nova),
        fingerprint=model_fingerprint(nova),
    )
    print(f"[读取] 载入 {store}  （schema={store.schema.digest[:12]} 指纹={store.fingerprint[:12]} 外来={store.foreign}）")
    sess = MemorySession(store, nova, max_len=args.max_len, top_k=args.top_k)

    filler = build_filler(args.turns)
    hist_ids = tok(render(tok, filler, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    hist_t = torch.tensor([hist_ids], device="cuda")
    print(f"[读取] 可见历史 {len(hist_ids)} token = {args.turns} 轮无关对话（第 1 轮已被记忆替代）")
    print(f"[读取] 记忆插入位置：当前轮之前（place={args.place}）")

    n_hit = 0
    for qi, (qtext, want, accept) in enumerate(QUESTIONS, 1):
        full_text = render(tok, filler + [{"role": "user", "content": qtext}], add_generation_prompt=True)
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(hist_ids)] == hist_ids, "历史不是前缀（模板变了？）"
        qspan = find_span(tok, full_text, qtext)
        cur_t = torch.tensor([full_ids[len(hist_ids) :]], device="cuda")

        print(f"\n=== 第 {args.turns + 2} 轮（写入后第 {args.turns} 轮）问题 {qi}/{len(QUESTIONS)}：{qtext}")
        print(f"    期望：{want}")
        for cond in ("auto", "off", "swap"):
            force = None
            if cond == "swap":
                force = ([f[0] for f in FACTS].index(want) + 1) % len(FACTS)
            t0 = time.perf_counter()
            info = sess.prefill(
                hist_t, cur_t,
                use_memory=(cond != "off"),
                force_index=force,
                query_span=qspan,
                place=args.place,
            )
            out = sess.generate(args.max_new, STOP_IDS)
            dt = time.perf_counter() - t0
            text = tok.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
            ok = all(k in text for k in accept)
            if cond == "auto":
                n_hit += int(ok)
                rank = ""
                if info.scores is not None:
                    order = torch.argsort(info.scores, descending=True)
                    rank = " 打分排序 " + " > ".join(
                        f"{store.items[int(i)].label}({float(info.scores[int(i)]):.4f})" for i in order
                    )
                print(
                    f"    [auto] 注入 {info.prefix_len} token · 检索 {info.query_ms:.0f}+{info.score_ms:.0f}ms"
                    f" + prefill {info.prefill_ms:.0f}ms{rank}"
                )
            else:
                inj = "—" if cond == "off" else f"{store.items[force].label}"
                print(f"    [{cond:4s}] 注入 {info.prefix_len} token（{inj}）")
            print(f"           {'✓' if ok else '✗'} {text}   [{dt:.2f}s]")

    print(f"\n[汇总] auto 条件下答案正确 {n_hit}/{len(QUESTIONS)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Nova S4 记忆最小实现演示")
    ap.add_argument("--phase", choices=["write", "ask", "all"], default="all")
    ap.add_argument("--mem", default=str(ROOT / ".tmp" / "s4-memory" / "demo.safetensors"))
    ap.add_argument("--turns", type=int, default=20, help="写入之后插入多少轮无关对话")
    ap.add_argument("--paths", type=int, default=2, choices=(1, 2))
    ap.add_argument("--norm", choices=["exact", "triton"], default="triton")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--place", choices=["turn", "front"], default="turn")
    ap.add_argument("--log", default=None, help="同时把输出写进这个文件（UTF-8，报告证据用）")
    args = ap.parse_args()

    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(args.log)

    if args.phase in ("write", "all"):
        cmd_write(args)
    if args.phase in ("ask", "all"):
        cmd_ask(args)


if __name__ == "__main__":
    main()
