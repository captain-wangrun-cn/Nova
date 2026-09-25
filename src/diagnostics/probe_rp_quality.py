r"""D19 前置 · 英文 RP 输出的**机器可判指标**（主观评分仍然要人）。

## 这个脚本能回答什么、不能回答什么

**能**：把"明显弱"里**不需要人**的那部分挡掉 —— 复读、中途换语言、拒答、格式崩、
长度塌缩、EOS 太早。这些只要有一样大面积出现，就不必再花人力去评分了；
反过来它们全过，**也不等于**质量好（"读起来像不像人"只能人来判）。

**不能**：语气、人物一致性、有没有重复上一轮的词、英文是不是地道。这些留给人工评分。

指标定义（都由 `reports/baseline-outputs.jsonl` 现算，不引入外部判官模型 ——
用模型当判官会把"教师偏好"混进基座选型里）：

| 指标 | 怎么看 |
|---|---|
| 词数 / 字符数 | 太短说明塌缩 |
| `stopped` | `eos` = 自己收住了；`max_new_tokens` = 被截断（RP 长文常见，未必是坏事） |
| 复读率 = 最长重复 n-gram 占比 | RP 最常见的崩法 |
| 中文字符占比 | 英文 prompt 下混中文 = 语言串味（D18 先英后中，这会让训练数据更难做） |
| 拒答关键词 | 命中就是硬伤 |

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_rp_quality.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "reports" / "baseline-outputs.jsonl"

_REFUSE = ("i can't", "i cannot", "i'm unable", "as an ai", "i won't", "cannot assist",
           "i'm sorry, but", "against my", "not appropriate")
_CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def repeat_ratio(text: str, n: int = 8) -> float:
    """最长重复 n-gram 的**覆盖占比**：把重复出现的 n-gram 命中的词数 / 总词数。"""
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    hit = 0
    for g, c in grams.items():
        if c > 1:
            hit += (c - 1) * n
    return hit / len(words)


def analyse(row: dict) -> dict:
    # ⚠️ 字段名是 `completion`（不是 `output`）：写错会得到"全是 0 词"的假通过 —— 踩过。
    text = row.get("completion") or ""
    if not text:
        raise ValueError(f"{row.get('id')} 的 completion 是空的 —— 字段名或数据有问题，别当成通过")
    words = text.split()
    cjk = len(_CJK.findall(text))
    low = text.lower()
    return {
        "id": row.get("id"),
        "tag": row.get("tag"),
        "label": row.get("label"),
        "stopped": row.get("stopped"),
        "words": len(words),
        "chars": len(text),
        "cjk_ratio": cjk / max(len(text), 1),
        "repeat8": repeat_ratio(text, 8),
        "refusal": [k for k in _REFUSE if k in low],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rp = [r for r in rows if str(r.get("id", "")).startswith("rp-")]
    if not rp:
        print(f"{args.path} 里没有 rp-* 行")
        return 1

    print(f"{args.path} · {len(rp)} 条英文 RP 输出\n")
    print(f"{'id':>7s} {'tag':>13s} {'stopped':>15s} {'词数':>6s} {'字符':>6s} "
          f"{'中文占比':>9s} {'复读(8-gram)':>12s}  拒答")
    worst = {"repeat8": 0.0, "cjk_ratio": 0.0}
    for r in rp:
        a = analyse(r)
        worst["repeat8"] = max(worst["repeat8"], a["repeat8"])
        worst["cjk_ratio"] = max(worst["cjk_ratio"], a["cjk_ratio"])
        flag = ",".join(a["refusal"]) or "-"
        print(f"{a['id']:>7s} {a['tag']:>13s} {a['stopped']:>15s} {a['words']:6d} "
              f"{a['chars']:6d} {a['cjk_ratio']:8.3%} {a['repeat8']:11.1%}  {flag}")

    print(f"\n最差：复读 {worst['repeat8']:.1%} · 中文占比 {worst['cjk_ratio']:.3%}")
    print("判据（机器可判部分，任意一条不满足就该先查原因再谈换基座）：")
    ok_rep = worst["repeat8"] < 0.20
    ok_cjk = worst["cjk_ratio"] < 0.01
    print(f"  · 复读率 < 20%：{'✅' if ok_rep else '❌'}（实 {worst['repeat8']:.1%}）")
    print(f"  · 英文 prompt 下中文占比 < 1%：{'✅' if ok_cjk else '❌'}（实 {worst['cjk_ratio']:.3%}）")
    print("\n⚠️ 这些指标全过 **不等于** RP 质量够用 —— 语气 / 人物一致性 / 英文地道度只能人读。")
    print("   机器的结论只能到这一步：'没有明显崩坏'。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
