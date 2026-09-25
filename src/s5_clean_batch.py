"""S5 批量清洗：语言、拒答、截断、思考段、角色边界、状态答案、工具调用。"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_RAW = REPO / "data/s5-samples/batch50-raw.jsonl"
DEFAULT_SCORED = REPO / "data/s5-samples/batch50-scored.jsonl"
DEFAULT_CLEAN = REPO / "data/s5-samples/batch50-clean.jsonl"
DEFAULT_REPORT = REPO / "reports/s5-batch50-cleaning.json"

CJK = re.compile(r"[\u4e00-\u9fff]")
REFUSAL_PATTERNS = [
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bi'm sorry\b",
    r"\bi am sorry\b",
    r"\bas an ai\b",
    r"\bi won't\b",
    r"\bi am unable\b",
    r"\bi'm unable\b",
    r"\bcannot assist\b",
    r"\bcan't help\b",
    r"\bunable to help\b",
]
THINKING_PATTERNS = [
    r"```(?:thinking|reasoning|thought)\b",
    r"\[(?:thinking|reasoning|thought)\]",
    r"<(?:thinking|reasoning|thought)>",
    r"\breasoning:\s",
    r"\banalysis:\s",
]
AI_DISCLAIMER_PATTERNS = [
    r"\bi am an ai\b",
    r"\bi'm an ai\b",
    r"\bas an ai\b",
]
ROLE_BOUNDARY_PATTERNS = [
    r"\b(?:buyer|dockmaster|officer|smuggler|stranger|woman|man)\b[^.\n]{0,80}\"[^\"]{2,}\"",
    r"\b(?:buyer|dockmaster|officer|smuggler|stranger|woman|man)\b[^.\n]{0,80}\u201c[^\u201d]{2,}\u201d",
]
TEMPLATE_PHRASES = [
    "didn't move",
    "didn't speak",
    "didn't look up",
    "didn't need to",
    "silence was thick",
    "like a stone",
    "like a thread",
    "like wet wool",
    "heavy with the weight",
    "breath caught",
    "knuckles white",
    "jaw tightened",
    "heart hammered",
    "a shiver ran",
    "the air was thick",
]
HARD_FLAGS = {
    "chinese",
    "empty_or_too_short",
    "refusal",
    "truncated",
    "thinking_leak",
    "role_boundary_suspect",
    "state_answer_missing",
    "tool_call_invalid",
    "tool_name_mismatch",
    "duplicate",
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"找不到输入：{path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def final_content(sample: dict[str, Any]) -> str:
    return str(sample["messages"][-1]["content"])


def check_sample(sample: dict[str, Any]) -> dict[str, Any]:
    content = final_content(sample)
    lower = content.lower()
    flags: list[str] = []
    details: dict[str, Any] = {"word_count": len(content.split())}

    cjk_count = len(CJK.findall(content))
    details["cjk_count"] = cjk_count
    if cjk_count:
        flags.append("chinese")
    min_words = {"rp": 40, "state": 2, "tool": 1}.get(str(sample.get("category")), 5)
    details["min_words"] = min_words
    if details["word_count"] < min_words:
        flags.append("empty_or_too_short")
    if any(re.search(pattern, lower) for pattern in REFUSAL_PATTERNS):
        flags.append("refusal")
    if any(re.search(pattern, lower) for pattern in AI_DISCLAIMER_PATTERNS):
        flags.append("ai_disclaimer")
    if any(re.search(pattern, lower) for pattern in THINKING_PATTERNS):
        flags.append("thinking_leak")
    if sample.get("meta", {}).get("finish_reason") != "stop":
        flags.append("truncated")

    template_hits = sum(lower.count(phrase) for phrase in TEMPLATE_PHRASES)
    details["template_hits"] = template_hits
    if template_hits >= 3:
        flags.append("template_heavy")

    category = sample.get("category")
    if category == "rp":
        if any(re.search(pattern, content) for pattern in ROLE_BOUNDARY_PATTERNS):
            flags.append("role_boundary_suspect")
    elif category == "state":
        expected = sample.get("expected") or []
        missing = [item for item in expected if str(item).lower() not in lower]
        details["state_missing"] = missing
        if missing:
            flags.append("state_answer_missing")
    elif category == "tool":
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
        if not match:
            flags.append("tool_call_invalid")
            details["tool_name"] = None
        else:
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                flags.append("tool_call_invalid")
                details["tool_name"] = None
            else:
                details["tool_name"] = payload.get("name")
                if payload.get("name") != sample.get("expected"):
                    flags.append("tool_name_mismatch")
            outside = (content[: match.start()] + content[match.end() :]).strip()
            details["tool_extra_text"] = outside
            if outside:
                flags.append("tool_extra_text")

    details["flags"] = flags
    details["hard_fail"] = any(flag in HARD_FLAGS for flag in flags)
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 S5 50 条批量样本")
    parser.add_argument("--raw", type=pathlib.Path, default=DEFAULT_RAW)
    parser.add_argument("--scored", type=pathlib.Path, default=DEFAULT_SCORED)
    parser.add_argument("--clean", type=pathlib.Path, default=DEFAULT_CLEAN)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    samples = read_jsonl(args.raw)
    seen: dict[str, str] = {}
    scored: list[dict[str, Any]] = []
    for sample in samples:
        content = final_content(sample)
        normalized = " ".join(content.split()).lower()
        details = check_sample(sample)
        if normalized in seen:
            details["flags"].append("duplicate")
            details["hard_fail"] = True
            details["duplicate_of"] = seen[normalized]
        else:
            seen[normalized] = sample["id"]
        scored.append({**sample, "quality": details})

    clean = [row for row in scored if not row["quality"]["hard_fail"]]
    write_jsonl(args.scored, scored)
    write_jsonl(args.clean, clean)

    flag_counts = Counter(
        flag for row in scored for flag in row["quality"]["flags"]
    )
    finish = Counter(str(row.get("meta", {}).get("finish_reason")) for row in scored)
    categories: dict[str, dict[str, int]] = {}
    for row in scored:
        category = str(row.get("category"))
        bucket = categories.setdefault(category, {"total": 0, "clean": 0})
        bucket["total"] += 1
        bucket["clean"] += int(not row["quality"]["hard_fail"])
    word_counts = [row["quality"]["word_count"] for row in scored]
    total_tokens = sum(
        (row.get("meta", {}).get("usage") or {}).get("total_tokens", 0) for row in scored
    )
    failures = [
        {
            "id": row["id"],
            "category": row.get("category"),
            "flags": row["quality"]["flags"],
            "excerpt": final_content(row)[:240],
        }
        for row in scored
        if row["quality"]["hard_fail"]
    ]
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "raw": str(args.raw),
        "total": len(scored),
        "clean": len(clean),
        "pass_rate": len(clean) / len(scored) if scored else 0.0,
        "by_category": categories,
        "finish_reason": dict(finish),
        "flag_counts": dict(flag_counts),
        "word_count": {
            "min": min(word_counts) if word_counts else None,
            "median": statistics.median(word_counts) if word_counts else None,
            "max": max(word_counts) if word_counts else None,
        },
        "total_tokens": total_tokens,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"total={report['total']} clean={report['clean']} pass_rate={report['pass_rate']:.1%}")
    print(f"by_category={categories}")
    print(f"finish_reason={dict(finish)}")
    print(f"flag_counts={dict(flag_counts)}")
    print(f"word_count={report['word_count']} total_tokens={total_tokens}")
    if failures:
        print("\nfailures:")
        for failure in failures[:20]:
            print(f"  {failure['id']} {failure['flags']} :: {failure['excerpt'][:120]!r}")
    print(f"\nscored={args.scored}\nclean={args.clean}\nreport={args.report}")


if __name__ == "__main__":
    main()
