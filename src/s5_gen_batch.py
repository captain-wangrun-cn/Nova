"""S5 批量数据生成：用 API 教师生成 50 条英文样本。

结构：
- RP / 对话 30 条（10 个场景 × 3 个变体）
- 状态跟踪 10 条（多轮，带客观 expected）
- 工具调用 10 条（要求输出 `<tool_call>` JSON）

输出：`data/s5-samples/batch50-raw.jsonl`（可断点续跑，已生成的 id 会跳过）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from s5_gen_samples import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_KEY_PATH,
    DEFAULT_MODEL,
    generate_messages,
    load_api_key,
)

DEFAULT_OUT = REPO / "data/s5-samples/batch50-raw.jsonl"

SYSTEM_RP = (
    "You are Elara Voss, a character in an ongoing roleplay. "
    "Stay in character at all times. Write in third-person past tense about Elara only. "
    "Never write the other characters' dialogue, thoughts, or actions. "
    "Favour concrete sensory detail over abstract summary."
)

SYSTEM_STATE = (
    "You are a careful assistant. Track the user's state across turns. "
    "Answer the latest question using the conversation so far. Be concise."
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string"},
            },
            "required": ["title", "start_time"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a local file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "time": {"type": "string"},
            },
            "required": ["message", "time"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between currencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text into a target language.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_language": {"type": "string"},
            },
            "required": ["text", "target_language"],
        },
    },
]

SYSTEM_TOOL = (
    "You have access to the following tools. To call a tool, output exactly one "
    "<tool_call> block containing JSON with keys `name` and `arguments`. "
    "Do not add any other text.\nTools: " + json.dumps(TOOL_SCHEMAS)
)

RP_SCENARIOS: list[tuple[str, str]] = [
    (
        "dockmaster",
        "Elara, the dockmaster says we still owe him for last week. "
        "He has two men with him and he is blocking the ramp.",
    ),
    (
        "manifest",
        "Elara, the manifest you signed was not for machine parts. "
        "Kessler is dead, and his name is on the last page.",
    ),
    (
        "buyer",
        "The buyer wants to renegotiate. He says the cargo arrived wet and he is "
        "offering forty percent of the agreed price. He is smiling while he says it.",
    ),
    (
        "storm",
        "Elara, the storm has passed. You are in the loft above the warehouse, and "
        "your shoulder is bleeding through the bandage. Someone is knocking on the "
        "door downstairs.",
    ),
    (
        "locket",
        "Elara, a woman you thought was dead is standing in the doorway. She is "
        "holding the locket you buried with her.",
    ),
    (
        "customs",
        "Elara, a customs officer is at the gate with a search warrant. He says he "
        "wants to inspect every crate in the warehouse.",
    ),
    (
        "missing",
        "Elara, the sealed container is empty. The manifest says it should hold forty "
        "rifles, and the buyer is waiting on the dock.",
    ),
    (
        "rival",
        "Elara, a rival smuggler offers you a deal: he will clear your debt if you "
        "hand over the harbor ledger.",
    ),
    (
        "letter",
        "Elara, you find a letter in Kessler's handwriting. It has your name on it, "
        "and the seal is already broken.",
    ),
    (
        "bell",
        "Elara, the harbor bell starts ringing. A storm is coming in fast, and the "
        "last cargo boat is still outside the breakwater.",
    ),
]

STATE_CASES: list[dict[str, Any]] = [
    {
        "slug": "coat",
        "history": [
            ("user", "I'm wearing a blue coat and carrying a black umbrella."),
            ("assistant", "Got it."),
        ],
        "question": "What color is my coat?",
        "expected": ["blue"],
    },
    {
        "slug": "meeting",
        "history": [
            ("user", "My meeting is at 3:30 PM in room 204."),
            ("assistant", "Got it."),
        ],
        "question": "What time is my meeting?",
        "expected": ["3:30"],
    },
    {
        "slug": "dog",
        "history": [
            ("user", "My dog's name is Milo and he is a beagle."),
            ("assistant", "Got it."),
        ],
        "question": "What kind of dog is Milo?",
        "expected": ["beagle"],
    },
    {
        "slug": "allergy",
        "history": [
            ("user", "I'm allergic to peanuts and shellfish."),
            ("assistant", "Got it."),
        ],
        "question": "Which foods should I avoid?",
        "expected": ["peanuts", "shellfish"],
    },
    {
        "slug": "osaka",
        "history": [
            ("user", "I'm flying to Osaka on Friday."),
            ("assistant", "Got it."),
        ],
        "question": "Where am I going on Friday?",
        "expected": ["osaka"],
    },
    {
        "slug": "safe",
        "history": [
            ("user", "The safe code is 4821."),
            ("assistant", "Got it."),
        ],
        "question": "What is the safe code?",
        "expected": ["4821"],
    },
    {
        "slug": "shift",
        "history": [
            ("user", "I work the night shift on Tuesday and Thursday."),
            ("assistant", "Got it."),
        ],
        "question": "Which nights do I work?",
        "expected": ["tuesday", "thursday"],
    },
    {
        "slug": "book",
        "history": [
            ("user", "I'm reading a book called The Silent Harbor."),
            ("assistant", "Got it."),
        ],
        "question": "What book am I reading?",
        "expected": ["silent harbor"],
    },
    {
        "slug": "seat",
        "history": [
            ("user", "My seat is 14C on the train."),
            ("assistant", "Got it."),
        ],
        "question": "What seat am I in?",
        "expected": ["14c"],
    },
    {
        "slug": "oven",
        "history": [
            ("user", "The oven should be set to 180 degrees Celsius."),
            ("assistant", "Got it."),
        ],
        "question": "What temperature should the oven be?",
        "expected": ["180"],
    },
]

TOOL_CASES: list[tuple[str, str, str]] = [
    ("weather", "What's the weather in Shanghai right now?", "get_weather"),
    (
        "calendar",
        "Schedule a meeting called 'S5 review' for tomorrow at 10:00.",
        "create_calendar_event",
    ),
    (
        "email",
        "Send an email to ana@example.com with subject 'Status' and body 'Pipeline works.'",
        "send_email",
    ),
    (
        "search",
        "Search the web for the latest Qwen3-VL release notes.",
        "search_web",
    ),
    ("calculate", "Calculate 48 * 17 + 5.", "calculate"),
    ("read_file", "Read the file /data/notes.txt.", "read_file"),
    ("reminder", "Remind me to call my sister at 19:30.", "set_reminder"),
    ("currency", "Convert 100 USD to CNY.", "convert_currency"),
    ("stock", "What's the current stock price of NVDA?", "get_stock_price"),
    (
        "translate",
        "Translate 'good morning' into Japanese.",
        "translate_text",
    ),
]


def build_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for i, (slug, user) in enumerate(RP_SCENARIOS, 1):
        for variant in range(1, 4):
            specs.append(
                {
                    "id": f"s5b-rp-{i:02d}-v{variant}",
                    "category": "rp",
                    "messages": [
                        {"role": "system", "content": SYSTEM_RP},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.8,
                    "max_tokens": 512,
                }
            )
    for i, case in enumerate(STATE_CASES, 1):
        messages = [{"role": "system", "content": SYSTEM_STATE}]
        for role, content in case["history"]:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": case["question"]})
        specs.append(
            {
                "id": f"s5b-state-{i:02d}",
                "category": "state",
                "messages": messages,
                "expected": case["expected"],
                "temperature": 0.2,
                "max_tokens": 128,
            }
        )
    for i, (slug, user, expected) in enumerate(TOOL_CASES, 1):
        specs.append(
            {
                "id": f"s5b-tool-{i:02d}",
                "category": "tool",
                "messages": [
                    {"role": "system", "content": SYSTEM_TOOL},
                    {"role": "user", "content": user},
                ],
                "expected": expected,
                "temperature": 0.1,
                "max_tokens": 256,
            }
        )
    return specs


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def generate_with_retry(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = 4,
) -> tuple[str, dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return generate_messages(
                client,
                base_url=base_url,
                model=model,
                api_key=api_key,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt + 1 >= retries:
                break
            wait = 2**attempt
            print(f"  retry {attempt + 1}/{retries - 1} after {wait}s ({status or type(exc).__name__})")
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="S5 批量生成 50 条英文样本")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-path", type=pathlib.Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    specs = build_specs()
    api_key = load_api_key(args.key_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.out)
    done = {s["id"] for s in existing}
    todo = [s for s in specs if s["id"] not in done]
    print(f"specs={len(specs)} existing={len(existing)} todo={len(todo)}")

    generated = 0
    with httpx.Client(timeout=120.0, follow_redirects=True) as client, args.out.open(
        "a", encoding="utf-8"
    ) as f:
        for i, spec in enumerate(specs, 1):
            if spec["id"] in done:
                print(f"[{i}/{len(specs)}] {spec['id']} skip")
                continue
            content, meta = generate_with_retry(
                client,
                base_url=args.base_url,
                model=args.model,
                api_key=api_key,
                messages=spec["messages"],
                temperature=spec["temperature"],
                max_tokens=spec["max_tokens"],
            )
            sample = {
                "id": spec["id"],
                "category": spec["category"],
                "messages": spec["messages"] + [{"role": "assistant", "content": content}],
                "expected": spec.get("expected"),
                "meta": {
                    **meta,
                    "teacher": args.model,
                    "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
                    "created": datetime.now(timezone.utc).isoformat(),
                    "temperature": spec["temperature"],
                    "max_tokens": spec["max_tokens"],
                },
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1
            print(
                f"[{i}/{len(specs)}] {spec['id']} finish={meta['finish_reason']} "
                f"words={len(content.split())}"
            )
            time.sleep(args.sleep)

    all_samples = read_jsonl(args.out)
    finish: dict[str, int] = {}
    total_tokens = 0
    for sample in all_samples:
        reason = str(sample.get("meta", {}).get("finish_reason"))
        finish[reason] = finish.get(reason, 0) + 1
        total_tokens += (sample.get("meta", {}).get("usage") or {}).get("total_tokens", 0)
    print(f"\ngenerated_this_run={generated} total={len(all_samples)}")
    print(f"finish_reason={finish} total_tokens={total_tokens}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
