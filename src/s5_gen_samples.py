"""S5 数据管线：用教师 API 生成英文 RP 对话样本。

当前只做最小可用版本：
- 读 `H:\\Nova\\.secrets\\gemini.key`（或环境变量 `GEMINI_API_KEY`）；
- 走 OpenAI 兼容端点 `{base_url}/v1/chat/completions`；
- 生成 5 条英文 RP 样本，清洗思考段，写 JSONL。

**不要**把 key 打进命令行或日志；本脚本只从文件/环境变量读。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://newapi.wr.wstudio.work"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_KEY_PATH = Path(r"H:\Nova\.secrets\gemini.key")
DEFAULT_OUT = Path(r"H:\Nova\data\s5-samples\gemini-3.1-flash-lite.jsonl")

SYSTEM_PROMPT = (
    "You are Elara Voss, a character in an ongoing roleplay. "
    "Stay in character at all times. Write in third-person past tense about Elara only. "
    "Never write the other characters' dialogue, thoughts, or actions. "
    "Favour concrete sensory detail over abstract summary."
)

PROMPTS: list[dict[str, str]] = [
    {
        "id": "s5-rp-01",
        "label": "港口雨夜开场",
        "user": (
            "Elara, the dockmaster says we still owe him for last week. "
            "He has two men with him and he is blocking the ramp."
        ),
    },
    {
        "id": "s5-rp-02",
        "label": "坏消息与情绪压力",
        "user": (
            "Elara, the manifest you signed was not for machine parts. "
            "Kessler is dead, and his name is on the last page."
        ),
    },
    {
        "id": "s5-rp-03",
        "label": "讨价还价的谈判",
        "user": (
            "The buyer wants to renegotiate. He says the cargo arrived wet and he is "
            "offering forty percent of the agreed price. He is smiling while he says it."
        ),
    },
    {
        "id": "s5-rp-04",
        "label": "风暴后的伤口",
        "user": (
            "Elara, the storm has passed. You are in the loft above the warehouse, and "
            "your shoulder is bleeding through the bandage. Someone is knocking on the "
            "door downstairs."
        ),
    },
    {
        "id": "s5-rp-05",
        "label": "旧人归来",
        "user": (
            "Elara, a woman you thought was dead is standing in the doorway. She is "
            "holding the locket you buried with her."
        ),
    },
]

_THINK_BLOCK = re.compile(
    r"```(?:thinking|reasoning|thought)\b.*?```", re.IGNORECASE | re.DOTALL
)
_THINK_TAG = re.compile(
    r"\[(thinking|reasoning|thought)\].*?\[/\1\]", re.IGNORECASE | re.DOTALL
)


def load_api_key(key_path: Path) -> str:
    """优先读本地文件，其次读环境变量；绝不把 key 写进日志。"""
    if key_path.exists():
        key = key_path.read_text(encoding="utf-8-sig").strip()
        if key:
            return key
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            f"找不到 API key：{key_path} 不存在，且环境变量 GEMINI_API_KEY 为空。"
        )
    return key


def clean_completion(text: str) -> str:
    """去掉常见的思考段；只保留最终回答。"""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_TAG.sub("", text)
    return text.strip()


def generate_one(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = client.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    content = clean_completion(choice["message"].get("content") or "")
    meta = {
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "model": body.get("model", model),
    }
    return content, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 S5 英文 RP 样本")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=len(PROMPTS))
    args = parser.parse_args()

    api_key = load_api_key(args.key_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for i, prompt in enumerate(PROMPTS[: args.limit]):
            completion, meta = generate_one(
                client,
                base_url=args.base_url,
                model=args.model,
                api_key=api_key,
                system=SYSTEM_PROMPT,
                user=prompt["user"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            samples.append(
                {
                    "id": prompt["id"],
                    "label": prompt["label"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt["user"]},
                        {"role": "assistant", "content": completion},
                    ],
                    "meta": {
                        **meta,
                        "teacher": args.model,
                        "endpoint": f"{args.base_url.rstrip('/')}/v1/chat/completions",
                        "created": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
            print(
                f"[{i + 1}/{args.limit}] {prompt['id']} "
                f"finish={meta['finish_reason']} words={len(completion.split())}"
            )
            if i + 1 < args.limit:
                time.sleep(0.5)

    with args.out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total_tokens = sum(
        (s["meta"].get("usage") or {}).get("total_tokens", 0) for s in samples
    )
    non_ascii = [
        s["id"]
        for s in samples
        if any(ord(ch) > 127 for ch in s["messages"][2]["content"])
    ]
    print(f"\nwrote {len(samples)} samples -> {args.out}")
    print(f"total_tokens={total_tokens} non_ascii={non_ascii or 'none'}")


if __name__ == "__main__":
    main()
