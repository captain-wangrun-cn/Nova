"""Nova · chat 模板与编码的唯一入口。

S1 验收（HANDOFF.md 第四节）：所有数据合成（S5）与推理（S2/S3/S4）都必须走本模块，
不允许在别处另写一套 prompt 拼接。

约束：只依赖 transformers；纯文本路径（视觉 processor 需要 torchvision，S1 不装）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DEFAULT_REPO = "Qwen/Qwen3-VL-4B-Instruct"

# --- 已核查常量（tokenizer_config.json / config.json，2026-09-21 实测） --------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
VISION_PAD = "<|vision_pad|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

# id 来自 tokenizer_config.json 的 added_tokens_decoder
SPECIAL_TOKEN_IDS: dict[str, int] = {
    END_OF_TEXT: 151643,
    IM_START: 151644,
    IM_END: 151645,
    VISION_START: 151652,
    VISION_END: 151653,
    VISION_PAD: 151654,
    IMAGE_PAD: 151655,
    VIDEO_PAD: 151656,
    TOOL_CALL_START: 151657,
    TOOL_CALL_END: 151658,
}
SPECIAL_TOKEN_ID_SET = frozenset(SPECIAL_TOKEN_IDS.values())
IM_START_ID = SPECIAL_TOKEN_IDS[IM_START]
IM_END_ID = SPECIAL_TOKEN_IDS[IM_END]

# 生成终止符：eos_token_id=151645 (<|im_end|>)；151643 (<|endoftext|>) 也在 eos 集合里
EOS_ID = IM_END_ID
EOS_ID_ALT = SPECIAL_TOKEN_IDS[END_OF_TEXT]

VALID_ROLES = ("system", "user", "assistant", "tool")

SYSTEM_RP = (
    "You are a character in an ongoing roleplay. Stay in character at all times. "
    "Write in third-person past tense. Never write the user's dialogue, thoughts, or actions. "
    "Favour concrete sensory detail over abstract summary."
)


def load_tokenizer(repo_id: str = DEFAULT_REPO, **kwargs: Any):
    """加载 Qwen3-VL 的 text tokenizer（纯文本路径）。"""
    from transformers import AutoTokenizer

    kwargs.setdefault("use_fast", True)
    return AutoTokenizer.from_pretrained(repo_id, **kwargs)


def build_messages(
    system: str | None, turns: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """把 (system, turns) 规范成 HF messages 列表。

    turns 元素为 {'role': 'user'|'assistant', 'content': str 或 content list}。
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in turns:
        role = turn["role"]
        if role not in VALID_ROLES:
            raise ValueError(f"unsupported role: {role!r}")
        messages.append({"role": role, "content": turn["content"]})
    return messages


def render(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = True,
    tools: Any = None,
) -> str:
    """messages -> prompt 字符串（tokenize=False 路径）。"""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(list(messages), **kwargs)


def encode(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = True,
    tools: Any = None,
    return_tensors: str | None = None,
) -> dict[str, Any]:
    """messages -> {'input_ids', 'attention_mask'}（tokenize=True 路径）。"""
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    return tokenizer.apply_chat_template(list(messages), **kwargs)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """裸文本编码，不额外插入特殊 token（训练数据构造用）。"""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def decode(tokenizer: Any, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


def find_span(tokenizer: Any, text: str, needle: str, *, occurrence: int = 1) -> tuple[int, int]:
    """在 `text` 里找到 `needle`（第 occurrence 次出现），返回对应的 **token 区间** `[start, end)`。

    用于 S4 记忆写入：把"哪句话值得记"映射成 token 下标（见 src/nova/memory.py）。
    `text` 必须是**喂给 tokenizer 的同一个字符串**（否则 offset 对不上）。
    边界上被切开的 token 会算进来（宁可多记一个 token，也不要漏）。
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    c0 = -1
    for _ in range(int(occurrence)):
        c0 = text.find(needle, c0 + 1)
        if c0 < 0:
            raise ValueError(f"在文本里找不到第 {occurrence} 处 {needle!r}")
    c1 = c0 + len(needle)
    idx = [i for i, (a, b) in enumerate(offsets) if b > c0 and a < c1]
    if not idx:
        raise ValueError(f"{needle!r} 没有落到任何 token 上")
    return idx[0], idx[-1] + 1


def render_turns(tokenizer: Any, system: str | None, turns: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
    return render(tokenizer, build_messages(system, turns), **kwargs)


def encode_turns(tokenizer: Any, system: str | None, turns: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return encode(tokenizer, build_messages(system, turns), **kwargs)


def ids_to_special_names(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    """把 id 序列里出现的特殊 token 还原成可读名字（调试 / 断言用）。"""
    out: list[str] = []
    for i in token_ids:
        name = None
        for content, tid in SPECIAL_TOKEN_IDS.items():
            if i == tid:
                name = content
                break
        if name is not None:
            out.append(name)
    return out
