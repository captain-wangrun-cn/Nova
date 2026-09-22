"""pytest 全局配置：环境变量必须在 import transformers **之前**设好。

不设 `HF_HOME`，模型会下到 `C:\\Users\\<user>\\.cache\\huggingface`（约 9GB）——
本项目**禁止写 C 盘**。Triton / Inductor 同理。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".tmp" / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".tmp" / "inductor-cache"))
(ROOT / ".tmp").mkdir(exist_ok=True)

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    """整个测试会话只加载一次基座 + Nova（加载约 12s，显存约 4.6 GiB）。"""
    from nova.loader import load_nova

    nova, hf, tok = load_nova(norm_impl="exact")
    return nova, hf, tok


@pytest.fixture(scope="session")
def prompt_ids(bundle):
    _, _, tok = bundle
    text = tok.apply_chat_template(
        [{"role": "user", "content": "用一句话说明什么是潮汐。"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
