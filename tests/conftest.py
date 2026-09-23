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
def nova_swa(bundle):
    """按需**重挂**一层 Nova 外壳（`swa_window` 可配）。

    复用同一个 HF 基座对象（`build_nova` 直接引用它的层），所以几乎不额外吃显存 ——
    这条很重要：再 `load_nova()` 一次会多占 ~2.5 GiB，8 GB 装不下。
    """
    _, hf, _ = bundle
    from nova.loader import build_nova

    def make(window: int, chunk: int = 0, num_paths: int = 1, **kw):
        # ⚠️ 必须显式 `num_paths`：`build_nova` 的默认是**双通路**，那会 deepcopy 24 个通路层
        # （每次调用 +2.4 GiB）—— 在 8 GB 上再建一个双通路模型直接 OOM（实测踩过）。
        return build_nova(hf, norm_impl="exact", num_paths=num_paths,
                          swa_window=window, swa_chunk=chunk, **kw)

    return make


@pytest.fixture(scope="session")
def prompt_ids(bundle):
    _, _, tok = bundle
    text = tok.apply_chat_template(
        [{"role": "user", "content": "用一句话说明什么是潮汐。"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
