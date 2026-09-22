"""从 HF 的 Qwen3-VL-4B 构建 Nova 模型。

**重要：返回的 Nova 会**复用** HF 模型的模块对象**（通路 0 的线性层直接指向 HF 的原层），
所以调用方必须让 `hf_model` 保持存活，不能提前释放。
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from .config import NovaConfig
from .model import NovaForCausalLM, NovaTextModel

DEFAULT_REPO = "Qwen/Qwen3-VL-4B-Instruct"


def bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_hf_base(repo: str = DEFAULT_REPO, quant: str = "4bit"):
    """加载基座。`quant` ∈ {"4bit", "bf16"}。"""
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if quant == "4bit":
        kwargs["quantization_config"] = bnb_4bit_config()
        kwargs["dtype"] = torch.float16
    elif quant == "bf16":
        kwargs["dtype"] = torch.bfloat16
    else:
        raise ValueError(f"不支持的 quant: {quant!r}")

    model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
    model.eval()
    return model


def build_nova(hf_model, norm_impl: str = "exact", **config_overrides) -> NovaForCausalLM:
    """把已加载的 HF 基座改造成 Nova 双通路模型。"""
    config = NovaConfig.from_hf(hf_model.config.text_config, **config_overrides)
    text_model = NovaTextModel(hf_model.model.language_model, config, norm_impl=norm_impl)
    model = NovaForCausalLM(text_model)
    model.eval()
    return model


def load_nova(
    repo: str = DEFAULT_REPO,
    quant: str = "4bit",
    norm_impl: str = "exact",
    **config_overrides,
) -> tuple[NovaForCausalLM, Any, Any]:
    """一步到位：返回 `(nova, hf_model, tokenizer)`。"""
    hf_model = load_hf_base(repo, quant)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    nova = build_nova(hf_model, norm_impl=norm_impl, **config_overrides)
    return nova, hf_model, tokenizer


def enable_lm_head_4bit(
    model: NovaForCausalLM,
    quant_type: str = "nf4",
    compress_statistics: bool = True,
) -> Any:
    """给输出投影装一份 **4-bit 副本**（写到 `model.lm_head4`）。

    动机（已核查，见 [reports/s4-nf4-gemv.md](../../reports/s4-nf4-gemv.md)）：
    `lm_head` 复用 fp16 的 `embed_tokens.weight` 时，每 token 要读 **778 MB**
    （151936x2560x2B），实测 **3.11 ms**。这已经是 250 GB/s 的 DRAM 上限
    （"打满带宽"），但**降位宽能把要搬的数据砍到 1/4** —— 换 bnb `Linear4bit`
    后实测 **1.02 ms**（3.04x）。

    ⚠️ **fp16 的 `embed_tokens.weight` 原样保留**：它是输入 embedding 的表，
    动了会让输入也掉精度。所以这是**额外**的一份 4-bit 副本（约 200 MB）。

    ⚠️ 与 `quant.convert_to_nf4` 的关系：`convert_to_nf4` 默认**跳过** `lm_head4`
    （自写 Triton kernel 在 lm_head 这种 N=151936 的形状上比 bnb 慢 1.34x）。
    """
    import bitsandbytes as bnb
    from bitsandbytes.functional import quantize_4bit

    W = model.model.embed_tokens.weight.detach()
    n_out, n_in = int(W.shape[0]), int(W.shape[1])

    packed, quant_state = quantize_4bit(
        W,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
        quant_storage=torch.float16,
    )
    lin = bnb.nn.Linear4bit(
        n_in,
        n_out,
        bias=False,
        compute_dtype=torch.float16,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
        quant_storage=torch.float16,
        device="meta",
    )
    lin.weight = bnb.nn.Params4bit(
        packed,
        requires_grad=False,
        quant_state=quant_state,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
        quant_storage=torch.float16,
        bnb_quantized=True,
    )
    lin.eval()
    model.lm_head4 = lin
    return lin
