"""训练路径的最小回归：梯度检查点只该省显存，不该改前向输出。"""

from __future__ import annotations

import torch


def test_gradient_checkpointing_forward_matches(bundle, prompt_ids):
    nova, _, _ = bundle
    was_training = nova.training
    nova.train()
    try:
        with torch.no_grad():
            plain = nova(input_ids=prompt_ids, cross_mode="on", gradient_checkpointing=False)
            ckpt = nova(input_ids=prompt_ids, cross_mode="on", gradient_checkpointing=True)
        assert torch.equal(plain, ckpt), (
            f"梯度检查点改变了前向输出，max|diff|={(plain.float() - ckpt.float()).abs().max().item():.3e}"
        )
    finally:
        nova.train(was_training)
