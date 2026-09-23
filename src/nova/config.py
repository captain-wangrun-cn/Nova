"""Nova 配置：基座文本塔超参 + 双通路超参。

设计原则：
- **基座字段必须与 Qwen3-VL-4B 的 `text_config` 完全一致**，否则权重装不进去。
- **双通路字段全部可配置**（02-architecture.md 第八节第 3 条硬要求）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NovaConfig:
    # ---- 基座文本塔（对齐 Qwen3-VL-4B-Instruct / text_config）----
    vocab_size: int = 151936
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    # ---- 双通路 ----
    # 层切分：prefix（共享） → path（复制成 num_paths 条） → suffix（共享）
    num_prefix_layers: int = 6
    num_suffix_layers: int = 6
    num_paths: int = 2
    # 双通路段内每 cross_every 层插一处交叉注意力
    cross_every: int = 4
    # 交叉注意力是新增参数，头数独立于自注意力
    cross_num_heads: int = 8
    # 预测编码门控：预测器隐藏维（新增参数）
    gate_hidden: int = 256

    # ---- 滑动窗口注意力（E2，见 reports/swa-window.md）----
    # `swa_window = 0` ⇒ 关闭（每层都看全上下文，= 现在的行为）。
    # > 0 时：层号能被 `swa_global_every` 整除的是**全局层**，其余是**局部层**（只看最近 `swa_window` 个 token）。
    # 层号用**变换器层号**（前缀 i / 通路 prefix+i / 后缀 j），不是 cache 槽位号 ——
    # 双通路的两条路必须用同一套窗口，否则两条路的表示会漂。
    swa_window: int = 0
    swa_global_every: int = 4
    # prefill 的分块大小（0 = 取 `swa_window`）。**必须 ≤ swa_window**：
    # 局部层的 ring 只有 2W 个槽，"上一块的尾巴 + 本块" ≤ 2W 才装得下。
    swa_chunk: int = 0

    def __post_init__(self) -> None:
        if self.num_prefix_layers + self.num_suffix_layers >= self.num_hidden_layers:
            raise ValueError("prefix + suffix 层数必须小于总层数，否则没有可复制的中间层")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须是 num_key_value_heads 的整数倍")
        if self.q_dim % self.num_attention_heads != 0:
            raise ValueError("q_dim 必须能被 num_attention_heads 整除")

    # ---- 派生量 ----

    @property
    def q_dim(self) -> int:
        """Q 投影的输出维度。

        ⚠️ **注意：Qwen3-VL-4B 是"扩张注意力"** —— `num_attention_heads * head_dim`
        = 32 × 128 = **4096**，而 `hidden_size` 只有 **2560**。两者**不相等**，
        所以不能套用"hidden = heads × head_dim"的常见假设。
        """
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """K / V 投影的输出维度。"""
        return self.num_key_value_heads * self.head_dim

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def num_path_layers(self) -> int:
        """单条通路占用的层数（= 被复制的中间层数）。"""
        return self.num_hidden_layers - self.num_prefix_layers - self.num_suffix_layers

    @property
    def prefix_range(self) -> range:
        return range(0, self.num_prefix_layers)

    @property
    def suffix_range(self) -> range:
        """共享后段在原模型里的绝对层号。"""
        start = self.num_prefix_layers + self.num_path_layers
        return range(start, start + self.num_suffix_layers)

    def path_layer_indices(self) -> list[int]:
        """双通路段内、需要插交叉注意力的**相对**层号。

        例：num_path_layers=24, cross_every=4 → [0, 4, 8, 12, 16, 20]
        """
        return list(range(0, self.num_path_layers, self.cross_every))

    def window_for_layer(self, t_idx: int) -> int:
        """变换器层 `t_idx` 的注意力窗口（0 = 全局层 / 关闭）。

        `t_idx` 是**变换器层号**：前缀 `0..5`、通路层 `prefix+i`、后缀 `prefix+path+j`。
        两条通路的同一个 `t_idx` 拿到同一个窗口 —— 否则两条路的表示空间会分叉。
        """
        if not self.swa_window:
            return 0
        every = max(1, int(self.swa_global_every))
        return 0 if int(t_idx) % every == 0 else int(self.swa_window)

    # ---- 构造 ----

    @classmethod
    def from_hf(cls, text_config, **overrides) -> "NovaConfig":
        """从 HF 的 `text_config` 取基座字段，双通路字段用默认值或 overrides。"""
        get = lambda k, d=None: getattr(text_config, k, d)  # noqa: E731
        base = dict(
            vocab_size=int(get("vocab_size", 151936)),
            hidden_size=int(get("hidden_size", 2560)),
            num_hidden_layers=int(get("num_hidden_layers", 36)),
            num_attention_heads=int(get("num_attention_heads", 32)),
            num_key_value_heads=int(get("num_key_value_heads", 8)),
            head_dim=int(get("head_dim", 128)),
            intermediate_size=int(get("intermediate_size", 9728)),
            hidden_act=str(get("hidden_act", "silu")),
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            rope_theta=float(get("rope_theta", 5_000_000.0)),
            max_position_embeddings=int(get("max_position_embeddings", 262144)),
            attention_bias=bool(get("attention_bias", False)),
            tie_word_embeddings=bool(get("tie_word_embeddings", True)),
        )
        base.update(overrides)
        return cls(**base)
