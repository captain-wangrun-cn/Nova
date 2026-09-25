"""S4 · 记忆最小实现：L0「精确 KV」潜空间记忆（latent memory）。

对应文档：[docs/03-memory-system.md](../../docs/03-memory-system.md)（三层记忆的第二层）、
[docs/09-speculative.md](../../docs/09-speculative.md)（记忆精度分级 L0）、
报告 [reports/s4-memory-min.md](../../reports/s4-memory-min.md)、决策 **D31**。

## 为什么不是 docs/03 里那个 `[128, 2560]` 记忆令牌

那是**要训练**的写入器 / 读取器（模型得学会"什么值得记""怎么生成记忆令牌"），属于里程碑 2。
S4 的验收标准（第 1 轮写入 → 第 20 轮取回；存盘重启后仍可取回）**不训练**也要能达成，
所以先落地 [docs/09-speculative.md](../../docs/09-speculative.md) 的 **L0 精确 KV 层**：

| 环节 | S4 的做法 |
|------|------|
| 写入 | 抓记忆区间内**每个 cache 槽位**的 K（RoPE 之前）/ V |
| 寻址 | 用模型**自己的** Q/K 做键值联想（按**注入后的真实位置**旋转后的 `Q·K` 注意力质量） |
| 读取 | 把取回的 K/V 按新位置旋转后写进静态 KV cache 的**最前面**，模型照常生成 |

## 三条硬约束的落点

- **D06 全在模型内部**：记忆是模型自己的 K/V —— 不转文本、不重新 tokenize、不查外部库。
- **D07 张量持久化**：`.safetensors`；K/V 是记忆内容，`_meta` 只放校验信息。
- **D09 表示空间冻结**：`MemorySchema` 固定槽位布局与维度，文件带 `schema_digest` 与
  `model_fingerprint`，不匹配**拒绝加载**（宁可"想不起来"，也不要读串味）。

## 已知边界（不要当成已解决）

- **无损，不是压缩**：一个记忆项 = 该区间全部 token 的 K/V，体积随区间线性增长
  （L1 高保真嵌入 / L2 压缩摘要留给里程碑 3）。
- **键未训练**：区分度是**实测**出来的，不是设计保证的 —— 见报告。
- **LoRA / 全参微调会移动 K/V 空间**（D09 的"表示漂移"），记忆匹配质量**待实测**。
- **通路 0/1 各存一份**：门控关闭时两条通路逐位相同（已实测），去重可省一半空间，
  留待交叉注意力真正打开后再定形（否则格式会白改一次）。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .layers import apply_rotary_pos_emb

MEMORY_FORMAT = "nova-memory"
MEMORY_FORMAT_VERSION = 1

# safetensors 的 dtype 字符串 → torch dtype（只列本项目会写出的类型）
_SF_DTYPES: dict[str, torch.dtype] = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class MemoryFormatError(ValueError):
    """记忆文件本身不合法（缺字段 / 版本不认识 / 形状不对）。"""


class MemorySchemaMismatch(MemoryFormatError):
    """记忆的表示空间与当前模型不一致（D09）—— 拒绝加载。"""


# ---------------------------------------------------------------- 表示空间


@dataclass(frozen=True)
class MemorySchema:
    """记忆表示空间的形状。**冻结**：改动必须升 `version`，否则老记忆会读串味。"""

    hidden_size: int
    num_kv_heads: int
    head_dim: int
    num_cache_layers: int
    num_paths: int
    dtype: str = "float16"
    version: int = MEMORY_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "hidden_size": self.hidden_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "num_cache_layers": self.num_cache_layers,
            "num_paths": self.num_paths,
            "dtype": self.dtype,
        }

    @property
    def digest(self) -> str:
        """表示空间的指纹（写进文件，加载时比对）。"""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @classmethod
    def for_model(cls, model) -> "MemorySchema":
        text = model.model
        cfg = text.config
        return cls(
            hidden_size=int(cfg.hidden_size),
            num_kv_heads=int(cfg.num_key_value_heads),
            head_dim=int(cfg.head_dim),
            num_cache_layers=int(text.num_cache_layers),
            num_paths=int(cfg.num_paths),
            dtype=str(text.embed_tokens.weight.dtype).replace("torch.", ""),
        )


_FINGERPRINT_SAMPLE = 512


def _fingerprint_tensors(model):
    """决定 K/V 的那些权重：embedding + 每层的 k/v 投影与 k 归一化。"""
    text = model.model
    yield "embed_tokens", text.embed_tokens.weight
    groups = [
        ("prefix", list(enumerate(text.prefix_layers))),
        ("suffix", list(enumerate(text.suffix_layers))),
    ]
    for p, per_path in enumerate(text.path_layers):
        groups.append((f"path{p}", list(enumerate(per_path))))
    for name, layers in groups:
        for i, layer in layers:
            attn = layer.self_attn
            yield f"{name}.{i}.k_proj", attn.k_proj.weight
            yield f"{name}.{i}.v_proj", attn.v_proj.weight
            yield f"{name}.{i}.k_norm", attn.k_norm.weight


def model_fingerprint(model, sample: int = _FINGERPRINT_SAMPLE) -> str:
    """记忆表示空间的权重指纹：**换了模型 / 换了精度要被发现**，不是密码学校验。

    只对每个张量抽样 `sample` 个元素（顺序读取太慢），够用且快。
    """
    h = hashlib.sha256()
    for name, t in _fingerprint_tensors(model):
        flat = t.detach().reshape(-1)
        stride = max(1, flat.numel() // sample)
        probe = flat[::stride][:sample].to(torch.float32).cpu().numpy().tobytes()
        h.update(name.encode("utf-8"))
        h.update(probe)
    return h.hexdigest()[:32]


# ---------------------------------------------------------------- 捕获


def slot_to_attention(text_model) -> dict[int, Any]:
    """KV cache 槽位号 -> 该槽位的注意力模块（顺序即文件里的层顺序）。"""
    out: dict[int, Any] = {}
    for i, layer in enumerate(text_model.prefix_layers):
        out[text_model.cache_slot_prefix(i)] = layer.self_attn
    for p, per_path in enumerate(text_model.path_layers):
        for i, layer in enumerate(per_path):
            out[text_model.cache_slot_path(p, i)] = layer.self_attn
    for i, layer in enumerate(text_model.suffix_layers):
        out[text_model.cache_slot_suffix(i)] = layer.self_attn
    return out


@contextlib.contextmanager
def _capturing(text_model, span: tuple[int, int]):
    targets = slot_to_attention(text_model)
    for attn in targets.values():
        attn.capture_slice = (int(span[0]), int(span[1]))
        attn.captured = None
    try:
        yield targets
    finally:
        for attn in targets.values():
            attn.capture_slice = None


@torch.inference_mode()
def capture_qkv(text_model, input_ids: torch.Tensor, span: tuple[int, int], cross_mode: str = "off"):
    """抓取 `span` 区间在**每个 cache 槽位**上的 Q / K / V（都是 **RoPE 之前** 的）。

    返回 `(q, k, v)`，形状 `[num_cache_layers, heads, m, head_dim]`。

    ⚠️ 必须在**不带 cache** 的一次完整前向上做：K/V 依赖因果上下文，
    输入要放"整段对话"，抓到的才是这段记忆在**真实上下文**里的状态。
    """
    if input_ids.shape[0] != 1:
        raise ValueError("S4 只支持 batch=1")
    with _capturing(text_model, span) as targets:
        text_model(input_ids=input_ids, cross_mode=cross_mode)
    qs, ks, vs = [], [], []
    for slot in sorted(targets):
        got = targets[slot].captured
        if got is None:
            raise RuntimeError(f"槽位 {slot} 没抓到 K/V —— span 越界？")
        qs.append(got[0].squeeze(0))
        ks.append(got[1].squeeze(0))
        vs.append(got[2].squeeze(0))
    return torch.stack(qs, 0), torch.stack(ks, 0), torch.stack(vs, 0)


@torch.inference_mode()
def capture_kv(text_model, input_ids: torch.Tensor, span: tuple[int, int], cross_mode: str = "off"):
    """只要 K / V 的版本（写入路径用）。"""
    _q, k, v = capture_qkv(text_model, input_ids, span, cross_mode=cross_mode)
    return k, v


@torch.inference_mode()
def query_vectors(
    text_model,
    input_ids: torch.Tensor,
    span: tuple[int, int] | None = None,
    n_last: int = 4,
    cross_mode: str = "off",
) -> torch.Tensor:
    """"检索查询"：问题那一段 token 的 **Q**（RoPE 之前），形状 `[层, Q头, t, head_dim]`。

    用 **Q** 而不是 hidden state 或 K：注意力打分就是 `Q·K` ——
    查询与记忆在**同一个空间**里比相似度，不需要任何训练，也不需要额外的投影。

    ⚠️ **`span` 一定要给对**（用 `chatfmt.find_span` 定位问题那句话）。实测踩过的坑：
    取 prompt 最后 4 个 token 时，拿到的是 `<|im_start|>assistant\\n`（**没有内容**），
    寻址直接退化成"恒选第一条"。取不到 span 时才回退到"最后 n_last 个 token"。
    """
    t = int(input_ids.shape[1])
    if t < 1:
        raise ValueError("空输入")
    if span is None:
        span = (max(0, t - int(n_last)), t)
    s0, s1 = int(span[0]), int(span[1])
    if not 0 <= s0 < s1 <= t:
        raise ValueError(f"query span {span} 超出 prompt 长度 {t}")
    q, _k, _v = capture_qkv(text_model, input_ids, (s0, s1), cross_mode=cross_mode)
    return q


# ---------------------------------------------------------------- 记忆项


@dataclass
class MemoryItem:
    """一个记忆项：一段对话在**每个 cache 槽位**上的 K / V。

    `k` / `v`：`[num_cache_layers, kv_heads, n_tokens, head_dim]`，`k` 是 **RoPE 之前** 的。
    `label`：只给人看（报告 / 调试），**不是记忆内容**，不参与检索。
    """

    k: torch.Tensor
    v: torch.Tensor
    label: str | None = None

    @property
    def n_tokens(self) -> int:
        return int(self.k.shape[2])

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


@dataclass
class RetrievalInfo:
    """一次 prefill 的检索结果（供报告 / 调试打印）。"""

    hits: list[tuple[int, float]] = field(default_factory=list)
    scores: torch.Tensor | None = None
    prefix_len: int = 0
    forced: int | None = None
    query_ms: float = 0.0
    score_ms: float = 0.0
    prefill_ms: float = 0.0

    @property
    def used_memory(self) -> bool:
        return self.prefix_len > 0


# ---------------------------------------------------------------- 记忆库


class MemoryStore:
    """一组记忆项 + 冻结的表示空间：写入 / 联想检索 / safetensors 读写 / 注入 KV cache。"""

    def __init__(self, schema: MemorySchema, fingerprint: str | None = None) -> None:
        self.schema = schema
        self.fingerprint = fingerprint
        self.items: list[MemoryItem] = []
        self.foreign = False  # True = 加载时模型指纹不匹配，但调用方显式允许了

    @classmethod
    def for_model(cls, model) -> "MemoryStore":
        """按当前模型建一个空记忆库（表示空间取自模型，指纹取自权重）。"""
        return cls(MemorySchema.for_model(model), model_fingerprint(model))

    # ---- 基本信息 ----

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f"MemoryStore(items={len(self.items)}, tokens={self.n_tokens}, "
            f"{self.nbytes() / 1024 ** 2:.2f} MiB, schema={self.schema.digest[:8]})"
        )

    @property
    def n_tokens(self) -> int:
        return sum(it.n_tokens for it in self.items)

    def nbytes(self) -> int:
        return sum(it.nbytes() for it in self.items)

    def describe(self) -> list[str]:
        return [
            f"[{i}] {it.n_tokens} token · {it.nbytes() / 1024 ** 2:.2f} MiB"
            + (f" · {it.label}" if it.label else "")
            for i, it in enumerate(self.items)
        ]

    # ---- 写入 ----

    def add(self, k: torch.Tensor, v: torch.Tensor, label: str | None = None) -> int:
        want = (self.schema.num_cache_layers, self.schema.num_kv_heads, self.schema.head_dim)
        for name, t in (("k", k), ("v", v)):
            if t.dim() != 4 or t.shape[0] != want[0] or t.shape[1] != want[1] or t.shape[3] != want[2]:
                raise MemorySchemaMismatch(
                    f"{name} 形状 {tuple(t.shape)} 与 schema 不符"
                    f"（期望 [层={want[0]}, KV头={want[1]}, m, head_dim={want[2]}]）"
                )
        if k.shape[2] != v.shape[2]:
            raise MemorySchemaMismatch("k / v 的 token 数不一致")
        self.items.append(MemoryItem(k.contiguous(), v.contiguous(), label))
        return len(self.items) - 1

    def write(self, model, input_ids: torch.Tensor, span: tuple[int, int], label: str | None = None) -> int:
        """从 `input_ids` 的 `span` 区间写入一条记忆（在**真实上下文**里抓 K/V）。"""
        k, v = capture_kv(model.model, input_ids, span)
        return self.add(k, v, label=label)

    # ---- 联想检索 ----

    def logits(self, query_rot: torch.Tensor, key_rot: torch.Tensor) -> torch.Tensor:
        """`Q·K / sqrt(head_dim)` —— 就是注意力打分本身 → `[层, Q头, t, m]`。

        用 fp32 算，避免 fp16 下的精度噪声。
        ⚠️ GQA：Q 头数（32）是 KV 头数（8）的整数倍，必须按 `repeat_kv` 的同一映射分组，
        否则 einsum 直接报维度不匹配。
        """
        q = query_rot.float()
        k = key_rot.float()
        groups = q.shape[1] // k.shape[1]
        qg = q.view(q.shape[0], k.shape[1], groups, q.shape[2], q.shape[3])
        out = torch.einsum("lhgtc,lhmc->lhgtm", qg, k) * (self.schema.head_dim ** -0.5)
        return out.reshape(q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    def scores(
        self,
        query_pre: torch.Tensor,
        item_starts: list[int],
        query_starts: list[int],
        rotary_emb,
        standardize: bool = False,
        normalize_by_length: bool = True,
        temperature: float = 1.0,
        per_layer: bool = False,
    ):
        """按**注入后的真实位置**旋转 Q / K 再打分 → `[n_items]`。

        为什么必须按真实位置旋转（实测，见 [reports/s4-memory-min.md](../../reports/s4-memory-min.md)）：
        RoPE 让 `Q·K` 依赖相对距离。把查询和记忆放在**不同的位置坐标系**里打分
        （例如查询在位置 600、记忆在位置 0）会让打分退化成噪声 —— 实测 top-1 只有 1/3。

        打分本身是**注意力质量**：把查询对所有候选记忆的 token 做 softmax，按记忆项汇总。
        pre-RoPE 的键有一个很强的**公共分量**（裸余弦全部 > 0.8，几乎不区分内容），
        softmax 会把它减掉（对候选是同一个加性偏移）。

        三个默认值都是**实测选出来的**（6 条记忆 × 8 个问题，含改写问法，见报告）：

        | 配置 | top-1 |
        |------|:---:|
        | `frame="alone"` + 按长度归一 + 不标准化（**默认**） | **8/8** |
        | `frame="alone"` + 不按长度归一 | 7/8 |
        | `frame="stack"` | 2/8 |

        - `normalize_by_length=True`：除以该条记忆的 token 数。不除的话，**长记忆天然占便宜**
          （质量是按 token 累加的）。
        - `standardize=True` 会按 (层, 头) 标准化。它在**小候选集**上看着有用（3 条时 2/3），
          但候选变多后反而变差（4/8）—— 所以默认关掉。

        `item_starts[i]` / `query_starts[i]`：第 i 条记忆与查询在**注入后**的起始位置。
        逐条给是因为"每条单独注入"的帧里，查询位置会随该条记忆的长度变化。
        """
        if not self.items:
            return torch.zeros(0, device=query_pre.device)
        parts = []
        for item, item_start, query_start in zip(self.items, item_starts, query_starts):
            q = _rotate(query_pre, int(query_start), self.schema.hidden_size, rotary_emb)
            k = _rotate(item.k, int(item_start), self.schema.hidden_size, rotary_emb)
            parts.append(self.logits(q, k))  # [层, Q头, t, m]
        all_logits = torch.cat(parts, dim=-1)  # [层, Q头, t, Σm]
        if standardize:
            mu = all_logits.mean(dim=-1, keepdim=True)
            sd = all_logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
            all_logits = (all_logits - mu) / sd
        weights = torch.softmax(all_logits / float(temperature), dim=-1)
        by_layer, start = [], 0
        for item in self.items:
            mass = weights[..., start : start + item.n_tokens].sum(dim=-1).mean(-1).mean(-1)
            if normalize_by_length:
                mass = mass / item.n_tokens
            by_layer.append(mass)
            start += item.n_tokens
        stacked = torch.stack(by_layer, dim=-1)  # [层, n_items]
        return (stacked, stacked.mean(0)) if per_layer else stacked.mean(0)

    def retrieve(
        self,
        scores: torch.Tensor,
        top_k: int = 1,
        threshold: float | None = None,
    ) -> list[tuple[int, float]]:
        """按打分取前 `top_k` 条；低于 `threshold` 的直接丢掉（= "想不起来"）。"""
        if scores.numel() == 0:
            return []
        order = torch.argsort(scores, descending=True)[: int(top_k)]
        hits = [(int(i), float(scores[i])) for i in order]
        if threshold is not None:
            hits = [(i, s) for i, s in hits if s >= float(threshold)]
        return hits

    # ---- 注入 KV cache ----

    def inject_length(self, items: list[MemoryItem]) -> int:
        """这组记忆项占用的前缀长度（token 数）。"""
        return sum(it.n_tokens for it in items)

    def inject(
        self,
        cache,
        items: list[MemoryItem],
        rotary_emb,
        offset: int = 0,
        batch: int = 1,
    ) -> int:
        """把记忆项写进静态 KV cache 的 `[offset, offset+Σm)` 槽位，返回新的前缀长度。

        ⚠️ 记忆里的 K 是 **RoPE 之前** 的，这里按**新位置**重新旋转 ——
        于是"那段话"在模型看来就位于位置 `offset..` 上（= 把记忆拼在上下文最前面）。
        """
        if len(cache.key_cache) != self.schema.num_cache_layers:
            raise MemorySchemaMismatch(
                f"cache 槽位数 {len(cache.key_cache)} 与记忆的 {self.schema.num_cache_layers} 不一致"
            )
        pos = int(offset)
        with torch.no_grad():
            for item in items:
                m = item.n_tokens
                cos, sin = _rope_cos_sin(rotary_emb, item.k, pos, m, self.schema.hidden_size, batch)
                k_rot = apply_rotary_pos_emb(item.k, item.k, cos, sin)[0]
                for slot in range(self.schema.num_cache_layers):
                    cache.key_cache[slot][:, :, pos : pos + m, :] = k_rot[slot].unsqueeze(0)
                    cache.value_cache[slot][:, :, pos : pos + m, :] = item.v[slot].unsqueeze(0)
                pos += m
        return pos

    # ---- 持久化（D07）----

    def meta(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """`_meta`：**只放校验与调试信息**，不含记忆内容。"""
        meta: dict[str, Any] = {
            "format": MEMORY_FORMAT,
            "version": MEMORY_FORMAT_VERSION,
            "schema": self.schema.to_dict(),
            "schema_digest": self.schema.digest,
            "model_fingerprint": self.fingerprint,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_items": len(self.items),
            "items": [{"label": it.label, "n_tokens": it.n_tokens} for it in self.items],
        }
        if extra:
            meta.update(extra)
        return meta

    def save(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
        """写 `.safetensors`：`item{i}.k` / `item{i}.v` 是记忆内容，`_meta` 在文件头。"""
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        for i, item in enumerate(self.items):
            tensors[f"item{i}.k"] = item.k.detach().cpu().contiguous()
            tensors[f"item{i}.v"] = item.v.detach().cpu().contiguous()
        save_file(
            tensors,
            str(path),
            metadata={"nova_memory": json.dumps(self.meta(extra_meta), ensure_ascii=False)},
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        schema: MemorySchema | None = None,
        fingerprint: str | None = None,
        device: str | torch.device = "cuda",
        allow_foreign: bool = False,
    ) -> "MemoryStore":
        """读 `.safetensors` 并**校验表示空间**（D09）。

        `schema` / `fingerprint` 给了就比对；不一致直接抛 `MemorySchemaMismatch`，
        除非 `allow_foreign=True`（此时 `store.foreign = True`，调用方自负后果）。
        """
        from safetensors import safe_open

        path = Path(path)
        if not path.exists():
            raise MemoryFormatError(f"记忆文件不存在：{path}")
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            raw = f.metadata() or {}
            blob = raw.get("nova_memory")
            if not blob:
                raise MemoryFormatError("文件头里没有 nova_memory 元数据 —— 不是 Nova 记忆文件？")
            meta = json.loads(blob)
            # 校验逻辑**只写一份**（`validate_memory_meta`）：safetensors 头路径与预取路径
            # 必须用同一套判据，否则两条路会飘（D09 的"宁可拒绝加载"就守不住了）。
            file_schema, fp, foreign = validate_memory_meta(meta, schema, fingerprint, allow_foreign)
            store = cls(file_schema, fp)
            store.foreign = foreign
            for i in range(int(meta["n_items"])):
                label = None
                items_meta = meta.get("items") or []
                if i < len(items_meta):
                    label = items_meta[i].get("label")
                store.add(f.get_tensor(f"item{i}.k"), f.get_tensor(f"item{i}.v"), label=label)
        return store

    @classmethod
    def load_prefetched(
        cls,
        path: str | Path,
        schema: MemorySchema | None = None,
        fingerprint: str | None = None,
        device: str | torch.device = "cuda",
        seg_bytes: int = 4 << 20,
        ring: int = 4,
        allow_foreign: bool = False,
        verify: bool = False,
    ) -> "MemoryStore":
        """从磁盘加载记忆，走 **`SegmentPrefetcher` 双缓冲**（D41 的唯一正确写法）。

        ## 与 `load()` 的区别

        | | `load()`（`safe_open` 路径） | `load_prefetched()`（本方法） |
        |---|---|---|
        | 读法 | 逐张量 `get_tensor()` | **整段**原始字节（pin + `readinto` + 双缓冲 + `non_blocking`） |
        | 盘读与 H2D | 串行 | **重叠**（先发 H2D，再读下一段） |
        | 显存张量 | 每个张量独立分配 | **一块 uint8 缓冲 + 零拷贝视图** |
        | D07 | ✅ safetensors | ✅ 同一个 safetensors，**不另存裸 blob**（按数据区偏移搬） |

        ## 为什么快

        侧会话实测：裸 `read()` + H2D **1.76 GiB/s**；pin + `readinto` + 双缓冲 + `non_blocking`
        **4.49 GiB/s（2.55x）**。端到端数字见 [reports/memory-prefetch-load.md](../../reports/memory-prefetch-load.md)。

        `verify=True` 额外用 `safe_open` 逐张量读一遍做**逐位比对**（验收用，会拖慢）。
        """
        from .prefetch import SegmentPrefetcher

        path = Path(path)
        if not path.exists():
            raise MemoryFormatError(f"记忆文件不存在：{path}")
        meta, data_start, layout = read_safetensors_layout(path)
        file_schema, fp, foreign = validate_memory_meta(meta, schema, fingerprint, allow_foreign)

        n_items = int(meta["n_items"])
        want: list[tuple[str, tuple]] = []
        for i in range(n_items):
            for suf in ("k", "v"):
                name = f"item{i}.{suf}"
                if name not in layout:
                    raise MemoryFormatError(f"文件头里缺少张量 {name}")
                want.append((name, layout[name]))

        store = cls(file_schema, fp)
        store.foreign = foreign
        if not want:
            return store

        lo = min(t[2] for _, t in want)
        hi = max(t[3] for _, t in want)
        total = hi - lo
        buf = torch.empty(total, dtype=torch.uint8, device=device)
        pf = SegmentPrefetcher(path, seg_bytes=seg_bytes, ring=ring,
                               offset=data_start + lo, length=total)
        moved = pf.stream_into(buf)
        if moved != total:
            raise MemoryFormatError(f"只搬运了 {moved} / {total} 字节")
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()

        # 显存内**零拷贝建视图**：切片已连续，`view(dtype)` / `view(shape)` 都不复制
        views = {name: buf[s - lo : e - lo].view(dt).view(shape)
                 for name, (dt, shape, s, e) in want}
        for i in range(n_items):
            label = None
            items_meta = meta.get("items") or []
            if i < len(items_meta):
                label = items_meta[i].get("label")
            store.add(views[f"item{i}.k"], views[f"item{i}.v"], label=label)

        if verify:
            _verify_against_safe_open(path, store)
        return store


def _verify_against_safe_open(path: Path, store: "MemoryStore") -> None:
    """验收用：用 `safe_open` 独立读一遍，与预取加载的结果**逐位比对**。"""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        names = set(f.keys())
        for i, item in enumerate(store.items):
            for suf, got in (("k", item.k), ("v", item.v)):
                name = f"item{i}.{suf}"
                if name not in names:
                    raise MemoryFormatError(f"safe_open 里没有 {name}")
                if not torch.equal(got.cpu(), f.get_tensor(name)):
                    raise MemoryFormatError(f"预取加载的 {name} 与 safe_open 读到的**不一致**")


# ---------------------------------------------------------------- safetensors 布局 / 预取加载


def read_safetensors_layout(path: str | Path) -> tuple[dict, int, dict[str, tuple]]:
    """只读**文件头**，返回 `(头部 JSON, 数据区起点, {张量名: (dtype, shape, rel_start, rel_end)})`。

    `.safetensors` 布局 = `[8 字节头长 u64 LE][JSON 头][数据区]`；每个张量的
    `data_offsets` 是**相对数据区起点**的字节区间。已核查（`probe_safetensors_layout.py`）：
    本项目写出的文件里张量**连续、无夹缝**，所以"整段搬数据区 + 显存内建视图"是等价的。
    """
    import struct

    path = Path(path)
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise MemoryFormatError(f"文件太短，读不出 safetensors 头长：{path}")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise MemoryFormatError(f"safetensors 头长异常：{header_len}")
        header = json.loads(fh.read(header_len).decode("utf-8"))

    tensors: dict[str, tuple] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dt = _SF_DTYPES.get(spec["dtype"])
        if dt is None:
            raise MemoryFormatError(f"张量 {name} 的 dtype {spec['dtype']!r} 不支持")
        s, e = spec["data_offsets"]
        tensors[name] = (dt, tuple(spec["shape"]), int(s), int(e))
    meta_raw = (header.get("__metadata__") or {}).get("nova_memory")
    if not meta_raw:
        raise MemoryFormatError("文件头里没有 nova_memory 元数据 —— 不是 Nova 记忆文件？")
    # `safe_open().metadata()` 给的是字符串，这里保持一致（load() 也是 json.loads 字符串）
    return json.loads(meta_raw), 8 + header_len, tensors


def validate_memory_meta(
    meta: dict,
    schema: MemorySchema | None = None,
    fingerprint: str | None = None,
    allow_foreign: bool = False,
) -> tuple[MemorySchema, str | None, bool]:
    """校验 `_meta`（格式 / 版本 / D09 表示空间），返回 `(文件 schema, 指纹, 是否外来)`。"""
    if meta.get("format") != MEMORY_FORMAT:
        raise MemoryFormatError(f"未知的记忆格式：{meta.get('format')!r}")
    if int(meta.get("version", 0)) > MEMORY_FORMAT_VERSION:
        raise MemoryFormatError(
            f"记忆格式版本 {meta.get('version')} 比本代码（{MEMORY_FORMAT_VERSION}）新，拒绝读取"
        )
    file_schema = MemorySchema(**meta["schema"])
    if schema is not None and file_schema.digest != schema.digest:
        raise MemorySchemaMismatch(
            f"记忆表示空间与当前模型不一致（D09）：文件 {file_schema.digest[:12]} vs 当前 {schema.digest[:12]}"
        )
    fp = meta.get("model_fingerprint")
    if fingerprint is not None and fp is not None and fp != fingerprint and not allow_foreign:
        raise MemorySchemaMismatch(
            f"记忆的模型指纹 {fp[:12]} 与当前权重 {fingerprint[:12]} 不一致（D09）——"
            " 确认无误可传 allow_foreign=True 强制加载"
        )
    return file_schema, fp, bool(fp is not None and fingerprint is not None and fp != fingerprint)


def _rope_cos_sin(rotary_emb, ref: torch.Tensor, start: int, n: int, hidden_size: int, batch: int = 1):
    """按 `NovaTextModel.forward` 的同一方式取 cos/sin（Qwen3-VL 的 mrope 要 `[3, b, t]`）。"""
    positions = torch.arange(start, start + n, device=ref.device).view(1, 1, -1).expand(3, batch, -1)
    dummy = torch.zeros(batch, n, hidden_size, dtype=ref.dtype, device=ref.device)
    cos, sin = rotary_emb(dummy, positions)
    return cos, sin


def _rotate(x_pre: torch.Tensor, start: int, hidden_size: int, rotary_emb) -> torch.Tensor:
    """把一段 **pre-RoPE** 的 Q / K（`[层, 头, t, head_dim]`）按真实位置旋转。"""
    cos, sin = _rope_cos_sin(rotary_emb, x_pre, int(start), x_pre.shape[2], hidden_size)
    return apply_rotary_pos_emb(x_pre, x_pre, cos, sin)[0]


# ---------------------------------------------------------------- 会话封装


class MemorySession:
    """`MemoryStore` + `GraphDecoder` 的薄封装：写入一段 -> 之后多轮取回。

    每轮 `prefill()` 做两件事：
    1. **检索**：对当前 prompt 跑一次（不带记忆的）前向，取问题那段 token 的 Q 打分；
    2. **注入 + 预填充**：把 top-k 记忆项写进 KV cache，再 prefill 当前 prompt。

    于是每轮多一次短前向的开销（实测见报告）；不注入时（`use_memory=False`）与原来完全一样。
    """

    def __init__(
        self,
        store: MemoryStore,
        model,
        max_len: int = 1024,
        top_k: int = 1,
        threshold: float | None = None,
        query_last: int = 4,
        device: str | torch.device = "cuda",
    ) -> None:
        from .decode import GraphDecoder

        self.store = store
        self.model = model
        self.text = model.model
        self.dec = GraphDecoder(model, max_len=max_len, device=device)
        self.top_k = int(top_k)
        self.threshold = threshold
        self.query_last = int(query_last)
        self._captured = False

    # ---- 写入 ----

    def write(self, input_ids: torch.Tensor, span: tuple[int, int], label: str | None = None) -> int:
        return self.store.write(self.model, input_ids, span, label=label)

    # ---- 检索 + prefill ----

    @torch.inference_mode()
    def prefill(
        self,
        history_ids: torch.Tensor,
        current_ids: torch.Tensor | None = None,
        use_memory: bool = True,
        force_index: int | None = None,
        query_span: tuple[int, int] | None = None,
        place: str = "turn",
        frame: str = "alone",
    ) -> RetrievalInfo:
        """把 `history_ids`(+`current_ids`) 写进 cache，顺带做一次记忆检索。

        - `place="turn"`（**默认**）：记忆插在 `history_ids` 与 `current_ids` **之间**。
          与问题的距离只有"当前轮的长度"。实测：这样寻址 3/3 全对。
        - `place="front"`：记忆插在最前面（docs/03 的原始写法）。距离 = 整段历史，
          实测寻址退化（20 轮历史时 1/3）—— RoPE 长距离把 `Q·K` 抹平了。
        - `frame`：打分时假定的布局。`"alone"` = 每条记忆单独注入（与 `top_k=1` 的真实情况一致）；
          `"stack"` = 全部候选按顺序堆在一起。
        - `query_span`：问题在 **full prompt**（history+current）里的 token 区间，
          用 `chatfmt.find_span` 定位。**只取它的末尾 `query_last` 个 token** ——
          实测整句取平均更差（8/8 → 7/8）；不给就退回"prompt 最后 `query_last` 个 token"
          （那多半是 `<|im_start|>assistant`，**没有内容**，寻址会退化）。
        """
        info = RetrievalInfo()
        if place not in ("turn", "front"):
            raise ValueError(f"place 必须是 'turn' / 'front'，收到 {place!r}")
        full = history_ids if current_ids is None else torch.cat([history_ids, current_ids], dim=1)
        history_len = int(history_ids.shape[1])
        if place == "turn" and current_ids is None:
            raise ValueError("place='turn' 需要把当前轮单独给出来（current_ids）")

        if force_index is not None:
            use_memory = True
        if use_memory and len(self.store):
            scores = self.rank(history_ids, current_ids, query_span, place, frame, info=info)
            if force_index is not None:
                info.forced = int(force_index)
                info.hits = [(int(force_index), float(scores[int(force_index)]))]
            else:
                info.hits = self.store.retrieve(scores, top_k=self.top_k, threshold=self.threshold)
            if info.hits:
                items = [self.store.items[i] for i, _ in info.hits]
                info.prefix_len = self.store.inject_length(items)
                t_pf = time.perf_counter()
                if place == "turn":
                    self.dec.prefill(history_ids)  # 位置 0..h-1
                    self.store.inject(self.dec.cache, items, self.text.rotary_emb, offset=history_len)
                    self.dec.prefill(current_ids, offset=history_len + info.prefix_len, reset=False)
                else:
                    self.dec.prefill(
                        full,
                        prefix_writer=lambda cache: self.store.inject(
                            cache, items, self.text.rotary_emb
                        ),
                    )
                info.prefill_ms = (time.perf_counter() - t_pf) * 1000
                return info
        t_pf = time.perf_counter()
        self.dec.prefill(full)
        info.prefill_ms = (time.perf_counter() - t_pf) * 1000
        return info

    # ---- 检索（只打分，不注入）----

    @torch.inference_mode()
    def rank(
        self,
        history_ids: torch.Tensor,
        current_ids: torch.Tensor | None = None,
        query_span: tuple[int, int] | None = None,
        place: str = "turn",
        frame: str = "alone",
        info: RetrievalInfo | None = None,
    ) -> torch.Tensor:
        """给每条记忆打分，**不 prefill、不注入** —— 返回 `[n_items]` 的分数。

        检索与注入必须用**同一套位置坐标系**，所以这段逻辑只写一份，`prefill` 也走它。
        （教训：在测试里另手搓一遍打分，`query_span` 忘了截成末尾 `query_last` 个 token，
        命中率就从 8/8 掉到 7/8。）

        给了 `info` 就把 `query_ms` / `score_ms` / `scores` 填进去（供报告归因）。
        """
        if place not in ("turn", "front"):
            raise ValueError(f"place 必须是 'turn' / 'front'，收到 {place!r}")
        if place == "turn" and current_ids is None:
            raise ValueError("place='turn' 需要把当前轮单独给出来（current_ids）")
        full = history_ids if current_ids is None else torch.cat([history_ids, current_ids], dim=1)
        base = int(history_ids.shape[1]) if place == "turn" else 0
        if query_span is not None:
            query_span = (max(int(query_span[0]), int(query_span[1]) - self.query_last), int(query_span[1]))
        t0 = time.perf_counter()
        query = query_vectors(self.text, full, span=query_span, n_last=self.query_last)
        query_ms = (time.perf_counter() - t0) * 1000
        q0 = int(query_span[0]) if query_span is not None else max(0, full.shape[1] - self.query_last)
        item_starts, query_starts = self._frames(base, q0, frame)
        t1 = time.perf_counter()
        scores = self.store.scores(query, item_starts, query_starts, self.text.rotary_emb)
        if info is not None:
            info.query_ms = query_ms
            info.score_ms = (time.perf_counter() - t1) * 1000
            info.scores = scores
        return scores

    def _frames(self, base: int, q0: int, frame: str) -> tuple[list[int], list[int]]:
        """打分用的位置坐标系：第 i 条记忆的起点 + 查询的起点。

        `"alone"`：每条单独注入（与 `top_k=1` 的真实情况一致）→ 查询位置随该条长度变化。
        `"stack"`：全部候选按顺序堆在 `base`。
        """
        items = self.store.items
        if frame == "alone":
            return [base] * len(items), [q0 + it.n_tokens for it in items]
        if frame != "stack":
            raise ValueError(f"frame 必须是 'alone' / 'stack'，收到 {frame!r}")
        starts, pos = [], base
        for it in items:
            starts.append(pos)
            pos += it.n_tokens
        return starts, [q0 + pos] * len(items)

    # ---- 生成 ----

    @torch.inference_mode()
    def generate(self, n_new: int, stop_ids: set[int] | None = None) -> list[int]:
        """贪心生成：返回**新生成**的 token（prefill 已经算出的第一个 token 也算在内）。"""
        if not self._captured:
            self.dec.capture()
            self._captured = True
        out: list[int] = []
        while len(out) < int(n_new):
            tok = int(self.dec.input_ids.item())
            if stop_ids and tok in stop_ids:
                break
            out.append(tok)
            if len(out) >= int(n_new):
                break
            self.dec.step()
        return out

    def reset_cache(self) -> None:
        self.dec.cache.reset()
