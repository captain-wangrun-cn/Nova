r"""② 前置核查 · **safetensors 的字节布局**能不能直接当"原始字节段"搬。

## 为什么必须先验这一条

`SegmentPrefetcher`（D41）搬的是**裸字节**，而 D07 要求记忆以 **safetensors** 持久化。
要把两者接起来，只有一条不违反 D07 的路：**按数据区偏移搬原始字节，再在显存里 `view` 成张量**
—— 不另存一份裸 blob（那就是"同一份记忆存两遍"）。

这条路能不能走，取决于三件事，本脚本逐条验：

1. `.safetensors` 的真实布局是不是 `[8 字节头长][JSON 头][数据区]`，且张量的
   `data_offsets` 相对**数据区起点**（不是文件起点）；
2. 把数据区**整体**拷进一块 `uint8` 显存缓冲后，能不能用 `slice.view(dtype).view(shape)`
   得到**逐位相同**的张量（`view(dtype)` 对非零 `storage_offset` 是否允许）；
3. 张量在数据区里是不是**连续**的（有没有夹缝 / 对齐填充）。

## 结论（已核查）

三条都成立：布局如上、`view(dtype)` 对任意偏移可用、张量连续无夹缝。
所以 `MemoryStore.load_prefetched` 可以"一次 H2D 搬完整段 + 全零拷贝建视图"。

跑法：
    & .\.venv\Scripts\python.exe src\diagnostics\probe_safetensors_layout.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

_DTYPES = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BF16": torch.bfloat16,
}


def read_layout(path: Path):
    """返回 `(meta, data_start, {name: (dtype, shape, 数据区相对起止)})`。"""
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError("文件太短，读不出 safetensors 头长")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
    meta = header.get("__metadata__") or {}
    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        s, e = spec["data_offsets"]
        tensors[name] = (_DTYPES[spec["dtype"]], tuple(spec["shape"]), int(s), int(e))
    return meta, 8 + header_len, tensors


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = Path(tempfile.mkdtemp(dir=str(ROOT / ".tmp")))
    path = tmp / "layout.safetensors"

    torch.manual_seed(0)
    tensors = {
        "item0.k": torch.randn(3, 2, 5, 4, dtype=torch.float16),
        "item0.v": torch.randn(3, 2, 5, 4, dtype=torch.float16),
        "item1.k": torch.randn(3, 2, 7, 4, dtype=torch.float16),
        "item1.v": torch.randn(3, 2, 7, 4, dtype=torch.float16),
    }
    save_file(tensors, str(path), metadata={"nova_memory": json.dumps({"format": "nova-memory"})})

    meta, data_start, layout = read_layout(path)
    file_size = path.stat().st_size
    print(f"文件 {file_size} B · 头 {data_start} B · 数据区 {file_size - data_start} B")
    assert meta.get("nova_memory"), "头部元数据没读到"

    lo = min(v[2] for v in layout.values())
    hi = max(v[3] for v in layout.values())
    total = hi - lo
    # 张量连续、无夹缝：区间总和 == 数据区长度，且首尾相接
    covered = sum(v[3] - v[2] for v in layout.values())
    print(f"数据区 {total} B · 张量覆盖 {covered} B · 连续无夹缝 = {covered == total}")
    assert covered == total, "数据区里有无法解释的字节（夹缝 / 对齐填充）"

    with open(path, "rb") as fh:
        fh.seek(data_start + lo)
        blob = fh.read(total)
    buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device)

    ok = True
    for name, (dtype, shape, s, e) in layout.items():
        view = buf[s - lo : e - lo].view(dtype).view(shape)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            want = f.get_tensor(name)
        same = torch.equal(view.cpu(), want)
        ok = ok and same and view.is_contiguous()
        print(f"{name:9s} {str(shape):14s} view==safe_open {same} · 连续 {view.is_contiguous()}")
    assert ok, "视图与 safe_open 不一致 —— 这条路走不通"

    print("\n已核查：布局 = [8B 头长][JSON 头][数据区]，张量连续，偏移视图逐位一致")
    print("⇒ 记忆段可以「按数据区偏移搬原始字节 + 显存内零拷贝建视图」，不违反 D07")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
