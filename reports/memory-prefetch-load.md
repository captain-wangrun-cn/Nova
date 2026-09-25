# 第六轮 ② · 记忆段接 `SegmentPrefetcher`：2.70–3.37x，2.3 GiB 打到 4.49 GiB/s（2026-09-25）

> **一句话结论（已核查）**：`MemoryStore.load_prefetched()` 把 D41 的双缓冲预取接进了记忆加载。
> 2.34 GiB 的合成记忆从 **1376.9 → 510.1 ms（2.70x）**，实测带宽 **4.49 GiB/s**
> —— 正好打到 D41 侧会话测出的上限；小文件（16 MiB）也有 **3.08x**。
> **两条路读出的张量逐位相同**（不是"接近"：这条路径只搬字节，不算数）。
> **不违反 D07**：不另存裸 blob，直接按 safetensors **数据区偏移**搬原始字节，再在显存里零拷贝建视图。
> `pytest tests -q` → **125 passed**（新增 11 条）。

---

## 一、上一轮卡住的那个设计问题（现在有答案了）

E6（D41）把预取写法落地了，但**接不进 `memory.py`** —— `SegmentPrefetcher` 搬的是**裸字节**，
而 D07 要求记忆以 **safetensors** 持久化（`[8 字节头长][JSON 头][数据区]`）。当时留了两条路：

**(a)** 读 safetensors 数据区偏移、搬原始字节再 `view` 成张量；
**(b)** 另存一份裸 blob 供预取。

**选 (a)**。理由：**(b) 等于把同一份记忆存两遍**（S4 的记忆是 240 KiB/token，
1 万 token 就是 2.3 GiB，存两份没有道理），而且两份文件必须同步，迟早会出现"预取的那份是旧的"。

**(a) 能不能走，取决于三件必须先验的事**（`src/diagnostics/probe_safetensors_layout.py`，已核查）：

| # | 假设 | 实测 |
|---|---|---|
| 1 | 布局 = `[8B 头长 u64 LE][JSON 头][数据区]`，`data_offsets` 相对**数据区起点** | ✅ |
| 2 | 数据区里张量**连续、无夹缝**（没有对齐填充） | ✅ 覆盖字节数 == 数据区长度 |
| 3 | 整段拷进一块 `uint8` 显存缓冲后，`slice.view(dtype).view(shape)` 能还原 | ✅ 与 `safe_open` **逐位相同** |

> 第 3 条是这条路的关键：`view(dtype)` 对**非零 `storage_offset`** 可用（探针里 `lo=0`，
> 但张量切片本身就是非零偏移的视图，实测可用）。所以"一次 H2D + 全零拷贝建视图"成立。

---

## 二、实现

### 2.1 `SegmentPrefetcher` 加两个能力（`src/nova/prefetch.py`）

| 新增 | 为什么必须加 |
|---|---|
| `offset` / `length` | safetensors 的数据区不在文件开头（前面有头），"只搬数据区"要求能从**任意偏移**流式读 |
| `stream_into(dst)` | `stream()` 每段 `to(device)` 会**新分配**显存张量；搬 GB 级记忆时这是分配器抖动。`stream_into` 写进预分配缓冲，只做一次 H2D |

`stream_into` 保持 D41 定死的**顺序**：**先发起当前段的 H2D，再读下一段**（反了 PCIe 就藏不住）。

### 2.2 `MemoryStore.load_prefetched()`（`src/nova/memory.py`）

```python
meta, data_start, layout = read_safetensors_layout(path)   # 只读文件头
file_schema, fp, foreign = validate_memory_meta(meta, schema, fingerprint, allow_foreign)
buf = torch.empty(total, dtype=torch.uint8, device="cuda")  # 一块缓冲
SegmentPrefetcher(path, seg_bytes=4<<20, ring=4,
                  offset=data_start + lo, length=total).stream_into(buf)
views = {name: buf[s - lo : e - lo].view(dt).view(shape) ...}   # 零拷贝
```

**D09 的校验只写一份**：新增 `validate_memory_meta()`，`load()`（`safe_open` 路径）与
`load_prefetched()` 共用同一套判据 —— 两条路各写一份迟早会飘，那"宁可拒绝加载"就守不住了。

---

## 三、验收证据

### 3.1 端到端（`src/diagnostics/bench_memory_load.py`，`clocks.sm` 见各行）

合成记忆按**真实记账**造：双通路 **60 个 cache 槽位** × 8 KV 头 × 128 head_dim × (k+v) × fp16
= **240 KiB/token**（与 HANDOFF 第七节的实测量级一致）。

| 文件 | 大小 | `safe_open` 逐张量 | `prefetch` 双缓冲 | 倍数 | prefetch 带宽 | 逐位一致 |
|---|---:|---:|---:|---:|---:|:---:|
| 合成 1000 token | 234.4 MiB | 162.3 ms | **48.1 ms** | **3.37x** | 4.75 GiB/s | ✅ |
| 合成 4000 token | 937.5 MiB | 725.6 ms | **239.0 ms** | **3.04x** | 3.83 GiB/s | ✅ |
| 合成 10000 token | 2343.8 MiB | 1376.9 ms | **510.1 ms** | **2.70x** | **4.49 GiB/s** | ✅ |
| S4 demo（真实记忆） | 16.4 MiB | 17.0 ms | **5.5 ms** | **3.08x** | 2.90 GiB/s | ✅ |

- **2.34 GiB 那行的 4.49 GiB/s 正好等于 D41 的上限**（侧会话实测：pin + `readinto` + 双缓冲
  + `non_blocking` = 4.49 GiB/s；盘顺序读 5.09、PCIe H2D 12.2）—— 说明这条路径**已经打满**，
  再优化只能去动 D41 记的"并行上限 5.3%"那一小块。
- 小文件倍数反而更高（3.08–3.37x）：`safe_open` 的**逐张量开销**（12 次 `get_tensor` +
  每次一次分配）在小文件上占比更大。
- 4000 那行的带宽（3.83）低于 10000（4.49）：文件还没大到让双缓冲完全盖住启动开销。
- ⚠️ **`clocks.sm` 在这条基准里不是主角**：瓶颈在盘读 + PCIe，不在 SM。开始采样到 210 MHz
  （GPU 空闲）、结束 2505 MHz，**两次的吞吐数字仍可比**（同一块盘、同一条 PCIe）。
  这一点与算力基准相反，写在这里免得以后误读。

### 3.2 正确性（`tests/test_memory_prefetch.py`，11 条）

| 测试 | 判据 |
|---|---|
| `test_layout_matches_safe_open` | 头部解析出的偏移逐张量与 `safe_open` **逐位一致**；数据区末尾 == 文件末尾 |
| `test_prefetched_load_is_bit_identical` | `load_prefetched()` vs `load()` 每个张量逐位相同（含 label / schema / 指纹） |
| `test_prefetched_load_survives_segment_boundaries` | 段长 64 / 1000 / 4096 / 1 MiB（**不整除**文件、最后一段不满）都必须一致 |
| `test_prefetched_load_checks_schema` | D09 校验在预取路径上**同样生效**（schema 与指纹不匹配都拒绝） |
| `test_prefetched_load_allows_foreign` | `allow_foreign=True` 放行且 `store.foreign=True` |
| `test_prefetched_views_are_independent` | 多个记忆项从同一块缓冲切出，偏移算错就会互相覆盖 |
| `test_prefetch_range` / `test_prefetch_range_matches_stream` | `offset`/`length` 语义；多写一字节都算失败；`stream_into` 与 `stream` 结果相同 |

---

## 四、踩到的坑（写下来免得重踩）

1. **记账写错会得出好看的假数字**：第一版按**单通路 36 槽位**（144 KiB/token）造合成记忆，
   带宽数字比现在更漂亮。真实记忆是双通路 **60 槽位 = 240 KiB/token**（2× 冗余，
   见 HANDOFF 第七节的已知边界）。**基准的输入规模必须对着真实记账算**。
2. **造 2.3 GiB 合成记忆时宿主内存要留够**：`make_memory()` 先在 CPU 上建整份张量再 `save_file`，
   一次要 2.3 GiB 宿主内存 + 写盘缓冲。第一次跑（同一进程里连造 3 个文件）在
   `safe_open` 上抛 `Attempted to access the data pointer on an invalid python storage`，
   **单独跑同一个文件就正常** —— 是宿主内存压力，不是代码 bug。
3. **`stream()` 与 `stream_into()` 都要保留**：前者给"逐段消费"的调用方，
   后者给"一次性搬进大缓冲"。两者的段顺序必须一致，测试里专门卡了这一条。

---

## 五、待实测 / 已知边界

1. **端到端**"记忆加载 → 检索 → 注入 → 出 token"的**整条**链路延迟**待实测**：
   本报告量的是**加载段**（`SegmentPrefetcher` 的职责边界）。整条链路还包含
   取 Q 的一次完整前向（HANDOFF 第七节记的 ≈1× 基线前向）+ 打分 + 注入。
2. **L0 的 2× 冗余没去掉**：通路 0/1 各存一份（D31 的已知边界）。去掉它等于让记忆体积砍半，
   但要等交叉注意力真正打开后再定形，否则格式白改一次。
3. **记忆压缩未做**：现在是无损 KV，240 KiB/token 随区间线性增长（L1/L2 留给里程碑 3）。
4. **`load_prefetched` 目前要求张量连续无夹缝**：本项目自己写的文件满足；
   若以后用别的工具生成 safetensors，可能带对齐填充，那时要按张量分别搬（多几次 H2D）。

---

## 六、复现命令

```powershell
$env:HF_HOME=(Resolve-Path .).Path+'\.hf-cache'; $env:TMP=(Resolve-Path .).Path+'\.tmp'; $env:TEMP=$env:TMP
$env:HF_HUB_OFFLINE='1'

& .\.venv\Scripts\python.exe src\diagnostics\probe_safetensors_layout.py    # 布局前提（秒级）
& .\.venv\Scripts\python.exe -m pytest tests\test_memory_prefetch.py -q     # 11 条
& .\.venv\Scripts\python.exe -u src\diagnostics\bench_memory_load.py --tokens 1000 4000 --reps 3
& .\.venv\Scripts\python.exe -u src\diagnostics\bench_memory_load.py --tokens 10000 --reps 3
```
