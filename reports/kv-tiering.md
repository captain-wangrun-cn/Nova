# E6 · PCIe / 主机内存分层（2026-09-23 · **A 段落地，B 段上限确认 = 5.3%**）

> **一句话结论（已核查）**：**预取加载器的正确写法确实值 2.5x**（朴素 1.76 → **4.49 GiB/s**）；
> **"显存读 + PCIe 搬运并行"也完全成立**（并行效率 **1.00x max**，两条路互不抢带宽）。
> 于是侧会话的公式被实测钉住：**可以白放主机内存的 KV 比例上限 = PCIe / 显存带宽 = 5.3%**。
> 结论是**二级优先**：先修"有效带宽只有 60 GiB/s"那件事（P0 已做掉一部分），分层只值 5–20%。

## 一、A 段：预取加载器（`src/nova/prefetch.py`）

| 写法 | 1 GiB 耗时 | 带宽 | 相对 |
|---|---:|---:|---:|
| 朴素 `read()` + H2D | 569 ms | **1.76 GiB/s** | 1.00x |
| **pin + `readinto` + 双缓冲 + `non_blocking`** | 223 ms | **4.49 GiB/s** | **2.55x** |

- 判据"端到端 ≥ 4.5 GiB/s"：实测 **4.49**，**达标**（侧会话同口径实测 4.8，同一量级）。
- **工程约束（进生产路径）**：加载记忆段**只有这一种写法**。`SegmentPrefetcher.stream()` 每次
  先发起 H2D（`non_blocking`）再去读下一段 —— **顺序不能反**，反了就退化成串行。
- `readinto` 需要可写 buffer，torch 张量不是 ⇒ 用 `memoryview(t.numpy())` 包一层（已封装在类里）。

## 二、B 段：两条路并行（`probe_kv_tiering.py`）

| 项 | 耗时 | 带宽 |
|---|---:|---:|
| 显存读 1 GiB（Triton reduce） | 4.3 ms | **231.1 GiB/s** |
| H2D 搬运 1 GiB（pin） | 80.4 ms | **12.4 GiB/s** |
| **两条 stream 同时跑** | **80.5 ms** | — |

- `max = 80.4 ms`、`sum = 84.7 ms`、实测 **80.5 ms** ⇒ **并行效率 1.00x max（完全重叠）**。
- ⇒ 侧会话的公式 `总时间 = max(A/显存带宽, B/PCIe带宽)` **成立**，
  可白放主机内存的 KV 比例上限 = `12.2 / 231.1` = **5.3%**（与侧会话的 5.3% 一致）。
- 若显存侧没打满（P0 之前 decode 的有效读只有 ~60 GiB/s），这个上限是 **~20%**；
  P0 把容量按桶分配之后，显存侧更接近打满，**白赚额度相应缩到 5% 量级**。

## 三、结论

1. **A 段进生产路径**：`SegmentPrefetcher` 作为记忆段/持久化 KV 的**唯一**加载写法。
2. **B 段是二级优先**：并行成立但只有 **5.3%（理想）/ ~20%（显存没打满时）** 的免费额度，
   而它要求"每步细粒度重叠传输"的工程复杂度 ⇒ **排在"把显存侧打满"之后**。
3. **不要做的三件事**（侧会话结论，本轮无异议）：不做通用 KV offload（全量卸载实测 325 ms/字）、
   不 pin 十几 GB（pin 只留给预取环）、不把预取池当计算台。
4. **待实测**：真实记忆段（`MemoryStore` 的 safetensors 段）走 `SegmentPrefetcher` 的端到端取回延迟 ——
   本轮只测了裸文件搬运，没接进 `memory.py` 的检索路径。

## 四、结论状态

| 结论 | 状态 |
|---|---|
| pin + 双缓冲 + non_blocking = 4.49 GiB/s（朴素 2.55x） | ✅ 已核查 |
| 显存读与 H2D 完全重叠（1.00x max） | ✅ 已核查 |
| 白赚上限 = PCIe/显存带宽 = 5.3% | ✅ 已核查 |
| 接进 `memory.py` 检索路径的端到端延迟 | 🧪 待实测 |

## 五、复现命令

```powershell
$env:HF_HOME=(Resolve-Path .).Path+'\.hf-cache'; $env:TMP=(Resolve-Path .).Path+'\.tmp'; $env:TEMP=$env:TMP
$env:TRITON_CACHE_DIR=$env:TMP+'\triton-cache'

& .\.venv\Scripts\python.exe -u src\diagnostics\probe_kv_tiering.py --gib 1.0 --reps 2
```

> ⚠️ 探针会在 `.tmp/` 生成一个 1 GiB 的测试文件（跑完可以删）；不需要加载模型。
