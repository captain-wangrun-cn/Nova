# 第六轮 ①.5 · int8 流式 KV cache 接进解码路径：4K 上下文 3.26x，KV 常驻 0.54x（2026-09-24）

> **一句话结论（已核查）**：`src/nova/kvattn.py` 的 `KVInt8Cache` 把 D42 的融合核接进了
> `GraphDecoder`，**图解码整步**在 4K 上下文上从 **115.68 → 35.49 ms/token（3.26x）**，
> KV 常驻从 **1920 → 1035 MiB（0.539x）**，峰值显存 **6.68 GiB（D17 的 7GB 内）**，
> `pytest tests -q` → **114 passed**（新增 21 条）。
> 与「先还原再算」的逐元素一致性：合成数据 **≤4 ULP**（最大 2.44e-4 绝对，fp16 在 0.1 量级
> 上 1 ULP = 6.1e-5）。
> **未做**：8K 上下文的端到端数字（待实测，见第五节）；prefill 期间 fp16 暂存区仍按整段分配。

---

## 一、这一步要解决的问题

D42 结束时核是好的，但**没接进解码路径**，两个卡点写在 D42 的"待做"里：

1. **流式写入时分组尺子还没定死** —— `int8t64` 的每组要等第 64 个 token 到齐才有一条尺子，
   而解码是**一个 token 一个 token** 写的；
2. **CUDA Graph 的形状恒定** —— 已量化的前缀长度每 64 个 token 涨一次，看起来必然要重捕。

第 2 条是这一步最关键的判断，也是本次唯一一次"改设计"：

| 做法 | 后果 |
|---|---|
| 把前缀长度当**编译期常量**冻结进图 | 图每 64 步就过期，**而且过期不是变慢、是静默算错**：刚写满的组仍被当 fp16 尾部，而环槽已被更新的 token 覆盖 ⇒ 那段历史凭空消失 |
| 每 64 步重捕一次图 | capture 代价远超收益（本机单次 capture 约 0.3–1s） |
| **把长度放进显存标量、核内 `tl.load` 读**（采用） | 形状（grid）按**桶长**固定，数值每步都能涨 —— 这才是"形状恒定"的正确解法 |

可行性先单独核查过：Triton 在核内 `tl.load` 一个 1 元素张量能取到**更新后**的值
（`src/diagnostics/probe_kvattn_scalar.py`）。

---

## 二、实现（`src/nova/kvattn.py` · `KVInt8Cache`）

三段结构，按 token 位置切：

```
[0, Q)      int8：Q = 64·(U//64)，尺子已定死的整组（U = 已写入总数）
[Q, U)      fp16 尾部环：最近 ≤64 个 token，槽位 = 绝对位置 mod 64
```

**解码步每层做的事**（全部图内安全，零 `.item()`）：

1. `update()`：K/V 写进环槽 `pos % 64`；
2. **无条件**把整条环（64 个 token）量化成 int8，写进第 `pos // 64` 组 ——
   部分填充时这一组虽然尺子不对，但**不会被读到**（`Q` 只到整组边界）；
   写满那一刻（`pos % 64 == 63`）同一层内**先写后量化**，尺子正好定死；
3. 用 `pos` 推出三个设备标量：`_written = pos+1`、`_qlen = 整组边界`、`_pg = 组数`；
4. `attend()`：int8 前缀（融合核，`used=_qlen`）+ 环尾部（SDPA，掩码由设备标量算）
   两段 `logsumexp` 合并。

**尾部环的掩码**（`ring_mask()`）：环槽 `i` 的绝对位置 `p = u - ((u-i) % 64)`，
`u = _written - 1`；有效条件 `_qlen ≤ p ≤ u`。下界是 `_qlen`，正是"已经在前缀里、
不能再算一遍"的那条线 —— 少了它，那批 token 会被**加权两次**。

**prefill**：融合核只处理 `q_len == 1`，prefill 是变长 + 因果掩码，另一套写法。
所以 prefill 期间用一块 fp16 暂存区（按**实际 prompt 长度**分配，不按 `max_len`），
段末 `finish_prefill()` 一次性量化整组 + 填环 + 释放暂存区。

---

## 三、验收证据

### 3.1 与「先还原再算」的一致性（合成数据，`tests/test_kvattn_stream.py`）

参考 = 前 `Q` 个 token 走 `quantize_int8(..., "token")` → `dequantize_int8` 还原成 fp16，
其余保持 fp16，再走 `SDPA(MATH)`：

| L | 前缀 Q | 最大绝对偏差 | 折算 |
|---:|---:|---:|---:|
| 40 | 0（全 fp16） | 0 | — |
| 65 | 64 | 2.44e-4 | ≈4 ULP |
| 100 | 64 | 2.44e-4 | ≈4 ULP |
| 128 | 128 | 6.10e-5 | ≈1 ULP |
| 192 | 192 | 1.22e-4 | ≈2 ULP |
| 320 | 320 | 3.05e-5 | <1 ULP |

**判据为什么是 4 ULP 而不是融合核那 2 ULP**（已核查）：融合核比的是**同一条**路径的两个实现；
流式 cache 多了一次**结构性**舍入 —— 前缀走 int8 核、尾部单独走一次 fp16 SDPA，
两者按 `logsumexp` 合并后再舍到 fp16。参考（一次性 SDPA 后舍一次）与它（合并后再舍）
不可能逐位对齐。判据取 4 ULP / 1e-3 绝对，与结构差异匹配。

### 3.2 端到端（`src/diagnostics/bench_int8_decode.py`，`clocks.sm 2475 MHz`）

| L | fp16 ms/token | int8 ms/token | 倍数 | fp16 KV | int8 KV | 比值 | int8 峰值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 46.47 | 33.06 | **1.41x** | 485 MiB | 273 MiB | 0.563x | 4.98 GiB |
| 2048 | 69.94 | 33.53 | **2.09x** | 960 MiB | 525 MiB | 0.547x | 5.54 GiB |
| 4096 | 115.68 | 35.49 | **3.26x** | 1920 MiB | 1035 MiB | 0.539x | 6.68 GiB |

- **int8 的 ms/token 几乎是平的**（33.1 → 33.5 → 35.5），fp16 随 L 线性涨 —— 这就是带宽收益的形状；
- 桶长：1K→2070、2K→4096、4K→8192（P0 的分桶）。KV 常驻按**桶长**算，不按真实长度；
- 峰值显存 6.68 GiB < D17 的 7GB。fp16 同条件 6.59 GiB。

### 3.3 开发中修掉的三个真 bug（都留了测试）

| # | 症状 | 根因 |
|---|---|---|
| 1 | 合并后的输出比参考差 **4.8e-2** | `logsumexp` 权重写成 `[1,H,1]`，与 `[1,H,1,D]` 广播时**对齐到了 query 维**，变成 `[1,H,H,D]`，每个元素被别的头污染。必须写成 `[1,H,1,1]` |
| 2 | 上下文 < 64 时输出 **NaN** | grid 按桶长固定、`used` 是运行期值 ⇒ 多余的 split 里 `l_run == 0`，核做 0/0。改为 `l_run > 0` 时才算除法，否则输出 0 / lse `-inf` |
| 3 | 图捕获直接失败 | `int8_attn_decode` 的 Python-int 路径**在图内分配** `torch.tensor([used])`。改成复用**常驻标量缓冲**，只做原地 `fill_` |

### 3.4 测试

```powershell
& .\.venv\Scripts\python.exe -m pytest tests -q          # 114 passed（原 93 + 流式 17 + 端到端 4）
& .\.venv\Scripts\python.exe -m pytest tests\test_kvattn_stream.py -q    # 17 条
& .\.venv\Scripts\python.exe -m pytest tests\test_graph_decode_int8.py -q # 4 条（加载模型 ~25s）
```

`test_graph_decode_int8.py::test_int8_decode_runs_and_agrees_on_tokens` 实测 int8 与 fp16 的
贪心 token **4/4 相同**（判据是 ≥3/4）。

---

## 四、顺手修掉的两个显存/时间坑

| 坑 | 修法 | 效果 |
|---|---|---|
| prefill 按**组**循环量化：36 层 × 64 组 × 2 张量 ≈ 4600 次小 kernel 调用 | 每层**一次**量化整段 | 4096 token 的 prefill **18.2s → 3.65s** |
| prefill 的加性掩码按**头**展开成 `[1,32,n,hist]` | 改回 `[1,1,n,hist]` 让 SDPA 广播 | 4096 token 上省 **1 GiB**，峰值 **7.62 → 6.68 GiB** |

---

## 五、待实测 / 已知边界（写清楚，别当已经做了）

1. **8K 上下文的端到端数字：待实测。** 4096 token 时暂存区（fp16，36 层 × 2 × 8 × 4096 × 128 × 2 = 604 MiB）
   与 int8 常驻同时在场；8K 时两者各 1.2 GiB，加上模型约 7.0 GiB —— 试跑时本机进入换页，未取到数字。
   **修法已明确**：prefill 按 64 的整数倍分块（如 512），暂存区只留当前块，块末量化；
   注意块大小必须**是 64 的整数倍**，否则分组的尺子会被块边界切坏（这一版就是为了避开它才整段保留）。
2. **真模型 K/V 的流式一致性：待实测。** 3.1 是合成数据；D42 的真数据复核是在**核**这一层做的
   （layer 0/18/35 = 1/2/31 ULP，判据通过）。流式 cache 多一次合并舍入，真数据上应重跑一遍。
3. **S4 记忆注入与 `quant="int8"` 不兼容**：`MemoryStore.inject` 直接写 `cache.key_cache[slot]`，
   而 int8 cache 的常驻是 int8 + 环，两条路不通用。**没有**在本轮解决。
4. **跨桶 `grow()` 不支持**：int8 缓冲、尺子、环、标量都要重建，`GraphDecoder.grow()` 在
   `quant="int8"` 下直接抛 `NotImplementedError`（明确报错，不静默算错）。
5. **`int8_attn_decode` 的 Python-int 路径**用的是**模块级常驻标量缓冲**（每个 device 一对）。
   这意味着同一 device 上多个 cache 交替调用时，缓冲会被互相覆盖 —— 但每次调用都会先
   `fill_` 再启核，且核在同一个 stream 上顺序执行，所以安全。真要并发必须各自持标量。

---

## 六、复现命令

```powershell
$env:HF_HOME=(Resolve-Path .).Path+'\.hf-cache'; $env:TMP=(Resolve-Path .).Path+'\.tmp'; $env:TEMP=$env:TMP
$env:TRITON_CACHE_DIR=$env:TMP+'\triton-cache'; $env:TORCHINDUCTOR_CACHE_DIR=$env:TMP+'\inductor-cache'
$env:HF_HUB_OFFLINE='1'

& .\.venv\Scripts\python.exe -m pytest tests -q                                   # 114 passed
& .\.venv\Scripts\python.exe -u src\diagnostics\bench_int8_decode.py --lens 1024 2048 4096 --reps 30
```
