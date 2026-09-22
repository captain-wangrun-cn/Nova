# S3 · 速度路径 ①（二）：CUDA Graph 解码 —— 把 6500 次 kernel 启动压成 1 次

> 日期：2026-09-22 · 阶段：**S3 收尾** · 结论标注：**已核查 / 待实测 / 推测**
> 代码：`src/nova/cache.py`（静态 KV cache）、`src/nova/decode.py`（`GraphDecoder`）
> 基准：`src/bench_graph.py` · 测试：`tests/test_graph_decode.py`（**12 passed**）
> 诊断：`src/diagnostics/microbench_op_cpu.py`、`probe_cuda_graph{,2,3}.py`、`check_graph_exactness.py`、`profile_graph_decode.py`
> 关联决策：**D29**（新增）、D27（CPU 派发受限）、D17（显存预算）

---

## 一、结论先行（已核查）

**同一份权重，只把"每步解码"包进一张 CUDA Graph，单通路从 63.0 ms/token 掉到 16.2 ms/token。**

| 配置 | 层数 | tok/s | ms/token | 峰值显存 |
|------|:---:|:---:|:---:|:---:|
| HF 基线 | 36 | 13.24 | 75.5 | 2.73 GiB |
| Nova 单通路 eager | 36 | 15.87 | 63.0 | 3.04 GiB |
| **Nova 单通路 + CUDA Graph** | 36 | **61.84** | **16.2** | 3.09 GiB |
| Nova 双通路 eager | 60 | 9.10 | 109.9 | 4.57 GiB |
| **Nova 双通路 + CUDA Graph** | 60 | **40.26** | **24.8** | 4.64 GiB |

- **单通路 4.7x**（对 HF 基线）/ **3.9x**（对 Nova eager）
- **双通路 3.0x**（对 HF 基线）/ **4.4x**（对 Nova eager）
- 正确性：单通路 **24/24**、双通路 **24/24** 个 token 与 eager 解码完全相同
- 显存 4.64 GiB，仍在 D17 的 7GB 预算内

> **路线图的 60+ tok/s 目标，单通路已达成（61.84）。** 双通路 40.26 还差一截，
> 差距来自 bnb 的 4-bit 路径本身（见第六节）。

---

## 二、根因：本机**单次 kernel 启动 13.5us CPU**（已核查）

`src/diagnostics/microbench_op_cpu.py` 实测（**不 sync** 连发 N 次，测 CPU 塞队列的时间）：

| 算子 | CPU 时间 |
|------|:---:|
| `x.view(1,1,2560)` | **1.29 us**（无 kernel 启动） |
| `x.to(fp16)`（no-op） | **0.32 us** |
| `x.contiguous()`（no-op） | **0.10 us** |
| **`x * 2.0`** | **15.96 us** ← 一个纯逐元素算子 |
| `nn.Linear` fp16 M=1 | 42.02 us |
| `bnb Linear4bit` NF4 M=1 | 70.80 us |
| eager RMSNorm（手写） | 138.33 us |
| `F.rms_norm` | 114.58 us |
| Triton 融合 RMSNorm | 61.62 us |
| `F.scaled_dot_product_attention` M=1 | **314.14 us** |

启动开销**严格线性**（`probe_cuda_graph.py`）：

```
    1 个 mul  cpu=  18.84us  -> 每次 18.84us
    4 个 mul  cpu=  53.58us  -> 每次 13.40us
   16 个 mul  cpu= 218.05us  -> 每次 13.63us
   64 个 mul  cpu= 887.48us  -> 每次 13.87us
```

**Nova 单通路每 token 有 ~6500 次算子调用** → 6500 × ~9.6us ≈ 62ms。
这与 D27 的 `CPU enqueue-only 76.92 ms/step ≈ wall 77.02 ms` 完全一致。
**CPU 不是"算得慢"，是"发不动"。** GPU 全程挨饿。

---

## 三、为什么 CUDA Graph 能救（已核查）

`probe_cuda_graph.py` / `probe_cuda_graph2.py` / `probe_cuda_graph3.py` 逐个验证了前提：

| 问题 | 实测 |
|------|------|
| 纯 aten 算子进图后 replay 多快？ | 256 个 `mul`：eager 3634us → **replay 273us（13.3x）** |
| **bnb `Linear4bit` 能进图吗？** | ✅ 能。eager 96.6us → replay **25.5us**，**数值逐位一致**（`max|diff| = 0`） |
| **Triton kernel 能进图吗？** | ✅ 能。eager 86.9us → replay **30.8us**，**数值逐位一致** |
| 两者能共存吗？ | ✅ 能。`RMSNorm + 7x Linear4bit` 图 wall = 157us（≈ 两者之和） |
| 图 replay 有固定底噪吗？ | 有，**约 28us CPU / 次**（与图内节点数无关） |

**结论：整步捕获后，CPU 从 62ms/token 降到 ~10us/token，瓶颈交还给 GPU。**

---

## 四、实现（已核查）

### 4.1 `StaticKVCache`（`src/nova/cache.py`）

`transformers.DynamicCache.update()` 用 `torch.cat` 增长张量，形状每步都变，**无法进图**。
替换为预分配的 `[batch, kv_heads, max_len, head_dim]`：

```python
self.key_cache[layer_idx].index_copy_(2, self.pos, key)   # 原地写一个槽位
```

- `self.pos` 是**GPU 上的** `[1]` int64 张量 —— 放 CPU 会触发图内同步
- 返回**整条**定长 cache（未写入的槽位是 0），由 attention mask 屏蔽
- prefill 走 `append_prefill()`（eager，多 token 一次写）

### 4.2 加性 mask 表（`GraphDecoder._build_mask_table`）

解码时注意力长度必须**恒定**，否则形状变了图就废了。做法：预计算 `[max_len, max_len]`
的加性 mask 表（第 `pos` 行 = 前 `pos+1` 位 0、其余 `-inf`），图内一次
`mask_table.index_select(0, pos)` 取当前行 —— **一个节点、形状恒定**。

### 4.3 整步在图内（`GraphDecoder._body`）

```
position_ids = pos.view(1,1,1).expand(3, 1, 1)
mask         = mask_table.index_select(0, pos).view(1,1,1,max_len)
hidden       = [36 / 60 层]（KV 原地写入槽位 pos）
logits       = lm_head(hidden)
input_ids.copy_(logits[:, -1].argmax(-1))    # 贪心 token 原地写回输入缓冲
pos.add_(1)
```

**CPU 每 token 只发一次 `graph.replay()`。** 输入缓冲、位置、mask 全部在图内自洽推进，
不需要 CPU 侧任何准备动作。

### 4.4 三个必须注意的点

1. **replay 必须在 `torch.inference_mode()` 内** —— 图内对输入缓冲做了原地写入，
   否则报 `Inplace update to inference tensor outside InferenceMode is not allowed`。
2. **`gqa_in_sdpa`**：HF 的 `use_gqa_in_sdpa` 在有 mask 时返回 False，于是退回 `repeat_kv`。
   图解码下那是每层每 token 白拷 8MB。实测 SDPA 带 mask 也支持 `enable_gqa`，
   已在 `LeanAttention` 里改成默认走 GQA。
3. **多个图并存会互相拖慢**：开发中曾出现"7 个线性层进图只要 148us、加上一个 norm 变 1184us"
   的假象 —— 原因是同时存活的多个图各自持有独立显存池。**测图要一个一个来，测完释放。**

---

## 五、开发中踩到的真 bug：prefill 把最后一个 prompt token 重复算了一遍（已修）

| 现象 | 根因 |
|------|------|
| 单通路 16/16 一致，**双通路只有 17/24** | `prefill()` 把 `input_ids` 填成**最后一个 prompt token**，而 `pos` 已经指向 `n`。图的第一步于是在位置 `n` 上**把这个 token 再算一遍** —— 输出看着通顺，序列是错的 |

**修法：** `prefill()` 结束后，输入缓冲填**第一个生成 token**（`argmax(prefill logits)`）。
修完单通路 **24/24**、双通路 **24/24**。

> **教训：** "输出看起来通顺"完全不能当作正确性证据。这类 bug 只有
> **逐 token 对照 eager** 才能发现 —— 已固化成 `tests/test_graph_decode.py::test_prefill_does_not_duplicate_last_token`。
> 这是本项目第三次踩到"看着对、其实错"（前两次见 [s3-dual-path-skeleton.md](s3-dual-path-skeleton.md) 第三节）。

### 数值一致性（`check_graph_exactness.py`）

强制两边吃同一个 token，逐步比 logits：

```
prefill logits 差异   : max|diff| = 7.812e-03
step 1..16           : max|diff| 3.1e-02 ~ 8.6e-02   argmax 16/16 相同
```

logits 量级约 40，偏差 3e-2 ~ 8e-2 即 **fp16 舍入级别**（padding 让 softmax 分母多算了
255 个 `-inf` 项，累加顺序与 eager 不同）。**不是 bug。**

---

## 六、剩下的 16.2ms 花在哪（已核查 · 局部图分解）

CUPTI **看不到图内 kernel**（profiler 只报 0.01 ms/token），所以改用**局部图**测量：

| | 内容 | wall |
|:-:|------|:---:|
| **A** | 整图（36 层 + lm_head） | **17.01 ms/token** |
| **B** | 只有 36 层（不含 lm_head） | **12.90 ms**（75.9%） |
| **C** | 只有 lm_head（151936×2560） | **3.12 ms**（18.3%） |
| | 其它（embed / argmax / mask） | ~1.0 ms（5.8%） |

**怎么读：**

1. **lm_head 已经打满带宽**：`151936×2560` fp16 = **0.78 GB**，3.12 ms → **250 GB/s**
   （本机实测带宽 234 GB/s）。**它没有优化空间**，除非把 lm_head 也量化 —— 但那会
   改变 `tie_word_embeddings` 的语义，需要单独评估。
2. **36 层的 12.90 ms 是主要缺口**：4-bit 权重总量 **1.23 GB**，带宽下界 **5.3 ms**，
   实测 **12.90 ms → 2.4x 差距**。
3. **差距的来源（已核查）**：bnb 的 `Linear4bit` 在 M=1 时**根本没有走 packed 4-bit GEMV** ——
   同样形状下单算子 GPU 耗时 bnb **29.2us** vs fp16 `nn.Linear` **27.9us**，几乎一样。
   说明它在 **dequant 到 fp16 workspace + cublas**。这既多搬运了约 2 倍数据，
   又白白丢掉 4-bit 的全部带宽优势。

> **注意（待实测）：** 上面 C 的 3.12 ms 与 B 的 12.90 ms 是在 `max_len=256` 下测的。
> `max_len` 增大时 attention 成本会随之上升（每 token 都要读满整条 cache），需另行测量。

---

## 七、下一步（按收益排序）

| # | 动作 | 依据 | 预估 |
|:-:|------|------|:---:|
| 1 | **自写 Triton NF4 dequant+GEMV**，替掉 bnb 的 `Linear4bit` | 36 层 12.90ms vs 带宽下界 5.3ms，**2.4x 缺口**；且 bnb 未用 packed kernel | **最大** |
| 2 | 量化 `lm_head` / `embed_tokens` | 3.12 ms/token，占整图 18.3% | 中（需先解决 tie_word_embeddings 语义） |
| 3 | 融合 RoPE、去掉冗余 `_to_copy` / `view` / `as_strided` | 图内仍逐节点执行，GPU 侧每节点约 1us | 中 |
| 4 | 融合注意力（当前 SDPA math 回退，M=1 时 CPU 派发高达 314us/次） | 图内已不痛，但 GPU 侧仍可省 | 低 |

**做完第 1 项后双通路的外推：** `12.90 × (60/36) ≈ 21.5ms` 的层成本降到约 `8.8ms`
→ 双通路约 `8.8 + 3.12×2 + 其它 ≈ 18ms` → **约 55 tok/s**（推测，需实测）。

---

## 八、复现命令

```powershell
$env:HF_HOME='H:\Nova\.hf-cache'; $env:TMP='H:\Nova\.tmp'; $env:TEMP=$env:TMP; $env:HF_HUB_OFFLINE='1'
$env:TRITON_CACHE_DIR='H:\Nova\.tmp\triton-cache'; $env:TORCHINDUCTOR_CACHE_DIR='H:\Nova\.tmp\inductor-cache'

# 验收
& .\.venv\Scripts\python.exe -m pytest tests -q                      # 12 passed

# 基准（含 HF 基线）
& .\.venv\Scripts\python.exe src\bench_graph.py --paths 1 --n 24
& .\.venv\Scripts\python.exe src\bench_graph.py --paths 2 --n 24 --no-hf

# 诊断
& .\.venv\Scripts\python.exe src\diagnostics\microbench_op_cpu.py    # 单算子 CPU 派发开销
& .\.venv\Scripts\python.exe src\diagnostics\probe_cuda_graph.py     # 启动线性 + 能否进图
& .\.venv\Scripts\python.exe src\diagnostics\check_graph_exactness.py --paths 2
& .\.venv\Scripts\python.exe src\diagnostics\profile_graph_decode.py --paths 1
```

---

## 九、本次新增/改动的文件

| 文件 | 说明 |
|------|------|
| `src/nova/cache.py` | **新增** `StaticKVCache` |
| `src/nova/decode.py` | **新增** `GraphDecoder`（捕获 / 回放 / generate） |
| `src/nova/layers.py` | `LeanAttention.gqa_in_sdpa`：有 mask 时也走 SDPA 的 GQA |
| `src/bench_graph.py` | **新增** HF / Nova eager / Nova graph 三方对照基准 |
| `tests/test_graph_decode.py` | **新增** 4 条验收测试 |
| `src/diagnostics/microbench_op_cpu.py` | **新增** 单算子 CPU 派发微基准 |
| `src/diagnostics/probe_cuda_graph{,_2,_3}.py` | **新增** 进图可行性探针 |
| `src/diagnostics/check_graph_exactness.py` | **新增** 逐步 logits 对照 |
| `src/diagnostics/profile_graph_decode.py` | **新增** 局部图时间分解 |
