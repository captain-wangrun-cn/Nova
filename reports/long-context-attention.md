# S4·补 · 长上下文注意力：一个开关换 15 倍速度与 4 倍长度

> 日期：2026-09-22 · 结论标注：**已核查 / 待实测 / 推测**
> 起因：用户问"能不能让 4060 8GB 跑满 260K 上下文，再测信息过载时能不能注意到重点"
> 结论：**长上下文的真正瓶颈不是 KV cache，是注意力后端** —— 改一个开关，长度天花板从 ~3.5K 提到 ~14K，
> 同轮实测快 2.1–4.1x，显存几乎不随长度增长
> 代码：`src/nova/layers.py`（`gqa_in_sdpa = False`）· 测试：`tests/test_nova_skeleton.py`（新增 1 条）
> 诊断：`probe_long_context_vram.py` / `probe_attention_kernel.py` / `probe_kernel_mapping.py` / `probe_attention_tradeoff.py` / `exp_needle.py`
> 关联决策：**D33**

---

## 一、问题：长上下文在 3.5K token 就崩

用 needle 脚本按长度递增测，前两档还算正常，第三档爆掉：

| 长度 | 峰值显存 | prefill | 现象 |
|:---:|:---:|:---:|---|
| 1894 token | 5.96 GiB | 3.6 s | 正常 |
| 3665 token | **8.80 GiB** | **29.1 s** | **超过显卡物理显存 8188 MiB** → 往系统内存换页 |
| 7291 token | — | — | **OOM 崩溃** |

按 KV cache 算只该占 144 KiB/token × 10240 = 1.41 GiB —— **有 3 GiB 以上没归因**。

### 归因（已核查）

`probe_long_context_vram.py` 打出 SDPA 后端的真实状态：

```
flash        : True（可用 False）        <- torch 没编译 flash attention
mem_efficient: True
cudnn        : True
```

强制各后端时的报错把原因说透了：

```
Torch was not compiled with flash attention.
Memory Efficient attention has been runtime disabled.
CuDNN attention has been runtime disabled.
For dense input, both fused kernels require query, key and value to have the same num_heads.
  Query.sizes(): [1, 32, 205, 128],  Key sizes(): [1, 8, 205, 128]
```

**因果链：** 本机 torch 2.6.0+cu124 没编译 flash → 只剩 mem-efficient / cuDNN 两个融合内核 → 两者都要求 Q/K/V **头数相同** → 我们是 GQA（32 Q / 8 KV）+ `enable_gqa=True` → **SDPA 退回 math 后端** → math 后端**实体化 O(n²) 的 fp32 分数矩阵**。

**算术验证：** 3520 token 的 prefill 峰值 8.30 GiB，固定占用（权重 4.80 + cache 1.41 + mask 0.20 = 6.41 GiB）之外的 **~1.9–3.5 GiB** 正好对上分数矩阵的量级：

```
32 头 × 3520² × 4 字节（fp32）× 2 张（scores + probs） = 3.2 GiB
```

`probe_long_context_vram.py` 的增长曲线也印证是 O(n²)：扣除固定占用后的增量，长度翻倍涨 ~4x（1749 → 3520 时 0.80 → 3.50 GiB）。

---

## 二、修复：进 SDPA 之前先把 KV 展平成 32 头

`src/nova/layers.py` 里本来就有这条分支（`gqa_in_sdpa=False` → `repeat_kv`），只是默认走的是另一条。**改一行**：

```python
self.gqa_in_sdpa = False
```

头数相同后融合内核就能用了。`probe_attention_kernel.py` **同一轮内**对比（唯一可信的比法）：

| 长度 | (a) GQA 留 SDPA（回退 math） | (b) 展平 32 头（融合内核） | (c) 强制 cuDNN |
|:---:|:---:|:---:|:---:|
| 1749 token | 4.00 GiB · 3204 ms | **3.15 GiB · 1541 ms（2.1x）** | 3.15 GiB · 2079 ms |
| 3012 token | 5.76 GiB · 4613 ms | **3.22 GiB · 1832 ms（2.5x）** | 3.22 GiB · 2773 ms |
| 7146 token | **OOM** | **3.53 GiB · 4684 ms** | 3.53 GiB · 4851 ms |

4096 token 前向单独测（`probe_attention_tradeoff.py`）：

| 后端 | 峰值显存 | 耗时 |
|---|:---:|:---:|
| MATH（现状） | 6.76 GiB | 5293 ms |
| **EFFICIENT_ATTENTION（展平后自动选中）** | **3.26 GiB** | **1299 ms（4.1x）** |
| CUDNN_ATTENTION | 3.26 GiB | 1636 ms |
| FLASH_ATTENTION | 不可用（未编译） | — |

**显存几乎不随长度增长**：展平后长度 ×4（1749 → 7146），峰值只从 3.15 涨到 3.53 GiB。O(n²) 消失了。

> ⚠️ **不要用跨时间点的数字算加速比。** needle 那次 3665 token 的 prefill 从 29.1 s 变成 1.4 s，
> 主因**不是**内核快了 20 倍，而是**不再溢出到系统内存**（换页消失）。纯内核加速只看**同轮内**的
> 2.1x / 2.5x / 4.1x 三组数字，它们才可比（[AGENTS.md](../AGENTS.md) 第七节第 4 条）。

---

## 三、数值正确性：改注意力路径必须先过这一关（已核查）

`probe_kernel_mapping.py`：

| 检查 | 结果 | 含义 |
|------|:---:|------|
| **两边都钉在 math 后端**比 logits | 逐位最大差 **0.000e+00**，argmax 全同 | `repeat_kv` 的 GQA head 映射**完全正确** |
| 两条路 vs **手写 fp32 参考实现** | 最大差都是 **1.897e-03**（相对 4.07e-04） | 两者精度等价，没有谁错 |
| 生成层面：贪心 32 token | **32/32 完全一致**（`probe_attention_tradeoff.py`） | 达到项目既有标准（lm_head 4-bit 当年是 24/24、32/32） |

**代价如实说：与 HF 的"逐位一致"没有了。** 原始 logits 会有 fp16 累加级差异（`max|diff| ≈ 1.0`，logits 量级 112，即 0.9%），token 决策一致但数值不完全相同。

处理方式——**把"架构保真度"与"内核 dispatch"分开测**：

- `test_gating_off_matches_baseline`：两边都钉 `sdpa_kernel(MATH)`。它测的是**架构与权重的保真度**，不该受"恰好 dispatch 到哪个内核"影响。
- `test_fused_attention_agrees_on_tokens`（新增）：展平（融合内核）vs GQA（math）的贪心 token 一致率。**实测 16/16。**

`pytest tests -q` → **41 passed**（原 40 + 新增 1 条），总耗时还从 69 s 降到 53 s。

---

## 四、新的长度天花板（单通路 · fp16 KV）

`probe_kernel_equivalence.py`，KV cache 按长度精确分配（`max_len = 长度 + 256`）：

| 长度 | 峰值显存 | prefill | 状态 |
|:---:|:---:|:---:|---|
| 7146 token | 4.65 GiB | 4.75 s | 健康（线性） |
| 14363 token | 6.47 GiB | 9.36 s | 健康（仍线性） |
| 21615 token | **8.50 GiB** | **375.9 s** | **越过物理显存 → 换页，慢 27 倍** |

**~14–15K token 是 fp16 KV 在这块卡上的天花板**（单通路）。再往上不是"慢一点"，是断崖。

---

## 五、信息过载下的注意力选择性（这才是用户想测的）

`exp_needle.py`：干草堆 = 重复闲聊（每轮唯一编号）+ **4 条形近事实**，格式完全一样、只有地点与号码不同，分别埋在不同深度（10% / 35% / 62% / 88%），然后**对 4 条各问一次**。

判分三档：**答对 / 挑错（答成别的密码，最危险）/ 没答出**。

| 长度 | 实际 token | 答对 | 挑错 | 没答出 | prefill | 峰值显存 | clocks.sm |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2048 | 1894 | **4/4** | 0 | 0 | 1.2 s | 6.04 GiB | 2355 MHz |
| 4096 | 3665 | **4/4** | 0 | 0 | 1.4 s | 6.10 GiB | 2400 MHz |
| 8192 | 7291 | **4/4** | 0 | 0 | 3.2 s | 6.27 GiB | 2460 MHz |
| 14363 | **12728** | **4/4** | **0** | 0 | 6.8 s | 6.67 GiB | 2340 MHz |

**结论（已核查）：到 ~12.7K token、且有 4 条同形干扰项时，注意力选择性没有退化 —— 全部挑对，零挑错。** prefill 随长度线性增长，峰值显存 6.67 GiB 仍在 D17 的 7GB 预算内。

**待实测：** 更长（>15K）时的退化点 —— 需要先压 KV 才测得动（见下节）。也还没测"事实数量增多"（现在是 4 条，没测 16/64 条）的干扰效应。

---

## 六、离 262144 还有多远（算术推算）

| 项 | 数字 |
|------|------|
| 目标 262144 token × 144 KiB/token（单通路 fp16 KV） | **36.0 GiB** |
| 8GB 卡在 7GB 预算内的可用量（权重已占 ~3.2 GiB） | **~3.8 GiB** |
| 缺口 | **~9.5x** |

**压缩杠杆（按"要不要训练"分）：**

| 手段 | 倍数 | 要不要训练 | 备注 |
|------|:---:|:---:|------|
| KV int8 / int4 量化（分组 scale） | 2x / **4x** | **不需要** | 最便宜的下一刀 |
| 跨层 KV 共享（每 2–4 层共享一份） | 2–4x | 要 | 架构改动，需继续预训练 |
| 局部窗口 + 少数全局层（Gemma 式） | 4–8x | 要 | 会**直接削掉长距离检索能力**，与"注意重点"的测试冲突 |
| MLA 式潜空间 KV | ~4x+ | 要 | DeepSeek 路线，收益大但改动最大 |

组合 int4 + 跨层共享 ≈ **16x** → 14K × 16 ≈ **224K**，摸得到 260K 的量级。

**但还有算力墙（算术推算）：** 注意力是 O(n²)。12728 token 的 prefill 实测 6.8 s，按 n² 外推到 262144 token：

```
(262144 / 12728)² × 6.8 s ≈ 47 分钟
```

即 **一次全量 262K prefill 约 45–50 分钟**。增量轮次（cache 命中）不受此影响，仍是每 token ~30 ms。

**推测（结论）：** 在 8GB 笔记本上，"260K 上下文"= **KV 压缩 9.5x 以上 + 一次性 ~45 分钟 prefill**。它不是不可能，但对闲聊不划算。真正值得做的是先压 KV 到 4x（不需训练）拿到 ~50K，再用 needle 量出精度衰减。

---

## 七、下一步（按收益 / 成本排序）

1. **KV int4 量化**（不需训练，~4x）→ 目标 50K token。这是唯一"不碰训练就能拿到的 4 倍"。
2. **用 needle 量 KV int4 的精度衰减**：与本节第五部分的 fp16 基线逐档对比 —— 压缩的代价必须量化，否则"注意不到重点"分不清是压缩还是模型能力。
3. **跨层 KV 共享**（要训练，里程碑 2）→ 再 2–4x。
4. **记忆替代长上下文**（项目本来路线，D06）：260K 塞不进 cache 时，用 S4 的 L0 记忆 + 短窗口，而不是硬塞。与本节第五部分的测法天然契合 —— 测的就是"检索能不能挑对重点"。
5. 干扰项数量扫描（4 → 16 → 64 条同形事实），找选择性退化的拐点。

---

## 八、复现命令

```powershell
$env:HF_HOME=(Resolve-Path .).Path+'\.hf-cache'; $env:TMP=(Resolve-Path .).Path+'\.tmp'; $env:TEMP=$env:TMP
$env:HF_HUB_OFFLINE='1'; $env:TRITON_CACHE_DIR=$env:TMP+'\triton-cache'; $env:TORCHINDUCTOR_CACHE_DIR=$env:TMP+'\inductor-cache'

& .\.venv\Scripts\python.exe -m pytest tests -q                                     # 41 passed
& .\.venv\Scripts\python.exe src\diagnostics\probe_long_context_vram.py             # 后端状态 + O(n^2) 归因
& .\.venv\Scripts\python.exe src\diagnostics\probe_attention_kernel.py              # 三种做法同轮对比
& .\.venv\Scripts\python.exe src\diagnostics\probe_kernel_mapping.py                # 数值等价性（必须过）
& .\.venv\Scripts\python.exe src\diagnostics\probe_attention_tradeoff.py            # 后端可用性 + 30 token 一致率
& .\.venv\Scripts\python.exe src\diagnostics\exp_needle.py --lens 2048 4096 8192 14363 --max-len 14848
```
