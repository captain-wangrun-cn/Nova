# S3 · 双通路骨架：验收报告

> 日期：2026-09-22 · 阶段：**S3** · 结论标注：**已核查 / 待实测 / 推测**
> 代码：`src/nova/`（`config / norm / kernels / layers / cross / model / loader / generate`）
> 测试：`tests/test_nova_skeleton.py`（**8 passed**）· 基准：`src/bench_nova.py`
> 关联决策：**D27**（速度路径 ①）、**D17**（显存预算）、**D04/D09/D14**

---

## 一、验收结论（已核查）

| 判据 | 要求 | 实测 | 结论 |
|------|------|------|:---:|
| 前向 shape | 正确 | `(1, seq, 151936)`，全 finite | ✅ |
| **门控关闭 ≈ 基线** | 数值接近 | **逐位一致**（`max|diff| = 0.000e+00`，prefill + 8 步 decode 全部） | ✅ |
| 交叉注意力隔离 | 切断一条不影响另一条 | 关闭时扰动通路 1 → 通路 0 **逐位不变**；打开时**必须改变** | ✅ |
| 生成 20 token | 不崩 | 通过 | ✅ |
| 显存 | **< 7 GB**（D17） | **4.57 GiB** | ✅ |

**`pytest tests -q` → 8 passed in 27.78s。**

---

## 二、实现的结构（已核查）

```
tokens → embed_tokens → [共享前段 6 层] → 分叉
                                            ├─ 通路 0（层 6-29 副本 A）
                                            └─ 通路 1（层 6-29 副本 B）
                                   每 4 层一处交叉（相对层 0/4/8/12/16/20）
                                                    ↓
                                    融合 mean → [共享后段 6 层] → norm → lm_head
```

| 项 | 值 |
|------|:---:|
| 总层数 / 共享前段 / 双通路段 / 共享后段 | 36 / 6 / 24 / 6 |
| 通路数 | 2（可配置） |
| 交叉点 | 6 处（每 4 层） |
| **KV cache 槽位总数** | **60**（两条通路必须用不同槽位，否则互相污染） |
| `lm_head` | **不单独建参数**，直接复用 `embed_tokens.weight`（tie_word_embeddings=True，省 0.78 GB） |

**交叉通路模块**（`src/nova/cross.py`，新增参数）：`CrossPathAttention`（本通路做 Q，另一条做 K/V，**因果**）+ `Predictor`（预测编码的预测器）+ 可学习 `gate`。

**三种模式**：`off`（完全不执行，**验收判据用这个**）/ `on`（恒执行）/ `predictive`（门控乘以预测偏差）。

---

## 三、开发中真实踩到的两个 bug（值得记住）

这两个都是**"看着对、其实错"**的类型，只有数值对照才能发现：

| # | 现象 | 根因 |
|:-:|------|------|
| 1 | `impl="exact"` 与 HF **不一致**（`max|diff| ≈ 4.6e-3`），而 `triton` 反而**逐位一致** | `LeanRMSNorm` 用 `torch.ones(dim)` 建权重 → **fp32**；HF 的是 fp16。`weight(fp32) * h(fp16)` 被**提升到 fp32**。triton kernel 里显式 `w.to(fp16)` 恰好抵消了这个错误 |
| 2 | **prefill 逐位一致，decode 从第 1 步就开始偏**（`max|diff| ≈ 10`） | 自写前向里 `position_ids` 从 **0** 开始编号；解码第 8 步应该用位置 **7**。只测 prefill 完全发现不了 |

> **教训：** 验收必须**同时覆盖 prefill 与 decode**；"某个实现看起来更接近参考"不能当作它正确的证据。

---

## 四、路径 ①（精简前向）的实测收益（已核查）

`--paths 1` 是唯一干净的对照 —— Nova 单通路与 HF 基线**同为 36 层**。

| 配置 | 层数 | tok/s | ms/token | 峰值显存 |
|------|:---:|:---:|:---:|:---:|
| HF 基线 | 36 | 13.34 | 75.0 | 2.73 GiB |
| Nova 单通路 `exact` | 36 | 13.73 | 72.8 | 3.04 GiB |
| **Nova 单通路 `triton`** | 36 | **16.05** | **62.3** | 3.04 GiB |
| Nova 双通路 `exact` | 60 | 8.23 | 121.5 | 4.57 GiB |
| **Nova 双通路 `triton`** | 60 | **9.51** | **105.2** | **4.57 GiB** |

**结论：**
1. **Triton 融合 RMSNorm 单独带来 +20.3%**（13.34 → 16.05 tok/s）。这是路径 ① 的第一个实证收益。
2. 双通路耗时 ≈ 单通路 × 1.69（60 层 vs 36 层 = 1.67）→ **几乎严格随层数线性增长**，说明**仍然是 CPU 算子派发受限**（D27 的结论在 Nova 上依然成立）。
3. 显存 4.57 GiB，距 D17 的 7GB 预算还有 2.4 GiB。

### 算子数对照（16 token 解码，已核查）

| | 算子数 / token | ms / token |
|------|:---:|:---:|
| HF 基线 | **8275** | 75.0 |
| Nova 单通路 triton | **6529** | 62.3 |
| 变化 | **-21.1%** | **-16.9%** |

### Nova 单通路里剩下的开销 top（已核查）

| 算子 | 次数 / token | CUDA 时间 |
|------|:---:|:---:|
| `bitsandbytes::gemm_4bit` | **252** | **28.99 ms** ← 单项占 47% |
| `aten::view` | 871 | 7.97 ms |
| `aten::to` | 942 | 7.76 ms |
| `aten::_scaled_dot_product_attention_math` | 36 | 6.79 ms |
| `aten::as_strided` | 714 | 6.35 ms |

> ⚠️ profiler 的绝对时间有膨胀（见 D27），但**相对结构可信**。

---

## 五、下一步（按收益排序，均未开始）

| # | 动作 | 依据 | 预估 |
|:-:|------|------|:---:|
| 1 | **自写 Triton NF4 dequant+GEMV**，替掉 bnb 的 `Linear4bit` | bnb 单项占 47%；且它的 `Params4bit.__torch_dispatch__` 有 Python 派发开销（约 83 µs/次 × 252） | 最大 |
| 2 | **融合 RoPE**（`rotate_half` 那 4 个算子 × 2） | 每层约 14 个算子，36 层 ≈ 500/token | 中 |
| 3 | **去掉冗余 `_to_copy` / `view` / `as_strided`** | 三者合计 2500+ 次/token；`aten::view` 在 Nova 里反而从 581 涨到 871（Triton kernel 包装引入） | 中 |
| 4 | 融合注意力（当前 36 次 `_sdpa_math` 回退） | 6.79 ms/token；此前实测手工扩 KV 头只换来 1.14x，优先级低于 1-3 | 低 |

**注意：** 上表是**速度路径**，与 S4（记忆最小实现）并行推进，不互相阻塞。

---

## 六、复现命令

```powershell
$env:HF_HOME='H:\Nova\.hf-cache'; $env:TMP='H:\Nova\.tmp'; $env:TEMP=$env:TMP; $env:HF_HUB_OFFLINE='1'
& .\.venv\Scripts\python.exe -m pytest tests -q
& .\.venv\Scripts\python.exe src\bench_nova.py --mode hf
& .\.venv\Scripts\python.exe src\bench_nova.py --mode nova --paths 1 --norm triton
& .\.venv\Scripts\python.exe src\bench_nova.py --mode nova --paths 2 --norm triton
```
