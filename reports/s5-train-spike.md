# S5 本地 10 步训练 spike（2026-09-26）

> **一句话结论**：**本地 4060 能训**。4-bit bnb 基座反向可用，基座梯度为 0；
> 只训 `cross_blocks`（330,362,892 参数），10 步 loss 1.9595 → 0.7702；
> 峰值 allocated **5.45 GiB** / reserved **7.07 GiB**；中位 **0.889 s/step**
> （`clocks.sm` 2280-2475 MHz）。

---

## 一、这一步回答什么

D46 的第 1 步：在本地 4060 上验证自写 Nova 双通路能不能反向传播、显存能不能装下、速度大概多少。
**不训练质量，不跑 S5 数据集。**

---

## 二、配置

| 项 | 值 |
|---|---|
| 数据 | `data/s5-samples/gemini-3.1-flash-lite.jsonl`（D47，5 条） |
| 样本长度 | 301 / 307 / 325 / 356 / 334 token（不截断） |
| 基座 | Qwen3-VL-4B-Instruct，bnb 4-bit，`norm_impl="exact"` |
| 可训练参数 | `cross_blocks` 全部（交叉注意力 + 门控 + 预测器）**330,362,892** |
| 冻结 | 其余全部冻结；`base_grads=0` |
| 模式 | `cross_mode="predictive"`（必须；`"on"` 会让 24/102 个预测器张量没有梯度） |
| 梯度检查点 | 开（必须） |
| 优化器 | `bitsandbytes.optim.PagedAdam8bit`，lr 1e-4，grad clip 1.0 |
| batch / 步数 | batch 1；2 warmup + 10 measured |

---

## 三、10 步结果（已核查）

| step | 样本 | seq | loss | grad_norm | step_s | clocks.sm | peak alloc |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | s5-rp-03 | 325 | 1.9595 | 1.698 | 0.87 | 2310 | 5.36 |
| 2 | s5-rp-04 | 356 | 1.8568 | 1.551 | 1.28 | 2280 | 5.45 |
| 3 | s5-rp-05 | 334 | 1.9470 | 1.721 | 0.95 | 2280 | 5.39 |
| 4 | s5-rp-01 | 301 | 1.3854 | 1.699 | 0.83 | 2310 | 5.29 |
| 5 | s5-rp-02 | 307 | 1.1054 | 1.295 | 0.83 | 2355 | 5.31 |
| 6 | s5-rp-03 | 325 | 1.3683 | 1.582 | 0.86 | 2475 | 5.36 |
| 7 | s5-rp-04 | 356 | 1.3063 | 1.311 | 0.93 | 2475 | 5.45 |
| 8 | s5-rp-05 | 334 | 1.5394 | 1.781 | 0.91 | 2475 | 5.39 |
| 9 | s5-rp-01 | 301 | 1.0479 | 1.709 | 1.05 | 2460 | 5.29 |
| 10 | s5-rp-02 | 307 | 0.7702 | 1.457 | 0.82 | 2415 | 5.31 |

汇总：

- loss first/last：**1.9595 → 0.7702**
- `grad_none`：**0/102**（predictive 模式下交叉模块每个张量都有梯度）
- `base_grads`：**0**（4-bit 基座全程冻结）
- `step_s`：中位 **0.889**，范围 0.821-1.278
- `clocks.sm`：**2280-2475 MHz**
- peak allocated：**5.45 GiB**；peak reserved：**7.07 GiB**
- checkpoint：`.tmp/s5-train-spike/cross_blocks.pt`（660,765,112 B）
- 原始结果：`reports/s5-train-spike-results.json`
- `pytest tests -q`：**126 passed**（新增 `tests/test_train_spike.py` 1 条）

---

## 四、四个关键发现

1. **4-bit bnb 基座反向可用。** 这是本轮最大的风险点；`base_grads=0` 证明梯度只流向新建的
   `cross_blocks`，没有把 4-bit 基座卷进优化器。
2. **梯度检查点是本地 8GB 的必选项。** 无检查点的诊断（seq 301）峰值 allocated 7.12 GiB /
   reserved 7.24 GiB；最长样本 356 用检查点后是 5.45 / 7.07 GiB。reserved 7.07 GiB 距离
   7.996 GiB 物理上限只剩约 0.93 GiB，**长上下文或 batch > 1 要上 Kaggle/云**。
3. **训练必须用 `cross_mode="predictive"`。** `"on"` 模式跳过预测器，24/102 个张量没有梯度；
   `"predictive"` 让交叉注意力、门控、预测器一起训。
4. **速度不是瓶颈，显存才是。** 5 条样本、seq 301-356、batch 1，中位 0.889 s/step；
   按 1000 步算约 15 分钟/epoch。真正限制是 reserved 7.07 GiB。

---

## 五、这一步**没有**做的事

1. **不是 LoRA**：本轮只训了新建的 `cross_blocks`（Stage A）；Stage B1 的 LoRA 栈还没接。
2. **不是完整 checkpoint**：只存了 `cross_blocks` state dict；`PagedAdam8bit` 的优化器状态没存，
   不能续训。
3. **不是 S5 数据集**：5 条样本、10 步，loss 下降不能当质量结论。
4. **没有评估回路**：还没有 held-out 集、复读/角色边界/拒答过滤器。

---

## 六、下一步

1. **S5 数据管线**：用 D24 的 API 教师生成 500-1000 条英文样本；清洗角色边界、复读、长度、
   拒答、思考段；落 JSONL。
2. **训练入口扩展**：把 `src/s5_train_spike.py` 扩成可配置的 SFT 入口；需要时再加 LoRA（Stage B1）。
3. **评估回路**：用 `qa-03` + 3 条 RP 的复读率/中文占比做固定对照。
4. **长上下文/大 batch**：本地 reserved 已接近上限，留给 Kaggle/云。

## 七、复现

```powershell
& .\.venv\Scripts\python.exe src\diagnostics\probe_train_backward.py `
  --sample-index 3 --checkpointing --cross-mode predictive
& .\.venv\Scripts\python.exe src\s5_train_spike.py --steps 10 --warmup 2
& .\.venv\Scripts\python.exe -m pytest tests -q
```
