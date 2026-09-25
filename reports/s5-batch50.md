# S5 50 条数据管线 smoke test（2026-09-26）

> **一句话结论**：**50/50 生成成功，50/50 清洗通过**；RP / 状态跟踪 / 工具调用三类
> 都能进训练入口。最坏情况工具样本 **636 token** 在本地 4060 也能训
> （峰值 **6.29 GiB allocated / 6.49 GiB reserved**）。
> 这是**管线验收**，不是数据质量验收。

---

## 一、生成

| 项 | 值 |
|---|---|
| 模型 | `gemini-3.1-flash-lite` |
| 端点 | `http://newapi.wr.wstudio.work/v1/chat/completions` |
| 数量 | **50** = 30 RP + 10 状态跟踪 + 10 工具调用 |
| `finish_reason` | **50/50 `stop`**（无截断） |
| 总 token | **17,197** |
| 原始数据 | `data/s5-samples/batch50-raw.jsonl`（gitignored） |
| 复现 | `& .\.venv\Scripts\python.exe src\s5_gen_batch.py` |

---

## 二、清洗结果

| 指标 | 结果 |
|---|---|
| 清洗通过 | **50/50（100%）** |
| 分类 | RP 30/30 · 状态 10/10 · 工具 10/10 |
| 中文 | **0** |
| 拒答 | **0** |
| 截断 | **0** |
| 思考段泄漏 | **0** |
| 角色越界（启发式） | **0** |
| 模板句警告 | **1**（`s5b-rp-04-v3`，3 处） |
| AI 免责声明警告 | **1**（`s5b-state-04`，末尾 "I am an AI, not a doctor"） |
| 词数 | min 4 · median 170.5 · max 238 |

产物：

- `data/s5-samples/batch50-scored.jsonl`（全部样本 + 质检标记，gitignored）
- `data/s5-samples/batch50-clean.jsonl`（50 条干净样本，gitignored）
- `reports/s5-batch50-cleaning.json`（机器可读报告）

复现：`& .\.venv\Scripts\python.exe src\s5_clean_batch.py`

---

## 三、50 条验收线（全部满足）

| # | 判据 | 结果 |
|---:|---|---|
| 1 | 50/50 请求成功 | ✅ |
| 2 | ≥90% 正常结束、不截断 | ✅ 50/50 = 100% |
| 3 | 100% 英文、无中文 | ✅ |
| 4 | 0 条拒答或空输出 | ✅（1 条 AI 免责声明，仅警告） |
| 5 | 0 条思考段混入正文 | ✅ |
| 6 | 角色越界 ≤1 且能被清洗抓出 | ✅ 0 条 |
| 7 | 50 条都能进训练入口 | ✅ 全部 tokenize；RP 2 步 + 工具 1 步已实跑 |

---

## 四、训练入口实跑

### 4.1 RP 两类样本（2 步）

| 项 | 结果 |
|---|---|
| 样本 | `s5b-rp-01-v1` 324 token · `s5b-rp-01-v2` 361 token |
| loss | 2.1957 → 1.8448 |
| step_s | 中位 1.44（1.07-1.82） |
| `clocks.sm` | **2490 MHz** |
| 峰值 | 5.47 GiB allocated / **6.09 GiB reserved** |
| `grad_none` / `base_grads` | 0 / 0 |

### 4.2 最坏情况：工具样本（1 步）

| 项 | 结果 |
|---|---|
| 样本 | `s5b-tool-01` **636 token**（工具 schema 很长） |
| loss | 0.2623 |
| step_s | 2.38 |
| `clocks.sm` | **2490 MHz** |
| 峰值 | **6.29 GiB allocated / 6.49 GiB reserved** |
| `grad_none` / `base_grads` | 0 / 0 |

**结论：** 本地 4060 + 梯度检查点 + batch 1，RP / 状态 / 工具三类都能训；
但工具样本已到 636 token，**再长就要上 Kaggle/云**（推测，待实测）。

---

## 五、已知问题（不阻塞 50 条管线，但 500-1000 条要修）

1. **工具参数语义没校验。** `s5b-tool-02` 结构正确、工具名正确，但 `start_time`
   写成 `2025-08-07 10:00`，不是"明天"。50 条只查结构；500-1000 条要加参数级校验。
2. **模板句仍有。** `s5b-rp-04-v3` 命中 3 处模板句（`sharp, rhythmic pulse` 等）；
   500-1000 条要么改教师 prompt，要么提高模板句过滤阈值。
3. **状态题偏简单。** 目前是"说一次 → 问一次"，500-1000 条要加多步状态变化、
   矛盾修正、跨轮指代。
4. **只用一个教师。** 本轮全部是 `gemini-3.1-flash-lite`；500-1000 条按 D24
   v4 七层做教师路由，不要重新调研教师池。

---

## 六、下一步

**S5 500-1000 条数据管线**：

1. 按 D24 教师池生成 500-1000 条，覆盖 RP / 状态跟踪 / 工具调用；
2. 清洗规则升级：工具参数级校验、模板句阈值、状态题难度；
3. 用干净数据跑 Stage A（交叉注意力 + 门控），再进 Stage B1 LoRA；
4. 评估回路：`qa-03` + RP 复读率 / 中文占比。

## 七、复现

```powershell
& .\.venv\Scripts\python.exe src\s5_gen_batch.py
& .\.venv\Scripts\python.exe src\s5_clean_batch.py
& .\.venv\Scripts\python.exe src\s5_train_spike.py `
  --data data\s5-samples\batch50-clean.jsonl --steps 2 --warmup 0 `
  --ckpt .tmp\s5-train-spike\cross_blocks-batch50.pt `
  --out reports\s5-train-spike-batch50-results.json
```
