# S5 API 教师 smoke test（2026-09-26）

> **一句话结论**：OpenAI 兼容网关 + `gemini-3.1-flash-lite` **可用**；
> 5 条英文 RP 样本已落盘；**rp-03 的角色越界（D45）在样本侧被新 system prompt 修掉**。

---

## 一、端点与配置（已核查）

| 项 | 值 |
|---|---|
| 网关 | `http://newapi.wr.wstudio.work` |
| 实际调用 | `POST /v1/chat/completions`（OpenAI 兼容） |
| 模型 | `gemini-3.1-flash-lite` |
| 原生端点（未用） | `/v1beta/models/gemini-3.1-flash-lite:generateContent` |
| key | `H:\Nova\.secrets\gemini.key`（`.gitignore` 已挡） |
| 采样 | `temperature=0.8`、`top_p=0.9`、`max_tokens=512`、`stream=false` |

复现：

```powershell
& .\.venv\Scripts\python.exe src\s5_gen_samples.py
```

产物：`data/s5-samples/gemini-3.1-flash-lite.jsonl`（5 行，9356 字节，`data/*` 已 gitignore）。

---

## 二、结果（已核查）

| id | 场景 | 词数 | finish_reason |
|---|---|---:|---|
| s5-rp-01 | 港口雨夜开场 | 158 | stop |
| s5-rp-02 | 坏消息与情绪压力 | 166 | stop |
| s5-rp-03 | 讨价还价的谈判 | 182 | stop |
| s5-rp-04 | 风暴后的伤口 | 190 | stop |
| s5-rp-05 | 旧人归来 | 177 | stop |

- **5/5 HTTP 200**，全部 `finish_reason=stop`，没有被 `max_tokens` 截断。
- **总 token 1537**（5 条合计，含 prompt）。
- **5/5 无中文**；非 ASCII 只有 `’` 和 `—`（英文排版字符）。
- `reasoning_content` 为 `null`，content 里也没有 thinking 块；清洗函数目前没有实际删除内容。

---

## 三、D45 的角色越界是否修掉（已核查）

新 system prompt 明确写：

> You are Elara Voss ... Write in third-person past tense about Elara only.
> Never write the other characters' dialogue, thoughts, or actions.

**rp-03 复测**：旧基线的 4bit/8bit 都把整段写成买家的动作和对白（D45 的失败模式）。
本轮 `s5-rp-03` 全程聚焦 Elara 的生理反应和动作，只观察"the man's lips curl upward"，
**没有代写买家对白**，也没有替买家做决定。角色边界这一条在样本侧通过。

---

## 四、安全边界（必须记住）

1. **网关 HTTPS 证书域名不匹配**：`https://newapi.wr.wstudio.work` 报
   `CERTIFICATE_VERIFY_FAILED: Hostname mismatch`。当前脚本走 HTTP。
2. **HTTP 下 key 在 Authorization 头里明文传输**。只在可信网络 / VPN / 自建网关上使用；
   不要把 key 提交到仓库，也不要写进报告。
3. 如果这个 key 有价值，建议本轮测试后轮换；如果网关有合法 HTTPS 域名，优先换成 HTTPS。

---

## 五、这一步**没有**做的事

1. **不是 S5 数据集**：只有 5 条、一个模型、一套采样参数，不能代表最终数据分布。
2. **没有做完整清洗**：角色边界、复读、长度、拒答等过滤器还没写成管线；这 5 条只是 smoke test。
3. **没有跑训练**：下一步才是 D46 的 **10 步训练 spike**，用这 5 条样本做最小输入。

---

## 六、下一步

> **2026-09-26 更新：已完成，见 [s5-train-spike.md](s5-train-spike.md)（D48）。**

用这 5 条样本跑 **本地 4060 的 10 步训练 spike**（D46）：

1. 装/验证 LoRA 栈（`peft` 或最小自定义 LoRA 包装）；
2. 写训练入口：`labels → loss`、梯度检查点、8-bit Adam；
3. batch 1、seq 512、`norm_impl="exact"`，跑 10 步；
4. 记录显存峰值 / step 时间 / `clocks.sm`，落 `reports/` + 新决策。

**这一步不碰蒸馏、不找公开数据集。**
