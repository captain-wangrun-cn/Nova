# 08 · 数据集与模型选型

---

## 一、基座模型候选评估

### TokenRhythm/NeoHorse-1-9B（用户最初看中的）

| 属性 | 值 |
|------|-----|
| 基座 | Qwen3.5-9B |
| 参数量 | ~9B |
| 上下文 | 262,144 原生，可扩至 1,010,000 |
| 权重 | BF16 Safetensors，Apache-2.0 |
| 后训练 | Routing-guided agentic post-training |
| **接口** | **纯文本输入输出（Vision weights are not included）** |
| 评测 | 十项基准宏平均 69.04 vs 基座 65.60（+3.44） |
| 量化版本 | 社区已有 13 个量化 |

**评估结论：**
- 优点：agentic 后训练充分、长上下文原生支持、许可宽松、有社区量化
- **致命问题：没有视觉能力**（若多模态需求保留，则不合适）
- 另一个问题：9B 双通路 = 18B，4060 装不下

**状态：作为参考，不作为基座（除非改回单通路方案）。**

### Qwen3-4B（本项目选择）

| 属性 | 值 |
|------|-----|
| 参数量 | ~4.0B |
| 层数 | 36 |
| 隐藏维度 | 2560 |
| 注意力 | GQA（8 个 KV 头） |
| 上下文 | 32K 原生，YaRN 可扩展 |
| 中文能力 | 优秀 |
| 生态 | 完善（量化、工具链） |

**为什么选它：**
- 双通路后约 7.9B，Q4 约 4.7GB，4060 8GB 舒适
- 中文 RP 生态成熟
- 36 层便于设计"共享 6 + 分叉 + 共享 6"的结构
- 社区量化多，便于对比基线

### 其他候选（备选）

| 模型 | 参数量 | 特点 | 备注 |
|------|:---:|------|------|
| Qwen3-8B | ~8.2B | 质量更高，隐藏维度 4096 | 双通路装不下，单通路可用 |
| Gemma 3 系列 | 4B / 12B | 原生 128K，SWA 省 KV | 多模态变体存在 |
| Phi-4-mini | 3.8B | 128K，效率高 | 中文稍弱 |

## 二、现有数据集盘点

### 中文 RP 数据集

| 数据集 | 规模 | 特点 | 适合度 |
|--------|------|------|:---:|
| `Johnson8187/role-play-chinese` | 22.4k 下载 | 更新活跃 | 好 |
| `LooksJuicy/Chinese-Roleplay-CAI` | 7.1k 下载 | Character.AI 抓取，多角色 | 好 |
| `LooksJuicy/Chinese-Roleplay-Novel` | 266 下载 | 小说风格 | 中 |
| `LooksJuicy/Chinese-Roleplay-SingleTurn` | 7.59k 下载 | 单轮 | 中 |
| `shibing624/roleplay-zh-sharegpt-gpt4-data` | 6.58k 下载 | **GPT-4 生成，质量高** | **很好** |
| `CausalLM/Kingfall-Roleplay` | 10k 下载 | 中文叙事 | 好 |
| `Tarklanse/Traditional_Chinese_roleplay_chat_Dataset` | 9.51k 下载 | 繁体中文 | 中 |
| `BigPancake01/roleplayLLM_Chinese` | 10.6k 下载 | 中文 RP | 中 |
| `raincandy-u/chinese-roleplay` | 2.42k 下载 | 中文 RP | 中 |
| `Seikaijyu/Classical-Chinese-Roleplay` | 579 下载 | 文言文风格 | 特殊 |

### 英文 RP 数据集

| 数据集 | 规模 | 特点 | 适合度 |
|--------|------|------|:---:|
| `agentlans/combined-roleplay` | 1.42M 下载 | 多语言 RP 合集，量最大 | 基础量大 |
| `beyoru/Aesir-Character-CoT-roleplay` | 1.07K 赞 | **角色 CoT 推理 RP** | **最创新** |
| `Gryphe/Sonnet3.5-Charcard-Roleplay` | 9.74k 下载 | Claude 生成，角色卡 | 风格好 |
| `lemonilia/roleplaying-forums-raw` | 244k 下载 | 真人论坛 RP，原始但真实 | 补充 |
| `rickRossie/bluemoon_roleplay_chat_data_300k_messages` | 261k 下载 | 30 万条消息 | 量大 |
| `MiniMaxAI/role-play-bench` | 6.37k 下载 | 基准测试集 | **用于评测** |
| `lazyweasel/roleplay-bench` | 2.36k 下载 | 基准 | 用于评测 |

### 重点推荐：Aesir-Character-CoT-roleplay

它让模型在 RP 前先输出**思维链**（角色此刻在想什么、情绪状态是什么），直接提升长对话的一致性。**这正好对应用户"注意力好"的需求**，值得深入研究其格式。

## 三、数据策略

### 核心理念：少而精

用户原话："我想要用很少的数据训练出更好的结果，现在都追求多数据，优数据，但是我想要聚焦于模型的学习质量。"

**这个直觉有研究支持：** LIMA 论文证明 1000 条精挑细选的数据效果超过 50000 条垃圾数据。对 RP 更是如此——风格多样性比数量重要。

### 数据来源组合

```
1. 现有数据集筛选（高赞、高质量子集）
2. 大模型 API 蒸馏合成（用户可提供 API 额度）
3. 云端模型蒸馏（用户可部署模型）
```

### 三类必需数据

| 类型 | 用途 | 来源 |
|------|------|------|
| 对话质量数据 | RP 风格、语气、角色一致性 | 现有数据集 + 蒸馏 |
| **状态跟踪数据** | 记忆写入 / 读取行为 | **必须合成（现实几乎没有）** |
| 工具调用数据 | 异步工具头训练 | 合成 |

### 状态跟踪数据（最关键，也最难）

需要构造大量如下模式的样本：

```
[多轮对话]
用户: 她推开门，换上了一件黑色礼服
助手: ...（自然回应，不刻意强调）
[继续 20 轮无关对话]
用户: 你今天真美
助手: ...（自然知道指的是黑色礼服）  <- 这就是要教的行为
```

**需要覆盖：**
- 服装 / 外观变化
- 地点变化
- 承诺 / 约定
- 人物关系变化
- 剧情伏笔与回收
- 物品 / 状态变化

## 四、语言配比与训练顺序

> **顺序是硬约束：先纯英文（阶段 0/1），再中文 LoRA（阶段 2）。** 见 D18 与 [15-language-plan.md](15-language-plan.md)。

| 顺序 | 阶段 | 语言 | 训练内容 | 数据量 |
|:---:|:---:|------|------|:---:|
| 1 | 0 | 纯英文 | 双通路对齐（交叉注意力 + 门控） | 少 |
| 2 | 1 | 纯英文 | 对话 / RP 地基 LoRA | 100%（基准） |
| 3 | 2 | 中文 + 中英混杂 | 中文 LoRA，叠加在阶段 1 之上 | 30-50% |
| 4 | 3 | 其他语言 | 按需 | —— |

### 英语（母语）数据

- 用于训练：通用对话能力 + RP 基础模式 + 双通路协作
- 来源：英文 RP 数据集 + 蒸馏
- **必须做语言过滤：不含中文**（否则地基不纯，见 15-language-plan.md 风险 L4）

### 中文 LoRA 数据

- **必须在英文阶段全部完成之后才训练**
- 叠加在英语 RP 基础之上训练
- **必须包含中英混杂样本**（英文人名 / 地名 + 中文对话）
- 因为基座已"会"中文，LoRA 只需教"怎么用中文做好 RP"
- 数据量需求比英语少（模式已在英语阶段学会）

## 五、上次训练乱码的教训

用户曾在云端用聊天记录训练，**输出乱码**。

**推测原因：chat template 不匹配。**

```
训练时：数据未用模型本身的对话模板包裹
        （如 Qwen 的 <|im_start|>...<|im_end|>）
        |
推理时：用了标准模板
        |
结果：模型学到的 token 分布与推理时对不上 -> 乱码
```

**规避方法：**
- 训练数据必须用基座模型**原生的 chat template**
- 训练后立刻做一次推理验证
- 保存训练用的模板，推理时严格一致

### 五·补 · 已核查：模板本身没问题，问题在"正文里混进控制符"（2026-09-21 · S1）

用 Qwen3-VL-4B-Instruct 原生 tokenizer 实测（`src/tokenizer_probe.py` → [reports/tokenizer-report.md](../reports/tokenizer-report.md)）：

| 检查 | 结果 |
|------|:---:|
| `<|im_start|>` / `<|im_end|>` 是否编码成**单个 id** | ✅ 全部是（151644 / 151645） |
| 3 轮 RP 对话渲染后块结构是否严格成对 | ✅ 7 个 `<|im_start|>` / 6 个 `<|im_end|>`，末尾补 `assistant` 续写位 |
| `decode(encode(prompt)) == prompt` 往返 | ✅ 一致 |
| `apply_chat_template(tokenize=True)` 与手工编码逐 id 对比 | ✅ 完全相同 |

**含义：** 模板渲染管线本身是干净的，"乱码"的根因**不在** chat template。

### 五·补2 · 新增根因嫌疑：正文里的特殊 token 注入

实测发现：**只要文本正文里字面出现 `<|im_start|>` / `<|im_end|>`，fast tokenizer 就会把它当成真正的控制符切出来**（测试串 14 个 token 里出现 2 个特殊 id：`[151645, 151644]`）。

教师 API 返回、用户输入、网页抓取都可能带这种串。**不清洗就进训练数据，等于把伪造的会话边界直接喂给模型**——这比"模板不匹配"更隐蔽，症状同样是串台 / 乱码 / 角色跳变。

**硬要求（S5 数据管线必须实现）：**

1. 所有训练样本在落盘前，剥离正文中的 `<|im_start|>`、`<|im_end|>`、`<|endoftext|>`、`<|vision_start|>`、`<|vision_end|>`、`<|image_pad|>`、`<|video_pad|>`、`<tool_call>`、`</tool_call>` 等字面串。
2. 清洗放在**数据管线**里，不放在 tokenizer 层（教师原始返回要留档可审计）。
3. 解析教师输出时**不要**用 `<|im_start|>` 之类的串当分隔符——用 JSON 字段或自定义分隔串。

### 五·补3 · 另外三条与模板/生成相关的实测事实

- `bos_token_id = None`、`add_bos_token = false` → **不要**在序列开头手动补 BOS。
- `eos_token_id = 151645`（`<|im_end|>`），但 `<|endoftext|>`(151643) 也是合法结束符 → 生成的 **stop 集合应同时包含 151645 与 151643**，否则长生成可能漏停。
- `pad_token_id = 151643`、`padding_side = "right"` → 批处理可直接 padding；训练侧若要 left padding 需显式设置。

## 六、蒸馏数据量与成本

### 按目标分类的需求量

| 目标 | 最少可行 | 推荐 | 说明 |
|------|:---:|:---:|------|
| RP 风格 / 语气 | 200 | 500-1000 | 风格学习效率高 |
| **状态跟踪**（核心） | 300 | 800-1500 | 行为学习，需多样性 |
| 记忆写入 / 读取 | 200 | 500-1000 | 全新行为，最难 |
| 工具调用 | 200 | 500-1000 | 格式学习为主 |
| 通用对话（防退化） | 300 | 800-1500 | 防灾难性遗忘 |
| **合计** | **约 1200** | **3000-6000** | |

Token 量估算：4000 条 x 约 2000 token = 约 8M token

### 成本与政策风险对比

| 方式 | 成本 | 政策风险 |
|------|:---:|:---:|
| 商业 API（强模型） | $30-80 | 中-高 |
| 商业 API（中等模型） | $10-25 | 中 |
| 商业 API + Batch（五折） | 减半 | 中 |
| **自部署开源模型（云 GPU + vLLM）** | **$5-20** | **无** |
| Kaggle 免费 GPU | $0 | **无** |

> ⚠️ **上表"商业 API"一行的估算偏低，已被 [六·补2](#六补2--前沿闭源模型gpt--claude--gemini能不能用) 的实测价目表取代**：全量前沿 API 实际为 **$50-660**（按档位），Batch 五折。政策风险一列仍然成立（三家条款均明文禁止用输出训练竞争模型）。

### 推荐路线：自部署蒸馏

**为什么自部署更优：**

1. **零政策风险** —— 不受 API 内容条款约束
2. **更便宜** —— vLLM 连续批处理可把吞吐提升 10-20 倍（这是关键：单条生成慢，批量后极快）
3. **同族教师更好** —— 用 Qwen 系列做教师，与学生（Qwen3-VL-4B）分布更匹配，蒸馏效率更高

**教师模型候选：** 见下方"教师模型选型"（2026-09-21 联网核查结果）。

---

## 六·补 · 教师模型选型（2026-09-21 联网核查）

> 全部结论来自联网核查（榜单抓取日期见下表），**不是**凭记忆写的。原始页面缓存于 `reports/web-*.txt`。

### 一、学生是谁，决定了教师的边界

学生 = **Qwen3-VL-4B-Instruct**（`qwen3_vl`，词表 151936，256K 上下文，Apache-2.0）。

| 事实 | 含义 |
|------|------|
| Qwen3.8 系列词表是 **248320**（与学生不同） | **不能做 logits 蒸馏**，只能做**输出蒸馏**（教师生成文本 → 学生 SFT）。好消息：输出蒸馏对词表不一致不敏感 |
| 学生只有 4B | 教师不宜过大。**7-9 倍（27B-36B）是甜点区**；千亿级巨兽的输出学生"学不动"（蒸馏鸿沟） |
| 学生是 Qwen 系 | 同族教师（Qwen3.8 / Qwen3.6）在风格先验、对话习惯上最接近，**默认主力教师** |

### 二、教师候选（按能否自己部署分两档）

**A 档 · 可自部署（单张 24GB 卡 4-bit 即可，零政策风险）**

| 模型 | 参数量 | 许可 | 多模态 | 核查数据 |
|------|:---:|:---:|:---:|------|
| **Qwen3.8-27B** | 27.8B dense | **Apache-2.0** | ✅ | 创意写作 Elo 1668（开源第 6） |
| **Qwen3.6-35B-A3B** | 36B MoE / 3B 激活 | **Apache-2.0** | ✅ | 速度快，适合大批量 |
| **GLM-4.7-Flash** | 31.2B | **MIT** | ❌ | BFCL 工具调用 **74.6%**（开源前五） |

**B 档 · 只能租用/调 API（datacenter 级，用于最难的部分，量少）**

| 模型 | 参数量 | 许可 | 多模态 | 核查数据 |
|------|:---:|:---:|:---:|------|
| **GLM-5.3-Flash** | 321B MoE | **MIT** | ✅ | 1M 上下文；创意写作 Elo 2064、Slop 1.2（极低） |
| **DeepSeek-V4-Flash** | 304B MoE | **MIT** | ❌ | 1M 上下文；主打长上下文记忆架构 |
| Qwen3.8-Flash-Next | 180B | 自定义 | ✅ | —— |
| Qwen3.8-2.4T-A95B | 2.4T MoE | 自定义 | ✅ | 创意写作 Elo 1841（开源最高） |
| GLM-5.3 / Kimi-K2.6 / Kimi-K3 | 750B+ / 1T+ | 自定义 | ✅ | 创意写作 Elo 2064-2071（接近闭源前沿） |

**已被排除：** Muse-Glimmer-30B（Meta，30B 多模态，创意写作 Elo 1790 本很有吸引力）—— **实测区域锁定**（`This model is not available in your region`）。

### 三、按蒸馏环节分配教师（核心结论）

| # | 数据部分 | 首选教师 | 备选 | 依据 |
|:-:|------|------|------|------|
| 1 | **RP 风格 / 语气** | GLM-5.3-Flash（少量，slop 1.2）+ **Qwen3.8-27B**（大量，自部署） | GLM-5.2（MIT，Elo 1753） | 创意写作 v3 榜 + **Slop 分** |
| 2 | **状态跟踪**（核心） | **GLM-5.3-Flash**（1M 上下文） | DeepSeek-V4-Flash（MIT，1M） | 长上下文榜（Llama 4 Scout 10M / DeepSeek V4 1M / Qwen3.5 1M） |
| 3 | **记忆写入 / 读取**（最难） | **GLM-5.3-Flash thinking 模式** | Qwen3.8-27B thinking | 全新协议，需要"读懂协议再生成"，量小、必须用最强 |
| 4 | **工具调用** | **GLM-4.7-Flash**（自部署，MIT） | Qwen3.8-27B（同族） | BFCL：GLM 系长期开源第一梯队（GLM-4.5-Thinking 76.7%、GLM-4.7-Flash 74.6%、Qwen3-32B 75.7%） |
| 5 | **通用对话防退化** | **Qwen3.8-27B** | Qwen3.6-35B-A3B | 同族分布最匹配 |

### 四、四条实操原则

1. **强模型当裁判，不当老师。** 榜首闭源模型（Claude / Gemini / GPT 系）适合做**质量筛选与打分**（把套路化的样本剔掉），不适合当教师——学生学不动，且条款风险高。
2. **看 Elo 也要看 Slop。** EQ-Bench 的 Slop 分衡量"套路化写作"程度，**越低越好**。教师 slop 高 → 蒸馏出来的学生也套路化。
3. **思考模式要处理干净。** 带思维链的教师输出会污染数据：要么只用最终答案，要么明确决定是否训练 `思考` 段落，不能混着用。
4. **分阶段换教师。** 先小量（500-1000 条）用 A 档自部署验证管线；管线通过后，只在最难的 2-3 个环节调用 B 档教师。

### 五、成本对照（与上一节估算一致）

| 方案 | 显存需求 | 成本 | 政策风险 |
|------|:---:|:---:|:---:|
| Qwen3.8-27B 4-bit 自部署（1×24GB 卡） | ~15-16GB | 约 $0.3-0.5/小时，全量约 **$5-20** | 无 |
| GLM-4.7-Flash 4-bit 自部署 | ~18GB | 同上 | 无 |
| GLM-5.3-Flash API（B 档，仅难样本） | —— | 按量，样本少所以可控 | 中（需读条款） |
| Kaggle 免费 2×T4 | 27B 4-bit 可分片 | **$0**（慢） | 无 |

### 六、数据来源（抓取于 2026-09-21）

| 来源 | 用途 |
|------|------|
| `arena.ai/leaderboard/text`（2026-09-13 快照，402 模型，814 万票） | 综合排名、开源/闭源标注 |
| `eqbench.com/creative_writing.html`（2026-09-07） | 创意写作 Elo + Slop + 重复度 |
| `eqbench.com/index.html`（EQ-Bench 4，2026-07-20 快照） | 多轮角色扮演的情商/社交能力 |
| `gorilla.cs.berkeley.edu/leaderboard.html`（BFCL V4，2026-04-12） | 工具调用 |
| `awesomeagents.ai` 长上下文榜 / `benchlm.ai` | MRCR / LongBench v2 / 长窗口 |
| `huggingface.co/api/models` | 参数量、许可、词表、上下文长度（逐个核对） |

> 原始页面已缓存在 `reports/web-*.txt`，可复查。

---

## 六·补2 · 前沿闭源模型（GPT / Claude / Gemini）能不能用？

> 用户的质疑很对："为什么全是小模型？" 核查后修正：**这不是能力问题，是条款问题**——而且结论要分层，不能一刀切。

### 一、条款原文（2026-09-21 抓取，缓存在 `reports/web-tos-*.txt`）

| 厂商 | 原文（节选） | 出处 |
|------|------|------|
| **OpenAI** | 禁止 "**Use Output to develop models that compete with OpenAI**"；同时禁止 "Automatically or programmatically extract data or Output" | Terms of Use |
| **Anthropic** | 禁止 "(a) access the Services to build a competing product or service, **including to train competing AI models**"，除非获得明确批准 | Commercial Terms |
| **Google** | "You may not use the Services to **develop models that compete with the Services** (e.g., Gemini API or Google AI Studio)" | Gemini API Terms |

**三家全部明文禁止。** 这不是保守推测，是条款原文。风险后果是**账号封禁**（不是诉讼），检测手段主要是用量异常与自动化提取行为。

**灰色地带：** 条款说的是"竞争模型"。一个**永远只自己用、不发布、不商用**的个人模型，算不算"竞争"？法律上模糊。这是灰色地带，**不是绿灯**。

### 二、按用法分层（关键区分）

| 用法 | 条款状态 | 风险 | 建议 |
|------|:---:|:---:|------|
| 前沿输出**进训练集**（全量） | ❌ 明确禁止 | 高 | 不做 |
| 前沿输出**少量进训练集**（风格锚点，<10%） | ⚠️ 灰色 | 中 | **由用户决定** |
| 前沿模型当**裁判 / 评测**（输出不进训练集） | ✅ 正常使用 | 低 | **推荐，且本来就该做** |
| 前沿生成 prompt / 场景，开源模型生成回答 | ⚠️ 灰色（"Output"字面仍覆盖） | 中-低 | 可用，量小 |

> **重要区分：把前沿输出写进训练集 = 违约；用前沿模型评测和对比 = 正常使用。**

### 三、成本不是阻碍（重新算账）

6000 条 × 约 2000 输出 token = **12M 输出 token**（+ 约 6M 输入 token）：

| 档位 | 单价（入/出，每 M） | 全量估算 | Batch 五折 |
|------|:---:|:---:|:---:|
| Gemini 3.8 Flash 级 | $0.75 / $3.75 | **约 $50** | 约 $25 |
| GPT-5.6-sol 级 | $4 / $20 | 约 $264 | 约 $132 |
| Claude Opus 4.6 级 | $5 / $25 | 约 $330 | 约 $165 |
| Claude Fable 5 级（榜首） | $10 / $50 | 约 $660 | 约 $330 |

**结论：全量前沿蒸馏是 $50-660，不是天价。** 所以拒绝前沿模型的理由**只有条款**，没有成本。

### 四、能力差距有多大（必须承认）

创意写作 v3（2026-09-07 快照）：

| 档 | 代表 | Elo |
|------|------|:---:|
| 闭源前沿 | gpt-6-astra / claude-fable-5-1 / claude-opus-5 | 2121-2164 |
| 开源可租（千亿级） | kimi-k3 / GLM-5.3 | 2064-2071 |
| 开源最强（2.4T） | Qwen3.8-2.4T-A95B | 1841 |
| **开源可自部署（27B）** | **Qwen3.8-27B** | **1668** |

**自部署教师与闭源前沿差约 500 Elo。** 这是本项目的现实上限差。

### 五、蒸馏鸿沟：强教师 ≠ 强学生

学生只有 4B。教师从 1668 换到 2164（+496 Elo），**学生能提升多少是未知的**——大概率远小于 496。教师越强，学生"够不着"的部分越多，边际收益递减。

**因此唯一正确的回答方式是做实验，而不是争论：**

| 实验 | 设计 | 成本 |
|------|------|:---:|
| 教师 A/B 测试 | 同一批 500 条 prompt，三个教师各生成一套（Qwen3.8-27B / GLM-5.3-Flash / 前沿 API），训三个 LoRA，对比学生表现 | 低（1500 条） |

**这是"要不要用 GPT/Claude"的正确答案：用数据回答，别猜。**

### 六、推荐的最终策略（三层）

| 层 | 内容 | 用什么 | 合规性 |
|:---:|------|------|:---:|
| 1 | **全量训练数据**（5000 条） | 开源 A 档自部署 | ✅ 零风险 |
| 2 | **评测与裁判**（不进训练集） | 闭源前沿 API | ✅ 正常使用 |
| 3 | **风格锚点**（200-300 条，<10%） | 闭源前沿 API | ⚠️ 灰色，**用户决定** |

**第 3 层值得单独说明：** 少量极高质量样本对风格的影响力远超其占比（LIMA 效应的经验）。用 200-300 条前沿样本当"风格锚点"，可能是性价比最高的一次冒险——**但这是用户的选择，不是默认动作**。

## 六·补3 · 混合配方落地：按环节落位 + 三个坑

> 承接 [六·补2](#六补2--前沿闭源模型gpt--claude--gemini能不能用) 与 D21。补的是**怎么落地**，不重复条款与成本。

### 一、补充条款核查（xAI / DeepSeek）

六·补2 只覆盖了 OpenAI / Anthropic / Google。另两家核查结果：

| 厂商 | 抓到的条款 | 结论 |
|------|------|:---:|
| **xAI** | 未见"禁止训练竞争模型"；只有"你的内容可用于训练我们"（可关闭） | ⚠️ 未找到禁止条款 |
| **DeepSeek** | 有"Requirements and Restrictions"章节，内容为内容合规类；4.2 条明确**输出权利归用户** | ⚠️ 未找到禁止条款 |

> 注意：这是"**未找到**"，不是"不存在"。两家的 API 专项条款未穷尽核查。原始页面缓存在 `reports/web-tos-*.txt`。

### 二、坑 1：闭源顶尖模型在 RP 场景会拒答

**这是六·补2 没覆盖、但会直接毁掉数据的坑。**

| 数据类型 | 能否用闭源顶尖 | 原因 |
|------|:---:|------|
| 状态跟踪 / 记忆协议 / 工具调用（SFW 结构类） | ✅ 放心用 | 内容安全，闭源模型表现最好 |
| 通用对话 / 日常聊天 | ✅ 可用 | —— |
| **NSFW / 成人向 RP 语气** | ❌ **绝对不要用** | 会拒答、说教、输出"我不能继续"，直接污染训练集 |

**所以 RP 语气与风格数据必须主要来自开源模型（含社区去审查版），闭源只做 SFW 结构锚点。**

### 三、按环节落位表（L0 / L1 / L2 + 裁判）

| 环节 | L0 锚点（闭源，少量） | L1 主体（自部署，零风险） | L2 难样本（开源 B 档，租用） |
|------|------|------|------|
| RP 风格 / 语气 | claude-opus-5、muse-spark（**仅 SFW**） | Qwen3.8-27B + 去审查版 | GLM-5.3-Flash |
| 状态跟踪 | **gemini-3.8-flash**（1M 上下文 + 便宜） | Qwen3.8-27B | GLM-5.3-Flash / DeepSeek-V4-Flash |
| 记忆写入 / 读取 | **claude-opus-5**（协议理解最强） | —— | GLM-5.3-Flash（thinking） |
| 工具调用 | gpt-5.6 / claude（格式最规范） | GLM-4.7-Flash | —— |
| 通用对话防退化 | —— | Qwen3.8-27B / Qwen3.6-35B-A3B | —— |
| **数据筛选（裁判）** | gemini-3.8-flash | —— | —— |

配比（以 5000 条为例）：**L0 = 200 条（4%）· L1 = 3800 条（76%）· L2 = 1000 条（20%）**。

### 四、裁判层怎么用（最划算的一层）

| 项 | 做法 |
|------|------|
| 用途 | 给 L1/L2 的自部署数据**打分**，剔除 slop / 拒答 / 重复 / 格式错 |
| 选型 | gemini-3.8-flash 档（$0.75 / $3.75 每 M） |
| 成本 | 5000 条 × (2000 入 + 200 出) ≈ 10M / 1M → **约 $11** |
| 合规 | 输出**只用于筛选、不写入训练集** → 完全合规（D21 第 2 层） |

### 五、坑 2：锚点数据必须人工过一遍

200-300 条是**人可以读完**的量。这一步能挡住"闭源模型说教腔""格式不匹配""角色设定漂移"三类问题，性价比最高。

### 六、坑 3：教师 A/B 实验必须先做（否则第 3 层可能白花钱）

六·补2 已提出这个实验，这里落成可执行设计：

| 项 | 设计 |
|------|------|
| 输入 | **同一批 500 条 prompt** |
| 教师 | 三组：Qwen3.8-27B（自部署）/ GLM-5.3-Flash（开源 B 档）/ 前沿 API（锚点） |
| 产出 | 三套数据 → 训三个 LoRA（同超参）→ 同一评测集对比 |
| 判定 | 若"前沿 LoRA"没有明显更好 → **第 3 层永久关闭**，省心省钱 |
| 成本 | 1500 条生成 + 3 次 LoRA 训练（本机可跑） |

### 七、生成锚点数据时的操作清单

1. **关闭"用我的数据训练"开关**（OpenAI / Anthropic / xAI 都提供）
2. **控制速率**：锚点只有 200-300 条，正常速率即可，不要并发轰炸
3. **保留调用记录**：请求 ID、时间、模型版本，便于追溯
4. **内容分级**：SFW 结构数据走闭源，NSFW / RP 语气走自部署
5. **国内渠道优先**：qwen3.8-max / glm-5.3-max / kimi-k3-max / deepseek-v4-pro 均可直连、价格 $1.4-6/M、无跨境支付问题

### 策略：先小后大

第一版只做 **500-1000 条**，验证格式与管线是否有效，再扩大规模。不要一上来就合成几千条。

### 若使用商业 API 的反封禁注意事项

1. **选对供应商** —— 中文 RP 内容，国内厂商通常比欧美厂商宽容
2. **控制速率** —— 避免短时间爆发式调用（正常付费使用不会因"蒸馏"本身被封）
3. **付费官方渠道** —— 不使用共享账号或异常账号
4. **内容分级** —— 把样本分成"安全"和"敏感"，敏感的走自部署或宽容供应商
5. **注意条款** —— 部分厂商明确禁止"用输出训练竞争模型"，需自查

**风险来源排序：** 内容违规 > 异常用量 > 条款问题。正常付费、正常速率、内容合规的使用不会被封。

## 六·补4 · 多厂商教师全景 + 混合蒸馏配方（2026-09-21 核查）

> 触发：用户质疑"为什么全是小模型，不考虑 GPT / Claude？" → 核查后结论：**教师从来不是小模型，只是"能合法自部署的"那一档偏小**。本节补上 六·补3 缺的**无审查教师层**，并给出混合配方 v2。

### 一、先纠正一个误解：学生 4B ≠ 教师 4B

| 角色 | 是谁 | 规模 |
|------|------|------|
| **学生（最终部署的）** | Nova = Qwen3-VL-4B + 双通路 + 记忆 | 4B（≈8B 计算量） |
| **教师（教它的）** | 见下面全景表 | **27B ~ 2.4T + 闭源前沿** |

学生小是为了能在 4060 上跑；教师大是为了教得好。**这两件事不冲突，也不该混为一谈。**

### 二、多厂商教师全景（2026-09 核查）

**开源可自部署（按 intelligence 排序；来源 modelgrep / OpenRouter）**

| # | 模型 | 厂商 | Intel | 许可 | 备注 |
|:-:|------|------|:---:|:---:|------|
| 1 | GLM 5.3 | 智谱 Z.ai | 44.8 | MIT | 开源第一 |
| 2 | Kimi K3 | 月之暗面 | 43.6 | 自定义 | |
| 3 | **GLM 5.3 Flash** | 智谱 Z.ai | **41.8** | **MIT** | **$0.09/M，性价比之王** |
| 4 | Qwen3.8 2.4T-A95B | 阿里 | 39.9 | 自定义 | |
| 5 | DeepSeek V4.1 Flash | 深度求索 | 39.5 | MIT | |
| 6-9 | DeepSeek V4 Pro / V4 Flash 系 | 深度求索 | 34.3-36.0 | MIT | |
| 10 | **Qwen3.8 27B** | 阿里 | **33.7** | **Apache-2.0** | 同族主力教师 |
| 11 | GLM 5.2 | 智谱 | 33.7 | MIT | |
| 12 | MiniMax M3 | MiniMax | 29.2 | 自定义 | |
| 14/21 | Inkling / Inkling Small | Thinking Machines | 27.8 / 25.0 | 开源权重 | 1M 上下文 |
| 17/20 | MiMo-V2.5 / -Pro | 小米 | 25.2 / 26.0 | 开源权重 | |
| 19 | Hy3 | 腾讯 | 25.3 | 开源权重 | |
| 22/23 | Ling 3.0 Flash (VL) | 蚂蚁 InclusionAI | 24.9 / 24.6 | 开源权重 | |
| 25 | Qwen3.5-27B | 阿里 | 22.9 | Apache-2.0 | BenchLM RP 榜第 1（proxy 95） |

**闭源（只能调 API）**

| 模型 | 厂商 | 备注 |
|------|------|------|
| GPT-5.x 系 | OpenAI | 工具调用 / 格式最规范 |
| Claude Opus 5 系 | Anthropic | 协议理解、长文一致性最强 |
| Gemini 3.8 Flash | Google | 1M 上下文 + $0.75/M，最便宜的顶尖 |
| Grok 4.6 | xAI | 44.3 intel；条款**未找到**禁止项 |
| Qwen3.8 Max (0902) | 阿里 | 45.4 intel，**无审查榜第 1**（服务商未加审核层） |

> **落选：** Muse-Glimmer-30B（Meta）区域锁定；**Mistral 系本期未进开源前 25**，不再作为主力候选。

### 三、本轮最重要的发现：无审查教师层已经"够强 + 够干净"

**1. 无审查生态已整体迁移到 Qwen3.8-27B**（Hugging Face 实测）

| 模型 | 基座 | 许可 | 下载 | likes |
|------|------|:---:|:---:|:---:|
| `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` | Qwen3.8-27B | **Apache-2.0** | 228 万 | 1193 |
| `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` | Qwen3.8-27B | **Apache-2.0** | 207 万 | 1331 |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | Qwen3.8-27B | **Apache-2.0** | 238 万（GGUF） | 835 |
| `OBLITERATUS/Qwen3.8-27B-OBLITERATED` | Qwen3.8-27B | **Apache-2.0** | 119 万 | 1286 |
| `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` | Qwen3.8-27B | — | 174 万 | 523 |
| `DavidAU/Qwen3.8-27B-TURBO-Fable-…-Heretic-Uncensored` | Qwen3.8-27B | Apache-2.0 | 135 万 | 1021 |

**2. 更关键：顶尖档也有"无审查 + 宽松许可"版**

| 模型 | 基座 | 许可 | Intel | 下载 |
|------|------|:---:|:---:|:---:|
| **`orcarouter/GLM-5.3-Flash-Uncensored-FP8`** | `zai-org/GLM-5.3-Flash` | **MIT** | **41.8** | 8.9 万 |
| **`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`** | `deepseek-ai/DeepSeek-V4.1-Flash` | **MIT** | **39.5** | 3.5 万 |

> **意义（本节最重要的一条）：**「顶尖质量 + NSFW 不拒答 + 零条款风险」**三者可以同时成立**，不必二选一。FP8 权重可直接起 vLLM 服务，不需要自己再训练。

### 四、RP 场景的独立证据：GPT / Claude 不是第一梯队

**RP-Leaderboard（psychosmiley，对抗式 RP 测试，325 分制，6 模型）**

| 排名 | 模型 | 得分 |
|:---:|------|:---:|
| 1 | kimi-k2.5 | 88.9% |
| 2 | gemini-3-pro-preview | 87.1% |
| 3 | grok-4.1-fast | 85.8% |
| 4 | **claude-opus-4.5** | **66.5%** |
| 5 | **gpt-5.1-codex** | **61.8%** |
| 6 | nemotron-3-ultra-550b-a55b | 56.6% |

**modelgrep RP 榜（按 OpenRouter 真实 RP 流量份额，非审核模型 + 32K 以上上下文）**

| 排名 | 模型 | RP 流量份额 |
|:---:|------|:---:|
| 1 | DeepSeek V4 Flash 0423 | **20.0%** |
| 2 | DeepSeek V4 Flash 0731 | 13.5% |
| 3 | DeepSeek V4.1 Flash | 6.3% |
| 4 | Gemini 2.5 Flash Lite | 5.3% |
| 5 | DeepSeek V3.2 | 4.5% |
| 6 | GLM 5.3 Flash | 2.5% |

> 两个榜样本都不大，不能当铁证；但它们**互相印证**且和社区实际用法一致：**RP 这一环，主力是 DeepSeek / Kimi / Gemini / GLM，不是 GPT / Claude。** 闭源里只有 Gemini、Grok 进过第一梯队。

### 五、混合蒸馏配方 v2（四层）

| 层 | 占比 | 教师 | 教什么 | 合规 |
|:---:|:---:|------|------|:---:|
| **L1 主体** | 55-65% | Qwen3.8-27B（Apache-2.0，自部署）+ GLM-4.7-Flash（MIT，自部署） | 通用对话、RP 骨架、工具调用、格式 | ✅ 零风险 |
| **L2 无审查** | 25-30% | GLM-5.3-Flash-Uncensored（MIT）+ DeepSeek-V4.1-Flash-UNCENSORED（MIT）+ Qwen3.8-27B-Uncensored 系（Apache-2.0） | **NSFW / 成人向 RP 语气**、情感强度、角色黏性 | ✅ 零风险 |
| **L3 顶尖锚点** | 5-10% | 闭源：GPT-5.x / Claude Opus 5 / Gemini 3.8 Flash（**仅 SFW**）；开源：GLM-5.3（MIT）/ Kimi K3 | 状态跟踪 schema、记忆读写协议、工具轨迹、长上下文一致性 | 闭源 ⚠️ 灰色；开源 ✅ |
| **L4 裁判** | —— | 闭源 API（**仅非 NSFW 样本**） | 打分、剔除 slop / 拒答 / 套路化 | ✅ 合规 |

**为什么 L3 必须限死 5-10%：**
1. **拒答** —— 闭源在 NSFW 上直接拒答（模型行为，不是条款问题）
2. **助手腔污染** —— 4B 学生会把"我很乐意帮您"放大成客服腔，这是最毁 RP 的失败模式
3. **蒸馏鸿沟** —— 2164 Elo 的教师，4B 学生吸收率很低（见 六·补2 第五节）
4. **条款** —— 见下节

**推荐做法：** 把 L3 的主力换成 **`orcarouter/GLM-5.3-Flash-Uncensored`（MIT，41.8 intel）**，闭源只留 2-3% 做"格式最规范的那一小撮"（工具调用轨迹、记忆协议）。

### 六、"无视条款"这条：我只能给事实，不给规避方案

| 事实 | 说明 |
|------|------|
| **拒答 ≠ 条款** | NSFW 拒答是**模型自身的对齐行为**。改条款也改不了它 —— 所以"无视条款"**解决不了拒答**，只有**无审查权重**能解决 |
| **条款写了什么** | OpenAI / Anthropic / Google 三家**明文禁止**用输出训练竞争模型（原文见 六·补2 与 `reports/web-tos-*.txt`） |
| **实际风险形态** | 低量使用基本不会被技术检测；**被处理的路径是账号**（封号 → 断供），不是法律追责 |
| **风险敞口在缩小** | L2 的 MIT/Apache 无审查教师已达 **39-42 intel**（闭源前沿约 45），**闭源教师的边际价值只剩 3-5 分** |
| **结论** | **要 NSFW 就必须自部署**（L2 本来也只能自部署）；闭源留给 SFW 结构数据 → 风险敞口自然接近 0 |

### 七、混合蒸馏的 5 个坑（新增）

1. **模板归一化** —— GLM / Kimi / DeepSeek 的对话模板与 Qwen 不同。**所有教师输出必须统一转成学生的 `qwen3_vl` chat 模板**再训练，否则学出一堆错位的角色标签
2. **不要 greedy 采样** —— 生成数据用 `T=0.7-1.0` + `top_p=0.9-0.95`；greedy 会让 5000 条数据高度同质
3. **同 prompt 多教师 = 选优，不是拼接** —— 多教师输出互相打架，必须由 L4 裁判选出最好的一条（或按环节路由到**唯一**教师）
4. **思考模式要处理干净** —— 见 六·补 原则 3
5. **先做 100 条小样对照** —— 混教师前确认风格差异在可接受范围内，再放大规模

### 八、数据来源（抓取于 2026-09-21）

| 来源 | 缓存文件 |
|------|------|
| modelgrep 开源榜 / 无审查榜 / RP 榜 | `reports/web-mg-os.txt`、`web-mg-unc.txt`、`web-mg-rp2.txt` |
| RP-Leaderboard（psychosmiley） | `reports/web-rp-lead2.txt` |
| BenchLM RP 榜 | `reports/web-benchlm-rp.txt` |
| Hugging Face API（无审查模型 + 许可核查） | `reports/web-hf-unc.txt`、`web-hf-q38.txt` |
| 条款原文 | `reports/web-tos-*.txt` |

## 六·补5 · 社区评价核查：RP 到底谁最好用（2026-09-21）

> 触发：用户要求"去看看社区评价，哪个模型最好用 RP"，并明确**条款与拒答不再作为约束，只看质量**。
> 全部结论来自联网抓取，原始页面缓存在 `reports/web-*.txt`。

### 一、结论先行：RP 的"第一"分属四个不同模型

**没有一个模型通吃。** 四个维度各有冠军，且互相不重叠：

| 维度 | 冠军 | 关键数据 | 来源 |
|------|------|------|------|
| **真实使用量** | **DeepSeek V4 Flash 系** | OpenRouter RP 流量：V4 Flash 0423 **20.0%** + 0731 **13.5%** + V4.1 Flash 6.3% ≈ **40%** | modelgrep roleplay 榜 |
| **对抗式 RP 测试** | **Kimi K2.5** | 88.9%，高于 gemini-3-pro 87.1%、grok-4.1 85.8% | RP-Leaderboard（325 分制，6 模型） |
| **角色沉浸口碑** | **GLM-5.x 系** | r/LocalLLaMA 原话："**picks up character better than anything short of Opus**" | 311 回复 megathread 合成 |
| **慢热 / 长线剧情** | **Kimi K2.6 / K3** | r/SillyTavernAI 帖题："**Kimi K2.6 is the best LLM for slowburn**" | rpfiend 周报 |
| 文笔 Elo | Claude 系 | BenchLM 明确写"Claude models hold the top Arena Elo scores" | benchlm.ai |
| 幽默 / 角色声音 | DeepSeek V4 | 社区原话："first time DeepSeek made me laugh HARD" | rpfiend 周报 |

### 二、⚠️ 一个必须先说的数据陷阱

modelgrep 的 `sillytavern` / `janitorai` / `chub-ai` / `roleplay` **四个页面排名完全相同**——因为它们都来自**同一个 OpenRouter RP 流量分类器**。

**它们是一个来源，不是四个。** 不能拿"四个平台都选 DeepSeek"当四倍证据。

同理，RP-Leaderboard 只有 6 个模型，样本极小，只能当方向性信号。

### 三、社区时间线（2026-04 → 2026-06，r/SillyTavernAI 视角）

| 日期 | 事件 | 社区反应 |
|------|------|------|
| 4/20 | Kimi K2.6 发布 | 先被骂"不值得"，两天后翻盘："slowburn 最佳"——**长线剧情不掉线** |
| 4/24 | DeepSeek V4 Flash / Pro 发布 | "**surprisingly good**"，幽默感与角色声音好评；V4 Pro 有"随机插入数字"的 bug |
| 6/9 | Fable 5 发布 | 立刻引爆审查争议（"new level of censorship in such short time"） |
| 6/13 | **Anthropic 给旧模型加安全层** | Claude 3.x 突然开始拒答，**破坏生产环境的角色一致性** |
| 6/16 | GLM 5.2 上线 OpenRouter | "GLM 5.2 is making me enjoy a card I normally only use for testing"；NSFW 测试整体正面 |

**Z.ai 的转向（重要）：** rpfiend 作者原话——GLM 曾是他的主力（"I said I was a GLM simp"），但 **Z.ai 涨价 + 降速 + 条款收紧**，他转去了 MiniMax。而 MiniMax 的代价是**审查明显更严**。

> 结论：**GLM 系的口碑很好，但 Z.ai 官方端点在收缩**。这反过来加强了"用 GLM 的开源权重自部署"这条路线——`orcarouter/GLM-5.3-Flash-Uncensored`（MIT）不受端点政策影响。

### 四、GPT / Claude 在 RP 上的真实位置（再确认）

| 来源 | 结论 |
|------|------|
| RP-Leaderboard | claude-opus-4.5 **66.5%**、gpt-5.1-codex **61.8%** —— 明显低于 Kimi / Gemini / Grok |
| r/SillyTavernAI 共识 | Claude 的文笔仍被认可，但**审查持续收紧**（6/13 事件），"coding benchmarks are not roleplay benchmarks" |
| modelgrep 流量榜 | 前 6 名里**没有** GPT / Claude |

**修正上一节的结论：** 用户解除约束后，Claude 的**文笔**值得用，但它的**角色沉浸与长线一致性不是第一梯队**，且官方端点会中途收紧。合理定位是"**文笔锚点**"，不是"RP 主力"。

### 五、质量优先下的最终教师表 v3

> ⚠️ **本表已被取代。** 当前生效的是 **v4**（[六·补6](#六补6--gemini-专项评估2026-09-21) 第五节）——Gemini 升入 L1，Claude 降为 L1 副手，教师分七层。此表保留供追溯。

| 层 | 占比 | 教师 | 为什么是它 | 合规 |
|:---:|:---:|------|------|:---:|
| **L1 · 角色沉浸** | 30-35% | **GLM-5.3-Flash-Uncensored**（MIT，41.8 intel） | 角色沉浸口碑第一 + 可自部署 + 成本极低 | ✅ |
| **L2 · 长线与状态跟踪** | 20-25% | **Kimi K3 / K2.6**（租用）+ **DeepSeek V4 Flash 系** | slowburn 冠军 + 真实使用量冠军；1M+ 上下文 | ✅ |
| **L3 · 文笔与情绪张力** | 15-20% | **Claude Opus 5**（渠道） | Arena Elo 文笔最高 | 用户自担 |
| **L4 · NSFW / 情感强度** | 15-20% | `GLM-5.3-Flash-Uncensored` + `DeepSeek-V4.1-Flash-UNCENSORED` + `Qwen3.8-27B-Uncensored` 系 | 不拒答 + 已实测许可 | ✅ |
| **L5 · 裁判** | —— | Gemini 3.8 Flash（便宜）/ Claude | 打分、剔除 slop 与套路化 | ✅ |
| **L6 · 同族保底** | 5% | Qwen3.8-27B（Apache-2.0） | 与学生同族，防分布漂移 | ✅ |

**与 v2 的差别：** v2 把闭源压到 5-10%，v3 因为约束解除，把 Claude 提到 15-20% 的**文笔层**；但 RP 主力仍然是 **GLM + DeepSeek + Kimi**，理由不再是条款，而是**社区口碑与榜单证据**。

### 六、必须实测的三件事（解除约束反而让假设变多了）

| # | 假设 | 为什么可疑 | 怎么测 |
|:-:|------|------|------|
| 1 | "渠道能拿到可用的闭源 NSFW 输出" | **绕过拒答 ≠ 输出质量不变**。常见副作用：委婉化、句式重复、角色漂移、说教残留 | 50 条同 prompt，闭源渠道 vs 无审查开源，人工盲评 |
| 2 | "Claude 的文笔优势能传递到 4B 学生" | 蒸馏鸿沟（六·补2 第五节）：2164 Elo → 4B 的传递率未知 | 教师 A/B 实验（500 条 × 3 教师 → 3 个 LoRA） |
| 3 | "GLM-5.3-Flash-Uncensored 的 abliteration 没损伤角色沉浸" | abliteration 会削弱边缘案例的连贯性（insiderllm 明确指出） | 对比 orcarouter 去审查版 vs 原版 GLM-5.3-Flash，各 100 条同 prompt |

### 七、数据来源（抓取于 2026-09-21）

| 来源 | 缓存文件 |
|------|------|
| modelgrep：roleplay / SillyTavern / JanitorAI / Chub AI 流量榜 | `web-mg-rp2.txt`、`web-mg-st.txt`、`web-mg-jan.txt`、`web-mg-chub.txt` |
| RP-Leaderboard（psychosmiley） | `web-rp-lead2.txt` |
| BenchLM RP 榜 | `web-benchlm-rp.txt` |
| r/LocalLLaMA megathread 合成（311 回复） | `web-bgos.txt` |
| rpfiend 周报（r/SillyTavernAI 合成） | `web-rpfiend-apr.txt`、`web-rpfiend-jun.txt` |
| rpfiend：GLM vs MiniMax | `web-rpfiend-glmmm.txt` |
| insiderllm：无审查技术路线（Dolphin / abliteration / Heretic） | `web-insider-unc.txt` |

## 六·补6 · Gemini 专项评估（2026-09-21）

> 触发：用户表示"**我觉得 Gemini 的 RP 挺好，我这里也是 Gemini 资源最多**"。本节评估 Gemini 能否从"裁判层"升级为"主力教师"。

### 一、结论先行

**能升级，但不是全部。Gemini 应该吃「散文 + 量产 + 裁判」，不该吃「状态跟踪」。**

理由见第四节的"提前遗忘"问题——那一条直接撞在 Nova 的核心卖点上。

### 二、Gemini 在 RP 上的真实位置（三个独立信号）

| 信号 | 数据 | 含义 |
|------|------|------|
| **OpenRouter RP 合集**（近 7 天真实用量） | Gemini 3 Flash Preview **4.5%**、Gemini 2.5 Flash Lite **3.6%** —— 两个条目进前十 | 社区确实在用；但量级是 DeepSeek V4.1 Flash（19.5%）/ GLM 5.3 Flash（7.8%）的 1/4 左右 |
| **RP-Leaderboard** 对抗测试 | gemini-3-pro-preview **87.1%（第 2 名）** | 对抗式 RP 理解力属第一梯队 |
| **跨厂商拒答对比** | Gemini "comparatively permissive for narrative content" | 四家宽松度排序：**Grok > Gemini > ChatGPT ≈ Claude** |

### 三、Gemini 的 RP 强项（社区原话，非推测）

| 强项 | 证据 |
|------|------|
| **散文质量** | "the prose is **the best the Gemini line has produced to date**"（rpfiend 评 3.5 Flash） |
| **"Less Geminisms"** | 公式化措辞（老 Gemini 病）明显减少，长会话中可感知 |
| **速度 + 价格** | Flash 档 + 1M 上下文；**cache read $0.075/M**（输入价的 1/10） |
| **指令遵循** | Stab's EDH 这类分层权威结构 preset 与 Gemini 搭配最自然 |
| **多模态输出** | HTML 视觉工具与 Gemini 配合最好 → 对 Nova 保留视觉塔（D14）是加分 |
| **成熟 preset 生态** | Stab's EDH / NemoEngine v10 / Marinara's Universal（Marinara 首选模型就是 Gemini 3.1 Pro） |

### 四、Gemini 的 RP 短板（必须记账）

| 短板 | 证据 | 对 Nova 的影响 |
|------|------|------|
| **长上下文"提前遗忘"** | Google 官方支持帖"Gemini seems to lose track of our long conversation"；memorylake"Why does Gemini forget my project context"；"Users say Gemini starts forgetting **long before it's supposed to**"（2026-06-04） | **致命**。Nova 的核心卖点就是长上下文注意力；用会遗忘的模型当状态跟踪教师 = **教学生遗忘** |
| **思考模式干扰 RP** | Marinara 建议"所有模型在 RP 时关闭 reasoning"；有"Gemini 3.1 Pro 忽略指令、思考过程外泄"的报告 | 教师输出必须清洗思考段 |
| **流式被过滤打断** | AI Studio 的 API 级过滤会在**流式输出中途**拦截消息 | 批量生成数据时**必须关掉 streaming** |
| **暴力内容有摩擦** | AI Studio 对暴力场景有过滤摩擦；**NanoGPT / OpenRouter 端点更干净** | 换端点，不是换模型 |
| Geminisms 残留 | 已大幅减少但未消失 | 裁判层要专门查这类套路 |

### 五、教师表 v4（Gemini 资源充足版）

| 层 | 占比 | 教师 | 为什么 |
|:---:|:---:|------|------|
| **L1 散文 / 日常对话** | 25-30% | **Gemini 3.8 Flash / 3.5 Flash**（资源多）+ Claude Opus 5 | 散文质量 + 便宜 + 量大 |
| **L2 角色沉浸** | 20-25% | GLM-5.3-Flash-Uncensored | 角色黏性口碑第一 |
| **L3 长线剧情 / 状态跟踪** | 20-25% | **Kimi K3·K2.6 + DeepSeek V4 Flash 系**（**不用 Gemini**） | 长线一致性冠军；Gemini 有提前遗忘问题 |
| **L4 NSFW / 情感强度** | 15-20% | 无审查开源（GLM / DeepSeek / Qwen 系） | 不拒答 |
| **L5 工具调用 / 结构化** | 5-10% | Gemini 3.8 Flash + GLM-4.7-Flash | 结构化输出 + 便宜 |
| **L6 裁判** | —— | **Gemini 3.8 Flash**（资源最多 → 最划算） | 只做筛选打分，不进训练集 |
| **L7 同族保底** | 5% | Qwen3.8-27B（Apache-2.0） | 防分布漂移 |

### 六、Gemini 资源多的三个用法（不只是当教师）

1. **裁判 + 数据清洗主力** —— cache read $0.075/M + 资源充足 → 5000 条数据的**全量打分**几乎零成本（比 六·补3 第四节估的 $11 更低）
2. **合成 prompt 库** —— 让 Gemini 批量生成 RP 场景 / 角色卡 / 多轮对话骨架（**只出题目，不写最终答案**，避免 Geminisms 污染）
3. **多模态数据标注** —— Nova 保留视觉塔（D14），Gemini 是唯一能大规模标注"图像 ↔ 对话"配对的多模态教师

### 七、⚠️ 一条针对 Nova 的硬警告

**不要把 Gemini 放进状态跟踪 / 记忆读写环节。**

该环节需要的正是"20 轮后仍记得角色换了衣服"，而这恰好是 Gemini 被反复报告会失效的地方（第四节第一行）。这一层留给 **Kimi / DeepSeek / GLM**。

### 八、数据来源（抓取于 2026-09-21）

| 来源 | 缓存文件 |
|------|------|
| OpenRouter RP 合集（真实用量，近 7 天） | `web-or-collection-rp.txt` |
| RP-Leaderboard（psychosmiley） | `web-rp-lead2.txt` |
| 跨厂商拒答对比（ChatGPT/Claude/Gemini/Grok） | `web-arcanum-refuse.txt` |
| rpfiend：Gemini 3.5 Flash Presets（含过滤/端点实操） | `web-rpfiend-gempreset.txt` |
| rpfiend：Gemini 文章索引 | `web-rpfiend-gem.txt` |
| modelgrep：Gemini 3.8 Flash / 2.5 Flash Lite | `web-mg-gem38.txt`、`web-mg-gemlite.txt` |
| Gemini 长上下文遗忘（多来源汇总） | `web-ddg-gemlimit.txt` |
