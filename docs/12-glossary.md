# 12 · 术语表

> 供快速查阅。按主题分组。

---

## 模型基础

| 术语 | 解释 |
|------|------|
| **参数量** | 模型权重总数，决定知识上限和推理深度天花板 |
| **权重（Weights）** | 模型训练后固化的参数，是"长期记忆"的载体 |
| **隐藏状态（Hidden State）** | 模型每一层输出的中间表征（张量），本架构中两条通路交换的就是它 |
| **前向传播（Forward Pass）** | 输入经过所有层得到输出的过程，生成一个 token 算一次 |
| **Transformer** | 主流模型架构，核心是自注意力机制 |
| **层（Layer）** | Transformer 的基本堆叠单元 |

## 注意力机制

| 术语 | 解释 |
|------|------|
| **自注意力（Self-Attention）** | 序列内部每个位置关注其他位置 |
| **交叉注意力（Cross-Attention）** | 一个序列关注另一个序列；本架构中两条通路互相读取 |
| **GQA（分组查询注意力）** | 多个查询头共享少量 KV 头，减少 KV Cache。Qwen3-4B 用 8 个 KV 头 |
| **MHA / MQA** | 多头注意力 / 多查询注意力（GQA 的两个极端） |
| **MLA（多头潜在注意力）** | DeepSeek 的 KV 压缩技术，Cache 缩小约 90% |
| **滑窗注意力（SWA）** | 只在局部窗口内做注意力，大幅减少 KV Cache。Gemma 系列使用 |
| **FlashAttention** | 融合的注意力算子，更快更省显存 |
| **PagedAttention** | vLLM 的 KV Cache 内存管理技术，减少碎片 |

## KV Cache 与缓存

| 术语 | 解释 |
|------|------|
| **KV Cache** | 缓存已计算的键值，避免重复计算。随上下文线性增长 |
| **前缀缓存（Prefix Caching）** | 相同前缀复用 KV Cache，按内容哈希索引 |
| **TTFT** | Time To First Token，首字延迟 |
| **TPS** | Tokens Per Second，生成速度 |
| **带宽受限（Memory-bandwidth-bound）** | 解码时瓶颈在显存带宽而非算力 |
| **块对角掩码（Block-diagonal Mask）** | 批处理中防止不同序列互相注意的掩码 |

## 训练

| 术语 | 解释 |
|------|------|
| **预训练（Pre-training）** | 从海量文本学习语言和知识 |
| **后训练（Post-training）** | SFT / RLHF，塑形行为，不是学新知识 |
| **SFT** | 监督微调，用问答对教模型听话 |
| **RLHF / RLVR** | 基于人类反馈 / 可验证奖励的强化学习 |
| **LoRA** | 低秩适配器，外挂式微调，不动原权重 |
| **QLoRA** | 量化 + LoRA，显存需求大幅降低 |
| **全量微调** | 直接更新所有模型权重 |
| **灾难性遗忘** | 学新任务时忘记旧任务 |
| **持续学习（Continual Learning）** | 不断吸收新知识而不遗忘 |
| **chat template** | 模型对话格式（如 Qwen 的 `<|im_start|>` / `<|im_end|>`）。**不匹配会导致输出乱码** |
| **梯度检查点（Gradient Checkpointing）** | 用计算换显存 |
| **梯度累积（Gradient Accumulation）** | 小 batch 累积成大 batch 效果 |

## 推理优化

| 术语 | 解释 |
|------|------|
| **计算图（Computation Graph）** | "张量怎么算"的算式结构，即架构本身。它在代码里，不在模型文件里（见 16 号文档） |
| **推理引擎** | 实现计算图的程序：transformers / llama.cpp / vLLM / 自研引擎。只认得它内置的架构 |
| **safetensors** | 权重存储格式，本质是 `{张量名: 张量}` 的扁平字典。**张量名必须与代码里的模块名逐字对应** |
| **GGUF** | llama.cpp 的单文件打包格式：量化权重 + 元数据 + 词表 + chat template。**不含代码** |
| **权重映射（Weight Mapping）** | 把旧结构的张量名搬到新结构对应位置。改架构复用权重时必须写 |
| **state_dict** | PyTorch 里模型权重与其名字的映射表，就是训练产物本身 |
| **量化（Quantization）** | 降低权重精度以省显存、提速 |
| **Q4_K_M / Q3_K_M / Q2_K** | GGUF 量化等级，数字越小越省显存但质量越低 |
| **投机解码（Speculative Decoding）** | 小模型草稿 + 大模型验证，一次前向出多个 token |
| **MTP（多头预测）** | 一次前向预测多个未来 token，DeepSeek-V3 使用 |
| **torch.compile** | PyTorch 编译优化 |
| **CUDA Graphs** | 减少 kernel 启动开销 |
| **Chunked Prefill** | 长上下文分块预填充，降低 TTFT |

## 记忆与脑启发

| 术语 | 解释 |
|------|------|
| **记忆令牌（Memory Token）** | 模型生成的、编码长期记忆的向量 |
| **Gist Token** | 把上下文压缩成少量"要旨"token（Mu et al. 2023） |
| **AutoCompressor** | 把长上下文压缩成摘要向量（2023） |
| **潜空间记忆（Latent Memory）** | 以张量而非文本形式存储的记忆 |
| **NTM / DNC** | 神经图灵机 / 可微神经计算机，带潜空间读写记忆（DeepMind） |
| **Infini-attention** | 每个注意力头维护压缩记忆矩阵（Google 2024） |
| **Titans** | 带神经长期记忆的架构，测试时通过梯度更新记忆（Google 2025） |
| **RETRO** | 潜空间检索增强（DeepMind 2022） |
| **记忆巩固（Consolidation）** | 把短期记忆沉淀为长期记忆，类比睡眠 |
| **表示漂移（Representation Drift）** | 模型更新后旧记忆失效的问题 |
| **状态跟踪（State Tracking）** | 持续维护变化中的世界状态（如服装、地点） |
| **预测编码（Predictive Coding）** | 只传输"意外的信息"（预测误差） |
| **全局工作空间（Global Workspace）** | 重要信息"点火"后广播给全脑 |
| **Free Energy Principle** | Karl Friston 的预测编码理论框架 |

## 本项目专有

| 术语 | 解释 |
|------|------|
| **双通路（Dual Pathway）** | Chat 通路 + Agent 通路并行，交叉注意力交换 |
| **Chat 通路** | 负责对话、情感、风格 |
| **Agent 通路** | 负责工具、推理、世界状态 |
| **工具头（Tool Head）** | Agent 通路的输出头，输出结构化 tool call，不走 tokenizer |
| **预测编码门控** | 偏差超阈值才触发交叉注意力 |
| **记忆精度分级** | L0 精确 / L1 高保真 / L2 摘要 / L3 句柄 |
| **并行假设推演** | 一次推演多条故事线，选最优 |
| **多时间尺度并行** | 快 / 中 / 慢 / 超慢通路并行 |
| **单用户单人格** | 一个模型只服务一个人，面向仿生人 |
| **并发自我** | 多个上下文并行（对话 / 环境 / 规划） |

## 硬件

| 术语 | 解释 |
|------|------|
| **RTX 4060 Laptop** | 本项目目标硬件，8GB 显存，约 256 GB/s 带宽 |
| **CPU + GPU 协同** | 本项目中指让 CPU 做工具执行 / IO，而非层卸载 |
| **层卸载（Offloading）** | 把部分层放 CPU，通常拖慢速度（负优化） |
