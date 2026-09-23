# Nova 项目设计文档

> 版本 0.2 · 2026-09-23 · 阶段：骨架验证（S0–S4 完成，第五轮 E1–E6 结案）

> **接手本项目的会话请按此顺序读：** [AGENTS.md](AGENTS.md)（工作守则 · 硬约束 · 环境事实）→ [HANDOFF.md](HANDOFF.md)（执行顺序 S0-S5 · 入手清单）→ [docs/13-decisions.md](docs/13-decisions.md)（决策日志）

---

## 这是什么

Nova 是一个以 **人脑为灵感**、同时 **充分发挥计算机算力优势** 的个人 AI 模型项目。

目标：用 4B 级模型，在消费级硬件上，实现接近闭源模型的对话体验、长上下文记忆一致性，以及"边说话边做事"的能力。

## 一句话概括

> 一个只属于一个人的模型——它有持续的记忆，像人一样遗忘与巩固，能一边对话一边行动，全部运行在模型内部，不依赖任何外部系统。

## 核心创新

| # | 创新点 | 一句话 |
|:-:|--------|--------|
| 1 | 双通路架构 | 对话通路与行动通路并行，通过交叉注意力实时交换隐藏状态 |
| 2 | 内部记忆系统 | 记忆令牌 + 磁盘持久化 + 巩固 + 遗忘，全部在模型内 |
| 3 | 异步工具调用 | 文本流永不中断，工具在后台并行执行 |
| 4 | 预测编码门控 | 只在需要时通信，闲聊时行动通路几乎零开销 |
| 5 | 单用户单人格 | 从架构上消除多租户问题，面向仿生人 / 具身智能 |

## 文档索引

| 文档 | 内容 |
|------|------|
| [AGENTS.md](AGENTS.md) | **工作守则：硬约束、环境事实、禁区、提交规范** |
| [HANDOFF.md](HANDOFF.md) | **执行顺序与交接：S0-S5、第一天完成标准** |
| [01-vision.md](docs/01-vision.md) | 愿景、需求、需求演进史 |
| [02-architecture.md](docs/02-architecture.md) | 核心架构设计 |
| [03-memory-system.md](docs/03-memory-system.md) | 记忆系统（核心创新） |
| [04-async-tools.md](docs/04-async-tools.md) | 异步工具调用 |
| [05-performance.md](docs/05-performance.md) | 速度优化 |
| [06-deployment.md](docs/06-deployment.md) | 部署、缓存层级、单用户 / 多用户 |
| [07-capability-bounds.md](docs/07-capability-bounds.md) | 能力边界与诚实评估 |
| [08-data.md](docs/08-data.md) | 数据集与训练数据策略 |
| [09-speculative.md](docs/09-speculative.md) | 待验证构想 |
| [10-risks.md](docs/10-risks.md) | 风险与未解决问题 |
| [11-roadmap.md](docs/11-roadmap.md) | 路线图与里程碑 |
| [12-glossary.md](docs/12-glossary.md) | 术语表 |
| [13-decisions.md](docs/13-decisions.md) | 决策日志与理由 |
| [14-hardware.md](docs/14-hardware.md) | 硬件选型与显存带宽 |
| [15-language-plan.md](docs/15-language-plan.md) | 语言策略与训练顺序（先英文后中文） |
| [16-model-anatomy.md](docs/16-model-anatomy.md) | 模型是怎么被实现的：文件 · 代码 · 引擎 |

## 关键决策速查

| 决策 | 选择 | 理由 |
|------|------|------|
| 基座模型 | **Qwen3-VL-4B-Instruct** | 4B 多模态，Apache-2.0，中文优秀（英文优先下是否重选见 D19，待定） |
| 多模态 | **保留** | 视觉编码器共享，不参与双通路复制 |
| 母语 | 英语 | 预训练占比高、可用数据多、内部推理更强 |
| 中文 | LoRA 适配器 | 独立训练、独立升级、不污染英文能力 |
| **训练顺序** | **先纯英文 → 后中文 LoRA** | 模式只教一遍、省 LoRA 容量、数据生态（D18，详见 15） |
| 硬件目标 | RTX 4060 Laptop 8GB | 用户现有设备 |
| 部署格式 | 待定 | GGUF / llama.cpp 不支持自定义架构 |
| 记忆存储 | safetensors（模型原生张量） | 非文本、非 JSON |
| 多用户 | 不支持（单用户设计） | 从架构上消除隔离问题 |

## 当前状态

- [x] 需求梳理
- [x] 架构设计
- [x] 可行性评估
- [x] 基座与语言策略定案（D16 / D18）
- [x] 核心骨架实现（S3 双通路 + S4 记忆最小实现，见 HANDOFF）
- [ ] 训练数据准备
- [ ] 训练
- [ ] 部署验证
