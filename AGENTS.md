# AGENTS.md · Nova 项目工作守则

> 更新：2026-09-23（第四节环境事实已同步）。

> 任何新会话（新 agent）进入本目录，**先读 [HANDOFF.md](HANDOFF.md)**，再读 [docs/13-decisions.md](docs/13-decisions.md) 与 [docs/11-roadmap.md](docs/11-roadmap.md)。
> 本文件是规则层，HANDOFF.md 是"现在做什么"。

---

## 一、项目一句话

Nova 是一个 **单用户、单人格、有持续记忆** 的脑启发式个人 AI 模型项目：4B 级多模态基座 + 双通路（对话 / 行动）+ 模型内部记忆，全部跑在 RTX 4060 Laptop 8GB 上，不依赖任何外部系统。

## 二、硬约束（不可违反；改动必须记录为新决策）

| 编号 | 约束 |
|:---:|------|
| D06 | **记忆全部在模型内部**——不做 RAG、不做外部数据库、不做外部 agent 框架 |
| D07 | 记忆以 **safetensors 张量** 持久化，不是 JSON / 文本 |
| D09 | 记忆模块与语言模型**解耦**，表示空间冻结（防表示漂移） |
| D13 | 手机端暂不考虑 |
| D14 | **保留多模态**（视觉塔共享，不参与双通路复制） |
| D17 | 硬件锁定 RTX 4060 Laptop 8GB，显存预算 **< 7GB** |
| D18 | 训练顺序：**先纯英文，后中文 LoRA**，不做中英同时训 |

## 三、工作方式

- **语言：** 面向用户的文档、报告、回复一律用中文。
- **决策日志：** [docs/13-decisions.md](docs/13-decisions.md) **只追加，不修改历史条目**。变更用新编号 + 标注旧条目的取代关系。
- **结论标注：** 每个结论必须分清 **已核查 / 待实测 / 推测**。不要把推测写成事实，不要美化可行性。
- **验收优先：** 先跑通最小前向，再谈训练；先有基线数字，再谈优化。
- **产物落盘：** 每完成一步，把结果与验收证据写进 `reports/`，并回写到对应 docs 章节。
- **风格：** 紧凑、直接、可执行。给出排序过的短方案，而不是长篇分析。

## 四、环境事实（已核查 · 2026-09-23）

| 项 | 值 |
|------|------|
| 项目路径 | `D:\360MoveData\Users\18889\Documents\Nova` 是**符号链接**，实体在 **`H:\Nova`** —— 项目内的缓存 / venv 实际都落在 H 盘 |
| Python（唯一可用） | `C:\Users\18889\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`（3.12.14）；本机 `python` / `py` **不在 PATH** |
| 项目 venv | `.venv\Scripts\python.exe` —— torch 2.6.0+cu124、transformers、triton-windows 3.2.0.post21 |
| pip / git | 26.2.1 / 2.49.0.windows.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，**8188 MiB**，驱动 610.88 |
| 磁盘 C: | **剩 6.48 GB** —— **绝对不要把任何下载 / 缓存 / venv 写到 C 盘** |
| 磁盘 H: | Fixed，剩 131.9 GB —— 项目实体、权重、缓存都在这里 |
| 权重 | Qwen3-VL-4B-Instruct，**8.89 GB 已缓存**在项目内 `.hf-cache/` |

**每次会话开始必须设置的环境变量**（缓存一律放**项目内**；不要再指到 `H:\hf-cache`，那会另建一份 9GB 重复缓存）：

```powershell
$env:HF_HOME        = (Resolve-Path .).Path + '\.hf-cache'
$env:PIP_CACHE_DIR  = (Resolve-Path .).Path + '\.pip-cache'
$env:TMP            = (Resolve-Path .).Path + '\.tmp'
$env:TEMP           = $env:TMP
$env:TRITON_CACHE_DIR        = $env:TMP + '\triton-cache'
$env:TORCHINDUCTOR_CACHE_DIR = $env:TMP + '\inductor-cache'
```

> 不设 HF_HOME，模型会下到 `C:\Users\18889\.cache\huggingface`（约 9GB），**直接把 C 盘塞爆**。

- 联网受限：访问 Hugging Face / PyPI 若失败，请用 `require_escalated` 重新发起。
- 中国大陆网络可试镜像，**必须带协议头**（机器级 `HF_ENDPOINT` 缺协议头会让所有 HF 请求报 `UnsupportedProtocol`）：`$env:HF_ENDPOINT = 'https://hf-mirror.com'`。

## 五、禁区（现阶段）

1. 不要先训练（双通路骨架未验证，数据与算力都是浪费）
2. 不要先做量化 / 部署格式 / GGUF 转换
3. 不要使用中文训练数据（违反 D18 顺序）
4. 不要引入任何外部记忆、检索、agent 框架（违反 D06）
5. 不要把重心放在"模型多大"上——本项目赌的是**架构**，不是参数量

---

## 六、Git 提交规范

**仓库：** <https://github.com/captain-wangrun-cn/Nova>（**public**，默认分支 `main`，远端名 `origin`）。

**格式：**

```
type(scope): msg
```

| 段 | 取值 | 说明 |
|------|------|------|
| `type` | `feat` `fix` `perf` `refactor` `test` `docs` `exp` `chore` | 小写，**固定枚举，不要自己造** |
| `scope` | `model` `decode` `quant` `kernels` `loader` `graph` `cli` `bench` `env` `docs` | 小写，模块 / 主题 |
| `msg` | **中文，≤ 25 字** | 写**结果**，不写"改了哪些文件" |

**规则：**

1. **一个提交只做一件事。** 跨 type 就拆开——塞在一起以后 `git log` 查不出东西。
2. 提交前 `pytest tests -q` 必须**全绿**（`docs` / `exp` 类可例外）。
3. 结论按第三节标注 **已核查 / 待实测 / 推测**；要展开就空一行写在正文。
4. **速度数字必须带 `clocks.sm`。** 本机 GPU 空闲时会停在 ~780 MHz（上限 3105 MHz），
   同一条命令实测差 **1.9x**（14.0 → 26.7 ms/token）。**不同时间点的数字不能直接对比。**
5. 不提交 `.venv/` `.hf-cache/` `.pip-cache/` `.tmp/` `__pycache__/`（见 `.gitignore`）。
6. 碰硬约束（D06/D07/D09/D13/D14/D17/D18）的改动必须带 `Dxx` 引用。

**示例：**

```
fix(model): lm_head_forward 无限递归，补逐位一致测试
perf(model): lm_head 4-bit，单通路 16.0→14.0 ms/token
exp(quant): 自写 Triton NF4 GEMV 证伪，速度路径 ① 结案
feat(cli): 图解码交互 CLI，支持多轮与采样
docs: D30 + reports/speed-path1-nf4-gemv.md + HANDOFF 更新
```
