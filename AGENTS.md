# AGENTS.md · Nova 项目工作守则

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

## 四、环境事实（已核查 · 2026-09-21）

| 项 | 值 |
|------|------|
| Python（唯一可用） | `C:\Users\18889\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`（3.12.14）；本机 `python` / `py` **不在 PATH** |
| pip | 26.2.1（随上述运行时） |
| git | 2.49.0.windows.1 |
| torch | 未安装（需装进项目 venv） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，**8188 MiB**，驱动 610.88 |
| 磁盘 C: | Fixed，**仅剩 0.4 GB** —— **绝对不要把任何下载 / 缓存 / venv 写到 C 盘** |
| 磁盘 D: | Fixed，剩 32 GB —— 放**代码与 venv** |
| 磁盘 H: | Fixed，剩 131.9 GB —— 放**模型权重、数据集、HF 缓存、pip 缓存** |

**每次会话开始必须设置的环境变量：**

```powershell
$env:HF_HOME       = 'H:\hf-cache'
$env:PIP_CACHE_DIR = 'H:\pip-cache'
$env:TMP           = 'H:\tmp'
```

> 不设 HF_HOME，模型会下到 `C:\Users\18889\.cache\huggingface`（约 9GB），**直接把 C 盘塞爆**。

## 五、写文件的方法（重要）

本目录文件用 `apply_patch` 修改。但**直接调用 `apply_patch.bat` 会因 UTF-8 中文内容报错**，必须走 exe。可用模板（`（文件内容）` 换成正文）：

```powershell
$exe="C:\Users\18889\AppData\Local\OpenAI\Codex\bin\eab8377aebac6c07\codex.exe"
$body = @'
（文件内容）
'@
$l = (($body.TrimEnd("`r","`n") -split "`r?`n") | ForEach-Object {"+"+$_}) -join "`n"
& $exe --codex-run-as-apply-patch ("*** Begin Patch`n*** Add File: 路径`n" + $l + "`n*** End Patch")
```

**两个必须注意的坑：**
1. 上面 `'@` 的位置，真实使用时是一个**独占一行的单引号加 @ 符号**（本文档里用占位符，因为直接写会提前结束字符串）。
2. here-string 的收尾符**必须独占一行**；漏掉它会得到"exit code 1 且无任何输出"的空错误，很难查。

- **修改已有文件**：把 `*** Add File: 路径` 换成 `*** Update File: 路径`，正文里用上下文行 + `-`/`+` 标注增删。
- 联网受限：访问 Hugging Face / PyPI 若失败，请用 `require_escalated` 重新发起。
- 中国大陆网络可试镜像：`$env:HF_ENDPOINT = 'https://hf-mirror.com'`。

## 六、禁区（现阶段）

1. 不要先训练（双通路骨架未验证，数据与算力都是浪费）
2. 不要先做量化 / 部署格式 / GGUF 转换
3. 不要使用中文训练数据（违反 D18 顺序）
4. 不要引入任何外部记忆、检索、agent 框架（违反 D06）
5. 不要把重心放在"模型多大"上——本项目赌的是**架构**，不是参数量

---

## 七、Git 提交规范

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
docs: D30 + reports/s4-nf4-gemv.md + HANDOFF 更新
```
