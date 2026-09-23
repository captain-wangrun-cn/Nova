# HANDOFF.md · 交接与执行顺序

> 面向**新会话的 agent**。先读 [AGENTS.md](AGENTS.md)（规则 + 环境事实），再看本文件（做什么、按什么顺序）。
> 更新于 **2026-09-22**。

---

## 一、现在的位置

| 项 | 状态 |
|------|------|
| 设计文档 | ✅ **16 份，3359 行**（`docs/01` ~ `docs/16`） |
| 决策 | ✅ **35 条**（`docs/13-decisions.md`）；D19 待定，D21/D22 已被 D23/D24 取代，**D26 技术归因已被 D27 更正**，**D30 取代 D29 第 2 条的路径排序**，**D32 更正 D31 的解读**，**D33 补充 D28 的成立条件**；新增 D31（S4 记忆最小实现）、D32（S4 对照实验）、D33（长上下文注意力）、D34（KV int4 量化：精度无退化，但 3.46x 收益未到手）、**D35（拆 O(n²) 掩码表 + 容量分桶：decode 少搬 48–60%，并推翻"重捕爆显存"）** |
| 教师选型 | ✅ **已冻结（v4 七层，D24）**，S5 直接执行，不要重新调研 |
| 代码 | ✅ **S0-S4 全部完成 + 速度路径 ①/② + KV int4 测量 + P0 解码分桶**：**`src/nova/`**（双通路骨架 + 静态 KV cache + CUDA Graph 解码（**分桶：`BUCKETS`/`for_length`/`grow`**）+ 4-bit lm_head + L0 记忆 `memory.py` + KV 量化 `kvquant.py`）、**`src/chat.py`（交互 CLI，`--paths 1/2`）**、**`src/s4_memory_demo.py`**、`tests/`（**60 passed**）、`src/bench_nova.py`、`src/bench_graph.py`、`src/bench_memory.py`、`src/diagnostics/`（速度归因 + 记忆诊断 + KV 量化诊断 + **重捕/分桶诊断**） |
| 环境 | ✅ torch 2.6.0+cu124 + 权重 **8.89 GB 已缓存**（`.hf-cache`）；基线 4-bit 峰值 **2.79 GiB**；**Nova 单通路图解码 14.0 ms/token（71.3 tok/s，3.28 GiB）/ 双通路 22.7 ms/token（44.0 tok/s，4.83 GiB）** |
| 代码托管 | ✅ **<https://github.com/captain-wangrun-cn/Nova>**（**public**，默认分支 `main`）。提交规范见 [AGENTS.md](AGENTS.md) 第七节 |
| 下一步 | **第五轮**：**E1** ✅ 不 adopt → **E2** ✅ 不 adopt → **E3** ✅ 通过（改选 int8）→ **E5** ✅ 结案（尺子升级到 ≥16 条）→ **E4** 窄化 Triton 证伪（**靶子改成 int8**）→ **E6** PCIe/主机内存分层；然后 **S5 · 数据与蒸馏**（里程碑 2 起点） |

**一句话：设计做完了，现在要开始证明"双通路 + 内部记忆"在 8GB 显存上真的能跑。**

---

> ## ⚠️ 2026-09-21 环境事实更正（新会话必读；`AGENTS.md` 第四节尚未同步）
>
> 1. **`D:\360MoveData\Users\18889\Documents\Nova` 是一个符号链接 → `H:\Nova`。** 项目实体在 **H 盘**上，不是 D 盘。写"D 盘"实际消耗的是 H 盘空间。因此第三节"**D 盘低于 10GB 时把 `.venv` 挪到 H 盘**"的预案**不适用**（它已经在 H 盘上了）。
> 2. **缓存不再放 `H:\hf-cache`。** 本次按用户指示"直接在 nova 文件夹下"，改为项目内 `.hf-cache/` / `.pip-cache/` / `.tmp/`（已进 `.gitignore`）。因为项目实体就在 H 盘，这些目录同样**不在 C 盘**。⚠️ **若照抄 `AGENTS.md` 的 `HF_HOME='H:\hf-cache'`，会另建一份重复的 9GB 缓存**——本次不要照抄。
> 3. **C 盘余量是 6.48GB**，不是文档写的 0.4GB。仍然不写入 C 盘。
> 4. **本机机器级环境变量 `HF_ENDPOINT=hf-mirror.com` 缺少协议头**，会让所有 HF 请求直接报 `UnsupportedProtocol`。**必须显式覆盖为带协议的形式：** `$env:HF_ENDPOINT='https://hf-mirror.com'`。
> 5. ~~`AGENTS.md` 第五节的 `codex.exe` 硬编码路径已过期~~ —— **2026-09-23 起不再需要**：`apply_patch` 工具已可直接改文件（含中文），AGENTS.md 第五节已改成"直接用 `apply_patch`"，绕行脚本与硬编码路径都已删除。
> 6. **`triton-windows 3.2.0.post21` 已装入 `.venv`**（与 torch 2.6.0 兼容；`3.8.0` 不兼容）。Triton / Inductor 缓存目录必须显式指向 H 盘（`TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR`），否则报 `WinError 5`。**但 `torch.compile` 目前对本模型不可用**，见 **D27**。
> 7. **命名澄清（易误判）：`reports/speed-path1-nf4-gemv.md` 与 `tests/test_nf4_linear.py` 属于「速度路径 ①」，不是路线图的 S4。** 路线图的 **S4 = 记忆最小实现**，已在 **2026-09-22 完成**（见第七节）。这两处此前误标了 "S4"，已改名/改标题。

---

## 二、执行顺序总表（S0 → S5）

| 阶段 | 目标 | 预估 | 验收标准 |
|:---:|------|:---:|------|
| **S0** | 环境就绪 | 半天 | ✅ **已完成** —— [reports/s0-environment.md](reports/s0-environment.md) |
| **S1** | Tokenizer 探针 | 1 小时 | ✅ **已完成** —— [reports/tokenizer-report.md](reports/tokenizer-report.md) |
| **S2** | 单通路基线 | 半天 | ✅ **已完成** —— [reports/baseline-qwen3vl4b.md](reports/baseline-qwen3vl4b.md)（4-bit 10.64 tok/s / 8-bit 4.91 tok/s / 峰值 2.79 GiB） |
| **S3** | **双通路骨架** | 2-4 天 | ✅ **已完成** —— [reports/s3-dual-path-skeleton.md](reports/s3-dual-path-skeleton.md)（8 passed / 门控关闭**逐位一致** / 峰值 4.57 GiB） |
| **S4** | 记忆最小实现 | 2-3 天 | ✅ **已完成** —— [reports/s4-memory-min.md](reports/s4-memory-min.md)（**8/8 取回** / 不注入 0/8 / 跨进程可复现 / 峰值 5.17 GiB） |
| **S5** | 数据与蒸馏 | 里程碑 2 | 500-1000 条跑通管线（届时再开） |

**排序原则：**
1. **最便宜、最能证伪的先做** —— S1 只花 1 小时就能关掉文档里挂着的"待实测"。
2. **最大风险独占整块时间** —— S3 是"架构到底行不行"的唯一关键，之前不要碰训练。
3. **训练排最后** —— 骨架没验证前，数据集和 GPU 时间都是浪费。

> **S5 的教师池已经定好了**（**D24**，2026-09-21 核查，配方 **v4 七层**）：L1 散文/对话 = **Gemini 3.8 Flash（资源最多）** + Claude Opus 5 25-30%；L2 角色沉浸 = `orcarouter/GLM-5.3-Flash-Uncensored-FP8`（MIT）20-25%；L3 长线/状态 = Kimi K3·K2.6 + DeepSeek V4 Flash 系 20-25%（**禁止用 Gemini**）；L4 NSFW = 无审查开源 15-20%；L5 工具/结构 = Gemini + GLM-4.7-Flash；L6 裁判 = Gemini 3.8 Flash；L7 同族保底 = Qwen3.8-27B 5%。**Gemini 三条实操：生成数据关 streaming、走 NanoGPT/OpenRouter 端点、清洗思考段。** 完整配方见 [08-data.md](docs/08-data.md) 第六·补6。**开 S5 时不要再重新调研一遍教师。**

---

## 三、S0 · 环境（✅ 已完成 2026-09-21）

> **实测结果：** torch **2.6.0+cu124**，`cuda.is_available()==True`，RTX 4060 Laptop 8.0 GiB（sm_89）。
> 完整证据与本次实际执行的命令见 [reports/s0-environment.md](reports/s0-environment.md)。
> 下面的命令块是**原始配方**，保留备查；与实测的差异是缓存目录（见第一节的更正框）。

```powershell
cd D:\360MoveData\Users\18889\Documents\Nova
$env:HF_HOME='H:\hf-cache'; $env:PIP_CACHE_DIR='H:\pip-cache'; $env:TMP='H:\tmp'
New-Item -ItemType Directory -Force -Path reports,src,tests,data,models | Out-Null

$base="C:\Users\18889\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $base -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -U transformers accelerate safetensors pytest pillow

.\.venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- 需要联网；命令失败就用 `require_escalated` 重发。
- 预期输出：`2.x.x True NVIDIA GeForce RTX 4060 Laptop GPU`。
- 若 cu124 轮子不可用，依次试 cu126 / cu128。
- torch + CUDA 库装完约 **3-6GB**，占 D 盘（D 剩 32GB）。**D 盘低于 10GB 时，把 `.venv` 挪到 H 盘。**

> **要用户批准的事：** 本机网络受限，`pip install` 与模型下载都需要用户点同意（`require_escalated`）。建议一次给出可复用的前缀规则，例如 `pip install`、`hf download`，免得每条命令都问一遍。

**下载基座（S2 之前完成）：**

```powershell
.\.venv\Scripts\python.exe -m pip install -U "huggingface_hub[cli]"
$env:HF_ENDPOINT='https://hf-mirror.com'   # 直连不通时启用
#
# 新版 CLI 是 hf.exe，旧版是 huggingface-cli.exe；两者都不确定时用下面的 Python 一行（版本无关）：
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct')"
```

约 9GB，落在 `H:\hf-cache`。**下载前先确认 `$env:HF_HOME` 已设为 `H:\hf-cache`**，否则会灌进只剩 0.4GB 的 C 盘。

---

## 四、S1 · Tokenizer 探针（✅ 已完成 2026-09-21）

> **验收结果：全部通过。** 报告 [reports/tokenizer-report.md](reports/tokenizer-report.md) · 原始数字 [reports/tokenizer-probe.json](reports/tokenizer-probe.json)
>
> - 中英 token 比 **1.02x**（"中文费 2-3 倍"作废 → 新决策 **D25**）；中文 **0.796 tok/汉字**
> - chat template **9 项检查全过**：`<|im_start|>` / `<|im_end|>` 严格成对，且全部是**单 id**
> - **新发现（对 S5 是硬约束）：** 正文里字面出现 `<|im_start|>` 会被 tokenizer 当**真正的控制符**切出来 → 数据管线必须清洗，见 [docs/08-data.md](docs/08-data.md) 五·补2
> - 已回写：[docs/15-language-plan.md](docs/15-language-plan.md) 第 2.5 节、[docs/13-decisions.md](docs/13-decisions.md) **D25**

产出：`src/tokenizer_probe.py` → `reports/tokenizer-report.md`

必须包含：

1. **中英 token 效率实测** —— 同一段内容的中英对照，给出 tokens/字符、tokens/词，得出比值。
   → 结论回写 [docs/15-language-plan.md](docs/15-language-plan.md) 第 2.5 节，把"待实测"改成实测值。
2. **chat template 验证** —— 构造 3 轮 RP 对话 → `apply_chat_template` → 解码回看 `<|im_start|>` / `<|im_end|>` 是否成对。
   → 这是**上次训练输出乱码的根因排查**（见 [docs/08-data.md](docs/08-data.md) 第五节），必须先坐实。
3. **关键常量** —— 词表大小、特殊 token、最大上下文、vision token 占位符、pad / eos 行为。

验收：报告落盘；把可复用的模板包装写成 `src/chatfmt.py`（后续所有数据与推理都用它，不许另写一套）。

---

## 五、S2 · 单通路基线（✅ 已完成 2026-09-21 · 需要被超越的基准）

> **结果：数字已落盘并冻结。** 报告 [reports/baseline-qwen3vl4b.md](reports/baseline-qwen3vl4b.md) · 原始输出 [reports/baseline-outputs.jsonl](reports/baseline-outputs.jsonl)
>
> | 指标 | 4-bit（速度基准） | 8-bit（质量对照） |
> |------|:---:|:---:|
> | decode 速度（6 条均值） | **10.64 tok/s** | 4.91 tok/s |
> | TTFT 均值 | 0.130 s | 0.264 s |
> | 峰值显存 | **2.79 GiB** | 4.58 GiB |
>
> - ✅ **显存达标**：4-bit 峰值 2.79 GiB，双通路（+约 2 GB）仍在 D17 的 7GB 预算内。
> - ⚠️ **速度不达标（10.6 vs 目标 60）**：已归因，**瓶颈是 bitsandbytes 的 4-bit kernel**（有效带宽 47 GB/s vs fp16 的 234 GB/s），不是模型/显存/配置。详见 [reports/s2-speed-diagnosis.md](reports/s2-speed-diagnosis.md) 与决策 **D26**。
> - ⚠️ **新确立的项目级结论**：fp16 在这张卡上上限只有 **~29 tok/s**，**60+ tok/s 必须靠 4-bit 级有效位宽** —— 量化不是优化项，是必要条件。
> - ❌ **qa-03（唯一有客观答案的题）答错**：输出 **18:17**，正确 **18:27** —— 中间步骤把 `3h45m` 误算成 215 分钟（正确 225），此后一路错到底。**不是不会做题，是中间状态算错后不自查** → 正是 Nova 想解决的那类问题，可作里程碑 2 的固定对照点。
> - **D19（是否换基座）** 还需要人工主观评分才能定；机器只能给速度。质量对照输出已落盘，见报告第三节。

- 固定 6 个 prompt（英文 RP ×3、短问答 ×3），存 `data/eval/baseline-prompts.json`（固定随机种子）。
- **速度用 4-bit 测**（TTFT、tok/s、峰值显存）；**质量用 8-bit 对照**。
  8GB 装不下 BF16 —— 权重本身就 8GB。
- 产物：`reports/baseline-qwen3vl4b.md` + `reports/baseline-outputs.jsonl`。
- 若 `bitsandbytes` 在 Windows + py3.12 装不上 → 备选：GGUF + llama.cpp 测速度；CPU 跑质量对照（慢但可用）。
- **这一组数字就是后续所有优化的比较基准**，也是 D19（是否换基座）的一半证据。

---

## 六、S3 · 双通路骨架（最大风险，慢一点也要做对）

> ### ✅ 2026-09-22 已完成 —— 结论见 [reports/s3-dual-path-skeleton.md](reports/s3-dual-path-skeleton.md) 与 **D28**
>
> **验收全过：** `pytest tests -q` → **8 passed**；**门控关闭时与基线逐位一致**（`max|diff| = 0.000e+00`，prefill + 8 步 decode）；峰值显存 **4.57 GiB** < 7GB；生成 20 token 正常。
>
> **代码在 `src/nova/`**（`config / norm / kernels / layers / cross / model / loader / generate`），测试在 `tests/test_nova_skeleton.py`，基准在 `src/bench_nova.py`。
>
> **顺手拿到的速度收益：** Triton 融合 RMSNorm → 同 36 层下 **13.34 → 16.05 tok/s（+20.3%）**；算子数 **8275 → 6529 / token（-21.1%）**。
>
> **开发中踩到的两个 bug（新会话必读）：**
> 1. `LeanRMSNorm` 权重建成了 fp32 → `exact` 反而与 HF 不一致；`triton` 因 kernel 内显式 `.to(fp16)` 而"侥幸"逐位一致。**"某个实现看起来更接近参考"不能当作它正确的证据。**
> 2. 自写前向的 `position_ids` 从 0 编号 → **prefill 对、decode 错**。**验收必须同时覆盖 prefill 与 decode。**
>
> **后续（第三轮）：** 在此基础上加了 **CUDA Graph 解码**（`src/nova/cache.py` + `src/nova/decode.py`），
> 单通路 **61.84 tok/s**。见 [reports/s3-graph-decode.md](reports/s3-graph-decode.md) 与 **D29**。

<details>
<summary>原始设计要求（保留备查）</summary>

**结构：** 共享 embedding + LM head；中间层复制成两条通路；每隔 4 层插一处交叉注意力；门控控制通信开关。

**正确性判据（最重要）：**

> **门控关闭时，双通路模型的输出必须与单通路基线一致。**

这一条通过，才算"骨架没写错"。做不到就说明权重复制、位置编码或 attention mask 有问题。

**单元测试 4 条：**
1. 前向 shape 正确
2. 门控关闭 ≈ 基线（数值级别接近）
3. 交叉注意力隔离（切断一条通路不影响另一条）
4. 生成 20 token 不崩

**显存验收：< 7GB**（留 1GB 给 KV cache 与系统）。

**失败预案（显存超限时按顺序退让）：** 只复制后半层 → 降 LoRA / 权重精度 → 减少交叉注意力层数 → 降上下文长度。

</details>

---

## 六·补 · 速度路径 ①：Triton NF4 GEMV（✅ 2026-09-22 结案 —— **方向证伪**）

> 完整报告：[reports/speed-path1-nf4-gemv.md](reports/speed-path1-nf4-gemv.md) · 决策 **D30**（取代 D29 第 2 条的路径排序）

**结论：自写 Triton NF4 GEMV 打不过 bnb。路径 ① 结案，不再投入。**

| 证据 | 数字 |
|------|------|
| 7 个投影形状 | 自写 kernel 比 bnb **慢 1.31x ~ 2.75x**，**无一胜出** |
| 距 DRAM 下界 | bnb **1.19x**（9.79 ms）vs 自写 **1.99x**（16.28 ms）；下界 8.20 ms（2.04 GB @ 249 GB/s） |
| 36 组 launch 参数 | `block_n × num_warps × num_stages` **没有一个**快过 bnb |
| 更激进结构 | `tl.dot`/MMA 848us vs bnb 234us；`tl.gather` **编译失败**；全宽 BLOCK_N 568us；fp16 码本 394us |
| 端到端（36 层图解码） | `--quant nf4` **28.8 ms/token** vs `--quant bnb` **16.0 ms/token** |

**顺手捡到的真收益：`lm_head` 换 4-bit（`loader.enable_lm_head_4bit()`，现在是默认配置）。**

| 配置 | 图解码 ms/token | tok/s | 峰值显存 |
|------|:---:|:---:|:---:|
| 单通路 · fp16 lm_head | 16.0 | 62.35 | 3.09 GiB |
| **单通路 · 4-bit lm_head** | **14.0** | **71.34** | 3.28 GiB |
| 双通路 · fp16 lm_head | 24.8 | 40.38 | 4.65 GiB |
| **双通路 · 4-bit lm_head** | **22.7** | **43.99** | 4.83 GiB |

代价 **+0.187 GiB**（**fp16 的 `embed_tokens.weight` 必须保留** —— 它同时是输入 embedding 的表）；贪心解码 **24/24、32/32 个 token 与 fp16 一致**。

**新会话必读的两条：**
1. **微基准证明不了端到端收益。** eager 下 bnb 16.32 vs nf4 16.56 tok/s（几乎一样，甚至 nf4 略快），只有**图解码**（纯 GPU 时间）才暴露 **1.79x** 的真实差距。
2. **「已打满带宽」不等于「没得优化」。** lm_head 的 3.11 ms 确实打满了 250 GB/s（访存效率到顶），但**降位宽把要搬的数据砍到 1/4** —— 访存效率与数据量是**两件正交的事**。S3 报告曾据此下"没得优化"的结论，已更正。

**剩余空间很小：** 单通路 36 层 bnb **9.79 ms** vs DRAM 下界 **8.20 ms**，只剩 **1.19x**。要再大幅提速只能走 **③ 融合 RoPE / 去冗余拷贝 → ④ 融合注意力**。

**验收：** `pytest tests -q` → **28 passed**（含 `tests/test_nf4_linear.py` 11 条、`tests/test_lm_head4.py` 5 条）。

---

## 六·补二 · 长上下文注意力（✅ 2026-09-22 —— **一个开关换来 4 倍长度**）

**报告：[reports/long-context-attention.md](reports/long-context-attention.md) · 决策 D33**

起因是用户提的"让 4060 8GB 跑满 260K 上下文，再测信息过载时能不能注意到重点"。开工前先量天花板，结果**瓶颈不在原先设想的地方**。

**已核查（诊断脚本 `probe_long_context_vram.py` / `probe_attention_kernel.py` / `probe_kernel_mapping.py` / `probe_attention_tradeoff.py` / `exp_needle.py`）：**

1. **瓶颈是注意力后端，不是 KV cache。** 本机 torch 2.6.0+cu124 **没编译 flash attention**，而 mem-efficient / cuDNN 两个融合内核**都要求 Q/K/V 头数相同**；GQA（32 Q / 8 KV）+ `enable_gqa=True` → SDPA **退回 math 后端 → 实体化 O(n²) 的 fp32 分数矩阵**。3665 token 的 prefill 峰值 **8.80 GiB**（超过物理 8188 MiB，换页，29.1 s），7291 token 直接 OOM —— 而按 KV cache 算只该占 1.41 GiB。
2. **改一行就解决**：`gqa_in_sdpa = False`（进 SDPA 前先 `repeat_kv` 展平成 32 头）。**同轮内**对比：1749 token **2.1x**、3012 token **2.5x**、4096 token **4.1x**（6.76 → 3.26 GiB）。长度 ×4（1749 → 7146）峰值只从 3.15 涨到 3.53 GiB。
3. **数值正确性已过**：两边钉 math 后端时 `repeat_kv` 与 `enable_gqa` **逐位一致（0.000e+00）**；vs 手写 fp32 参考误差相同（4.07e-04）；**贪心 32/32 token 一致**。
4. **新天花板 ~14–15K token**（单通路 fp16 KV）。14363 健康（6.47 GiB / 9.36 s）；21615 峰值 8.50 GiB、prefill **375.9 s**（换页断崖）。
5. **信息过载下的选择性没有退化**：4 条**同形干扰事实**（格式相同，只有地点与号码不同）埋在不同深度、各问一次 → **1894 / 3665 / 7291 / 12728 token 全部 4/4 答对，零挑错**，峰值 6.67 GiB，prefill 6.8 s（clocks.sm 2340–2460 MHz）。

**`pytest tests -q` → 41 passed**（新增 `test_fused_attention_agrees_on_tokens`），总耗时 69 s → 53 s。

**新会话必读的三条：**

1. **`test_gating_off_matches_baseline` 两边钉在 `sdpa_kernel(MATH)`。** 它测的是**架构与权重保真度**，不该受"恰好 dispatch 到哪个内核"影响。**D28 的"逐位一致"今后必须注明是在 math 后端下** —— 展平后与 HF 不再逐位相同（原始 logits 有 fp16 累加级差异，`max|diff| ≈ 1.0` / 量级 112）。
2. **不要用跨时间点的数字算加速比。** needle 那次 3665 token 从 29.1 s 变 1.4 s，主因是**不再溢出到系统内存**，不是内核快了 20 倍。纯内核加速只看同轮的 2.1x / 2.5x / 4.1x。
3. **`GraphDecoder.capture()` 每次都会新建一张 CUDA Graph 并分配新内存池。** 每个问题都重捕（起点位置不同）会让显存爆掉 —— 实测重捕 9 次后 7905/8188 MiB 崩溃。只生成几十个 token 时直接用 `dec._body()`（eager）。

**离 262144 还有多远（算术推算）：** 262144 × 144 KiB = **36.0 GiB**，而 D17 预算内只剩约 **3.8 GiB** → 需 **~9.5x** KV 压缩。杠杆：**int4 KV（4x，不需训练）** → 跨层 KV 共享（2–4x，要训练）→ MLA（~4x+，要训练）；组合 ~16x 可摸到 224K。另加**算力墙**：注意力 O(n²)，12728 token 实测 6.8 s → 262144 token 约 **47 分钟/次全量 prefill**（增量轮次不受影响，仍 ~30 ms/token）。

**下一步（排序）：** ① ~~KV int4 量化~~ → ✅ **已完成，见六·补三** ② ~~用 needle 逐档量 int4 的精度衰减~~ → ✅ **已完成（12.7K 内零退化）** → ③ 跨层 KV 共享（里程碑 2）→ ④ 干扰项数量扫描（4 → 16 → 64 条同形事实）找退化拐点（**已提到最前，见六·补三**）。

---

## 六·补三 · KV int4 量化（✅ 2026-09-22 —— **精度没掉，但收益也一分没拿到**）

**报告：[reports/kv-int4.md](reports/kv-int4.md) · 决策 D34 · 代码 `src/nova/kvquant.py` · 测试 `tests/test_kvquant.py`（13 passed，全套 54 passed）**

**已核查：**

1. **记账 3.46x**：fp16 **144 KiB/token** → int4 **41.6 KiB/token**（K 按 `head_dim` 每 32 通道一组、V 按整条 128 一组；`min`/`step` 各 fp16，元数据只占 13.5%）。
2. **`--kv int4` 一开始 0/4 是接线 bug，不是精度损失。** 根因：`QuantRoundTripCache.update` 把已写入范围写成 `pos + 1`，而多 token 时 `StaticKVCache.update` 内部转调 `append_prefill` 且 `pos` 不推进 → prefill 时只有第 0 个位置被拷进工作区。**决定性线索是"本该恒等却不等"的对照组**（`residual=128` 在 15 token 上 `keep == 0`）。两条回归测试已入库。
3. **端到端 needle 逐档零退化**：**1894 / 3665 / 7291 / 12728 token 全部 4/4、零挑错**，与 fp16 基线逐档相同（含贴着 D33 天花板的 12728）。
4. **但误差真实存在**：K 相对 L2 误差最坏 **11.8%**；**第 0 层 K 通道离群 65x**（中间层 4.3–7.2x），该层噪声达通道激活的 **55%**。**"int4 无损"是错的，只是 4 道题的尺子测不出来。**
5. **开销**：朴素"先还原再算"同轮实测 **prefill 1.0–1.5x / decode 2.0–5.0x**；在 `max_len=18432`（贴近 8GB 上限）的 needle 环境里 prefill 涨到 **8.4x → 25.8x**（现象已核查，**归因标 `推测`**）。
6. **模拟版不省显存**：SDPA 必须吃 fp16 → 必须有整段 fp16 工作区。收益只按公式记账，**一分未到手**。

**新会话必读的三条：**

1. **别把"记账收益"当成"已实现收益"。** 目前 `src/nova/kvquant.py` 是一个**测量仪**，不是省显存的实现。真省只有融合核 / 分块 dequant + 在线 softmax 两条路，都没做。
2. **先别写融合核。** 顺序是：**①细尺子**（4→16→64 条干扰事实）→ **②试 8 位**（1.9x 但误差减半、torch 有现成路径，**性价比可能高于 int4**）→ **③K 分组方向对照**（改成按 **token 维**分组，每通道一条跨 token 的尺子；第 0 层 65x 离群说明现在这套在首层很吃亏）。否则会把一套可能要推倒重来的方案固化进 Triton。融合核的进场条件：写之前定死**与"先还原再算"的注意力输出逐位或 ULP 级一致**。
3. **记忆一律 fp16 存**（D34 第 4 条）：int4 只是**运行时 cache 的格式**，注入记忆时按当时格式量化 → **D09 的冻结判据不变**。要改这条必须新开决策 + 重测取回率。

**260K 的账（算术推算）：** 262144 × 41.6 KiB = **10.4 GiB**，D17 预算内只剩 ~3.8 GiB → **int4 单独还差 ~2.7x**，跨层 KV 共享 / MLA（都要训练）仍然必需。

---

## 六·补四 · P0：拆掉 O(n²) 掩码表 + 容量分桶（✅ 2026-09-22 —— **长上下文实验的前提，解码少搬 48–60%**）

**报告：[reports/decode-mask-bucket.md](reports/decode-mask-bucket.md) · 决策 D35 · 代码 `src/nova/decode.py` · 测试 `tests/test_decode_mask_bucket.py`（6 条，全套 60 passed）**

**已核查：**

1. **掩码整表拆掉了**：旧 `_build_mask_table` 常驻 `max_len²×2`（18432→0.63 / 32768→2.00 / 65536→**8.00 GiB**，是"上下文上限"的第一堵墙）；现在**图内即时构造一行**，常驻只剩 `8×max_len` 的 arange + 图内 `2×max_len`（65536 → **512 KiB + 128 KiB**）。**与旧表逐位一致**（测试钉住）。`_build_mask_table` 只留着做对照与核算旧体积。
2. **容量按桶分配**（`BUCKETS = 2070/4096/8192/16384/32768/65536`、`bucket_for()`、`GraphDecoder.for_length()`、`grow()`）：decode 读量随**桶**走，不随"随手给的 `max_len`"走。`used=2048` **139.1 → 56.1 ms/token（省 59.7%）**；`used=7291` **138.7 → 71.2（省 48.6%）**（单通路 triton，同轮交替两遍取小，`clocks.sm 2460`）。**耗时的判据仍是"只随 `max_len` 走"**。
3. **选桶与搬家都不改结果**：窄桶 vs 宽桶、`grow()` 前后，贪心 token **逐个相同**；跨桶重捕的显存增量**正好等于**多出来的 KV 容量（2026×144 KiB = 291.7 MiB / 4096×144 KiB = 576 MiB）⇒ 不漏。
4. **"重捕就爆显存"是误判**：三种图内存池策略的逐轮 allocated 基本平。**共享池反而会踩 PyTorch 裸 assert**（`CUDACachingAllocator.cpp:2225`，只在 pytest 全量跑中触发、脚本 5 种序列复现不出）⇒ `capture()` 默认 `pool="off"`（不共享）。

**没解决的（就是第五轮的门）：**

- **64K 仍装不下**：fp16 KV 在 65536 是 **9.00 GiB** ⇒ 必须靠 **E2 滑窗层**。
- decode **固定地板 ~59 ms/token**（`max_len=2070` 时 KV 只占 ~1 ms）、prefill 的 **O(n²) 算力墙**（12.7K=6.8 s → 32K≈45 s → 64K≈180 s）都没动。
- 逐轮 **+8.1 MiB** 的缓慢增长来源未定位（**待实测**）；共享池 assert 的触发条件**推测**（依赖分配器状态），处置是默认绕开。

> 第六·补三留下的三条**仍然有效**：①别把"记账收益"当"已实现收益"；②**先别写融合核**（顺序：细尺子 → 8 位 → K 分组方向 → 再谈融合核）；③记忆一律 fp16 存。

---

## 六·补五 · E1 记忆路由（✅ 2026-09-22 —— **判据未达，不 adopt**）

**报告：[reports/memory-routing.md](reports/memory-routing.md) · 决策 D36 · 探针 `src/diagnostics/probe_memory_routing.py`**

**已核查（三档段长，干草堆 7291 token / 4 条形近事实 / `clocks.sm` 2460）：**

1. **段级静态代表 K 路由达不到判据**：`recall@4` 在 seg=1024/256/64 上分别是 **0.75 / 0.25 / 0.75**（要求 ≥ 0.95）⇒ **不 adopt**。精确路径（现有 `MemoryStore.scores()`）三档**都是 1.00**。
2. **`top-‖K‖` 明显差于 `mean-K`**：侧会话"用范数最大的、不要用平均"**被证伪**（最差一档名次 **93/112**）。段级平均 `‖K‖` 的最大/中位只有 **1.00x** ⇒ 连"挑出 attention sink 段"这件事它都做不到。
3. **旋转不是变量**：pre-RoPE / 注入帧旋转 / 旋转后平均，每档差距 ≤ 1 名。
4. 代价：精确 **3192–3278 ms/题** vs 路由器 **2.03 ms/题**（28 段）⇒ 路由器快约 1500x，但**不够准**；精确够准，但 **100K 记忆要扫 7.2 GB（显存放不下）** —— 这才是真问题。

**下一步的替代方向（都还没测）**：查询相关的粗筛 + 小集合内精确重排；或把 K 压小但**别丢 token**（**E3** 覆盖量化那条）。

---

## 六·补六 · E2 滑动窗口（✅ 2026-09-23 —— **不 adopt：质量崩、代价还是反的**）

**报告：[reports/swa-window.md](reports/swa-window.md) · 决策 D37 · 代码 `src/nova/cache.py`（`WindowedKVCache`）+ `decode.py`/`layers.py`/`model.py` · 测试 `tests/test_swa.py`（6 条）**

**已核查：**

1. **质量门控未过**：每 4 层 1 全局 + W ∈ {1024,2048,4096}，在 **7.3K 与 14.5K** 上下文上 **全部 0/4**（全量基线 4/4）；32K 档 0/4。
2. **不是单纯的"截断"**：**W=8192（盖住整段 7.3K）时 4/4** ⇒ ring/掩码/分块实现都对；而 W=4096 时**落在窗口内的两条事实也没答出** ⇒ 检索需要足够多的层同时看到关键 token（9/36 不够）。机制归因标 **推测**。
3. **提高全局层比例会把"答不出"变成"答错"**：每 2 层 1 全局 + W=4096 → 1/4 正确、**3/4 挑错**（答成另一条形近事实）。信息过载下这比"想不起来"更糟。
4. **滑窗下的 S4 记忆取回 7/8 → 5/8**（3710 token 上下文，注入的记忆掉出局部层窗口）。
5. **代价方向是反的**：KV 省 40–67%，但 **prefill 更慢**（加性窗口掩码把因果稀疏性变成稠密读）：7.3K 档 5.6 s vs 全量 4.1 s；29K 档 **180.3 s**（同长度因果外推约 35 s）。峰值也没降（7.63 GiB @32K）。
6. 顺带修一个真 bug：`prefill` 把 ring 的 `first` 覆盖成 `offset` ⇒ S4 记忆注入路径上局部层窗口被清空（**所有问题都答不出**）；`test_second_prefill_keeps_window` 已钉住。

**对 260K 的结论**：**"少看 token"这条路（滑窗 / 丢 token）在不训练的前提下被证伪**，重心回到
**"不看少、但看便宜"** ⇒ **E3（8 位 KV，不丢 token）→ E4（融合核）**，以及里程碑 2 的训练侧压缩。
另有一条实现约束：**带状注意力不能靠加性掩码实现**，真要做窗口得让内核跳过窗口外的块。

---

## 七、S4 · 记忆最小实现（✅ 已完成 2026-09-22 —— **8/8 取回，跨进程可复现**）
## 六·补七 · E3 8 位 KV（✅ 2026-09-23 —— **通过，改选 int8**）

**报告：[reports/kv-quant-8bit.md](reports/kv-quant-8bit.md) · 决策 D38 · 探针 `src/diagnostics/probe_kvquant_bits.py` · 测试 `tests/test_kvquant.py`（19 条）**

**已核查（真实 K/V 3665 token + needle 8K 档，同轮）：**

1. **`int8`（K 按 token 维分组，`int8t64`）与 fp16 逐题一致**：4 干扰 4/4、16 干扰 13/16、64 干扰 54/64，含挑错的那几条都相同。
2. **误差只有 int4 的 1/50**：K relL2 0.0018（int4 0.0910）；**通道最坏 0.0037**（int4 0.7705）。
3. **记账 1.86–1.91x**（int4 3.46x）；prefill 与 fp16 基本同速，**远好于"int4 先反量化再喂 SDPA"**。
4. **D34 第 3 条被验证**：K 改按 token 维分组后通道最坏误差 **18x** 变好。
5. `fp8` 不进默认路径（精度不占优、记账更差、per-tensor 更糟）；要写融合核就写 **int8** 的。
6. 顺手修一个真 OOM：`QuantRoundTripCache` 必须用 `base=dec.cache` 复用已有 cache（否则两份 cache 同时在显存里，18432 槽位下 8.3 GiB 直接爆）。

---

## 六·补八 · E5 干扰项扫描（✅ 2026-09-23 —— **尺子本身就是结论**）

**报告：[reports/interference-scan.md](reports/interference-scan.md) · 决策 D39 · 尺子 `exp_needle.make_needles(n)`**

| 干扰项 | fp16 | `int8` | `int8t64` | `fp8` |
|---:|---|---|---|---|
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| 16 | **13/16** | 13/16 | 13/16 | 12/16 |
| 64 | **54/64** | — | 54/64 | — |

- **4 条干扰测不出任何东西**（当年 int4 的"零退化"就是这个原因）；16 条时**基线自己就掉 3 题**。
- **失败形态全是"挑错"**（答成另一条同形事实），"没答出"为 0 ⇒ 要防的是**混淆**，不是遗忘（与 D32 一致）。
- 判据升级：**以后"是否损伤检索"一律在 ≥16 条干扰下判**。

---

**报告：[reports/s4-memory-min.md](reports/s4-memory-min.md) · 决策 D31 · 演示 `src/s4_memory_demo.py` · 基准 `src/bench_memory.py`**

```powershell
$env:HF_HOME=(Resolve-Path .).Path+'\.hf-cache'; $env:TMP=(Resolve-Path .).Path+'\.tmp'; $env:TEMP=$env:TMP
$env:HF_HUB_OFFLINE='1'; $env:TRITON_CACHE_DIR=$env:TMP+'\triton-cache'; $env:TORCHINDUCTOR_CACHE_DIR=$env:TMP+'\inductor-cache'

& .\.venv\Scripts\python.exe -m pytest tests -q          # 40 passed（新增 tests/test_memory.py 12 条）
& .\.venv\Scripts\python.exe src\s4_memory_demo.py --phase write --mem .tmp\s4-memory\demo.safetensors
& .\.venv\Scripts\python.exe src\s4_memory_demo.py --phase ask   --mem .tmp\s4-memory\demo.safetensors
& .\.venv\Scripts\python.exe src\bench_memory.py --reps 7 --warmup 2
```

**验收标准达成（已核查）：** 第 1 轮写入 6 条事实 → 第 1 轮从可见历史里去掉（压缩进记忆）→ 20 轮无关对话（608 token）→ 第 22 轮 8 个问题 **8/8 全对**；不注入 **0/8**（反问 / 编造）、故意注入另一条 **0/8**（原答案不再出现）；**write / ask 是两个独立进程**，只通过 `.safetensors` 传递 → 8/8。峰值显存 **5.17 GiB**（D17 内）。

> ⚠️ **不要误读这条（D32 已核查）：** `不注入 = 0/8` 是"第 1 轮被**移出上下文**"的因果对照，**不是**"原版模型会忘"。
> 实测把第 1 轮**留在**上下文里、不注入任何记忆：**20 轮 683 token → 8/8，60 轮 1915 token → 8/8，120 轮 3785 token → 8/8**（该模型上下文上限 **262144**）。
> **所以"20 轮就忘了"这个说法作废** —— 20 轮这个尺度上记忆**没有收益**。记忆真正不可替代的地方是：**跨会话持久**（上下文在关掉程序时清零，双进程演示才是它对应的场景）、**突破显存**（7GB 预算下模型占 4.83 GiB，按 240 KiB/token 双通路，KV cache 约 **9000 token** 到顶）、以及作为里程碑 2/3 巩固 / 遗忘机制的地基。
> 复现：`& .\.venv\Scripts\python.exe src\diagnostics\exp_recall_baseline.py --turns 20 60 120`

**机制：L0「精确 KV」层，不是 docs/03 的 `[128, 2560]` 记忆令牌。** 后者是**要训练**的写入器 / 读取器，属里程碑 2（D31 第 1 条）。S4 的写入 = 抓区间的 K（RoPE 前）/ V；寻址 = 模型**自己的** `Q·K`；读取 = 按新位置重新旋转后写进 KV cache。**零新增参数、零训练**。

**新会话必读的四条（都是实测逼出来的）：**

1. **记忆插在"当前轮之前"（`place="turn"`），不是最前面。** 插最前面在 20 轮历史下寻址退化到 1/3，**且生成质量崩**（复读"今天多云转晴"）—— RoPE 长距离把 `Q·K` 抹平。**这条推翻了 docs/03 的原始写法。**
2. **寻址必须按注入后的真实位置旋转 Q/K 再算 `Q·K`，并用 softmax 质量。** pre-RoPE 裸余弦对 6 条候选**全部 > 0.8**，几乎不区分内容。
3. **查询取"问题那句话的末尾 4 个 token"。** 取 prompt 最后 4 个拿到的是 `<|im_start|>assistant\n`（**没有内容**）→ 退化；整句取平均 8/8 掉 7/8。
4. **检索逻辑只写一份**（`MemorySession.rank()`），`prefill` 与测试都走它。第一版测试自己手搓打分、忘了截末尾 4 token，命中率立刻从 8/8 掉 7/8。

**开销（速度数字带 `clocks.sm`）：** 检索 ≈ **1x 一次基线前向**（与时钟无关的比值 0.94–0.98x，因为取 Q 要跑一次完整无 cache 前向）；打分 40–90 ms；**注入 ≈ 0**（三次独立测量 −41 / −40 / −12 ms）。记忆体积 **240 KiB/token**。
⚠️ **连测 7 轮会把笔记本 GPU 从 2160 MHz 压到 ~870 MHz（94 W → 35 W），同一条件耗时翻倍** —— 基准已内置预热与逐轮 `clocks.sm` 采样。

**已知边界：** 无损不压缩；键未训练（区分度是实测，不是设计保证）；**LoRA / 微调会让 K/V 空间漂移，匹配质量衰减待实测**（里程碑 2 必测）；通路 0/1 各存一份（2× 冗余）；检索不能跨轮复用。

**本阶段不做：** 遗忘 / 巩固 / 多时间尺度（里程碑 3）。

---

## 八、待决策（不阻塞开工）

| 事项 | 何时决定 | 说明 |
|------|:---:|------|
| ~~**速度路径 ①/②**~~ | ✅ **已定（D30）** | ①=自写 NF4 GEMV **已证伪并结案**（比 bnb 慢 1.31~2.75x）；②=量化 lm_head **已落地**（−2.0 ms/token，见第六·补节）。**余下：③ 融合 RoPE / 去冗余拷贝 → ④ 融合注意力** |
| D19 是否换基座 | **S2 基线出来之后** | 只有实测显示英文 RP 明显弱，才值得付出"中文保底"的代价 |
| 双通路代码怎么起步：改 transformers 的 `modeling_qwen3_vl.py`，还是自己写一份 `nn.Module` | **S3 开始前** | 改动量大、要跟上游版本，但省掉权重映射；自写更干净但要自己写映射（见 [docs/16-model-anatomy.md](docs/16-model-anatomy.md) 第二节） |
| 部署格式（PyTorch / 自定义引擎 / GGUF） | 里程碑 2 之后 | 自定义架构大概率不能用 GGUF |
| 训练平台（4060 / Kaggle / 云） | S3 通过之后 | 先本地小规模，云端按需 |

---

## 九、未验证事实清单（写代码前先确认）

1. ✅ ~~`transformers` 是否已支持 `qwen3_vl` 架构~~ —— **已核查：支持**（transformers 5.17.0，`qwen3_vl` 在 `CONFIG_MAPPING` 与 `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` 中，无需 `trust_remote_code`）。见 [reports/s0-environment.md](reports/s0-environment.md) 第三节
2. `bitsandbytes` 在 Windows + py3.12 是否可用
3. Qwen3-VL-4B 在 4060 上的真实速度（S2 才知道）
4. Hugging Face 是否可直连（不通走镜像）
5. ✅ ~~中英 token 效率比值（S1）~~ —— **已实测 1.02x**（[reports/tokenizer-report.md](reports/tokenizer-report.md)）
6. D 盘 32GB 是否够用（不够则 venv 也挪到 H）
7. Qwen3-VL-4B 的视觉塔参数量与显存占用（影响双通路预算）
8. transformers 里 `qwen3_vl` 的实现是否适合直接改造成双通路（决定 S3 的起步方式）

---

## 十、文件地图

| 路径 | 内容 |
|------|------|
| `AGENTS.md` | 规则、硬约束、环境事实、写文件方法 |
| `HANDOFF.md` | 本文件：执行顺序与交接 |
| `README.md` | 项目总览与文档索引 |
| `docs/01` ~ `docs/16` | 设计文档（15 愿景 / 02 架构 / 03 记忆 / 11 路线图 / 13 决策 / 15 语言 / 16 模型解剖） |
| `src/` | 代码（S0-S4 全部完成；`nova/` 是双通路骨架 + 图解码（**分桶**）+ **滑动窗口 `WindowedKVCache`** + L0 记忆 + KV 量化 `kvquant.py`，`chat.py` 交互 CLI，`diagnostics/` 是速度归因 + 记忆诊断 + KV 量化诊断 + 分桶/重捕诊断 + **路由/滑窗实验（`probe_memory_routing.py` / `exp_swa.py` / `probe_swa_memory.py`）**） |
| `tests/` | 单元测试（**66 passed**：`test_nova_skeleton.py` 9 条 + `test_graph_decode.py` 4 条 + `test_nf4_linear.py` 11 条 + `test_lm_head4.py` 5 条 + `test_memory.py` 12 条 + `test_kvquant.py` 13 条 + `test_decode_mask_bucket.py` 6 条 + **`test_swa.py` 6 条**） |
| `reports/` | 每步的产物与验收证据（`s0-environment` / `tokenizer-report` / `baseline-qwen3vl4b` / `s2-speed-diagnosis` / `s3-dual-path-skeleton` / `s3-graph-decode` / `speed-path1-nf4-gemv` / `s4-memory-min` / `long-context-attention` / `kv-int4` / `decode-mask-bucket` / `memory-routing` / `swa-window` / **`kv-quant-8bit`** / **`interference-scan`**） |
| `data/` | 评测集、训练数据（待建，**放 H 盘更大的话用软链接**） |
| `models/` | 本地权重（建议只放软链接，实体在 `H:\hf-cache`） |

---

> **进度（2026-09-21 本次会话）：** 第 1、2 条 ✅ **已完成** —— S0 通过、S1 报告落盘并已回写 [docs/15-language-plan.md](docs/15-language-plan.md) 第 2.5 节与 [docs/13-decisions.md](docs/13-decisions.md) D25。第 3 条（S2 基线）**未开始**，是下一步。

---

> **进度（2026-09-22 会话）：** S0 / S1 / S2 全部完成并落盘。**S2 的速度归因被推翻并重做**：原 D26 判定"bnb 4-bit kernel 是瓶颈（疑似回退 CPU）"，实测证明 **bnb 确实在跑 CUDA**，真凶是 **CPU 算子派发**（`CPU enqueue-only 76.92 ms/step` ≈ `wall 77.02 ms/step`，GPU 忙 ≈ 0.10 ms，约 8000 次 kernel 启动/token）。同时实测排除了 torchao、transformers 官方编译入口、`torch.compile`（default / reduce-overhead）四条路。完整证据见 [reports/s2-speed-diagnosis.md](reports/s2-speed-diagnosis.md) **第八节**与 **D27**。
>
> **意外收获（对 S3 直接有用）：** 自写的逐层精简前向与 HF 官方前向**数值完全一致**（`max|diff| = 0.0000`）→ **S3 的"自写 `nn.Module`"路线已验证可行**。
>
> **下一步：** ~~S3 · 双通路骨架~~ → ✅ **已完成**，见下方。

---

> **进度（2026-09-22 会话 · 第二轮）：S3 双通路骨架 ✅ 完成。**
>
> 按用户选定的**路径 ①（自写精简前向）**推进 —— 自写前向同时是双通路骨架的载体和速度路径的地基，一份工作两处收益。
>
> | 验收项 | 结果 |
> |------|------|
> | `pytest tests -q` | ✅ **8 passed** |
> | **门控关闭 ≈ 基线** | ✅ **逐位一致**（`max|diff| = 0.000e+00`，prefill + 8 步 decode 全覆盖） |
> | 交叉注意力隔离 | ✅ 关闭时扰动通路 1 → 通路 0 逐位不变；打开时必须改变 |
> | 生成 20 token | ✅ 不崩 |
> | 显存（D17 < 7GB） | ✅ **4.57 GiB** |
> | 路径 ① 首个收益 | ✅ Triton 融合 RMSNorm：同 36 层 **13.34 → 16.05 tok/s（+20.3%）**；算子数 **8275 → 6529 / token（-21.1%）** |
>
> 代码 `src/nova/`，测试 `tests/test_nova_skeleton.py`，基准 `src/bench_nova.py`，报告 [reports/s3-dual-path-skeleton.md](reports/s3-dual-path-skeleton.md)，决策 **D28**。
>
> **下一步：S4 · 记忆最小实现**（速度路径 ① 并行，不互相阻塞）。速度侧下一步按收益排序：**自写 Triton NF4 dequant+GEMV 替掉 bnb**（`bitsandbytes::gemm_4bit` 单项占 Nova 单通路耗时的 **47%**）→ 融合 RoPE → 去掉冗余 `_to_copy`/`view`/`as_strided` → 融合注意力。

> **进度（2026-09-22 会话 · 第三轮）：速度路径 ①（二）✅ CUDA Graph 解码打通。**
>
> **先做的测量（已核查）：** `src/diagnostics/microbench_op_cpu.py` 实测本机**单次 kernel 启动约 13.5us CPU** ——
> 连 `x * 2.0` 都要 15.96us，且**严格线性**（64 个 mul = 887us）。Nova 每 token ~6500 次算子 × ~9.6us ≈ 62ms。
> **不是"算得慢"，是"发不动"。**
>
> 于是把**整步解码**（embed → 36/60 层 → lm_head → argmax → 写回输入 → pos+1）全部捕获成一张 CUDA Graph：
>
> | 配置 | 层数 | tok/s | ms/token | 峰值显存 |
> |------|:---:|:---:|:---:|:---:|
> | HF 基线 | 36 | 13.24 | 75.5 | 2.73 GiB |
> | Nova 单通路 eager | 36 | 15.87 | 63.0 | 3.04 GiB |
> | **Nova 单通路 + CUDA Graph** | 36 | **61.84** | **16.2** | 3.09 GiB |
> | Nova 双通路 eager | 60 | 9.10 | 109.9 | 4.57 GiB |
> | **Nova 双通路 + CUDA Graph** | 60 | **40.26** | **24.8** | 4.64 GiB |
>
> - `pytest tests -q` → ✅ **12 passed**；**单通路 24/24、双通路 24/24 个 token 与 eager 完全相同**
> - **路线图的 60+ tok/s 目标，单通路已达成（61.84）**；双通路 40.26 的缺口在 bnb 的 4-bit 路径
> - 代码 `src/nova/cache.py` + `src/nova/decode.py`，基准 `src/bench_graph.py`，报告 [reports/s3-graph-decode.md](reports/s3-graph-decode.md)，决策 **D29**
>
> **新会话必读的两个坑：**
> 1. **replay 必须在 `torch.inference_mode()` 内** —— 图内对输入缓冲做了原地写入，否则报 `Inplace update to inference tensor`。
> 2. **`prefill()` 后输入缓冲要填"第一个生成 token"，不是最后一个 prompt token** —— 填错会让图的第一步把最后一个 prompt token **再算一遍**。单通路"看着 16/16 一致"是巧合，双通路才暴露成 17/24。已固化成测试。
>
> **剩余开销分解（局部图测量，单通路 17.0ms）：** 36 层 **12.90ms（75.9%）** / lm_head **3.12ms（18.3%，已打满带宽）** / 其它 1.0ms。
> 36 层的 4-bit 权重下界是 5.3ms，实测 12.90ms → **2.4x 缺口**，原因是 **bnb 在 M=1 时没走 packed 4-bit GEMV**（dequant 到 fp16 workspace + cublas）。
>
> **下一步：** ① **自写 Triton NF4 dequant+GEMV 替掉 bnb**（最大收益）→ ② 量化 lm_head → ③ 融合 RoPE / 去冗余拷贝 → ④ 融合注意力。
> S4（记忆最小实现）与此并行，不互相阻塞。

> **进度（2026-09-22 会话 · 第四轮）：速度路径 ① ✅ 结案 —— 方向证伪，但顺手捡到 lm_head 的 2ms。**
>
> **① 自写 Triton NF4 GEMV 打不过 bnb，不再投入**（完整证据 [reports/speed-path1-nf4-gemv.md](reports/speed-path1-nf4-gemv.md)，决策 **D30**）：
>
> | 证据 | 数字 |
> |------|------|
> | 7 个投影形状 | 自写 kernel 比 bnb **慢 1.31x ~ 2.75x**，**无一胜出** |
> | 距 DRAM 下界（8.20 ms） | bnb **1.19x**（9.79 ms）vs 自写 **1.99x**（16.28 ms） |
> | 36 组 launch 参数 | `block_n × num_warps × num_stages` **没有一个**快过 bnb |
> | 更激进结构 | `tl.dot`/MMA 848us vs bnb 234us；`tl.gather` **编译失败**；全宽 BLOCK_N 568us；fp16 码本 394us |
> | 端到端（36 层图解码） | `--quant nf4` **28.8 ms/token** vs `--quant bnb` **16.0 ms/token** |
>
> **② 量化 lm_head —— 落地，白赚 2ms：** `loader.enable_lm_head_4bit()`（**现在是默认配置**）
>
> | 配置 | 图解码 ms/token | tok/s | 峰值显存 |
> |------|:---:|:---:|:---:|
> | 单通路 · fp16 lm_head | 16.0 | 62.35 | 3.09 GiB |
> | **单通路 · 4-bit lm_head** | **14.0** | **71.34** | 3.28 GiB |
> | 双通路 · fp16 lm_head | 24.8 | 40.38 | 4.65 GiB |
> | **双通路 · 4-bit lm_head** | **22.7** | **43.99** | 4.83 GiB |
>
> 代价 **+0.187 GiB**（fp16 `embed_tokens.weight` **必须保留** —— 它同时是输入 embedding 的表）；贪心解码 **24/24、32/32 个 token 与 fp16 一致**；`pytest tests -q` → **28 passed**。
>
> **新会话必读的两条：**
> 1. **微基准证明不了端到端收益** —— eager 下 bnb 16.32 vs nf4 16.56 tok/s 几乎一样（甚至 nf4 略快），**图解码**（纯 GPU 时间）才暴露 **1.79x** 的真实差距。
> 2. **「已打满带宽」不等于「没得优化」** —— lm_head 的 3.11ms 确实打满了 250 GB/s（**访存效率**到顶），但**降位宽把要搬的数据砍到 1/4**。上一轮"S3 报告说 lm_head 已打满带宽、没得优化"的结论**已更正**。
>
> **剩余空间：** 单通路 36 层 bnb **9.79 ms** vs DRAM 下界 **8.20 ms**，**只剩 1.19x**。
>
> **下一步：** **S4 · 记忆最小实现**已在 2026-09-22 完成（见第七节）；速度侧 **③ 融合 RoPE / 去冗余拷贝 → ④ 融合注意力**。

---

## 十一、第一天的完成标准

1. ✅ S0 跑通：`torch.cuda.is_available()` 为 True
2. ✅ S1 报告落盘，并把 [docs/15-language-plan.md](docs/15-language-plan.md) 第 2.5 节的"待实测"替换为实测数字
3. ✅ 时间够就做 S2：基线数字落盘

**这三条完成 = 本次会话合格。** 不要越过 S3 去碰训练数据。
