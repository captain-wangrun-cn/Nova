# S0 · 环境就绪报告

> 日期：2026-09-21 · 阶段：**S0（环境）** · 依据：`AGENTS.md` 第四节 + `HANDOFF.md` 第三节
> 结论标注：**已核查** / 待实测 / 推测

---

## 一、验收结论（已核查）

| 验收项 | 标准（HANDOFF 第二节） | 实测 | 结论 |
|------|------|------|:---:|
| torch CUDA 可用 | `torch.cuda.is_available()==True` | `True` | ✅ |
| GPU 识别 | RTX 4060 Laptop | `NVIDIA GeForce RTX 4060 Laptop GPU` / 8.0 GiB / sm_89 | ✅ |
| 缓存不落 C 盘 | HF_HOME 指向非 C 盘 | `D:\360MoveData\Users\18889\Documents\Nova\.hf-cache` | ✅ |
| transformers 支持 qwen3_vl | 未验证事实 #1 | 支持（见第三节） | ✅ |

**S0 通过。**

---

## 二、实测环境（已核查 · 2026-09-21）

| 项 | 值 |
|------|------|
| 虚拟环境 | `D:\360MoveData\Users\18889\Documents\Nova\.venv`（Python 3.12.14） |
| 基础解释器 | `C:\Users\18889\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` |
| pip | 26.2.1 |
| torch | **2.6.0+cu124** |
| transformers | **5.17.0** |
| accelerate | 1.15.0 |
| safetensors | 0.8.0 |
| tokenizers | 0.23.2 |
| huggingface_hub | 1.32.0 |
| numpy | 2.5.3 |
| pytest | 9.1.1 |
| pillow | 12.3.0 |
| GPU | RTX 4060 Laptop，8188 MiB，capability (8, 9)，驱动 610.88 |

**磁盘（实测，与文档记载有出入）：**

| 盘 | 剩余 | 备注 |
|:--:|------|------|
| C: | **6.48 GB** | 文档记 0.4GB，已变；仍按禁区对待，不写入 |
| D: | 32.83 GB | 代码 + venv + 本项目缓存 |
| H: | 131.87 GB | 本次未使用 |

---

## 三、未验证事实 #1 结论（已核查）

**问题：** `transformers` 是否已支持 `qwen3_vl` 架构。

**实测（transformers 5.17.0，离线枚举 718 个 config type）：**

```
qwen3_vl      : config=True  base=True  ImageTextToText=True  CausalLM=False
qwen3_vl_text : config=True   (文本子模型独立注册)
qwen3_vl_vision: config=True  (视觉塔独立注册)
```

**结论：**
1. `qwen3_vl` 已在 `CONFIG_MAPPING` 与 `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` 中 → `AutoConfig` / `AutoModelForImageTextToText` 可直接加载，**不需要 `trust_remote_code`**。
2. 它**不在** `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`（合理：VL 模型走 ImageTextToText 接口）。
3. 文本子模块与视觉塔都被**拆成独立 config type**（`qwen3_vl_text` / `qwen3_vl_vision`）——这一点对 S3 有利：双通路只需复制文本塔，视觉塔天然是独立可共享组件（呼应 **D14**）。

**对 S3 起步方式的意义（待 S3 前定）：** 架构被拆成 text / vision 两个子模型，自写 `nn.Module` 时的权重映射面比预期小；但具体接口仍需读 `modeling_qwen3_vl.py` 源码确认，见 `docs/16-model-anatomy.md` 第二节。

---

## 四、本次会话的两处环境偏离（需回写文档）

| # | 文档原记载 | 实际执行 | 原因 |
|:-:|------|------|------|
| 1 | 缓存目录 `H:\hf-cache` / `H:\pip-cache` / `H:\tmp` | 全部改为项目内 `.hf-cache/`、`.pip-cache/`、`.tmp/` | **用户指示**："可以直接在 nova 文件夹下"；同时避开跨盘授权的摩擦。C 盘仍未被写入 |
| 2 | torch 用 `cu124` 索引 | 一致，装成 `2.6.0+cu124` | — |

> 偏离 1 已写入 `.gitignore`。**D 盘余量 32.83GB**，需在装完模型权重（约 9GB）后复核；**D 盘低于 10GB 时**按 HANDOFF 第三节把 `.venv` 与缓存迁到 H 盘。

---

## 五、复现命令

```powershell
cd D:\360MoveData\Users\18889\Documents\Nova
$env:HF_HOME='D:\360MoveData\Users\18889\Documents\Nova\.hf-cache'
$env:PIP_CACHE_DIR='D:\360MoveData\Users\18889\Documents\Nova\.pip-cache'
$env:TMP='D:\360MoveData\Users\18889\Documents\Nova\.tmp'; $env:TEMP=$env:TMP

$base="C:\Users\18889\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $base -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -U pip
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& .\.venv\Scripts\python.exe -m pip install -U transformers accelerate safetensors pytest pillow "huggingface_hub[cli]"

& .\.venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 六、下一步（S1）

Tokenizer 探针 → `src/tokenizer_probe.py` → `reports/tokenizer-report.md`。
