# S2 · 速度归因诊断：4-bit 为什么只有 12 tok/s

> 日期：2026-09-21 · 阶段：**S2** · 结论标注：**已核查 / 待实测 / 推测**
> 证据：`reports/baseline-outputs.jsonl`（6 条 prompt × 2 精度）、`reports/bnb-variant-probe.json`（5 种加载配置）
> 关联决策：**D26**

---

## 一、结论先行（已核查）

**瓶颈是 bitsandbytes 的 4-bit kernel。不是模型、不是显存、不是 transformers、不是配置。**

| 事实 | 实测值 |
|------|:---:|
| 实测显存带宽（`copy_` 1 GiB） | **~234 GB/s** |
| 4-bit 权重总量（文本塔 4.02B × 0.5 B） | **2.01 GB** |
| 4-bit 带宽决定的理论上限 | **~116 tok/s** |
| **实际解码速度（4-bit）** | **10.6 tok/s** |
| **效率** | **≈ 9%** |

---

## 二、算子级证据（决定性）

用模型的真实形状（2560×2560）做单算子基准：

| 算子 | 权重体积 | 耗时 | 有效带宽 | 相对 fp16 |
|------|:---:|:---:|:---:|:---:|
| fp16 `nn.Linear` | 13.1 MB | **56.2 µs** | **234 GB/s** | 1.00x（打满带宽，正常） |
| bnb `Linear4bit` NF4 | 3.3 MB | **70.8 µs** | **47 GB/s** | **1.26x（更慢）** |

**怎么读：** 4-bit 权重只有 fp16 的 **1/4** 大小，本该快 4 倍；实测**反而慢 1.26 倍**。
唯一解释是 bnb 的 4-bit 矩阵乘走的是**低效回退路径**（每步反量化），它实际搬运的数据约是 4-bit 权重的 4 倍以上。

> 顺带得到一个有用的副产品：**这台机器的显存带宽实测 234 GB/s，fp16 算子能打满。** 后面所有速度预算都可以用这个数算。

---

## 三、bitsandbytes 自身状态（已核查）

```
python -m bitsandbytes                ->  "SUCCESS!"        # 官方自检通过
from bitsandbytes.cextension import lib
type(lib)                             ->  CudaBNBNativeLibrary
lib.get_compute_capabilities()
  -> RuntimeError: Method 'get_compute_capabilities' not available
                   in CPU-only version of bitsandbytes.
```

**即：bnb 加载到了 CUDA 版 DLL（类名是 `CudaBNBNativeLibrary`），但 CUDA 专属符号取不到**，实际执行落在 CPU-only 回退上。

### 排查过的修法（全部已实测，全部无效）

| # | 尝试 | 结果 |
|:-:|------|------|
| 1 | `BNB_CUDA_VERSION` = 118 / 121 / 124 / 126 / 128 / 130 | 全部回退 CPU-only（118、130 明确报"依赖缺失"） |
| 2 | `PATH` 前置 torch 的 CUDA 运行库目录 | 无效 |
| 3 | `os.add_dll_directory(torch/lib)`（句柄保活） | 无效 |
| 4 | 手动 `ctypes.CDLL("libbitsandbytes_cuda124.dll")` | **能加载成功** → 问题不在 DLL 本身，在 bnb 的选择逻辑 |
| 5 | `device_map="auto"` vs 不设 | 12.34 vs 12.17 tok/s（**无差别**） |
| 6 | 关掉 double quant | 12.48 tok/s（**无差别**） |
| 7 | 量化类型换 `fp4` | 12.48 tok/s（**无差别**） |
| 8 | 8-bit 不带 device_map | 加载直接失败：`OSError 1455 页面文件太小` |

**结论：配置层救不回来，必须换执行路径或修 bnb 本体。**

---

## 四、比"bnb 慢"更重要的发现（已核查）

就算 bnb 修好、或换用完美的 fp16，这张卡的上限是：

| 精度 | 权重体积 | 带宽决定的速度上限 |
|------|:---:|:---:|
| fp16 / bf16 | 8.04 GB | **~29 tok/s** |
| 8-bit | 4.02 GB | ~58 tok/s |
| **4-bit** | 2.01 GB | **~116 tok/s** |

**两条结论：**

1. **fp16 在这张卡上永远到不了路线图的 60+ tok/s 目标（上限 29）。**
2. **目标线（60）正好卡在 8-bit 上限（58）附近** —— 所以 **8-bit 只是"刚好不够"**，必须做到 **4-bit 级别的有效位宽**。

> **量化不是优化项，是能不能达成目标的必要条件。** 这条应写进项目级共识。

### 对 S3 的直接影响

双通路会把文本塔翻倍（4.02B → ~7.3B）：

| 项 | 数值 | 判断 |
|------|:---:|------|
| 4-bit 权重 | ~3.65 GB | ✅ 8GB 装得下 |
| 8-bit 权重 | ~7.3 GB | ❌ 超预算（D17 要求 < 7GB） |
| 按当前 bnb 效率的速度 | **~6 tok/s** | 很慢，但**不影响正确性测试** |

**因此 S3 的验收只测正确性与显存，不测速度**（速度留到 S3 通过后专门解决，见 D26）。

---

## 五、下一步候选路径（待用户选择）

| # | 路径 | 预期收益 | 代价 / 风险 |
|:-:|------|------|------|
| **A** | **GGUF + llama.cpp**（HANDOFF 原定备选） | 4B Q4 在 4060 上通常 **60-90 tok/s** | 失去 PyTorch 计算图 → **S3 双通路无法在其上验证**；只能作"速度天花板"的参照 |
| **B** | **继续修 bnb**（换版本 / 从源码编译 Windows 轮子） | 修好即约 **5x → 55-65 tok/s** | 不确定；可能要装 MSVC + CUDA Toolkit（体积大） |
| **C** | **换量化后端**（torchao / quanto / AWQ-Marlin） | 未知 | Windows 支持同样存疑；AWQ/Marlin 需要编译自定义算子 |
| **D** | **先不管速度，直接进 S3 正确性** | 0 成本，推进主线 | 速度债后移，里程碑 2 会撞墙 |

> **A 与 D 不冲突**：A 可以作为"速度参照"单独跑，D 是主线。B 若成功则 A 也不必做。
> **建议顺序：D（主线不等）→ A（拿参照数字）→ B（有空再试）。** 详见 D26。

---

## 六、复现命令

```powershell
cd D:\360MoveData\Users\18889\Documents\Nova
$env:HF_HOME='D:\360MoveData\Users\18889\Documents\Nova\.hf-cache'
$env:HF_HUB_OFFLINE='1'
$env:TMP='D:\360MoveData\Users\18889\Documents\Nova\.tmp'; $env:TEMP=$env:TMP

# 完整基线（4-bit 测速 + 8-bit 测质量）
& .\.venv\Scripts\python.exe src\baseline.py --mode 4bit --tag speed-4bit
& .\.venv\Scripts\python.exe src\baseline.py --mode 8bit --tag quality-8bit
& .\.venv\Scripts\python.exe src\baseline.py --report-only

# bnb 自检
& .\.venv\Scripts\python.exe -m bitsandbytes
```

---

## 七、附：S2 基线原始数字（已核查）

| 指标 | 4-bit（速度基准） | 8-bit（质量对照） |
|------|:---:|:---:|
| decode 速度（6 条均值） | **10.64 tok/s** | **4.91 tok/s** |
| TTFT 均值 | **0.130 s** | 0.264 s |
| prefill 速度 | 196-636 tok/s | — |
| 峰值显存 | **2.79 GiB** | 4.58 GiB |
| 权重占用 | 2.62 GiB | 4.50 GiB |
| 注意力实现 | `sdpa` | `sdpa` |

**显存结论：4-bit 峰值 2.79 GiB，距离 D17 的 7GB 预算还有 4.2 GiB 余量 —— 双通路（+~2 GB）在显存上是安全的。**

---

## 八、⚠️ 2026-09-22 更正：本文第一~五节的归因**已被推翻**

> **本节是权威结论。第一~五节保留为历史记录，其中"bnb 回退到 CPU 路径"的判断是错的。**
> 触发原因：另一位模型基于第一~五节给出了"bitsandbytes 静默回退 CPU"的诊断，用户转达后重新核查，发现原归因从根上就错了。

### 8.1 被推翻的结论

| 原文结论 | 核查结果 |
|------|------|
| 「bnb 加载了 CUDA DLL 但取不到 CUDA 符号，执行落在 CPU-only 回退」 | ❌ **错**。`cextension.lib._lib` 就是 `libbitsandbytes_cuda124.dll`；`cgemm_4bit_inference_naive_fp16` / `cquantize_blockwise_fp16_nf4` / `cdequantize_blockwise_fp16_nf4` 全在导出表里 |
| 「`get_compute_capabilities()` 报 CPU-only version → CUDA 符号不可用」 | ❌ **误导性文案**。该符号**根本不在 0.50.2 的 DLL 导出表里**（不是加载失败）；bnb 的 `BNBNativeLibrary.__getattr__` 对**任何**缺失符号都返回同一句硬编码的 "CPU-only version" 提示 |
| 「bnb 4-bit kernel 是主瓶颈」 | ❌ **错**。真实占比：bnb gemm **17.1%**、attention **4.0%**、**其余 78.9% 是 eager 逐元素/拷贝算子** |

### 8.2 决定性证据（已核查）

**证据 1 · profiler 里 4-bit 矩阵乘确实在 GPU 上跑**

```
bitsandbytes::gemm_4bit   Self CUDA = 14.964 ms / 200 次调用（74.8 µs/次）
```

CUPTI 的 CUDA 计时器只统计真正在设备上执行的 kernel。CPU 回退不会出现在这里。

**证据 2 · 吞吐量物理上排除 CPU**

`Linear4bit(2560,2560)` bs=512：**319.6 µs → 21.0 TFLOPS**。本机 CPU 不可能达到这个量级。

**证据 3 · 真凶：100% CPU 发射受限**

```
CPU enqueue-only : 76.92 ms/step     ← 只把算子塞进队列，不等 GPU
wall (with GPU)  : 77.02 ms/step
=> GPU 忙 ≈ 0.10 ms
```

约 **8000 次 kernel 启动 / token**，平均 **9.6 µs/次**。GPU 几乎全程空闲。

**证据 4 · 重新 profile（16 token，4-bit 解码）**

| 桶 | CUDA 占比 |
|------|:---:|
| eager 逐元素 / 拷贝（`to` / `_to_copy` / `copy_` / `mul` / `pow` / `mean` / `view` / `as_strided` …） | **78.9%** |
| bnb gemm_4bit | 17.1% |
| attention（`_scaled_dot_product_attention_math`） | 4.0% |

单算子 CPU 开销最贵的是 `bitsandbytes::gemm_4bit` 自己：**83 µs CPU / 次 × 252 次 = 21 ms/token** —— 这是 bnb 的 **Python 派发开销**，不是 kernel 时间。

### 8.3 尝试过并已排除的修法（全部已实测）

| # | 尝试 | 结果 |
|:-:|------|------|
| 1 | `torchao` 0.9.0 `int4_weight_only(g=128)`（bf16） | ❌ bs=1 **152 µs**，比 bnb 的 70 µs 还慢 2.2x；bs=512 只有 5.4 TFLOPS（bnb 21.0）。且报 `No module named 'triton'` |
| 2 | `torchao` `int8_weight_only` | ❌ 123.8 µs |
| 3 | transformers 官方编译入口（`cache_implementation="static"` + `CompileConfig`） | ❌ 被 `Bnb4BitHfQuantizer.is_compileable = False` **直接跳过** |
| 4 | 自写前向 + `torch.compile(mode="default", dynamic=False)` | ❌ **1.70 tok/s**（比 eager 慢 7x） |
| 5 | 自写前向 + `torch.compile(mode="reduce-overhead")`（CUDA graphs） | ❌ **0.21 tok/s**（慢 60x）。cudagraphs 被 StaticCache 挡掉：`cpu device (arg5_1)` + `mutated inputs` |

### 8.4 新的环境事实（已核查）

- **`triton-windows 3.2.0.post21` 在 torch 2.6.0 + py3.12 + Windows 上可用**（已装入 `.venv`）。`3.8.0` **不行**：`cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'`。
- Triton / Inductor 缓存目录必须显式指向 H 盘：`TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR`（默认会写 `C:\Users\18889\.triton`，报 `WinError 5`）。
- **Triton 可用 ≠ `torch.compile` 可用** —— 见 8.3 的 4/5。

### 8.5 修订后的速度账（已核查）

| 项 | 数值 |
|------|:---:|
| 4-bit 权重读取（1.82 GB，模型内实测有效带宽 ~135 GB/s） | ≈ **13.4 ms** / token |
| attention（SDPA math 回退） | ≈ 3 ms / token |
| ~8000 个微 kernel 的 **GPU 侧**启动开销 | ≈ 20-40 ms / token（估） |
| **CPU 发射开销** | ≈ **77 ms** / token ← **当前主项** |

**三条修订结论：**

1. **60+ tok/s 的障碍不是 kernel 效率，而是"每 token 约 8000 次算子派发"这件事本身。** 换量化位宽、换 bnb 版本、换 attention 实现，**都动不了它**。
2. **即使把 CPU 开销清零，也必须再减算子数量** —— 8000 个微 kernel 的 GPU 侧启动开销本身就有 20-40 ms。
3. **`torch.compile` / CUDA graphs 在当前 transformers + bnb + Windows 组合上不可用**（8.3）。要走到 60+，需要**自写精简前向**（融合 RMSNorm、消除冗余 dtype 转换）或**换运行时**。

### 8.6 意外的正面结果（对 S3 直接有用）

自写的精简前向与 HF 官方前向**数值完全一致**：

```
sanity: max|diff| = 0.0000    argmax 104198 vs 104198
```

即：**逐层调用 `model.model.language_model.layers[i]`、自己组织 forward，可以做到与官方前向逐位一致。** 这正是 S3「门控关闭时 ≈ 基线」判据需要的起点，也说明 **"自写 `nn.Module`"路线可行**（对应 [16-model-anatomy.md](16-model-anatomy.md) 第二节的待定项）。
