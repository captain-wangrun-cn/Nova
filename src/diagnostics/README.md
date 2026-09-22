# src/diagnostics · 速度归因复现脚本

> 产生于 **2026-09-22**，用于推翻 [reports/s2-speed-diagnosis.md](../../reports/s2-speed-diagnosis.md) 第一~五节的归因。
> 结论见该报告**第八节**与 **D27**。原始数字见 [reports/s2-speed-evidence-2026-09-22.json](../../reports/s2-speed-evidence-2026-09-22.json)。

**跑之前必须设的环境变量**（否则会往 C 盘写缓存）：

```powershell
$env:HF_HOME='H:\Nova\.hf-cache'; $env:TMP='H:\Nova\.tmp'; $env:TEMP=$env:TMP
$env:HF_HUB_OFFLINE='1'
$env:TRITON_CACHE_DIR='H:\Nova\.tmp\triton-cache'
$env:TORCHINDUCTOR_CACHE_DIR='H:\Nova\.tmp\inductor-cache'
```

| 脚本 | 证明什么 | 命令 |
|------|------|------|
| `probe_bnb_cuda_path.py` | bnb 加载的是 **CUDA** DLL，`cgemm_4bit_inference_naive_fp16` 等符号齐全；profiler 里 `bitsandbytes::gemm_4bit` 有真实 Self CUDA 时间 | `& .\.venv\Scripts\python.exe src\diagnostics\probe_bnb_cuda_path.py` |
| `probe_cpu_vs_gpu_bound.py` | **决定性**：`CPU enqueue-only` ≈ `wall` → 100% CPU 发射受限；并测 `torch.compile` 可用性 | 同上 |
| `profile_decode_ops.py` | 16 token 解码的算子级占比（eager 逐元素 78.9% / bnb 17.1% / attention 4.0%） | 同上 |
| `bench_quant_backends.py` | fp16 vs bnb vs torchao 的 bs=1 单算子基准（torchao 更慢，已排除） | 同上 |

## 第二轮（2026-09-22 · CUDA Graph 解码，见 [reports/s3-graph-decode.md](../../reports/s3-graph-decode.md) 与 **D29**）

| 脚本 | 证明什么 | 命令 |
|------|------|------|
| `microbench_op_cpu.py` | **单次 kernel 启动约 13.5us CPU**，且严格线性（64 个 `mul` = 887us）—— Nova 每 token ~6500 次算子，光启动就 ~62ms | `& .\.venv\Scripts\python.exe src\diagnostics\microbench_op_cpu.py` |
| `probe_cuda_graph.py` | 启动开销线性；256 个 aten 算子进图后 replay 快 **13.3x**；**bnb `Linear4bit` 与 Triton kernel 都能进图且数值逐位一致** | `& .\.venv\Scripts\python.exe src\diagnostics\probe_cuda_graph.py` |
| `probe_cuda_graph2.py` | 用"图 replay 的 wall = 纯 GPU 时间"当尺子：bnb 4-bit 与 fp16 的 GPU 耗时几乎相同（29.2 vs 27.9us）→ **bnb 没走 packed 4-bit GEMV** | 同上，文件名换成 `probe_cuda_graph2.py` |
| `probe_cuda_graph3.py` | 隔离"Triton + bnb 同图"是否有问题 —— **没有**（157us ≈ 两者之和） | 同上，文件名换成 `probe_cuda_graph3.py` |
| `check_graph_exactness.py` | 强制两边吃同一个 token，逐步比 logits：偏差 3e-2~8e-2（fp16 舍入级），argmax 全同 | `... check_graph_exactness.py --paths 2` |
| `profile_graph_decode.py` | **局部图分解**剩余 GPU 时间：36 层 12.90ms（75.9%）/ lm_head 3.12ms（18.3%，已打满带宽）/ 其它 1.0ms | `... profile_graph_decode.py --paths 1` |

**第二轮新增的坑：**

- **CUPTI 看不到 CUDA Graph 内的 kernel** —— profiler 只报 0.01 ms/token。要分解图内耗时只能用**局部图**（`profile_graph_decode.py`）。
- **图 replay 的 `wall` 就是纯 GPU 时间**（CPU 只发一次），这是本机目前唯一可靠的 GPU 时间尺子。
- **多个图并存会互相拖慢**：测图要一个一个来，测完 `del` + `empty_cache()`。否则会看到"7 个线性层 148us、加一个 norm 变 1184us"这种假象。
- **replay 必须在 `torch.inference_mode()` 内**（图内有原地写入）。

**已知坑：**

- `lib.get_compute_capabilities()` 会报 `CPU-only version`，**这是误导性文案**：该符号不在 0.50.2 的 DLL 导出表里，而 bnb 的 `BNBNativeLibrary.__getattr__` 对**任何**缺失符号都返回同一句话。**不要据此判断回退 CPU。**
- `triton-windows` 必须用 **3.2.0.post21**（配 torch 2.6.0）；`3.8.0` 会报 `cannot import name 'AttrsDescriptor'`。
- 判据要用 **CUPTI 的 Self CUDA 时间**或 **CPU enqueue vs wall 的对比**，不要用单算子小形状的"有效带宽"——小形状下测到的是启动延迟，不是带宽。
- ⚠️ **测速前先看 GPU 有没有降频**：`nvidia-smi --query-gpu=pstate,clocks.sm,power.draw,temperature.gpu --format=csv`。
  本机 SM 上限 **3105 MHz**，但笔记本在空闲 / 切软件 / 温度管理后会长时间停在 **~780 MHz**，
  此时**同一条命令会慢 1.9x**（实测：`bench_graph.py --paths 1 --no-hf --quant bnb --lm-head4`
  从 **14.0 ms/token 变 26.7 ms/token**）。**不同时间点测出来的数字不能直接对比**，必须连同 `clocks.sm` 一起记录。

## 第三轮（2026-09-22 · 速度路径 ①，见 [reports/s4-nf4-gemv.md](../../reports/s4-nf4-gemv.md) 与 **D30**）

| 脚本 | 证明什么 | 命令 |
|------|------|------|
| `probe_nf4_format.py` | 用 `ctypes` 读 bnb 的 DLL 导出表，**逐位复现 NF4 打包格式**（最大误差 **0.000e+00**） | `& .\.venv\Scripts\python.exe src\diagnostics\probe_nf4_format.py` |
| `bench_nf4_vs_bnb.py` | **决定性**：7 个投影形状自写 kernel 全部慢于 bnb（1.31~2.75x）；bnb 距 DRAM 下界 **1.19x**、自写 **1.99x** | 同上，换文件名 |
| `bench_nf4_lmhead.py` | lm_head 形状（N=151936, K=2560）：fp16 **3110us** / bnb **1021us** / 自写最佳 **1367us** | 同上 |
| `bench_nf4_gemv.py` | 自写 kernel 的数值正确性 + launch 参数扫描 | 同上 |
| `exp_nf4_launch.py` | **36 组 launch 参数没有一个快过 bnb** | 同上 |
| `exp_nf4_final.py` | 更激进结构全部证伪：`tl.dot`/MMA 848us、`tl.gather` 编译失败、全宽 BLOCK_N 568us、fp16 码本 394us（bnb 234us） | 同上 |
| `exp_nf4_stages.py` / `exp_nf4_struct.py` / `exp_nf4_isolate.py` | 中间探索（分段计时、结构变体、延迟隔离），保留作证据 | 同上 |

**第三轮新增的坑：**

- **微基准证明不了端到端收益。** eager 下 bnb 16.32 vs nf4 16.56 tok/s 几乎一样；**只有图解码（纯 GPU 时间）才暴露 1.79x 的真实差距**（16.0 vs 28.8 ms/token）。单算子尺度上时间被启动开销淹没。
- **`@triton.jit` 里引用模块级 Python 常量会报 `NameError: Cannot access global variable`** —— 即使标了 `: tl.constexpr` 也不行（实测）。**kernel 内直接用字面量。**
