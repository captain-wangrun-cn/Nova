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

## 第三轮（2026-09-22 · 速度路径 ①，见 [reports/speed-path1-nf4-gemv.md](../../reports/speed-path1-nf4-gemv.md) 与 **D30**）

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
## 第四轮（2026-09-22 · S4 记忆最小实现，见 [reports/s4-memory-min.md](../../reports/s4-memory-min.md) 与 **D31**）

| 脚本 | 证明什么 | 命令 |
|------|------|------|
| `probe_memory.py` | 捕获 / 注入的**逐位一致性**：整段 prompt 抓成记忆再注入，KV cache 与真实 prefill `max\|diff\| = 0.000e+00` | `& .\.venv\Scripts\python.exe src\diagnostics\probe_memory.py` |
| `probe_memory_keys.py` | **pre-RoPE 裸余弦不可用**：6 条候选记忆的键裸余弦**全部 > 0.8**，几乎不区分内容 | 同上，换文件名 |
| `probe_memory_addressing.py` | 寻址第一版：为什么必须按**注入后真实位置**旋转 Q/K 再算 `Q·K`（不旋转则 top-1 只有 1/3） | 同上 |
| `probe_memory_addressing2.py` | **决定性扫描**：6 条 × 8 问 × 12 组配置 → `alone`+按长度归一+不标准化 = **8/8**；`stack` 2/8；标准化 4/8 | 同上 |
| `probe_memory_addressing3.py` | 查询取哪几个 token：整句取平均 8/8 掉 7/8；**末尾 4 个 token** 才对 | 同上 |
| `probe_memory_addressing4.py` | **注入位置**：`place="front"`（插最前面）在 20 轮历史下退化到 1/3 且生成崩；`turn` 才对 | 同上 |
| `exp_recall_baseline.py` | **对照实验（重要）**：第 1 轮**留在**上下文里、不注入记忆 → **20 轮 683 token / 60 轮 1915 / 120 轮 3785 全部 8/8**。证明 **S4 演示的 `off=0/8` 是"第 1 轮被移出上下文"的因果对照，不是"原版模型会忘"** | `& .\.venv\Scripts\python.exe src\diagnostics\exp_recall_baseline.py --turns 20 60 120` |

**第四轮新增的坑：**

- **同一套检索逻辑写两份必然漂移。** 第一版 `tests/test_memory.py` 自己在测试里手搓了一遍打分，忘了把 `query_span` 截成末尾 4 个 token，命中率立刻从 8/8 掉 7/8。现在检索只写一份（`MemorySession.rank()`），`prefill` 与测试都调它。
- **取"prompt 最后 4 个 token"拿到的是 `<|im_start|>assistant\n`**（没有内容）→ 寻址退化成"恒选第一条"。必须用 `chatfmt.find_span` 定位问题那句话。
- **连测 7 轮会让笔记本 GPU 从 2160 MHz 掉到 ~870 MHz（94 W → 35 W），同一条件耗时翻倍。** 跨条件对比必须在同一时钟区间内**逐轮交替**取差值中位 —— 否则会算出"注入耗时 −40 ms"这种负数。
- **"模型忘了"必须实测，别当成前提。** S4 的验收标准写着"第 20 轮取回"，很容易读成"原版模型 20 轮就忘了"；实测 120 轮（3785 token）仍然 8/8，该模型上下文上限是 262144。**验收标准里的"答对"只能证明记忆通路可用，不能证明它比"留在上下文里"更好。**

## 第五轮（2026-09-22 · 长上下文注意力，见 [reports/long-context-attention.md](../../reports/long-context-attention.md) 与 **D33**）

| 脚本 | 证明什么 | 命令 |
|------|------|------|
| `probe_long_context_vram.py` | **瓶颈归因**：SDPA 后端真实状态（flash 未编译 / 融合内核要求头数相同）、峰值随长度的增长曲线（扣掉固定占用后翻倍涨 ~4x = O(n^2)）、固定占用拆解 | `& .\.venv\Scripts\python.exe src\diagnostics\probe_long_context_vram.py` |
| `probe_attention_kernel.py` | **决定性**：同轮内对比 GQA 留 SDPA（math 回退）/ 展平 32 头 / 强制 cuDNN —— 1749 token **2.1x**、3012 **2.5x**、7146 现状 OOM 而展平只要 3.53 GiB | 同上，换文件名 |
| `probe_kernel_mapping.py` | **数值正确性（改注意力路径必过）**：两边钉 math 时逐位一致（0.000e+00）→ `repeat_kv` 的 GQA 映射正确；与手写 fp32 参考误差相同（4.07e-04） | 同上 |
| `probe_attention_tradeoff.py` | 后端可用性（MATH 6.76 GiB / 5293 ms vs EFFICIENT **3.26 GiB / 1299 ms**）+ **贪心 32/32 token 一致** | 同上 |
| `probe_kernel_equivalence.py` | 展平后的**长度天花板**：7146 / 14363 健康，21615 峰值 8.50 GiB + prefill **375.9 s**（换页断崖） | 同上 |
| `exp_needle.py` | **信息过载下的选择性**：4 条同形事实（只有地点与号码不同）埋在不同深度、各问一次 -> 1894/3665/7291/**12728** token 全部 **4/4、零挑错** | `... exp_needle.py --lens 2048 4096 8192 14363 --max-len 14848` |

**第五轮新增的坑：**

- **`GraphDecoder.capture()` 每次都新建一张 CUDA Graph 并分配新内存池。** 若每个问题都重捕（起点位置不同），几次之后显存就爆 —— 实测重捕 9 次后 7905/8188 MiB 崩溃。只生成几十个 token 时直接调 `dec._body()`（eager），别建图。
- **GQA + `enable_gqa=True` 在本机会静默退回 math 后端**，实体化 O(n^2) 的 **fp32** 分数矩阵（`32 头 x n^2 x 4 字节 x 2 张`）。它不报错，只让显存与耗时突然涨一个量级 —— **看到长 prefill 峰值异常，先查 SDPA 走了哪个后端**（`torch.backends.cuda.is_flash_attention_available()`）。
- **改注意力路径必须先过"两边钉同一个后端"的等价性测试。** 直接比 logits 会被内核精度差异误导（`max|diff| ~ 1.0` 看着像 bug，其实是 fp16 累加）；钉住后端后是 0.000e+00。
- **别用跨时间点的数字算加速比。** 3665 token 从 29.1 s 变 1.4 s 是"不再换页"的功劳，不是内核快 20 倍；同轮内的数字才可比。
