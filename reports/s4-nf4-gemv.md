# S4 · 速度路径 ①：Triton NF4 GEMV —— 方向证伪，但顺手捡到 lm_head 的 2ms

> 日期 **2026-09-22** · 状态：**路径 ① 已证伪并结案**；附带收益（4-bit lm_head）已落盘
> 代码：[src/nova/kernels.py](../../src/nova/kernels.py)、[src/nova/quant.py](../../src/nova/quant.py)、[src/nova/loader.py](../../src/nova/loader.py)
> 证据：[probe_nf4_format.py](../../src/diagnostics/probe_nf4_format.py)、[bench_nf4_vs_bnb.py](../../src/diagnostics/bench_nf4_vs_bnb.py)、[bench_nf4_lmhead.py](../../src/diagnostics/bench_nf4_lmhead.py)、[exp_nf4_final.py](../../src/diagnostics/exp_nf4_final.py)、[exp_nf4_launch.py](../../src/diagnostics/exp_nf4_launch.py)
> 验收：`tests/test_nf4_linear.py`（11 条）+ `tests/test_lm_head4.py`（5 条）· 全量 **28 passed**

---

## 一、结论先行

| 问题 | 结论 | 标注 |
|------|------|:---:|
| 自写 Triton NF4 GEMV 能替掉 bnb 吗？ | ❌ **不能**。7 个投影形状**全部更慢**（1.31x ~ 2.75x），距 DRAM 下界 1.99x | 已核查 |
| 36 组 launch 参数里有没有能翻盘的？ | ❌ 没有。**36 个组合没有一个快过 bnb** | 已核查 |
| 换更激进的结构（`tl.dot` / `tl.gather`）呢？ | ❌ 更差：`tl.dot` 848us vs bnb 234us；`tl.gather` 直接编译失败 | 已核查 |
| 那这条路白走了吗？ | ❌ **没白走** —— 顺手把 `lm_head` 换成 4-bit，每 token 省 **2.0 ms** | 已核查 |

### 1.1 端到端（`bench_graph.py --n 32 --no-hf`，图解码，32/32 token 与 eager 一致）

| 配置 | 层数 | eager tok/s | 图 tok/s | **图 ms/token** | 峰值显存 |
|------|:---:|:---:|:---:|:---:|:---:|
| 单通路 · fp16 lm_head | 36 | 15.18 | 62.35 | 16.0 | 3.09 GiB |
| **单通路 · 4-bit lm_head** | 36 | 16.32 | **71.34** | **14.0** | 3.28 GiB |
| 单通路 · 自写 NF4 kernel | 36 | 16.56 | 34.74 | **28.8** | 3.31 GiB |
| 双通路 · fp16 lm_head | 60 | 9.20 | 40.38 | 24.8 | 4.65 GiB |
| **双通路 · 4-bit lm_head** | 60 | 9.24 | **43.99** | **22.7** | 4.83 GiB |

**净收益：单通路 −2.0 ms/token（+14%），双通路 −2.1 ms/token（+9%），代价 +0.19 GiB。**
峰值 **4.83 GiB**，D17 预算（< 7 GB）内。

---

## 二、路径 ① 为什么失败（已核查）

### 2.1 七个投影形状，无一胜出

`src/diagnostics/bench_nf4_vs_bnb.py`（本次重跑，CUDA Graph 计时 = 纯 GPU 时间）：

| shape | N | K | 4bit MB | bnb us | nf4 us | nf4/bnb | bnb GB/s | nf4 GB/s |
|-------|--:|--:|--:|--:|--:|:--:|--:|--:|
| q_proj | 4096 | 2560 | 5.90 | 29.7 | 42.6 | 1.43x | 199 | 139 |
| k_proj | 1024 | 2560 | 1.47 | 10.6 | 29.0 | 2.73x | 139 | 51 |
| v_proj | 1024 | 2560 | 1.47 | 10.5 | 28.9 | 2.75x | 141 | 51 |
| o_proj | 2560 | 4096 | 5.90 | 27.2 | 53.8 | 1.98x | 217 | 110 |
| gate_proj | 9728 | 2560 | 14.01 | 66.4 | 87.5 | 1.32x | 211 | 160 |
| up_proj | 9728 | 2560 | 14.01 | 66.0 | 86.6 | 1.31x | 212 | 162 |
| down_proj | 2560 | 9728 | 14.01 | 61.7 | 123.8 | 2.01x | 227 | 113 |

```
DRAM 读带宽（512MB 归约，本机实测）: 249 GB/s
每层 4-bit 权重 56.8 MB  ->  bnb 272 us  |  nf4 452 us
36 层                    ->  bnb 9.79 ms |  nf4 16.28 ms
DRAM 下界（2.04 GB @ 249 GB/s）: 8.20 ms
bnb 距下界 1.19x   ;   nf4 距下界 1.99x
```

**关键读数**：bnb **已经把 4-bit GEMV 打到带宽下界的 1.19x** —— 留给"更快的 GEMV"的空间本来就只有 19%，而自写 kernel 差了 2 倍。

### 2.2 为什么慢（机制）

GEMV（M=1）是**纯延迟受限**负载：算术强度≈0，要打满 249 GB/s 靠的是**足够多的在途 load**，不是算力。

- **码本查表是唯一的硬成本**：一次发散访存约 **3.5 SM-cycle/warp**，而 ALU 侧只 1 —— 所以"每字节查一次表"（低 16 位 = 偶数元素、高 16 位 = 奇数元素）已经是能做到的最省查表方案，仍然顶不住。
- **program 数不够**：`block_n=64` 时 `k_proj`（N=1024）只有 **16 个 program**，喂不满整卡的 SM。
- 内层每轮要等表加载，形成依赖链。

（标注：**延迟量级为已核查**（来自 `exp_nf4_isolate.py` / `exp_nf4_struct.py` 的分解实验）；"喂不满 SM"为**推测**，但被 `k_proj`/`v_proj` 恰为最差的 2.7x 这一事实支持。）

### 2.3 调参救不了（已核查）

`src/diagnostics/exp_nf4_launch.py` 本次重跑，扫描 `block_n ∈ {32,64,128,256} × num_warps ∈ {2,4,8} × num_stages ∈ {2,3,4}` = **36 个组合**：

```
2560x2560: bnb = 19.8 us     9728x2560: bnb = 67.3 us
最佳组合 block_n=64 num_warps=2 num_stages=3 -> 34.1 us / 87.1 us   （仍慢 1.7x / 1.3x）
最差组合 block_n=256 num_warps=8               -> 234.3 us / 408.9 us
```

**36 个组合没有一个快过 bnb。** 端到端同理：`--quant nf4` 稳定在 **28.8 ms/token**（vs bnb 16.0）。

### 2.4 更激进的结构实验也证伪（已核查）

`src/diagnostics/exp_nf4_final.py` 本次重跑（同一脚本、同一测量条件）：

| 方案 | 合计 us | vs bnb |
|------|--:|:--:|
| **bnb** | **233.9** | 1.00x |
| A：全宽 BLOCK_N 一次载入 | 567.6 | 2.43x |
| FP：一字节一次查表（表里存 fp16x2） | 394.5 | 1.69x |
| G：`tl.gather` | **inf（编译失败）** | — |
| DOT：join/permute/reshape 成 `[BLOCK_K, BLOCK_N]` 走 `tl.dot`/MMA | 848.0 | 3.63x |

`tl.dot` 反而最慢 —— MMA 是给**大 M** 准备的，M=1 时把 4-bit 解包成 fp16 tile 再乘，等于把省下来的带宽又还回去了。

---

## 三、NF4 格式逐位复现（已核查）

`src/diagnostics/probe_nf4_format.py` 用 `ctypes` 直接读 DLL 导出表，逐位复现 bnb 的打包格式，`max|diff| = 0.000e+00`：

```
一字节装 2 个 4-bit 索引：元素 2j = 高半字节，元素 2j+1 = 低半字节
每 64 个元素共享一个 absmax
double quant 时：absmax = state2.code[q_absmax] * state2.absmax[b // 256] + qs.offset
权重 = nf4_code[索引] * absmax[元素下标 // 64]
```

布局选择：`packed [N, K//2]` uint8（沿 K 连续，**与 bnb 原始一致，解包时不再转置**）+ `absmax [K//64, N]` fp32（转置后沿 N 连续）+ `lut [256]` int32。`tests/test_nf4_linear.py` 的 `test_extract_matches_bnb_dequant` 断言与 `bnb.functional.dequantize_4bit` **逐位一致**。

---

## 四、真正的收益：`lm_head` 换成 4-bit（已核查）

### 4.1 为什么这里能赚

`lm_head` 复用 fp16 的 `embed_tokens.weight`：**151936 × 2560 × 2 B = 778 MB**，每 token 都要完整读一遍。

`src/diagnostics/bench_nf4_lmhead.py` 本次重跑：

| 实现 | us | 说明 |
|------|--:|------|
| fp16 `nn.Linear` | **3109.9** | 778 MB / 3.11 ms = **250 GB/s ≈ DRAM 上限** |
| **bnb `Linear4bit`** | **1020.7** | 219 MB / 1.02 ms = 215 GB/s → **3.05x** |
| 自写 nf4 最佳（bn=128 nw=2） | 1367.1 | 比 bnb 慢 1.34x |

**这里是本次最重要的认知修正**：S3 报告说"lm_head 3.12ms 已打满带宽、没得优化"—— **数字是对的，结论是错的**。"打满带宽"只说明**不能再靠优化访存效率提速**，但**降位宽能把要搬的数据砍到 1/4**。这两件事是正交的。

### 4.2 精度与显存代价（已核查）

| 项 | 实测 |
|------|------|
| 额外显存 | **+0.187 GiB**（4-bit 权重 219 MB；fp16 `embed_tokens.weight` **原样保留**） |
| 装载耗时 | 0.07 s |
| logits 最大绝对差 | 1.0205（4-bit 的正常量化误差） |
| logits 平均绝对差 | 0.15598 |
| argmax 一致 | 随机 hidden 5/5；真实贪心解码 **24/24**、**32/32** |

**`embed_tokens.weight` 必须保留 fp16** —— 它同时是输入 embedding 的表，动它会让输入也掉精度。所以这是**额外**的一份副本，不是替换。

---

## 五、实现与验收

| 文件 | 改动 |
|------|------|
| `src/nova/kernels.py` | `_nf4_gemv_kernel` + `fused_nf4_linear` + `build_nf4_lut` + `set_nf4_launch_config`（保留，作为**证伪证据**与后续参考） |
| `src/nova/quant.py` | `extract_nf4` / `NF4Linear` / `convert_to_nf4`；`convert_to_nf4` 默认 **跳过 `lm_head4`** |
| `src/nova/loader.py` | 新增 `enable_lm_head_4bit()` |
| `src/nova/model.py` | `NovaForCausalLM.lm_head_forward()`：有 `lm_head4` 用它，否则走 fp16 `embed_tokens.weight` |
| `src/nova/decode.py` | 图内 `_body()` 改走 `model.lm_head_forward()`（原来硬编码 `F.linear(hidden, embed_tokens.weight)`） |
| `src/bench_graph.py` | 新增 `--quant {bnb,nf4}`、`--lm-head4` |

### 5.1 开发中踩到的真 bug：`lm_head_forward` 无限递归

一次**全局字符串替换**把 fallback 那行也换掉了：

```python
def lm_head_forward(self, hidden):
    if self.lm_head4 is not None:
        return self.lm_head4(hidden)
    return self.lm_head_forward(hidden)      # ← 无限递归
```

`test_lm_head_forward_uses_embed_tokens` 专门锁死它：断言 `lm_head4 is None` 时结果**逐位等于** `F.linear(hidden, embed_tokens.weight)`。

### 5.2 验收（全量 28 passed）

| 测试 | 判据 |
|------|------|
| `test_extract_matches_bnb_dequant` | 自写解包与 bnb 逐位一致 |
| `test_nf4_linear_matches_bnb_decode` / `_prefill` | M=1 / M=7 差 < 10 ULP(fp16) |
| `test_lm_head_forward_uses_embed_tokens` | **锁死无限递归** |
| `test_lm_head4_greedy_matches_fp16` | 4-bit lm_head 的 **24/24 个贪心 token 与 fp16 一致** |
| `test_lm_head4_memory_cost` | 额外显存 < 0.30 GiB |
| `test_convert_to_nf4_skips_lm_head4` | `lm_head4` 不被误换成自写 kernel |
| `test_graph_decode_matches_eager` | 图解码 vs eager **token 逐个相同**（含 4-bit lm_head） |
| `test_graph_memory_budget` | 峰值 < 7 GiB（D17） |

---

## 六、两条教训（写给下一个会话）

1. **微基准证明不了端到端收益。** eager 下 bnb 16.32 tok/s vs nf4 16.56 tok/s —— 看着几乎一样（甚至 nf4 略快），因为单算子尺度上时间被启动开销淹没。**只有图解码（纯 GPU 时间）才暴露 1.79x 的真实差距**（16.0 vs 28.8 ms/token）。同理，`lm_head` 的 2ms 收益在 eager 下只看到 1.1ms。
2. **"已打满带宽"不等于"没得优化"。** 打满带宽说的是**访存效率**，降位宽改的是**数据量**。见第四节。

---

## 七、对后续路径的影响

- **速度路径 ① 结案**，不再投入。自写 kernel 保留在 `quant.py`（`convert_to_nf4` 仍可用），但**默认路径是 bnb**。
- 单通路 36 层 bnb 9.79 ms vs DRAM 下界 8.20 ms —— **只剩 1.19x 空间**。想要大幅提速必须从**别处**拿：
  - ③ 融合 RoPE / 去冗余拷贝
  - ④ 融合注意力（`F.scaled_dot_product_attention` M=1 在 eager 下 314us，图内待测）
  - 削减算子数量（双通路 60 层的算子数更多，图内 wall 24.8ms 里层间开销占比更高）
- **下一步主线仍是 S4 · 记忆最小实现**；速度优化继续并行、不阻塞。
