# P0 · 拆掉 O(n²) 掩码表 + 容量分桶（2026-09-22）

> **一句话结论（已核查）**：掩码表从 `max_len²×2` 降到"常驻 `8×max_len` + 图内 `2×max_len`"
> （65536 档 **8.00 GiB → 512 KiB + 128 KiB**），逐位一致；容量改成按桶分配后，
> **同样一份代码，used=2048 从 139.1 → 56.1 ms/token（省 59.7%），used=7291 从 138.7 → 71.2（省 48.6%）**。
> 顺带**推翻**了"重捕就爆显存"这条旧结论 —— 真凶是图没被正确释放，不是"没共享池"。

## 一、为什么要先做这一刀

它是后面所有实验的**前提**，不是优化项：

| max_len | 旧掩码整表 `max_len²×2 B` | 后果 |
|---:|---:|---|
| 18432 | 0.63 GiB | 现在就在吃这个 |
| 32768 | **2.00 GiB** | 加上 KV(4.50) + 权重(3.28) 已经超 8 GB |
| 65536 | **8.00 GiB** | **完全不可能** —— 比 KV cache 更早爆 |

所以 E2（滑窗，32K/64K 档）、E5（干扰项扫描到 64 条）不做这一刀就开不了场。

另外一条：`GraphDecoder._body()` 传的掩码覆盖整个 `max_len`，而 `q_len == 1` 时
`layers.py` 里 `is_causal = False` ⇒ SDPA **读满 `max_len` 个槽位**。
实测 `used=2048` 却按 `max_len=18432` 读时，**89% 的读发生在没写过的零槽位上**。

## 二、改动一：图内即时构造掩码行

```python
def _mask_row(self, pos):
    return torch.where(self._arange <= pos, self._zero, self._ninf).view(1, 1, 1, self.max_len)
```

形状恒定（`(1,1,1,max_len)`）、图内安全、**与旧表逐位一致**。

| max_len | 旧：整表 | 新：常驻 arange(int64) | 新：图内掩码行(fp16) | 倍数 |
|---:|---:|---:|---:|---:|
| 18432 | 0.63 GiB | 144 KiB | 36 KiB | ~3700x |
| 32768 | 2.00 GiB | 256 KiB | 64 KiB | ~6600x |
| 65536 | **8.00 GiB** | **512 KiB** | **128 KiB** | ~13000x |

`_build_mask_table` 保留为**对照用**（算旧方案体积、给逐位一致测试当参考），
文档里明确写了解码路径不再用它。

**验收**：`tests/test_decode_mask_bucket.py::test_mask_row_matches_table_bitwise`
逐位比对 `pos ∈ {0, 1, 7, 63}`，含 `-inf` 位。

## 三、改动二：容量分桶

`BUCKETS = (2070, 4096, 8192, 16384, 32768, 65536)`，三个新接口：

| 接口 | 作用 |
|---|---|
| `bucket_for(n)` | 不小于 `n` 的最小桶；超过最大桶按 `n` 精确分配 |
| `GraphDecoder.for_length(model, n, reserve=64)` | 按"要用多少 token"选桶 —— 调用侧的推荐写法 |
| `GraphDecoder.grow(new_max_len)` | 跨桶搬家：把 `[0, used)` 的 KV 拷进更大的 cache + 要求重捕 |

### 验收（`probe_decode_maxlen.py`，单通路 triton，**同轮交替两遍取小**，`clocks.sm 2460`）

| used | max_len | ms/token | 等效读量 GiB/step | 有效带宽 GiB/s |
|---:|---:|---:|---:|---:|
| 2048 | 2070 | 60.0 | 0.28 | 4.7 |
| 2048 | **4096** | **56.1** | 0.56 | 10.0 |
| 2048 | 7323 | 65.7 | 1.01 | 15.3 |
| 2048 | 8192 | 71.1 | 1.12 | 15.8 |
| 2048 | 18432 | 139.1 | 2.53 | 18.2 |
| 7291 | 7323 | 66.0 | 1.01 | 15.2 |
| 7291 | **8192** | **71.2** | 1.12 | 15.8 |
| 7291 | 18432 | 138.7 | 2.53 | 18.2 |

**建议路径（`for_length`）**：

| used | 选中的桶 | ms/token | 与 `max_len=18432` 相比 |
|---:|---:|---:|---:|
| 2048 | 4096 | 56.1 | **省 59.7%** |
| 7291 | 8192 | 71.2 | **省 48.6%** |

- 判据成立：**耗时只随 `max_len` 走，几乎不随 `used` 走**（2048 与 7291 在同一个 `max_len` 下耗时相同）。
- 选桶**只改"读多少槽位"，不改结果** —— `test_bucketed_decode_matches_wide_decode`
  与 `test_grow_preserves_decode` 都逐 token 对照过。
- 地板仍在：`max_len=2070` 也要 60 ms/token，其中 KV 只占约 1 ms ⇒ **固定开销 ~59 ms/token**
  （Windows/WDDM 派发 + 36 层），这条与 D27/D29 一致，属速度路径 ③/④ 的范畴。

## 四、改动三：图内存池 —— 旧结论被推翻

旧报告写过"每题重捕，重捕 9 次 → 7905 / 8188 MiB 崩溃"，据此引入共享池。
`probe_graph_recapture.py` 实测（`clocks.sm 2475`，`max_len=2070`，每轮 `del` + `gc` + `empty_cache`）：

| 策略 | 逐轮 allocated（3 轮） | 峰值 |
|---|---|---|
| `off`（不传池） | 3087.5 → 3095.6 → 3103.7 MiB（+8.1/轮） | 3617.0 MiB |
| `shared`（死用一个池） | 3111.9 → 3120.0 → 3128.1 MiB（+8.1/轮） | 3453.5 MiB |
| `auto`（自适应） | 3136.2 → 3144.4 → 3152.5 MiB（+8.1/轮） | 3477.9 MiB |

**跨桶重捕**（同一个解码器 `2070 → 4096 → 8192`）：

| 桶 | allocated | 与上一档的差 | 该多出来的 KV 容量 |
|---:|---:|---:|---:|
| 2070 | 3453.9 MiB | — | — |
| 4096 | 3745.1 MiB | +291.2 MiB | 2026 token × 144 KiB = **291.7 MiB** ✅ |
| 8192 | 4329.2 MiB | +584.1 MiB | 4096 token × 144 KiB = **576.0 MiB** ✅ |

⇒ **增量正好等于多出来的 KV 容量**，即"搬家不漏"；峰值 4.33 GiB < D17 的 7 GB。

**结论（已核查）**：

1. **"重捕就爆显存"不成立** —— 三种策略逐轮都基本平，`off` 反而全程最低。
   真凶是图没被正确释放（图被别的引用吊着），不是"没共享池"。
2. **共享池反而会踩 PyTorch 的裸 assert**：`CUDACachingAllocator.cpp:2225`
   `it->second->use_count > 0 INTERNAL ASSERT FAILED`（没有消息，很难查）。
   它在 `tests/test_decode_mask_bucket.py` **全量跑时必中、单跑同一题不中**；
   脚本里换了 5 种调用序列（同长度连续 / 换长度 / 两张图并存 / 中间 `empty_cache` / 死用一个池）
   **一次都没复现**。⇒ 触发条件依赖分配器状态，无法可靠规避。
3. **处置**：`capture(pool="off")` 成为默认（不传池）；`"auto"` / `"shared"` 保留为可选，
   并在 docstring 里写明上面这个坑。P0 不需要共享池来省显存。

**待实测**：逐轮 **+8.1 MiB** 的缓慢增长（三轮 +24 MiB，来源未定位）。
按 8 MiB/轮外推，100 轮量级约 +800 MiB —— 实验循环跑到那个规模前要重新确认，
本轮不当作已解决。

## 五、这一刀**没有**解决的问题

- **64K 档还是装不下**：掩码表的墙拆了，但 fp16 KV 在 65536 是 **9.00 GiB**，
  加上权重 3.28 GiB 仍超预算。⇒ 64K 必须靠 **E2 的滑窗层**（本地层 ring buffer）。
- decode 的**固定地板 ~59 ms/token** 一分没动（与 `used`/`max_len` 都无关）。
- prefill 的 O(n²) 算力墙一分没动（12.7K=6.8 s → 32K≈45 s → 64K≈180 s，算术外推）。

## 六、结论状态

| 结论 | 状态 |
|---|---|
| 图内掩码行与旧表**逐位一致** | ✅ 已核查（测试） |
| 掩码常驻从 `max_len²×2` 降到 `8×max_len`（+图内 `2×max_len`） | ✅ 已核查（公式 + 测试） |
| 选桶把 used=2048 的 decode 从 139.1 降到 56.1 ms/token | ✅ 已核查（`clocks.sm 2460`，同轮交替） |
| 选桶/搬家**不改变解码结果** | ✅ 已核查（逐 token 对照测试） |
| "重捕就爆显存"是误判，共享池非必需 | ✅ 已核查（三策略对照） |
| 共享池的 assert 触发条件 | ⚠️ **推测**：依赖分配器状态（全量跑中、脚本不中） |
| 每轮 +8.1 MiB 的增长来源 | 🧪 待实测 |
| 64K 的可行性 | 🧪 待实测（依赖 E2 滑窗） |

## 七、复现命令

```powershell
$env:HF_HOME=(Resolve-Path .).Path+'\.hf-cache'; $env:TMP=(Resolve-Path .).Path+'\.tmp'; $env:TEMP=$env:TMP
$env:HF_HUB_OFFLINE='1'; $env:TRITON_CACHE_DIR=$env:TMP+'\triton-cache'; $env:TORCHINDUCTOR_CACHE_DIR=$env:TMP+'\inductor-cache'

# 逐位一致 + 选桶/搬家不改结果（60 passed）
& .\.venv\Scripts\python.exe -m pytest tests -q

# 掩码/分桶的验收表（used vs max_len + 建议路径），带 clocks.sm
& .\.venv\Scripts\python.exe -u src\diagnostics\probe_decode_maxlen.py

# 图内存池三策略 + 跨桶重捕的显存轨迹
& .\.venv\Scripts\python.exe -u src\diagnostics\probe_graph_recapture.py --rounds 3

# 各处显存固定占用（现在打印的是"arange + 图内掩码行"，不再是整表）
& .\.venv\Scripts\python.exe -u src\diagnostics\probe_long_context_vram.py
```

> ⚠️ 速度类数字必须带 `clocks.sm`；本次测量期间 GPU 从 765 MHz 爬到 2460 MHz，
> 表里的数字全部取自"同轮交替两遍取小"，**不要与其它时间点的数字直接比**。
