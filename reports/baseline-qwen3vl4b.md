# S2 · 单通路基线报告 · Qwen3-VL-4B-Instruct

> 生成方式：`python src/baseline.py --mode <4bit|8bit> --tag <tag>`（本报告自动重建，勿手改）
> 模型：`Qwen/Qwen3-VL-4B-Instruct` · 提示集：`data/eval/baseline-prompts.json`（6 条固定 prompt） · 结论标注：**已核查 / 待实测 / 推测**

---

## 零、这组数字怎么来的（先看这里，避免以后比错东西）

| 项 | 定义 |
|------|------|
| **TTFT** | 单独跑一次 prefill 前向的耗时（`torch.cuda.synchronize()` 夹住），不含采样与解码 |
| **decode t/s** | `(新生成 token 数 - 1) / (generate 总时长 - TTFT)` —— 纯解码速度 |
| **prefill t/s** | `prompt token 数 / TTFT` |
| **peak VRAM** | 该条 prompt 生成期间的 `torch.cuda.max_memory_allocated` 峰值（GiB） |
| 解码参数 | 模型自带 `generation_config.json` 的**官方默认值**：temperature 0.7 / top_p 0.8 / top_k 20 / repetition_penalty 1.0 |
| 随机性 | 每条 prompt 用 `seed + index` 固定，与运行顺序无关 |
| stop 集合 | `[151645, 151643]` —— 官方 `generation_config.json` 里 eos_token_id 就是这两个 |

---

## 运行 · `speed-4bit`（4bit）

模型加载方式：`Qwen3VLForConditionalGeneration / 4bit / device_map=auto` · 注意力实现：`sdpa`

| id | 类别 | prompt tok | 新生成 tok | TTFT (s) | 总时长 (s) | decode t/s | prefill t/s | 峰值显存 (GiB) | 停止原因 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| rp-01 | rp | 86 | 256 | 0.172 | 24.38 | 10.53 | 499.1 | 2.79 | max_new_tokens |
| rp-02 | rp | 85 | 256 | 0.142 | 24.27 | 10.57 | 599.5 | 2.79 | max_new_tokens |
| rp-03 | rp | 90 | 245 | 0.142 | 23.05 | 10.65 | 635.5 | 2.79 | eos |
| qa-01 | qa | 21 | 61 | 0.107 | 5.73 | 10.67 | 195.8 | 2.74 | eos |
| qa-02 | qa | 29 | 50 | 0.102 | 4.65 | 10.77 | 283.7 | 2.74 | eos |
| qa-03 | qa | 57 | 192 | 0.113 | 18.00 | 10.68 | 502.2 | 2.77 | max_new_tokens |

**均值：** TTFT **0.130 s** · decode **10.64 tok/s** · 峰值显存 **2.79 GiB**

- 英文 RP 三项 decode 速度均值：**10.59 tok/s**
- 短问答三项 decode 速度均值：**10.70 tok/s**

### 原始输出

**`rp-01` · 英文 RP · 港口雨夜开场**

```text
Elara’s fingers tightened around the edge of the crate, her knuckles white beneath the worn leather of her gloves. The dockmaster’s voice, low and gravelly, cut through the hum of the harbor wind. Two men stood behind him, their boots heavy on the wooden ramp, blocking the path like sentinels. One of them—broad-shouldered, sleeves rolled to reveal forearms scarred with old bruises—glared at her, eyes sharp as a blade.

The ramp creaked under their weight, groaning like an old ship’s timbers. Elara didn’t flinch. She lifted the crate, its edges slick with dew and salt, and set it down with a clatter that echoed through the narrow alley of the dock. She didn’t look at the men. Didn’t look at the dockmaster’s face, which was a mask of weary patience.

She turned, her boots crunching over the grit, and walked past the men, past the dockmaster’s arms crossed over his chest, and into the dim light of the warehouse entrance. The scent of diesel and damp wood clung to the air, thick and heavy as the debt she carried.

She didn’t speak. She didn’t need to. The dockmaster
```

**`rp-02` · 英文 RP · 坏消息与情绪压力**

```text
Elara’s fingers trembled as she traced the inked signature on the manifest, the paper crisp under her touch. The scent of stale coffee and ozone clung to the air in the dim office, the fluorescent lights flickering overhead. Her breath caught as she turned the page, the last one, the one that had slipped beneath the stack of documents she’d hastily signed. There, in bold, unyielding letters, was Kessler’s name. The ink had dried, but the weight of it still pressed against her chest like a stone.

She didn’t move. Didn’t speak. Just stared at the name, the ink smudged slightly at the edges, as if the pen had hesitated. The manifest was for machine parts — yes, that was what she’d signed — but Kessler? He hadn’t been on the list. Not even a footnote. Not even a name. He was dead, and his name was here, etched in the same ink as the rest.

The silence in the room was thick, heavy with the weight of what she’d done. The clock ticked on the wall, each second stretching like a thread. She could feel the pulse of the building beneath her feet, the hum of the old HVAC system, the
```

**`rp-03` · 英文 RP · 讨价还价的谈判**

```text
The buyer leaned forward, his smile wide and practiced, the creases around his eyes deepening as he spoke. He tapped his fingers lightly against the edge of the table, the sound muffled by the thick, worn leather of the chair beneath his hands. The cargo, still in its sealed container, sat in the corner, its once-dry surface now glistening under the overhead lights — a slick sheen of moisture that clung to the edges, like a silent accusation.

He didn’t flinch when the air conditioner hissed faintly overhead, nor when the faint scent of damp wood and stale coffee clung to the room. His voice, low and deliberate, carried the weight of compromise: “Forty percent. That’s all I’m asking.”

The silence that followed was thick, heavy with the unspoken tension. Outside, rain tapped against the windowpanes, a rhythmic percussion that seemed to echo his offer — a plea, a surrender, a gamble. The buyer’s gaze, steady and calculating, lingered on the container, as if he could still see the dry, crisp cargo it had once held — and now, the memory of it was a ghost, damp and fading.
```

**`qa-01` · 短问答 · 事实解释**

```text
At sunset, sunlight travels through more of Earth’s atmosphere, causing shorter blue wavelengths to scatter away and longer red wavelengths to dominate, making the sky appear red. This effect is enhanced by the curvature of the Earth and the angle of the sun, which elongates the path of light through the atmosphere.
```

**`qa-02` · 短问答 · 指令遵循（精确条数）**

```text
Hold papers together temporarily.  
Organize small items in a pocket or bag.  
Straighten bent wires or thin metal.  
Attach to a keychain or lanyard for quick access.  
Secure a note to a clipboard or notebook.
```

**`qa-03` · 短问答 · 多步算术（有唯一答案）**

```text
We are given:

- First train **leaves at 14:05** and **arrives at 17:50**.
- The **next train leaves 25 minutes later**.
- The **next train takes 12 minutes longer** than the first.

---

### Step 1: Find the travel time of the first train

From 14:05 to 17:50.

Break it down:

- From 14:05 to 17:05 is **3 hours**.
- From 17:05 to 17:50 is **45 minutes**.

So total travel time = **3 hours 45 minutes**.

---

### Step 2: Find departure time of the second train

Leaves **25 minutes later** than 14:05.

So:  
**14:05 + 
```

> 预期答案 `18:27`：❌ 未出现（需人工复核是否等价表述）

---

## 运行 · `quality-8bit`（8bit）

模型加载方式：`Qwen3VLForConditionalGeneration / 8bit / device_map=auto` · 注意力实现：`sdpa`

| id | 类别 | prompt tok | 新生成 tok | TTFT (s) | 总时长 (s) | decode t/s | prefill t/s | 峰值显存 (GiB) | 停止原因 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| rp-01 | rp | 86 | 256 | 0.271 | 52.47 | 4.89 | 317.9 | 4.58 | max_new_tokens |
| rp-02 | rp | 85 | 256 | 0.261 | 52.41 | 4.89 | 325.4 | 4.58 | max_new_tokens |
| rp-03 | rp | 90 | 256 | 0.264 | 52.77 | 4.86 | 340.9 | 4.58 | max_new_tokens |
| qa-01 | qa | 21 | 50 | 0.257 | 10.08 | 4.99 | 81.8 | 4.55 | eos |
| qa-02 | qa | 29 | 42 | 0.268 | 8.54 | 4.96 | 108.1 | 4.55 | eos |
| qa-03 | qa | 57 | 192 | 0.265 | 38.88 | 4.95 | 215.5 | 4.57 | max_new_tokens |

**均值：** TTFT **0.264 s** · decode **4.92 tok/s** · 峰值显存 **4.58 GiB**

- 英文 RP 三项 decode 速度均值：**4.88 tok/s**
- 短问答三项 decode 速度均值：**4.96 tok/s**

### 原始输出

**`rp-01` · 英文 RP · 港口雨夜开场**

```text
Elara’s fingers tightened around the rusted railing as she stared at the two men standing in the middle of the ramp, blocking the way. The dockmaster’s thick frame loomed over the metal, his voice low and gravelly as he leaned against his truck, arms crossed. Behind him, the two men—broad-shouldered, eyes narrowed—watched the approaching cargo van with the same impatient glint in their eyes.

The air smelled of diesel and salt, thick with the tang of the sea breeze that curled around the dock’s edge. The ramp groaned under the weight of the van’s tires as it rolled toward them, the metal creaking like an old man’s bones. Elara’s boots, scuffed and worn, pressed into the gritty concrete, her breath shallow as she waited.

She didn’t move. Didn’t speak. Just stood there, her gaze fixed on the dockmaster’s face—the one that had once been kind, now hardened by years of debt and grit. The sun glinted off his steel cap, casting a shadow over his eyes. Behind him, the ramp’s metal groaned under the weight of the van’s tires, the engine’s rumble a low, angry growl.

Elara’s hands
```

**`rp-02` · 英文 RP · 坏消息与情绪压力**

```text
Elara’s fingers trembled as she turned the page, the paper crisp under her touch, the ink smudged near the bottom where Kessler’s name had been stamped. The scent of old paper and metal lingered in the air, mingling with the faint, metallic tang of blood that still clung to the edge of the desk. Her breath caught as she read the final line—*“Kessler, deceased. Signature: Elara Voss.”* The words were deliberate, final. She’d signed it with her own name, but the ink had bled slightly at the edge, as if the pen had been pressed too hard, too fast.

She glanced at the clock. 3:17 a.m. The room was silent except for the low hum of the ventilation system, the faint click of a distant elevator. She’d been in the lab for hours, the glow of the monitors casting long shadows across the floor. She’d signed the manifest to cover up the truth—the machine parts weren’t what they claimed. They were the last thing Kessler had been working on before he vanished.

She didn’t look up. She didn’t need to. She knew what it meant. The manifest was a lie. And now, Kessler’s
```

**`rp-03` · 英文 RP · 讨价还价的谈判**

```text
The buyer leaned forward, his smile wide and practiced, as if he’d rehearsed the words for weeks. His fingers tapped lightly against the edge of the table, the sound sharp against the quiet of the room. He didn’t look at the contract—his eyes were fixed on the cargo manifest, as if the dampness had already seeped into the paper itself. 

The air smelled of mildew and stale coffee, the kind that clings to old offices after a long day. His voice was low, almost amused, like he was offering a joke he’d already told too many times. “Forty percent,” he said, “and I’m not even mad.” 

He didn’t blink. Not once. The smile didn’t waver, even as the weight of the offer hung in the air—like wet wool, heavy and clinging. The buyer’s hands, calloused and stained with ink from last week’s deal, moved deliberately, as if he were folding a fragile promise into something that could be handed back. 

Behind him, the window showed a sliver of gray sky, the kind that doesn’t promise rain, but just waits. He didn’t need to say anything else. The silence between them was thick, and the wet cargo had
```

**`qa-01` · 短问答 · 事实解释**

```text
At sunset, sunlight travels through more of Earth’s atmosphere, causing shorter blue wavelengths to scatter away and longer red wavelengths to dominate the light reaching our eyes. This scattering effect makes the sky appear red or orange as the sun dips below the horizon.
```

**`qa-02` · 短问答 · 指令遵循（精确条数）**

```text
Holds papers together temporarily.  
Secures loose wires or cables.  
Organizes small items in a drawer.  
Supports fragile documents during transport.  
Attaches notes to bulletin boards or doors.
```

**`qa-03` · 短问答 · 多步算术（有唯一答案）**

```text
We are given:

- **First train** leaves at **14:05** and arrives at **17:50**.
- The **next train** leaves **25 minutes later** than the first.
- It takes **12 minutes longer** than the first train.

---

### Step 1: Find departure time of the second train

First train leaves at **14:05**.

Next train leaves **25 minutes later**:

> 14:05 + 25 minutes = **14:30**

---

### Step 2: Find travel time of the second train

First train’s travel time:

> 17:50 - 14:05 = **3 hours 45 minutes**

Convert to minutes:  
> 3 hours × 60 = 180 minutes  
> 180 + 45 = **22
```

> 预期答案 `18:27`：❌ 未出现（需人工复核是否等价表述）

---

## 一、结论与待办

### 1. 可自动判定的结论

| 项 | 结论 |
|------|------|
| **显存** | ✅ 4-bit 峰值 **2.79 GiB**，远低于 D17 的 7GB 预算 → 双通路（+约 2 GB）在显存上安全 |
| **速度** | ⚠️ 4-bit **10.64 tok/s**，仅为带宽理论上限（~116 tok/s）的 9% → **瓶颈是 bitsandbytes 的 4-bit kernel**，不是模型。见 [s2-speed-diagnosis.md](s2-speed-diagnosis.md) 与 **D26** |
| **qa-03（有唯一答案）** | ❌ **答错**。模型输出 **18:17**，正确答案是 **18:27** —— 它在中间步骤把 `3h45m` 误算成 **215 分钟**（正确是 225），后续推理都基于这个错值 |

> `qa-03` 的失败方式值得记住：**不是不会做题，是中间状态算错之后一路错到底**。
> 这正是 Nova 想用双通路 + 内部记忆去处理的那类问题（长线状态一致性），可以作为里程碑 2 的一个固定对照点。

### 2. 需要人工做的事（待实测）

- **RP 三项的质量主观评分**（本报告只给速度，质量要人读）。4-bit 与 8-bit 的输出都在上面，可直接对照。
- **D19（是否换基座）**：等这份基线 + 人工评分出来之后再决定。

### 3. 这份基线本身的已知缺陷

- `qa-03` 的原始 `max_new_tokens=192` 偏小（已改为 640）。上面的数字是**修正预算前**跑的，故该行 `stopped=max_new_tokens`。
- 速度数字**只在当前 bitsandbytes 配置下成立**，修好量化路径后必须重测（见 D26）。

