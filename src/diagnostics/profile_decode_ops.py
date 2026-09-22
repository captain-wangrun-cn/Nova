import time, json, torch, collections
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, AutoTokenizer
from torch.profiler import profile, ProfilerActivity

REPO = "Qwen/Qwen3-VL-4B-Instruct"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForImageTextToText.from_pretrained(
    REPO,
    device_map="auto",
    dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
)
model.eval()
cfg = model.config
print("class:", type(model).__name__)
print("attn impl:", getattr(cfg, "_attn_implementation", None), getattr(model, "_attn_implementation", None))
print("footprint GiB:", round(model.get_memory_footprint()/1024**3, 2))
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
print("lm class:", type(lm).__name__)
print("layer0 attn class:", type(lm.layers[0].self_attn).__name__)

msgs = [{"role": "user", "content": "用两句话介绍一下你自己。"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

with torch.inference_mode():
    model.generate(**enc, max_new_tokens=8, do_sample=False)
torch.cuda.synchronize()

N = 40
with torch.inference_mode():
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=N, do_sample=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
n_new = out.shape[1] - enc["input_ids"].shape[1]
wall = t1 - t0
print(f"\ngenerate: {n_new} tok in {wall:.2f}s -> {n_new/wall:.2f} tok/s ; {wall/n_new*1000:.1f} ms/token")

# --- profiled decode ---
with torch.inference_mode():
    model.generate(**enc, max_new_tokens=4, do_sample=False)
torch.cuda.synchronize()
P = 16
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=P, do_sample=False)
    torch.cuda.synchronize()

ka = prof.key_averages()
evs = list(ka)
print(f"\n=== profiled {P} tokens ===")

def agg(keyfn):
    d = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for e in evs:
        k = keyfn(e.key)
        if k is None: continue
        d[k][0] += e.self_device_time_total / 1e3   # ms
        d[k][1] += e.self_cpu_time_total / 1e3
        d[k][2] += e.count
    return d

def bucket(k):
    if "scaled_dot_product_attention" in k or "flash" in k.lower() or "attention" in k.lower(): return "attention"
    if "gemm_4bit" in k or "bitsandbytes" in k or "cublas" in k or "cutlass" in k: return "linear/gemm"
    if k.startswith("aten::") or k.startswith("torch::") or k.startswith("cuda"): return "aten-other"
    return "other"

b = agg(bucket)
print("\n--- by bucket (CUDA ms / CPU ms / calls, per token) ---")
for k, v in sorted(b.items(), key=lambda x: -x[1][0]):
    print(f"  {k:14s} cuda={v[0]:8.2f}  cpu={v[1]:8.2f}  calls={v[2]:6d}   per-token: cuda={v[0]/P:6.2f}ms cpu={v[1]/P:6.2f}ms")

print("\n--- top 15 CUDA self time ---")
for e in sorted(evs, key=lambda e: -e.self_device_time_total)[:15]:
    print(f"  {e.key[:64]:64s} cuda={e.self_device_time_total/1e3:8.2f}ms cpu={e.self_cpu_time_total/1e3:7.2f}ms n={e.count}")

print("\n--- top 15 CPU self time ---")
for e in sorted(evs, key=lambda e: -e.self_cpu_time_total)[:15]:
    print(f"  {e.key[:64]:64s} cpu={e.self_cpu_time_total/1e3:8.2f}ms cuda={e.self_device_time_total/1e3:7.2f}ms n={e.count}")

tot_cpu = sum(e.self_cpu_time_total for e in evs)/1e3
tot_cuda = sum(e.self_device_time_total for e in evs)/1e3
print(f"\ntotal CPU {tot_cpu:.1f}ms  total CUDA {tot_cuda:.1f}ms  over {P} tokens")
print(f"per token: CPU {tot_cpu/P:.2f}ms  CUDA {tot_cuda/P:.2f}ms  (wall {wall/n_new*1000:.1f}ms)")
