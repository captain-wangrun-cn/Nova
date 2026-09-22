import time, torch, json
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, AutoTokenizer

REPO = "Qwen/Qwen3-VL-4B-Instruct"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForImageTextToText.from_pretrained(
    REPO, device_map="auto", dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
model.eval()

msgs = [{"role": "user", "content": "用两句话介绍一下你自己。"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
ids = enc["input_ids"]

with torch.inference_mode():
    model.generate(**enc, max_new_tokens=8, do_sample=False)
torch.cuda.synchronize()

# --- 1) CPU 发射时间 vs wall 时间 ---
import torch.nn.functional  # noqa
past = None
step_ids = torch.tensor([[tok("你好")["input_ids"][-1]]], device="cuda")

def one_step(input_ids, cache):
    with torch.inference_mode():
        return model(input_ids=input_ids, past_key_values=cache, use_cache=True)

# 先建好 cache
with torch.inference_mode():
    o = model(input_ids=ids, use_cache=True)
cache = o.past_key_values
o = None
torch.cuda.synchronize()

# CPU 发射（不 sync）
N = 30
t0 = time.perf_counter()
outs = []
for _ in range(N):
    outs.append(one_step(step_ids, cache))
t_cpu = (time.perf_counter() - t0) / N * 1000
torch.cuda.synchronize()
t_wall = (time.perf_counter() - t0) / N * 1000
print(f"CPU enqueue-only: {t_cpu:7.2f} ms/step")
print(f"wall (with GPU) : {t_wall:7.2f} ms/step")
print(f"-> GPU busy approx = wall - cpu = {t_wall-t_cpu:7.2f} ms  => {'CPU-BOUND' if t_cpu > 0.8*t_wall else 'GPU-BOUND'}")
print(f"-> implied max tok/s if CPU overhead removed: {1000/t_cpu:.1f}")
print(f"-> implied max tok/s if GPU work only: {1000/(t_wall-t_cpu) if t_wall>t_cpu else float('inf'):.1f}")

# --- 2) torch.compile 可用性（无 triton 时） ---
import torch._dynamo as dynamo
print("\ndynamo available:", True)
try:
    import triton
    print("triton:", triton.__version__)
except Exception as e:
    print("triton: NOT AVAILABLE ->", e)

def f(x):
    return (x * 2 + 1).relu()

try:
    cf = torch.compile(f, fullgraph=True)
    x = torch.randn(8, device="cuda")
    y = cf(x)
    print("torch.compile default backend: OK (silent fallback to eager if no triton)")
    print("  dynamo counters:", json.dumps({k: v for k, v in dynamo.utils.counters.get("frames", {}).items()}, default=str)[:300])
except Exception as e:
    print("torch.compile FAILED ->", type(e).__name__, str(e)[:300])

# --- 3) 纯 GPU 时间下界：把权重全部读一遍的时间 ---
w = sum(p.numel() for p in model.parameters())
print(f"\nparams={w/1e9:.3f}B  (4bit weight bytes ~{w*0.5/1e9:.2f} GB)")
for bw in (234, 352, 500):
    print(f"  at {bw} GB/s -> linear-only floor {w*0.5/bw/1e9*1000:.1f} ms/token = {bw/(w*0.5/1e9):.0f} tok/s")
