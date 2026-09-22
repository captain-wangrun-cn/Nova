import json, time, torch, ctypes
from bitsandbytes import cextension
from bitsandbytes.nn import Linear4bit
from torch.profiler import profile, ProfilerActivity

raw = cextension.lib._lib
print("raw CDLL:", type(raw).__name__, getattr(raw, "_name", None))
for s in ["cgemm_4bit_inference_naive_fp16", "cquantize_blockwise_fp16_nf4", "cget_context",
          "get_compute_capabilities", "cdequantize_blockwise_fp16_nf4"]:
    print(f"  sym {s}: {hasattr(raw, s)}")

torch.manual_seed(0)
lin = Linear4bit(2560, 2560, quant_type="nf4", compute_dtype=torch.float16).cuda()
x = torch.randn(1, 2560, dtype=torch.float16, device="cuda")
with torch.no_grad():
    for _ in range(20):
        y = lin(x)
torch.cuda.synchronize()

# timing
N = 300
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N):
        lin(x)
torch.cuda.synchronize(); t1 = time.perf_counter()
us = (t1 - t0) / N * 1e6
print(f"\nbnb Linear4bit 2560x2560 bs=1 : {us:.1f} us/call")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        for _ in range(200):
            lin(x)
    torch.cuda.synchronize()
print("\n--- CUDA kernels (top 12) ---")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

# also a bigger GEMM to make CPU-fallback absurd
xb = torch.randn(512, 2560, dtype=torch.float16, device="cuda")
with torch.no_grad():
    for _ in range(5):
        lin(xb)
torch.cuda.synchronize()
N2 = 100
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N2):
        lin(xb)
torch.cuda.synchronize(); t1 = time.perf_counter()
us2 = (t1 - t0) / N2 * 1e6
flops = 512 * 2560 * 2560 * 2
print(f"\nbnb Linear4bit bs=512 : {us2:.1f} us/call -> {flops/us2/1e6:.1f} TFLOPS")
