import time, torch, torch.nn as nn

torch.manual_seed(0)
DEV = "cuda"
D = 2560

def bench(fn, n=300, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6

x16 = torch.randn(1, D, dtype=torch.float16, device=DEV)
xbf = torch.randn(1, D, dtype=torch.bfloat16, device=DEV)

# --- fp16 baseline ---
lin16 = nn.Linear(D, D, bias=False).to(DEV, torch.float16)
with torch.no_grad():
    us16 = bench(lambda: lin16(x16))
wbytes16 = D * D * 2
print(f"fp16  nn.Linear  bs=1 : {us16:7.1f} us  eff-BW {wbytes16/us16/1e3:7.1f} GB/s")

# --- bnb 4bit ---
from bitsandbytes.nn import Linear4bit
lin4 = Linear4bit(D, D, bias=False, quant_type="nf4", compute_dtype=torch.float16).to(DEV)
with torch.no_grad():
    us4 = bench(lambda: lin4(x16))
print(f"bnb   Linear4bit bs=1 : {us4:7.1f} us  eff-BW {D*D*0.5/us4/1e3:7.1f} GB/s")

# --- torchao int4 (bf16) ---
import torchao
print("\ntorchao", torchao.__version__)
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only

for name, q in [("int4_weight_only(g=128)", int4_weight_only(group_size=128)),
                ("int4_weight_only(g=64)",  int4_weight_only(group_size=64)),
                ("int8_weight_only",        int8_weight_only())]:
    try:
        m = nn.Linear(D, D, bias=False).to(DEV, torch.bfloat16)
        quantize_(m, q)
        with torch.no_grad():
            us = bench(lambda: m(xbf))
        print(f"torchao {name:24s} : {us:7.1f} us")
    except Exception as e:
        print(f"torchao {name:24s} : FAILED -> {type(e).__name__}: {str(e)[:200]}")

# --- torchao int4 at bs=512 (prefill) ---
try:
    m = nn.Linear(D, D, bias=False).to(DEV, torch.bfloat16)
    quantize_(m, int4_weight_only(group_size=128))
    xb = torch.randn(512, D, dtype=torch.bfloat16, device=DEV)
    with torch.no_grad():
        usb = bench(lambda: m(xb), n=100)
    print(f"\ntorchao int4 bs=512  : {usb:7.1f} us -> {512*D*D*2/usb/1e6:.1f} TFLOPS")
except Exception as e:
    print("bs=512 failed:", e)
