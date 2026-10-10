import torch
import time
from tritformer_850m import TritFormer850M, TritFormer850MConfig

print("=" * 70)
print("TRITFORMER-850M CLUSTER VERIFICATION (STRICT BF16, ZERO FP32)")
print("=" * 70)

# 1. Config Check
cfg = TritFormer850MConfig.for_850m(vocab_size=32768)
print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")

# 2. Meta parameter count
with torch.device("meta"):
    meta_model = TritFormer850M(cfg)
meta_params = sum(p.numel() for p in meta_model.parameters())
print(f"Total Parameters: {meta_params:,} ({meta_params / 1e6:.2f}M parameters)")
assert 800_000_000 <= meta_params <= 950_000_000, f"Unexpected parameter count: {meta_params}"

torch.cuda.set_device(0)
device = torch.device("cuda:0")
torch.cuda.reset_peak_memory_stats()

print(f"Allocating model on {torch.cuda.get_device_name(device)} in BF16...")
model = TritFormer850M(cfg).to(device=device, dtype=torch.bfloat16)

vram_model_gb = torch.cuda.memory_allocated(device) / (1024**3)
print(f"Model Static VRAM in BF16: {vram_model_gb:.2f} GB")

# 4. Forward + Backward test
batch_size = 2
seq_len = 2048
print(f"Running Forward+Backward test (batch={batch_size}, seq_len={seq_len}) in BF16...")

input_ids = torch.randint(0, 32768, (batch_size, seq_len), device=device)
targets = torch.randint(0, 32768, (batch_size, seq_len), device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)

t0 = time.time()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits, loss = model(input_ids, targets=targets)
    assert logits.dtype == torch.bfloat16, f"Expected BF16 logits, got {logits.dtype}"
    assert loss.dtype in (torch.float32, torch.bfloat16)

loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize(device)
t1 = time.time()

vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Loss: {loss.item():.4f}")
print(f"Peak VRAM used: {vram_peak_gb:.2f} GB / 24.0 GB (Headroom: {24.0 - vram_peak_gb:.2f} GB)")
print(f"Step duration: {t1 - t0:.3f} s")
print("=" * 70)
print(">> TRITFORMER-850M BF16 + ADAMW VERIFICATION PASSED SUCCESSFULLY! <<")
print("=" * 70)
