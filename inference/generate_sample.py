import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tritformer_850m import TritFormer850M, TritFormer850MConfig

print("=" * 70, flush=True)
print("TRITFORMER-850M INFERENCE & TEXT GENERATION BENCHMARK", flush=True)
print("=" * 70, flush=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Inference Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Reconstruct Model Architecture
# With statically-compiled Triton Ternary GEMM kernels (triton_ternary_gemm_static),
# autoregressive token generation no longer freezes from dynamic autotuning recompilation!
cfg = TritFormer850MConfig(
    vocab_size=len(tokenizer),
    d_model=2048,
    n_layers=16,
    n_heads=16,
    head_dim=128,
    d_ff=5632,
    max_seq_len=2048,
    use_triton_gemm=True,
)

print(f"Building TritFormer-850M with {cfg.vocab_size} vocab, {cfg.n_layers} layers...", flush=True)
model = TritFormer850M(cfg)

# 3. Load Trained Checkpoint
ckpt_path = "/home/ubuntu/demsni/checkpoints/checkpoint_tritformer_850m_latest.pt"
print(f"Loading trained weights from {ckpt_path}...", flush=True)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device=device, dtype=torch.bfloat16)
model.eval()

step_trained = ckpt.get("step", "N/A")
tokens_trained = ckpt.get("tokens_seen", 0)
loss_achieved = ckpt.get("loss", "N/A")
print(f"Successfully loaded checkpoint: Step {step_trained} | Tokens: {tokens_trained:,} | Loss: {loss_achieved:.4f}", flush=True)

# 4. Text Generation Function
@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    
    generated = input_ids
    for step in range(max_new_tokens):
        cond = generated[:, -2048:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(cond)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

# 5. Prompts Test
prompts = [
    "The fundamental architecture of deep learning models",
    "In mathematics, the theory of",
    "Recent advances in neural network optimization have shown that",
]

print("\n--- SAMPLE GENERATIONS ---", flush=True)
for i, p in enumerate(prompts, 1):
    print(f"\n[Generating for Prompt {i}...]", flush=True)
    output = generate(model, p, max_new_tokens=40, temperature=0.7, top_k=40)
    print(f"[Prompt {i}]: '{p}'", flush=True)
    print(f"[Generated Continuation]:\n{output}", flush=True)

print("\n" + "=" * 70, flush=True)
print(">> INFERENCE & GENERATION TEST COMPLETE! <<", flush=True)
print("=" * 70, flush=True)
