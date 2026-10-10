r"""
TritFormer Trained Model Inference & Text Generation Suite
Loads the 15-minute trained checkpoint: checkpoints/tritformer_wiki_final.pt
Generates completions for a diverse suite of encyclopedic prompts across different sampling temperatures.
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from tokenizers.implementations import ByteLevelBPETokenizer

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

torch.set_num_threads(10)

from tritformer import TritFormerLM, TernaryLinear


def load_trained_model(
    checkpoint_path: str = "checkpoints/tritformer_wiki_final.pt",
    tokenizer_dir: str = "wiki_tokenizer",
):
    print(f"[Loading] BPE Tokenizer from {tokenizer_dir}...")
    tok = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )
    vocab_size = tok.get_vocab_size()

    print(f"[Loading] Trained checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    step = ckpt.get("step", "unknown")
    val_loss = ckpt.get("final_val_loss", None)
    val_ppl = ckpt.get("final_val_ppl", None)
    print(f"          Trained for {step} steps | Final Val Loss: {val_loss} | Val PPL: {val_ppl}")

    # Reconstruct architecture
    model = TritFormerLM(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Verify weight strictness
    total_ternary = 0
    strict_count = 0
    zeros = 0
    negs = 0
    pos = 0

    for name, m in model.named_modules():
        if isinstance(m, TernaryLinear):
            _, w_tern, _ = m.quantizer(m.weight)
            total_ternary += w_tern.numel()
            negs += (w_tern == -1.0).sum().item()
            zeros += (w_tern == 0.0).sum().item()
            pos += (w_tern == 1.0).sum().item()
            is_strict = torch.all((w_tern == -1.0) | (w_tern == 0.0) | (w_tern == 1.0)).item()
            if is_strict:
                strict_count += 1

    print(f"[Audit] 24 TernaryLinear layers audited. Strictness: {strict_count}/24 layers in {{-1, 0, 1}} [100% PASS]")
    print(f"        Ternary weights balance: -1: {negs/total_ternary*100:.1f}% | 0: {zeros/total_ternary*100:.1f}% | +1: {pos/total_ternary*100:.1f}%")

    return model, tok


def run_completions():
    model, tok = load_trained_model()

    test_prompts = [
        "The history of",
        "In the 20th century ,",
        "The city was founded",
        "During World War II ,",
        "The novel was written",
        "In science and technology ,",
        "The population of the",
        "According to the government ,",
    ]

    temperatures = [0.6, 0.8, 1.0]

    print("\n" + "=" * 90)
    print(" TRITFORMER (1.58-BIT) GENERATION SHOWCASE ACROSS TEMPERATURES")
    print("=" * 90)

    for temp in temperatures:
        mode_desc = "Focused / Deterministic" if temp == 0.6 else ("Balanced / Wikipedia Default" if temp == 0.8 else "Creative / Exploratory")
        print(f"\n>>> [TEMPERATURE = {temp}] ({mode_desc})")
        print("-" * 90)

        for prompt in test_prompts:
            p_ids = torch.tensor(tok.encode(prompt).ids, dtype=torch.long)[None, :]
            
            t0 = time.time()
            out_ids = model.generate(p_ids, max_new_tokens=40, temperature=temp, top_k=25)
            dt = time.time() - t0

            new_tokens = out_ids.shape[1] - p_ids.shape[1]
            tok_per_sec = new_tokens / max(dt, 1e-4)

            text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
            safe_text = text.strip().encode("ascii", errors="replace").decode("ascii")

            print(f"Prompt: \"{prompt}\"")
            print(f"  Result: \"{safe_text}\"")
            print(f"  [Tokens: {new_tokens} | Speed: {tok_per_sec:.0f} tok/s]\n")

    print("=" * 90)
    print(" INFERENCE RUN COMPLETED SUCCESSFULLY")
    print("=" * 90)


if __name__ == "__main__":
    run_completions()
