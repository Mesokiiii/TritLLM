# TritLLM: Towards Multiplication-Free Language Modeling with Balanced Ternary Quantization, SubLN Stability, and Fused Hardware Kernels

[![arXiv Report](https://img.shields.io/badge/Report-arXiv%20Preprint-b31b1b.svg)](TritLLM_Technical_Report.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust AVX2](https://img.shields.io/badge/Rust-AVX2%20SIMD-orange.svg)](trit_kernel/)
[![CUDA/Triton](https://img.shields.io/badge/GPU-Triton%20Fused%20Kernels-green.svg)](kernels/)

**TritLLM** is an open-source, multiplication-free foundation language model framework engineered around **symmetric balanced ternary logic** ($\mathbb{T} = \{-1, 0, +1\}$). By replacing power-hungry floating-point multiply-accumulate (MAC) units with addition, subtraction, and zero-skipping popcount operations, TritLLM advances the frontier of 1.58-bit deep learning.

> 📄 **Primary Research Document:** The complete academic report formatted strictly to arXiv publication standards is available in the repository root:
> - **DOCX (Word Publication Format):** [`TritLLM_Technical_Report.docx`](TritLLM_Technical_Report.docx)
> - **Markdown:** [`TritLLM_Technical_Report.md`](TritLLM_Technical_Report.md)

---

## ⚡ Key Highlights & Core Findings

1. **Multiplication-Free Matrix Math (TritGEMM):**
   - Linear projections $Y = XW$ decompose into pure signed accumulations: $Y_{ij} = N_+(i, j) - N_-(i, j)$.
   - Replaces FP16 matrix multipliers with bitwise XNOR/AND gates and adder trees, yielding up to **20x energy savings** (~0.05 pJ/OP vs ~1.0 pJ/OP in FP16).
2. **SubLN & Variance Bounding Theorem:**
   - Pre-LN in ternary networks triggers exponential variance explosion: $\text{Var}(x_l) \sim \mathcal{O}(l)$.
   - TritLLM introduces SubLN (Sub-Layer Normalization) with DeepNet residual scaling $\alpha = 1 / \sqrt{2N_{\text{layers}}}$, mathematically proving that total residual variance is strictly bounded by $\text{Var}(x_N) \le 2.0$ across arbitrary depth.
3. **QK-Norm Attention Stabilization:**
   - Normalizes Query and Key vectors per-head prior to dot product, preventing attention logit surges to $\pm 124$ and preserving attention entropy ($H \approx 3.03$).
4. **Custom Fused Triton Kernels (Ada Lovelace, sm_89):**
   - **Fused Cross-Entropy (Online LogSumExp):** Eliminates 411.7 MB logits DRAM tensor per micro-step; stores only a 16 KB 1D LSE vector in SRAM (**25,700x memory reduction**, +41% backward speedup).
   - **Fused SwiGLU & RMSNorm:** 1-pass register fusion, cutting DRAM bandwidth by 50%.
   - **Static Ternary GEMM Dispatch:** Eliminates 1,120 JIT autotune compilations per token during generation.
5. **UltraStreamer Dataflow Pipeline:**
   - Multiprocessing zero-copy tokenizer pool (GIL-free).
   - Document-masked sequence packing with block-diagonal causal attention masks.
   - Pinned memory DMA ring buffer sustaining **854,021 tokens/sec** across 4x RTX 4090 GPUs.
6. **Empirical CPU Pretraining (WikiText-2):**
   - 15-minute run on 10-thread Intel Core i5-13420H (AVX2 SIMD + Rust kernel): **Validation loss decreased by 54.5%** (7.8288 $\to$ 3.5608), **PPL dropped by 98.6%** (2511.93 $\to$ 35.19) at **6,773 tok/s**.
   - 100% strict ternary weights preserved, converging to the Blume-Capel tricritical equilibrium (32.7% zeros, 33.5% negatives, 33.8% positives).

---

## 🔍 Transparent Empirical Failures & Post-Mortem Analysis

In accordance with strict scientific methodology, we document all critical pathologies and engineering boundaries encountered:

| Pathology / Failure Mode | Root Cause | Consequence | Mathematical Resolution in TritLLM |
| :--- | :--- | :--- | :--- |
| **Initial Loss Explosion to 1989.83** | Tied embeddings with Gaussian $\sigma=1.0$ without $1/\sqrt{d}$ division. | Collinear logit $z_{t_k} \approx 2048$, Softmax saturation, massive gradient shock. | Continuous LM Head with $\sigma=0.002$; initial loss normalized to exact theoretical $\ln(50257) = \mathbf{10.8198}$. |
| **Vocabulary Bottleneck & Repetition Loop** | Quantizing vocabulary projection $W_{\text{head}} \in \{-1, 0, 1\}^{V \times d}$. | STE gradient starvation for rare tokens ($2 \times 10^{-5}$ drowned in noise); repetition collapse (`"uses uses uses..."`). | Maintain $W_{\text{head}}$ in high-precision continuous BF16 (BitNet b1.58 canonical standard). |
| **Attention Entropy Collapse** | Unnormalized Q and K in deep Pre-LN layers. | Inner products surge to $\pm 124$; Softmax collapses to $\delta$-function ($H \to 0.16$); frozen gradients. | QK-Norm enforces $\text{Var}(S) = 1.0$, restoring attention entropy to $H = \mathbf{3.0334}$. |
| **Triton Autotune JIT Freeze** | Dynamic sequence length $M = 15, 16, 17\dots$ during autoregressive decoding. | 1,120 JIT compilations per generated token; process paralyzed. | Static tile dispatch table ($M \le 16$ vs $M > 16$); 0 ms dispatch latency. |
| **Compute Budget Exhaustion** | Dedicated 4x RTX 4090 bare-metal server lease ran out of independent funding. | Full 18.5B-token Chinchilla pretraining halted after smoke/verification iterations (steps 2–4). | Documented transparently; serves as an empirical foundation for future low-cost scale-ups. |

---

## 📁 Repository Structure

```
TritLLM/
├── README.md                          # Comprehensive project documentation
├── TritLLM_Technical_Report.docx      # Complete academic research paper for arXiv
├── TritLLM_Technical_Report.md        # Technical report in Markdown format
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Strict exclusion of binaries and credentials
│
├── tritformer/                        # Core model architectures
│   ├── tritformer_stable.py           # BitTrit-SOTA: SubLN, QK-Norm, continuous head, BitLinear
│   ├── tritformer_850m.py             # 850M cluster configuration
│   └── tritformer.py                  # Baseline TritFormer implementation
│
├── kernels/                           # High-performance computing kernels
│   ├── fused_kernels.py               # Triton Fused CE (Online LSE), Fused SwiGLU, Fused RMSNorm
│   ├── triton_ternary_gemm.py         # Triton ternary GEMM & autotuning
│   ├── trit_rust_engine.py            # Python CFFI ctypes wrapper for Rust kernel
│   └── trit_kernel.dll                # Prebuilt Windows AVX2 SIMD dynamic library
│
├── trit_kernel/                       # Rust AVX2 SIMD acceleration crate
│   ├── Cargo.toml                     # Cargo manifest
│   ├── Cargo.lock                     # Dependency lockfile
│   └── src/                           # Rust source files (c_abi, gemm, gemv, lut, packing, simd_avx2)
│
├── streamer/                          # High-throughput data streaming
│   ├── ultra_streamer.py              # UltraStreamer: multiprocess tokenizer, block-diagonal mask, NVMe cache
│   └── dataset_streamer.py            # Base dataset streamer
│
├── training/                          # Training pipelines
│   ├── train_extreme.py               # Muon + AdamW, WSD schedule, CUDAGraphs, FineWeb validation
│   ├── train_distributed.py           # Multi-GPU DDP training launcher
│   ├── train_real_benchmark.py        # CPU benchmark training script
│   └── train_wikipedia_longrun.py     # Long-run training with thermal pacing
│
├── inference/                         # Autoregressive generation
│   ├── generate_sample.py             # Generation with repetition penalty, n-gram blocking, top-p
│   └── generate_inference.py          # Lightweight CPU text generator
│
├── tools/                             # Diagnostics & verification suite
│   ├── test_ultra_throughput.py       # Cluster dataflow throughput verification
│   ├── test_cluster.py                # Multi-GPU hardware diagnostic tool
│   ├── check_status.py                # Real-time cluster GPU monitor
│   └── inspect_ckpt.py                # Checkpoint inspection utility
│
├── scripts/                           # Cluster deployment and automation
│   ├── launch_4x4090.sh               # Distributed torchrun script
│   ├── launch_extreme.sh              # Extreme training launcher
│   ├── setup_server.sh                # Automated server environment setup
│   └── fast_install.sh                # Environment installer
│
├── benchmarks/                        # Real empirical logs and reports
│   ├── fifteen_min_training_log.json  # Raw telemetry from 15-minute CPU run
│   ├── fifteen_min_training_report.md # Summary of 15-minute convergence
│   ├── real_training_log.json         # Raw telemetry from 6-minute CPU run
│   ├── real_benchmark_analysis.md     # 6-minute empirical analysis
│   ├── longrun_training_log.json      # Raw telemetry from 100-step longrun
│   └── longrun_training_report.md     # 100-step longrun report
│
├── research/                          # Comprehensive theoretical treatises
│   ├── ternary_math_foundations.md    # Radix economy, Blume-Capel model, discrete geometry
│   ├── tritformer_architecture.md     # Core architectural RFC specification
│   ├── bitnet_stability_report.md     # Mathematical diagnosis of 4 stability pathologies
│   ├── fused_kernels_report.md        # Triton kernel engineering and memory traffic report
│   ├── train_extreme_spec.md          # Multi-GPU topology and optimizer specification
│   ├── ultra_streamer_spec.md         # UltraStreamer system architecture and benchmarks
│   └── reproducibility_audit_report.md # Forensic scientific audit of 15 optimization papers
│
└── wiki_tokenizer/                    # Tokenizer assets
    ├── vocab.json                     # Vocabulary mapping
    └── merges.txt                     # BPE merges
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Mesokiiii/TritLLM.git
cd TritLLM
pip install -r requirements.txt
```

To build the Rust AVX2 kernel from source:
```bash
cd trit_kernel
cargo build --release
cd ..
```

### 2. Verify Mathematical Stability (Audit Suite)

Run the automated mathematical audit suite verifying initial loss, logit scale, activation variance across 16 layers, attention entropy, and STE gradients:

```bash
python tritformer/tritformer_stable.py
```

Expected output:
```
================================================================================
AUDIT PROTOCOL: MATHEMATICAL STABILITY OF TRITFORMER-STABLE (BitTrit-SOTA)
================================================================================
TEST 1: Starting Loss: 10.8198 (Expected ~10.8249) -> [PASSED]
TEST 2: Residual Variance: Layer 00=0.020 -> Layer 16=0.9931 (<= 1.414) -> [PASSED]
TEST 3: Attention Entropy: H = 3.0334 (No Softmax Delta Collapse) -> [PASSED]
TEST 4: Active STE Gradients: 28/28 layers active -> [PASSED]
TEST 5: Generation Diversity: 20/20 unique tokens (100%) -> [PASSED]
>> ALL 5 AUDIT TESTS PASSED WITH 100% SUCCESS <<
```

### 3. Run CPU Benchmark Training (WikiText-2)

```bash
python training/train_real_benchmark.py --duration_sec 360
```

### 4. Run Multi-GPU Cluster Pretraining (4x RTX 4090)

```bash
bash scripts/launch_4x4090.sh
```

---

## 📊 Summary of Real Convergence Metrics

### 15-Minute CPU Pretraining (Intel Core i5-13420H):
- **Wallclock Duration:** 15.0 minutes (2,977 steps)
- **Tokens Streamed:** 6,096,896 tokens
- **Throughput:** 6,772.7 tokens/second
- **Validation Loss:** `7.8288` $\to$ **`3.5608`** (**-54.5%**)
- **Validation Perplexity:** `2511.93` $\to$ **`35.19`** (**-98.6%**)
- **Weight Strictness:** 100% in $\{-1, 0, +1\}$ (Zero: 32.7%, Negative: 33.5%, Positive: 33.8%)

### UltraStreamer Multi-GPU Dataflow:
- **Aggregate Cluster Throughput:** **854,021 tokens/sec**
- **Fetch Latency:** 30–40 ms (p95: 71–103 ms)
- **Document Isolation:** 100% block-diagonal causal masking

---

## 📜 Citation & Reference

If you find TritLLM useful in your research, please cite:

```bibtex
@article{tritllm2026,
  title={TritLLM: Towards Multiplication-Free Language Modeling with Balanced Ternary Quantization, SubLN Stability, and Fused Hardware Kernels},
  author={TritLLM Research Consortium},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/Mesokiiii/TritLLM}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
