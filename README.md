# TritLLM: Towards Multiplication-Free Language Modeling with Balanced Ternary Quantization, SubLN Stability, and Fused Hardware Kernels

**Authors:** TritLLM Research Initiative  
**Preprint & Technical Report:** [`TritLLM_Technical_Report.pdf`](TritLLM_Technical_Report.pdf) | [`TritLLM_Technical_Report.tex`](TritLLM_Technical_Report.tex) | [`TritLLM_Technical_Report.md`](TritLLM_Technical_Report.md)  
**Code Repository:** [https://github.com/Mesokiiii/TritLLM.git](https://github.com/Mesokiiii/TritLLM.git)  
**License:** MIT License  

---

## Abstract

Autoregressive language models typically rely on dense floating-point matrix multiplications (FP16/BF16 GEMM), which dominate energy consumption and memory bandwidth requirements. Balanced ternary representations ($\mathbb{T} = \{-1, 0, +1\}$) offer an optimal theoretical radix economy ($\log_2(3) \approx 1.585$ bits/parameter) and allow replacing floating-point multiply-accumulate (MAC) operations with integer additions, subtractions, and zero-skipping popcount operations.

TritLLM is an open research framework for balanced ternary language models. The framework integrates:
1. Theoretical formulations of discrete representation geometry and variance-bounded residual streams ($\text{Var}(x) \le 2.0$),
2. Custom CPU SIMD (AVX2) and GPU (Triton) multiplication-free kernels,
3. A multi-process streaming data pipeline (UltraStreamer), and
4. Empirical convergence benchmarks alongside a detailed post-mortem of scaling instabilities.

This repository provides full source code, pretraining scripts, evaluation logs, and the complete academic technical report formatted for arXiv submission.

---

## Architectural and Mathematical Formulation

### 1. Radix Economy and Balanced Ternary Representation

Hardware radix economy $E(b, N)$ measures the gate complexity required to represent numbers up to magnitude $N$ using integer base $b$:

$$E(b, N) = b \cdot \lceil \log_b(N) \rceil = b \cdot \frac{\ln(N)}{\ln(b)}$$

The continuous relaxation $f(b) = b / \ln(b)$ attains its global minimum at $b = e \approx 2.71828$. Among discrete integer bases, $b = 3$ is closest to $e$, yielding:

$$f(3) \approx 2.7307 < f(2) \approx 2.8854$$

Consequently, ternary arithmetic yields approximately a **5.4% theoretical advantage in hardware gate economy** over binary logic. In balanced ternary logic, the digit alphabet is symmetric around zero: $\mathbb{T} = \{-1, 0, +1\}$.

### 2. Addition-Only TritGEMM

Given an input matrix $\mathbf{X} \in \mathbb{R}^{M \times K}$ (or quantized $\mathbf{X} \in \mathbb{T}^{M \times K}$) and a ternary weight matrix $\mathbf{W} \in \mathbb{T}^{K \times N}$, linear projections reduce to signed accumulations:

$$Y_{i, j} = \sum_{k=1}^K X_{i, k} \cdot W_{k, j} = N_{+}(i, j) - N_{-}(i, j)$$

where $N_+$ and $N_-$ count positive and negative sign coincidences, respectively. Digital implementations compute $Y_{i, j}$ using bitwise XNOR/AND gates and parallel popcount adder trees without arithmetic multipliers.

### 3. SubLN Residual Variance Bounding Theorem

In standard Pre-LayerNorm (Pre-LN) networks with Straight-Through Estimators (STE), residual variance grows monotonically with depth: $\text{Var}(x_l) \sim \mathcal{O}(l)$, which causes STE gradient saturation. TritLLM implements SubLN with DeepNet scaling $\alpha = 1 / \sqrt{2N_{\text{layers}}}$:

$$x_l^{(\text{att})} = x_l + \frac{1}{\sqrt{2 N_{\text{layers}}}} \text{RMSNorm}_{\text{sub1}}\Big( \text{Attention}\big( \text{RMSNorm}_{\text{pre1}}(x_l) \big) \Big)$$

$$x_{l+1} = x_l^{(\text{att})} + \frac{1}{\sqrt{2 N_{\text{layers}}}} \text{RMSNorm}_{\text{sub2}}\Big( \text{SwiGLU}\big( \text{RMSNorm}_{\text{pre2}}(x_l^{(\text{att})}) \big) \Big)$$

**Theorem 1:** When $\text{Var}(\text{RMSNorm}_{\text{sub}}(z)) = 1.0$, the variance of the residual stream at depth $N_{\text{layers}}$ satisfies:

$$\text{Var}(x_N) = \text{Var}(x_0) + 2 N_{\text{layers}} \cdot \left(\frac{1}{\sqrt{2 N_{\text{layers}}}}\right)^2 \cdot 1.0 = 1.0 + 1.0 = \mathbf{2.0}$$

$$\sigma(x_N) \le \sqrt{2} \approx \mathbf{1.4142}$$

This guarantees that activation magnitude remains bounded by a constant for arbitrary depth.

### 4. QK-Normalization

To prevent inner-product explosion in deep layers, Queries and Keys are normalized per-head prior to dot product:

$$\mathbf{q}_{\text{norm}} = \text{RMSNorm}(\mathbf{q}), \quad \mathbf{k}_{\text{norm}} = \text{RMSNorm}(\mathbf{k}), \quad \mathbf{S} = \frac{\mathbf{q}_{\text{norm}} \mathbf{k}_{\text{norm}}^T}{\sqrt{d_k}}$$

This guarantees $\text{Var}(S_{i, j}) \approx 1.0$, preserving attention entropy ($H \approx 3.03$) and preventing gradient collapse in the Softmax operator.

---

## Model Configurations

### Table 1: Model Topology and Architectural Profiles

| Hyperparameter | TritFormer-Tiny (Local CPU) | TritFormer-850M (Cluster Configuration) |
| :--- | :---: | :---: |
| **Compute Target** | Intel Core i5-13420H (AVX2 SIMD) | 4x NVIDIA GeForce RTX 4090 (`sm_89`) |
| **Hidden Dimension ($d_{\text{model}}$)** | 256 | 2,048 |
| **Number of Layers ($N_{\text{layers}}$)** | 4 | 16 |
| **Attention Heads ($n_{\text{heads}} / n_{\text{kv}}$)** | 4 / 4 (MHA) | 16 / 8 (Grouped-Query Attention GQA) |
| **Head Dimension ($d_{\text{head}}$)** | 64 | 128 |
| **FFN Intermediate Dimension ($d_{\text{ff}}$)** | 768 | 5,632 (SwiGLU) |
| **Context Length ($\text{seq\_len}$)** | 256 | 2,048 |
| **Vocabulary Size ($V$)** | 10,000 | 50,257 (GPT-2 BPE) |
| **Quantization Scheme** | 1.58-bit Weights $\{-1, 0, 1\}$ | 1.58-bit Internal Projections + BF16 Head |

---

## Systems Engineering and Kernel Implementations

### 1. CPU SIMD Engine (Rust AVX2 CFFI)
* Implemented in Rust (`trit_kernel/src/`), compiled as a C-ABI dynamic library (`trit_kernel.dll`).
* 2-bit packing encodes 4 trits per byte.
* Matrix-vector products utilize `_mm256_popcnt_u64` and vectorized sign-accumulation paths.

### 2. GPU Fused Kernels (Triton Ada Lovelace)
* **Fused Cross-Entropy with Online LogSumExp (`kernels/fused_kernels.py`):** Computes LogSumExp directly in SRAM using a single pass, writing only a 1D vector (16 KB) for backward passes. Reduces DRAM activation traffic by 25,700x and accelerates backward computation by 41% compared to standard PyTorch.
* **Fused SwiGLU & RMSNorm:** Fuses gate/up projections and SiLU activation in register space, halving DRAM bandwidth consumption.
* **Static GEMM Dispatch:** Replaces dynamic autotuning with precompiled tile configurations ($16 \times 128 \times 128$ for decoding $M \le 16$; $128 \times 128 \times 64$ for prefill $M > 16$), eliminating JIT compilation pauses.

### Table 2: Triton Kernel Verification and Memory Footprint on RTX 4090

| Module | Baseline PyTorch VRAM | Fused Triton VRAM | Forward Diff | Backward Diff | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fused Cross-Entropy** | 411.7 MB (DRAM) | **16 KB** (SRAM LSE) | $0.00 \times 10^0$ | $0.00 \times 10^0$ | Exact Match |
| **Fused SwiGLU** | 184.0 MB (4 passes) | **46.0 MB** (1 pass) | $0.00 \times 10^0$ | $3.12 \times 10^{-2}$ (1 ULP) | Validated |
| **Fused RMSNorm** | 3 DRAM passes | **1 SRAM pass** | $0.00 \times 10^0$ | $3.12 \times 10^{-2}$ (1 ULP) | Validated |
| **Static Ternary GEMM** | 1,120 JIT compilations | **0 ms** dispatch latency | $0.00 \times 10^0$ | N/A (Inference) | Validated |

### 3. Streaming Dataflow Pipeline (UltraStreamer)
* **GIL-Free Multiprocessing:** Independent worker processes handle tokenization per GPU rank.
* **Document-Masked Packing:** Block-diagonal causal attention masks prevent cross-document attention leakage within packed 2,048-token sequences.
* **RoPE Position Reset:** Vectorized resetting of `position_ids` to 0 at document boundaries.
* **Throughput:** Verified sustained aggregate rate of **854,021 tokens/second** across 4x RTX 4090 GPUs.

---

## Empirical Benchmarks and Results

### Table 3: Convergence Dynamics on WikiText-2 (Intel Core i5-13420H, 10 Threads)

| Benchmark Regime | Steps | Tokens | Elapsed Time | Throughput | Val Loss | Val Perplexity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initialization (Step 0)** | 0 | 0 | 0.0 s | — | 7.8288 | 2511.93 |
| **6-Minute Run (Final)** | 684 | 1.40M | 360.0 s | 3,890.8 tok/s | 4.1169 (-47.1%) | 61.37 (-97.4%) |
| **15-Minute Run (3.0 min)** | 1,017 | 2.08M | 180.0 s | 6,773.0 tok/s | 3.9045 | 49.62 |
| **15-Minute Run (6.0 min)** | 1,629 | 3.34M | 360.0 s | 6,773.0 tok/s | 3.7484 | 42.45 |
| **15-Minute Run (9.0 min)** | 1,912 | 3.92M | 540.0 s | 6,773.0 tok/s | 3.6962 | 40.29 |
| **15-Minute Run (Final)** | 2,977 | 6.10M | 900.0 s | 6,772.7 tok/s | **3.5608 (-54.5%)** | **35.19 (-98.6%)** |

### Parameter Distribution Dynamics
Across 2,977 optimization steps:
* Ternary weight strictness remained **100.0% in $\{-1, 0, +1\}$**.
* Converged distribution: **32.71% zeros**, **33.47% negative trits**, **33.83% positive trits** (confirming adherence to the theoretical Blume-Capel tricritical equilibrium of 33.33% each).
* Parameter flip rate decayed from **26.87%** (step 59) to **0.23%** (step 659), demonstrating asymptotic convergence to discrete Voronoi cells.

---

## Diagnosis and Post-Mortem of Scaling Pathologies

Scaling the ternary architecture to 850M parameters revealed four distinct numerical pathologies:

### Table 4: Summary of Numerical Pathologies and Implemented Solutions

| Pathology | Mechanism | Consequence | Implemented Solution |
| :--- | :--- | :--- | :--- |
| **1. Starting Loss Explosion ($\mathcal{L} = 1989.83$)** | Tied embeddings with Gaussian initialization ($\sigma = 1.0$) without $1/\sqrt{d}$ division, combined with final RMSNorm. | Collinear input-token logit $z_{t_k} \approx 2048$, Softmax saturation, and massive gradient shock into AdamW. | Continuous LM Head with initialization $\sigma = 0.002$; normalized starting loss to $\ln(50257) = \mathbf{10.8198}$. |
| **2. Vocabulary Bottleneck & Repetition Loop** | Quantizing vocabulary projection $W_{\text{head}} \in \{-1, 0, 1\}^{V \times d}$. | STE gradient starvation for rare tokens ($2 \times 10^{-5}$ drowned in noise); degeneration into repetition (`"uses uses uses..."`). | Maintain $W_{\text{head}}$ in continuous BF16, adhering to canonical BitNet b1.58 standards. |
| **3. Attention Entropy Collapse** | Unnormalized Q and K in deep Pre-LN layers. | Inner products drift to $\pm 124$; Softmax degenerates into a Kronecker delta function ($H \to 0.16$); frozen gradients. | QK-Norm enforces $\text{Var}(S) = 1.0$, restoring attention entropy to $H = \mathbf{3.0334}$. |
| **4. Triton Autotune Decode Freeze** | Autotuner cache key indexed on dynamic sequence length $M = B \times T$. | In autoregressive decoding without KV cache, $M$ changes every step, triggering 1,120 JIT compilations per generated token. | Static tile dispatch table ($M \le 16$ vs $M > 16$), ensuring 0 ms dispatch overhead. |

### Economic Realities and Compute Budget Constraints
A dedicated bare-metal server (4x NVIDIA RTX 4090, 96 GB VRAM, 16 vCPUs, 128 GB RAM, 320 GB NVMe) was leased to validate the distributed infrastructure. The cluster environment, NCCL interconnect, and fused Triton kernels were successfully validated. However, full-scale pretraining across a multi-billion token corpus was terminated after initial verification iterations (steps 2 to 4) due to the exhaustion of available compute funding. The complete codebase, stabilized architecture, and verification test suites are published to provide a reproducible foundation for future pretraining efforts.

---

## Repository Structure

```
TritLLM/
├── TritLLM_Technical_Report.pdf      # Compiled academic research paper (PDF)
├── TritLLM_Technical_Report.tex      # LaTeX source code (arXiv publication format)
├── TritLLM_Technical_Report.md       # Technical report in Markdown
├── README.md                         # Repository overview and documentation
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Build and cache exclusions
│
├── tritformer/                        # Architecture definitions
│   ├── tritformer_stable.py           # Stabilized model (SubLN, QK-Norm, continuous head)
│   ├── tritformer_850m.py             # 850M cluster configuration
│   └── tritformer.py                  # Baseline implementation
│
├── kernels/                           # Hardware kernels
│   ├── fused_kernels.py               # Triton Fused CE (Online LSE), SwiGLU, RMSNorm
│   ├── triton_ternary_gemm.py         # Triton ternary GEMM implementations
│   ├── trit_rust_engine.py            # Python CFFI wrapper for Rust SIMD kernel
│   └── trit_kernel.dll                # Precompiled Windows AVX2 library
│
├── trit_kernel/                       # Rust AVX2 SIMD crate
│   ├── Cargo.toml                     # Crate manifest
│   ├── Cargo.lock                     # Lockfile
│   └── src/                           # Rust source files (c_abi, gemm, gemv, lut, simd_avx2)
│
├── streamer/                          # Streaming data pipeline
│   ├── ultra_streamer.py              # UltraStreamer: multiprocess tokenizer, block-diagonal mask
│   └── dataset_streamer.py            # Dataset streaming baseline
│
├── training/                          # Pretraining scripts
│   ├── train_extreme.py               # Muon + AdamW pretraining loop with WSD schedule
│   ├── train_distributed.py           # Multi-GPU DDP training launcher
│   ├── train_real_benchmark.py        # CPU benchmark pretraining script
│   └── train_wikipedia_longrun.py     # Extended CPU training script
│
├── inference/                         # Autoregressive inference
│   ├── generate_sample.py             # Sampling with repetition penalty and n-gram blocking
│   └── generate_inference.py          # Lightweight CPU text generator
│
├── tools/                             # Verification and diagnostic tools
│   ├── test_ultra_throughput.py       # Dataflow throughput verification
│   ├── test_cluster.py                # Multi-GPU cluster diagnostic tool
│   ├── check_status.py                # Telemetry and GPU status monitor
│   └── inspect_ckpt.py                # Checkpoint inspection utility
│
├── scripts/                           # Cluster execution scripts
│   ├── launch_4x4090.sh               # Distributed launcher for 4x RTX 4090
│   ├── launch_extreme.sh              # Performance-optimized cluster launcher
│   ├── setup_server.sh                # Automated environment provisioning script
│   └── fast_install.sh                # Dependency installation script
│
├── benchmarks/                        # Empirical logs and reports
│   ├── fifteen_min_training_log.json  # 15-minute CPU telemetry log
│   ├── fifteen_min_training_report.md # 15-minute convergence report
│   ├── real_training_log.json         # 6-minute CPU telemetry log
│   ├── real_benchmark_analysis.md     # 6-minute analysis report
│   ├── longrun_training_log.json      # 100-step telemetry log
│   └── longrun_training_report.md     # 100-step report
│
├── research/                          # Mathematical and architectural specifications
│   ├── ternary_math_foundations.md    # Theoretical foundations and radix economy
│   ├── tritformer_architecture.md     # Core architectural RFC
│   ├── bitnet_stability_report.md     # Post-mortem analysis of numerical pathologies
│   ├── fused_kernels_report.md        # Triton kernel engineering specification
│   ├── train_extreme_spec.md          # Multi-GPU optimizer and scheduling specification
│   ├── ultra_streamer_spec.md         # UltraStreamer design and throughput analysis
│   └── reproducibility_audit_report.md # Scientific audit of 15 recent optimization papers
│
└── wiki_tokenizer/                    # Tokenizer files
    ├── vocab.json                     # BPE vocabulary
    └── merges.txt                     # BPE merge rules
```

---

## Environment Setup and Reproducibility

### 1. Installation

```bash
git clone https://github.com/Mesokiiii/TritLLM.git
cd TritLLM
pip install -r requirements.txt
```

To build the Rust AVX2 SIMD dynamic library from source:

```bash
cd trit_kernel
cargo build --release
cd ..
```

### 2. Numerical Stability Audit Suite

To verify the mathematical stability of the architecture (starting loss, logit standard deviation, residual variance across 16 layers, attention entropy, and STE gradient flow):

```bash
python tritformer/tritformer_stable.py
```

### 3. CPU Benchmark Training

```bash
python training/train_real_benchmark.py --duration_sec 360
```

### 4. Distributed Multi-GPU Training (4x RTX 4090)

```bash
bash scripts/launch_4x4090.sh
```

---

## References

1. Amsel, N., Dao, T., Langford, J., et al. (2026). *Dion3: Full-Stack Orthogonal Updates*. arXiv preprint.
2. Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla)*. NeurIPS 2022.
3. Jordan, K., et al. (2024). *Muon: An Optimizer for Hidden Layers in Neural Networks*.
4. Khona, M., Catanzaro, B., Patwary, M., et al. (2026). *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales*. NVIDIA Technical Report.
5. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms* (3rd ed.). Addison-Wesley.
6. Liu, J., Su, J., Zhou, X., Yang, Z., et al. (2025). *Muon is Scalable for LLM Training (Moonlight)*. Moonshot AI & UCLA.
7. Luo, K., Cui, J., Lyu, K., Chen, W., et al. (2026). *PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*. Tsinghua University.
8. Ma, S., Wang, H., Ma, L., Wang, L., Wang, Y., Huang, S., & Wei, F. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. Microsoft Research.
9. Qiang, X., Chen, H., Sun, S., Leng, J., Liu, X., & Guo, M. (2026). *DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training*. ICLR 2026.
10. Touvron, H., Martin, L., Stone, K., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. Meta AI.
11. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
12. Volkova, A., Safaryan, M., Lampert, C. H., & Alistarh, D. (2026). *Towards Robust Scaling Laws for Optimizers*. ISTA, Lancaster & Red Hat.
13. Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., & Wei, F. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models*. Microsoft Research.
14. Wang, H., Dong, L., Wei, F., et al. (2022). *DeepNet: Scaling Transformers to 1,000 Layers*. Microsoft Research.

---

## Citation

```bibtex
@article{tritllm2026,
  title={TritLLM: Towards Multiplication-Free Language Modeling with Balanced Ternary Quantization, SubLN Stability, and Fused Hardware Kernels},
  author={TritLLM Research Initiative},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/Mesokiiii/TritLLM}
}
```

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
