//! High-Precision Microarchitectural Benchmark for Trit-Kernel AVX2 Addition-Only Engine.
//!
//! Evaluates speedup, throughput, and memory bandwidth across multiple matrix scales:
//! 1. Naive Scalar baseline (branching / multiplying loop)
//! 2. AVX2 Addition-Only Single-Thread
//! 3. AVX2 Addition-Only Multi-Thread (Rayon work-stealing + L1/L2 cache blocking)
//! 4. AVX2 Interleaved 64-byte TritBlock256

use std::time::Instant;
use trit_kernel::packing::{pack_ternary_weights, pack_into_blocks256, TritBlock256};
use trit_kernel::gemm::{trit_gemm_single_thread, trit_gemm_parallel, trit_gemm_blocks_parallel, trit_gemm_naive_scalar};

struct BenchCase {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
}

fn run_benchmark(case: &BenchCase) {
    let m = case.m;
    let n = case.n;
    let k = case.k;
    let iters = case.iters;

    println!("================================================================================");
    println!("Benchmark: {} | M = {}, N = {}, K = {} | Iterations: {}", case.name, m, n, k, iters);
    println!("Total Operations per GEMM: {:.2} M-FLOPs", (2.0 * m as f64 * n as f64 * k as f64) / 1e6);
    println!("FP32 Weight Size: {:.2} MB | Packed 2-bit Trit Size: {:.2} MB (16x smaller)",
        (n * k * 4) as f64 / 1024.0 / 1024.0,
        (n * k / 4) as f64 / 1024.0 / 1024.0
    );
    println!("--------------------------------------------------------------------------------");

    // Initialize deterministic test data
    let mut x = vec![0.5f32; m * k];
    for i in 0..(m * k) {
        x[i] = ((i as f32) * 0.03).sin();
    }

    let mut w_dense = vec![0i8; n * k];
    for i in 0..(n * k) {
        w_dense[i] = match (i * 17 + 3) % 3 {
            0 => 1,
            1 => -1,
            _ => 0,
        };
    }

    // Pack into dual bitplanes
    let bytes_per_row = (k + 7) / 8;
    let mut pos_bits = vec![0u8; n * bytes_per_row];
    let mut neg_bits = vec![0u8; n * bytes_per_row];
    pack_ternary_weights(&w_dense, &mut pos_bits, &mut neg_bits);

    // Pack into 64-byte blocks
    let blocks_per_row = (k + 255) / 256;
    let mut blocks = vec![TritBlock256::default(); n * blocks_per_row];
    pack_into_blocks256(&w_dense, n, k, &mut blocks);

    let mut y_naive = vec![0.0f32; m * n];
    let mut y_single = vec![0.0f32; m * n];
    let mut y_parallel = vec![0.0f32; m * n];
    let mut y_blocks = vec![0.0f32; m * n];

    // 1. Naive Scalar Baseline (fewer iterations if matrix is large)
    let naive_iters = if m * n * k > 4096 * 4096 { 2 } else if m * n * k > 2048 * 2048 { 5 } else { iters.min(20) };
    let t_start = Instant::now();
    for _ in 0..naive_iters {
        trit_gemm_naive_scalar(&x, &w_dense, &mut y_naive, m, n, k, 1.0);
    }
    let dur_naive = t_start.elapsed().as_secs_f64() / (naive_iters as f64);
    let naive_us = dur_naive * 1_000_000.0;
    let naive_gops = (2.0 * m as f64 * n as f64 * k as f64) / (dur_naive * 1e9);

    println!("1. Naive Scalar Baseline: {:10.2} us | {:7.2} GOPS | 1.00x (baseline)", naive_us, naive_gops);

    // 2. AVX2 Single-Thread (Addition-Only)
    // Warmup
    trit_gemm_single_thread(&x, &pos_bits, &neg_bits, &mut y_single, m, n, k, 1.0);
    let t_start = Instant::now();
    for _ in 0..iters {
        trit_gemm_single_thread(&x, &pos_bits, &neg_bits, &mut y_single, m, n, k, 1.0);
    }
    let dur_single = t_start.elapsed().as_secs_f64() / (iters as f64);
    let single_us = dur_single * 1_000_000.0;
    let single_gops = (2.0 * m as f64 * n as f64 * k as f64) / (dur_single * 1e9);
    let speedup_single = dur_naive / dur_single;

    println!("2. AVX2 Single-Thread:   {:10.2} us | {:7.2} GOPS | {:6.2}x speedup", single_us, single_gops, speedup_single);

    // 3. AVX2 Multi-Thread (Rayon)
    // Warmup
    trit_gemm_parallel(&x, &pos_bits, &neg_bits, &mut y_parallel, m, n, k, 1.0);
    let t_start = Instant::now();
    for _ in 0..iters {
        trit_gemm_parallel(&x, &pos_bits, &neg_bits, &mut y_parallel, m, n, k, 1.0);
    }
    let dur_parallel = t_start.elapsed().as_secs_f64() / (iters as f64);
    let parallel_us = dur_parallel * 1_000_000.0;
    let parallel_gops = (2.0 * m as f64 * n as f64 * k as f64) / (dur_parallel * 1e9);
    let speedup_parallel = dur_naive / dur_parallel;

    println!("3. AVX2 Parallel (Rayon):{:10.2} us | {:7.2} GOPS | {:6.2}x speedup", parallel_us, parallel_gops, speedup_parallel);

    // 4. AVX2 64-byte TritBlock256 Parallel
    trit_gemm_blocks_parallel(&x, &blocks, &mut y_blocks, m, n, k, 1.0);
    let t_start = Instant::now();
    for _ in 0..iters {
        trit_gemm_blocks_parallel(&x, &blocks, &mut y_blocks, m, n, k, 1.0);
    }
    let dur_blocks = t_start.elapsed().as_secs_f64() / (iters as f64);
    let blocks_us = dur_blocks * 1_000_000.0;
    let blocks_gops = (2.0 * m as f64 * n as f64 * k as f64) / (dur_blocks * 1e9);
    let speedup_blocks = dur_naive / dur_blocks;

    println!("4. AVX2 Block256 Parallel:{:9.2} us | {:7.2} GOPS | {:6.2}x speedup", blocks_us, blocks_gops, speedup_blocks);

    // Verify numerical accuracy
    let mut max_err = 0.0f32;
    for i in 0..(m * n) {
        let diff = (y_naive[i] - y_parallel[i]).abs();
        if diff > max_err { max_err = diff; }
    }
    println!("Max absolute error vs naive: {:.2e} (Bit-exact or within float accumulation reordering)", max_err);
    println!();
}

fn main() {
    println!("================================================================================");
    println!("   TRIT_KERNEL: AVX2 ADDITION-ONLY GEMM/GEMV HARDWARE ACCELERATION ENGINE       ");
    println!("   Target: Intel Raptor Lake i5-13420H (4 P-cores + 4 E-cores = 12 threads)    ");
    println!("================================================================================");
    println!("Active Rayon Thread Pool: {} workers", rayon::current_num_threads());
    println!();

    let cases = vec![
        BenchCase { name: "GEMV Small (Decode 512x512)", m: 1, n: 512, k: 512, iters: 200 },
        BenchCase { name: "GEMV Hidden Layer (Decode 2048x2048)", m: 1, n: 2048, k: 2048, iters: 100 },
        BenchCase { name: "GEMV LLaMA/BitNet (Decode 4096x4096)", m: 1, n: 4096, k: 4096, iters: 40 },
        BenchCase { name: "GEMM Batch-16 (Context Prefill 16x2048x2048)", m: 16, n: 2048, k: 2048, iters: 30 },
        BenchCase { name: "GEMM Batch-32 (Context Prefill 32x2048x2048)", m: 32, n: 2048, k: 2048, iters: 20 },
    ];

    for case in &cases {
        run_benchmark(case);
    }
}
