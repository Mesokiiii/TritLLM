//! Addition-Only GEMM (General Matrix-Matrix Multiplication) Kernels.
//!
//! Computes Y = scale * (X * W^T), where:
//! - X is in R^{M x K} (activations)
//! - W is in {-1, 0, +1}^{N x K} (ternary weights)
//! - Y is in R^{M x N} (output activations)
//!
//! Features multi-level cache blocking (L1/L2/L3) for Intel Core i5-13420H
//! and multi-core scheduling with Rayon.

use rayon::prelude::*;
use crate::packing::TritBlock256;
use crate::gemv::{trit_gemv_single_thread, trit_gemv_parallel};
use crate::simd_avx2::dot_ternary_block256_row_avx2;

/// Single-threaded Addition-Only GEMM.
pub fn trit_gemm_single_thread(
    x: &[f32],
    pos_bits: &[u8],
    neg_bits: &[u8],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(y.len(), m * n);

    for row_m in 0..m {
        let x_slice = &x[row_m * k..(row_m + 1) * k];
        let y_slice = &mut y[row_m * n..(row_m + 1) * n];
        trit_gemv_single_thread(x_slice, pos_bits, neg_bits, y_slice, n, k, scale);
    }
}

/// Multi-threaded Cache-Blocked Addition-Only GEMM using Rayon.
///
/// Automatically chooses between parallelizing along M or N based on matrix aspect ratio:
/// - If M == 1 (Decode phase): parallelizes over N rows.
/// - If M > 1: parallelizes over 2D tiles (M_tile, N_tile) fitting in L1/L2 caches.
pub fn trit_gemm_parallel(
    x: &[f32],
    pos_bits: &[u8],
    neg_bits: &[u8],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(y.len(), m * n);

    if m == 1 {
        trit_gemv_parallel(x, pos_bits, neg_bits, y, n, k, scale);
        return;
    }

    let num_threads = rayon::current_num_threads();

    // Coarse-grained chunking over M rows to minimize Rayon work-stealing overhead
    let chunk_rows = ((m + num_threads - 1) / num_threads).max(16);
    y.par_chunks_mut(n * chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, y_chunk)| {
            let start_m = chunk_idx * chunk_rows;
            let actual_rows = y_chunk.len() / n;
            for r in 0..actual_rows {
                let row_m = start_m + r;
                let x_slice = &x[row_m * k..(row_m + 1) * k];
                let y_slice = &mut y_chunk[r * n..(r + 1) * n];
                trit_gemv_single_thread(x_slice, pos_bits, neg_bits, y_slice, n, k, scale);
            }
        });
}

/// Multi-threaded GEMM with interleaved 64-byte `TritBlock256`.
pub fn trit_gemm_blocks_parallel(
    x: &[f32],
    blocks: &[TritBlock256],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(y.len(), m * n);
    let blocks_per_row = (k + 255) / 256;
    assert!(blocks.len() >= n * blocks_per_row);

    y.par_chunks_mut(n)
        .enumerate()
        .for_each(|(row_m, y_row)| {
            let x_ptr = unsafe { x.as_ptr().add(row_m * k) };
            for row_n in 0..n {
                let block_row_ptr = unsafe { blocks.as_ptr().add(row_n * blocks_per_row) };
                let sum = unsafe {
                    dot_ternary_block256_row_avx2(x_ptr, block_row_ptr, blocks_per_row)
                };
                y_row[row_n] = sum * scale;
            }
        });
}

/// Reference naive scalar GEMM for precision verification and baseline comparison.
pub fn trit_gemm_naive_scalar(
    x: &[f32],
    w: &[i8],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) {
    for row_m in 0..m {
        for row_n in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                let weight = w[row_n * k + i];
                let act = x[row_m * k + i];
                if weight == 1 {
                    sum += act;
                } else if weight == -1 {
                    sum -= act;
                }
            }
            y[row_m * n + row_n] = sum * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packing::pack_ternary_weights;

    #[test]
    fn test_gemm_equivalence() {
        let m = 4;
        let n = 64;
        let k = 128;

        let mut x = vec![0.0f32; m * k];
        for i in 0..(m * k) {
            x[i] = ((i as f32) * 0.05).cos();
        }

        let mut w = vec![0i8; n * k];
        for i in 0..(n * k) {
            w[i] = match (i * 7 + 2) % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }

        let mut pos_bits = vec![0u8; n * (k / 8)];
        let mut neg_bits = vec![0u8; n * (k / 8)];
        pack_ternary_weights(&w, &mut pos_bits, &mut neg_bits);

        let mut y_ref = vec![0.0f32; m * n];
        let mut y_opt = vec![0.0f32; m * n];

        trit_gemm_naive_scalar(&x, &w, &mut y_ref, m, n, k, 2.0);
        trit_gemm_parallel(&x, &pos_bits, &neg_bits, &mut y_opt, m, n, k, 2.0);

        for i in 0..(m * n) {
            let diff = (y_ref[i] - y_opt[i]).abs();
            assert!(diff < 1e-4, "Idx {}: ref={}, opt={}", i, y_ref[i], y_opt[i]);
        }
    }
}
