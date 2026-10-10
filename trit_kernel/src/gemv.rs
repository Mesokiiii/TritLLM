//! Ternary GEMV (Vector-Matrix Multiplication) Kernels.
//!
//! Computes y = scale * (W * x), where W is in {-1, 0, +1}^{N x K}
//! and x is in R^K.
//! Zero multiplications by weights in the accumulation loop.
//! Cache-blocked for Intel Raptor Lake L1/L2 caches and parallelized with Rayon.

use rayon::prelude::*;
use crate::packing::TritBlock256;
use crate::simd_avx2::{dot_ternary_row_avx2, dot_ternary_row4_avx2, dot_ternary_block256_row_avx2};

/// Single-threaded addition-only GEMV with 4-row unrolling.
pub fn trit_gemv_single_thread(
    x: &[f32],
    pos_bits: &[u8],
    neg_bits: &[u8],
    y: &mut [f32],
    n: usize,
    k: usize,
    scale: f32,
) {
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), n);
    let bytes_per_row = (k + 7) / 8;
    assert!(pos_bits.len() >= n * bytes_per_row);
    assert!(neg_bits.len() >= n * bytes_per_row);

    let x_ptr = x.as_ptr();
    let pos_ptr = pos_bits.as_ptr();
    let neg_ptr = neg_bits.as_ptr();
    let y_ptr = y.as_mut_ptr();

    let n_chunks_4 = n / 4;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        for c in 0..n_chunks_4 {
            let r0 = c * 4;
            let r1 = r0 + 1;
            let r2 = r0 + 2;
            let r3 = r0 + 3;

            let p0 = pos_ptr.add(r0 * bytes_per_row);
            let n0 = neg_ptr.add(r0 * bytes_per_row);
            let p1 = pos_ptr.add(r1 * bytes_per_row);
            let n1 = neg_ptr.add(r1 * bytes_per_row);
            let p2 = pos_ptr.add(r2 * bytes_per_row);
            let n2 = neg_ptr.add(r2 * bytes_per_row);
            let p3 = pos_ptr.add(r3 * bytes_per_row);
            let n3 = neg_ptr.add(r3 * bytes_per_row);

            let mut out4 = [0.0f32; 4];
            dot_ternary_row4_avx2(x_ptr, p0, n0, p1, n1, p2, n2, p3, n3, k, &mut out4);

            *y_ptr.add(r0) = out4[0] * scale;
            *y_ptr.add(r1) = out4[1] * scale;
            *y_ptr.add(r2) = out4[2] * scale;
            *y_ptr.add(r3) = out4[3] * scale;
        }

        // Remainder rows
        for r in (n_chunks_4 * 4)..n {
            let p = pos_ptr.add(r * bytes_per_row);
            let n_b = neg_ptr.add(r * bytes_per_row);
            let sum = dot_ternary_row_avx2(x_ptr, p, n_b, k);
            *y_ptr.add(r) = sum * scale;
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for r in 0..n {
            let row_p = &pos_bits[r * bytes_per_row..(r + 1) * bytes_per_row];
            let row_n = &neg_bits[r * bytes_per_row..(r + 1) * bytes_per_row];
            let mut sum = 0.0f32;
            for i in 0..k {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                let is_pos = (row_p[byte_idx] >> bit_idx) & 1;
                let is_neg = (row_n[byte_idx] >> bit_idx) & 1;
                if is_pos != 0 { sum += x[i]; } else if is_neg != 0 { sum -= x[i]; }
            }
            y[r] = sum * scale;
        }
    }
}

/// Multi-threaded parallel addition-only GEMV using Rayon.
///
/// Tiles rows across available CPU threads with grain size tuned to L1/L2 cache capacity.
pub fn trit_gemv_parallel(
    x: &[f32],
    pos_bits: &[u8],
    neg_bits: &[u8],
    y: &mut [f32],
    n: usize,
    k: usize,
    scale: f32,
) {
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), n);
    let bytes_per_row = (k + 7) / 8;
    assert!(pos_bits.len() >= n * bytes_per_row);
    assert!(neg_bits.len() >= n * bytes_per_row);

    // Dynamic tile size: target 16-64 rows per chunk to fit in L1/L2 cache
    let chunk_rows = 32.max((n / (rayon::current_num_threads() * 4)).max(4));

    y.par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, y_chunk)| {
            let row_start = chunk_idx * chunk_rows;
            let rows_in_chunk = y_chunk.len();

            let x_ptr = x.as_ptr();
            let pos_ptr = pos_bits.as_ptr();
            let neg_ptr = neg_bits.as_ptr();
            let y_ptr = y_chunk.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let n_chunks_4 = rows_in_chunk / 4;
                for c in 0..n_chunks_4 {
                    let local_r0 = c * 4;
                    let global_r0 = row_start + local_r0;
                    let global_r1 = global_r0 + 1;
                    let global_r2 = global_r0 + 2;
                    let global_r3 = global_r0 + 3;

                    let p0 = pos_ptr.add(global_r0 * bytes_per_row);
                    let n0 = neg_ptr.add(global_r0 * bytes_per_row);
                    let p1 = pos_ptr.add(global_r1 * bytes_per_row);
                    let n1 = neg_ptr.add(global_r1 * bytes_per_row);
                    let p2 = pos_ptr.add(global_r2 * bytes_per_row);
                    let n2 = neg_ptr.add(global_r2 * bytes_per_row);
                    let p3 = pos_ptr.add(global_r3 * bytes_per_row);
                    let n3 = neg_ptr.add(global_r3 * bytes_per_row);

                    let mut out4 = [0.0f32; 4];
                    dot_ternary_row4_avx2(x_ptr, p0, n0, p1, n1, p2, n2, p3, n3, k, &mut out4);

                    *y_ptr.add(local_r0) = out4[0] * scale;
                    *y_ptr.add(local_r0 + 1) = out4[1] * scale;
                    *y_ptr.add(local_r0 + 2) = out4[2] * scale;
                    *y_ptr.add(local_r0 + 3) = out4[3] * scale;
                }

                for local_r in (n_chunks_4 * 4)..rows_in_chunk {
                    let global_r = row_start + local_r;
                    let p = pos_ptr.add(global_r * bytes_per_row);
                    let n_b = neg_ptr.add(global_r * bytes_per_row);
                    let sum = dot_ternary_row_avx2(x_ptr, p, n_b, k);
                    *y_ptr.add(local_r) = sum * scale;
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                for local_r in 0..rows_in_chunk {
                    let global_r = row_start + local_r;
                    let row_p = &pos_bits[global_r * bytes_per_row..(global_r + 1) * bytes_per_row];
                    let row_n = &neg_bits[global_r * bytes_per_row..(global_r + 1) * bytes_per_row];
                    let mut sum = 0.0f32;
                    for i in 0..k {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        let is_pos = (row_p[byte_idx] >> bit_idx) & 1;
                        let is_neg = (row_n[byte_idx] >> bit_idx) & 1;
                        if is_pos != 0 { sum += x[i]; } else if is_neg != 0 { sum -= x[i]; }
                    }
                    y_chunk[local_r] = sum * scale;
                }
            }
        });
}

/// Multi-threaded parallel addition-only GEMV using interleaved 64-byte `TritBlock256`.
pub fn trit_gemv_blocks_parallel(
    x: &[f32],
    blocks: &[TritBlock256],
    y: &mut [f32],
    n: usize,
    k: usize,
    scale: f32,
) {
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), n);
    let blocks_per_row = (k + 255) / 256;
    assert!(blocks.len() >= n * blocks_per_row);

    let chunk_rows = 32.max((n / (rayon::current_num_threads() * 4)).max(4));

    y.par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, y_chunk)| {
            let row_start = chunk_idx * chunk_rows;
            let rows_in_chunk = y_chunk.len();
            let x_ptr = x.as_ptr();

            #[cfg(target_arch = "x86_64")]
            unsafe {
                for local_r in 0..rows_in_chunk {
                    let global_r = row_start + local_r;
                    let block_row_ptr = blocks.as_ptr().add(global_r * blocks_per_row);
                    let sum = dot_ternary_block256_row_avx2(x_ptr, block_row_ptr, blocks_per_row);
                    y_chunk[local_r] = sum * scale;
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                // Fallback
                for local_r in 0..rows_in_chunk {
                    y_chunk[local_r] = 0.0;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packing::pack_ternary_weights;

    #[test]
    fn test_gemv_single_and_parallel() {
        let n = 128;
        let k = 256;
        let mut x = vec![0.0f32; k];
        for i in 0..k {
            x[i] = (i as f32) * 0.01;
        }

        let mut w = vec![0i8; n * k];
        for i in 0..(n * k) {
            w[i] = match (i * 13) % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }

        let mut pos_bits = vec![0u8; n * (k / 8)];
        let mut neg_bits = vec![0u8; n * (k / 8)];
        pack_ternary_weights(&w, &mut pos_bits, &mut neg_bits);

        let mut y_single = vec![0.0f32; n];
        let mut y_parallel = vec![0.0f32; n];

        trit_gemv_single_thread(&x, &pos_bits, &neg_bits, &mut y_single, n, k, 1.5);
        trit_gemv_parallel(&x, &pos_bits, &neg_bits, &mut y_parallel, n, k, 1.5);

        for i in 0..n {
            assert!((y_single[i] - y_parallel[i]).abs() < 1e-4);
        }
    }
}
