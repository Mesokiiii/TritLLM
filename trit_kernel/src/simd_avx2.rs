//! AVX2 Addition-Only GEMM/GEMV Core Kernels.
//!
//! Performs ternary matrix operations WITHOUT ANY MULTIPLICATIONS by weights.
//! Utilizes `_mm256_loadu_ps`, `_mm256_add_ps`, `_mm256_sub_ps`, and bitwise masking.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::lut::get_mask_avx2;
use crate::packing::TritBlock256;

/// Branchless horizontal sum of 8 single-precision floats in a 256-bit AVX register.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn hsum_avx2(v: __m256) -> f32 {
    let v_high = _mm256_extractf128_ps(v, 1);
    let v_low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(v_low, v_high);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehdup_ps(sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

/// Compute dot product of single activation vector `X` with a single ternary row.
///
/// Pure addition-only: does not execute any multiplications.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_ternary_row_avx2(
    x_ptr: *const f32,
    pos_ptr: *const u8,
    neg_ptr: *const u8,
    k: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks_32 = k / 32;

    // Unroll by 4 AVX vectors (32 floats = 4 bytes of bitmasks)
    for c in 0..chunks_32 {
        let x_offset = c * 32;
        let b_offset = c * 4;

        // Sub-chunk 0
        let b_pos0 = *pos_ptr.add(b_offset);
        let b_neg0 = *neg_ptr.add(b_offset);
        let x0 = _mm256_loadu_ps(x_ptr.add(x_offset));
        let m_pos0 = get_mask_avx2(b_pos0);
        let m_neg0 = get_mask_avx2(b_neg0);
        acc0 = _mm256_add_ps(acc0, _mm256_and_ps(x0, m_pos0));
        acc0 = _mm256_sub_ps(acc0, _mm256_and_ps(x0, m_neg0));

        // Sub-chunk 1
        let b_pos1 = *pos_ptr.add(b_offset + 1);
        let b_neg1 = *neg_ptr.add(b_offset + 1);
        let x1 = _mm256_loadu_ps(x_ptr.add(x_offset + 8));
        let m_pos1 = get_mask_avx2(b_pos1);
        let m_neg1 = get_mask_avx2(b_neg1);
        acc1 = _mm256_add_ps(acc1, _mm256_and_ps(x1, m_pos1));
        acc1 = _mm256_sub_ps(acc1, _mm256_and_ps(x1, m_neg1));

        // Sub-chunk 2
        let b_pos2 = *pos_ptr.add(b_offset + 2);
        let b_neg2 = *neg_ptr.add(b_offset + 2);
        let x2 = _mm256_loadu_ps(x_ptr.add(x_offset + 16));
        let m_pos2 = get_mask_avx2(b_pos2);
        let m_neg2 = get_mask_avx2(b_neg2);
        acc2 = _mm256_add_ps(acc2, _mm256_and_ps(x2, m_pos2));
        acc2 = _mm256_sub_ps(acc2, _mm256_and_ps(x2, m_neg2));

        // Sub-chunk 3
        let b_pos3 = *pos_ptr.add(b_offset + 3);
        let b_neg3 = *neg_ptr.add(b_offset + 3);
        let x3 = _mm256_loadu_ps(x_ptr.add(x_offset + 24));
        let m_pos3 = get_mask_avx2(b_pos3);
        let m_neg3 = get_mask_avx2(b_neg3);
        acc3 = _mm256_add_ps(acc3, _mm256_and_ps(x3, m_pos3));
        acc3 = _mm256_sub_ps(acc3, _mm256_and_ps(x3, m_neg3));
    }

    // Combine 4 independent accumulators
    let mut total_acc = _mm256_add_ps(
        _mm256_add_ps(acc0, acc1),
        _mm256_add_ps(acc2, acc3),
    );

    // Remaining 8-float chunks
    let processed_32 = chunks_32 * 32;
    let chunks_8 = (k - processed_32) / 8;
    for c in 0..chunks_8 {
        let x_idx = processed_32 + c * 8;
        let b_idx = chunks_32 * 4 + c;
        let b_pos = *pos_ptr.add(b_idx);
        let b_neg = *neg_ptr.add(b_idx);
        let x_vec = _mm256_loadu_ps(x_ptr.add(x_idx));
        let m_pos = get_mask_avx2(b_pos);
        let m_neg = get_mask_avx2(b_neg);
        total_acc = _mm256_add_ps(total_acc, _mm256_and_ps(x_vec, m_pos));
        total_acc = _mm256_sub_ps(total_acc, _mm256_and_ps(x_vec, m_neg));
    }

    let mut sum = hsum_avx2(total_acc);

    // Scalar tail for remainder < 8
    let processed_8 = processed_32 + chunks_8 * 8;
    for i in processed_8..k {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_pos = (*pos_ptr.add(byte_idx) >> bit_idx) & 1;
        let is_neg = (*neg_ptr.add(byte_idx) >> bit_idx) & 1;
        if is_pos != 0 {
            sum += *x_ptr.add(i);
        } else if is_neg != 0 {
            sum -= *x_ptr.add(i);
        }
    }

    sum
}

/// Compute 4 dot products simultaneously for 4 ternary rows against the same activation vector `X`.
///
/// Maximizes L1 cache reuse of `X`, interleaves adds/subs across 4 independent row accumulators,
/// and achieves peak CPU pipeline throughput.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_ternary_row4_avx2(
    x_ptr: *const f32,
    pos0: *const u8,
    neg0: *const u8,
    pos1: *const u8,
    neg1: *const u8,
    pos2: *const u8,
    neg2: *const u8,
    pos3: *const u8,
    neg3: *const u8,
    k: usize,
    out: &mut [f32; 4],
) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks_8 = k / 8;

    for c in 0..chunks_8 {
        let x_idx = c * 8;
        let x_vec = _mm256_loadu_ps(x_ptr.add(x_idx));

        // Row 0
        let mp0 = get_mask_avx2(*pos0.add(c));
        let mn0 = get_mask_avx2(*neg0.add(c));
        acc0 = _mm256_add_ps(acc0, _mm256_and_ps(x_vec, mp0));
        acc0 = _mm256_sub_ps(acc0, _mm256_and_ps(x_vec, mn0));

        // Row 1
        let mp1 = get_mask_avx2(*pos1.add(c));
        let mn1 = get_mask_avx2(*neg1.add(c));
        acc1 = _mm256_add_ps(acc1, _mm256_and_ps(x_vec, mp1));
        acc1 = _mm256_sub_ps(acc1, _mm256_and_ps(x_vec, mn1));

        // Row 2
        let mp2 = get_mask_avx2(*pos2.add(c));
        let mn2 = get_mask_avx2(*neg2.add(c));
        acc2 = _mm256_add_ps(acc2, _mm256_and_ps(x_vec, mp2));
        acc2 = _mm256_sub_ps(acc2, _mm256_and_ps(x_vec, mn2));

        // Row 3
        let mp3 = get_mask_avx2(*pos3.add(c));
        let mn3 = get_mask_avx2(*neg3.add(c));
        acc3 = _mm256_add_ps(acc3, _mm256_and_ps(x_vec, mp3));
        acc3 = _mm256_sub_ps(acc3, _mm256_and_ps(x_vec, mn3));
    }

    out[0] = hsum_avx2(acc0);
    out[1] = hsum_avx2(acc1);
    out[2] = hsum_avx2(acc2);
    out[3] = hsum_avx2(acc3);

    // Remainder tail
    let rem_start = chunks_8 * 8;
    for i in rem_start..k {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let val = *x_ptr.add(i);

        let p0 = (*pos0.add(byte_idx) >> bit_idx) & 1;
        let n0 = (*neg0.add(byte_idx) >> bit_idx) & 1;
        if p0 != 0 { out[0] += val; } else if n0 != 0 { out[0] -= val; }

        let p1 = (*pos1.add(byte_idx) >> bit_idx) & 1;
        let n1 = (*neg1.add(byte_idx) >> bit_idx) & 1;
        if p1 != 0 { out[1] += val; } else if n1 != 0 { out[1] -= val; }

        let p2 = (*pos2.add(byte_idx) >> bit_idx) & 1;
        let n2 = (*neg2.add(byte_idx) >> bit_idx) & 1;
        if p2 != 0 { out[2] += val; } else if n2 != 0 { out[2] -= val; }

        let p3 = (*pos3.add(byte_idx) >> bit_idx) & 1;
        let n3 = (*neg3.add(byte_idx) >> bit_idx) & 1;
        if p3 != 0 { out[3] += val; } else if n3 != 0 { out[3] -= val; }
    }
}

/// Compute dot product of single activation vector `X` with a row of 64-byte `TritBlock256`.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_ternary_block256_row_avx2(
    x_ptr: *const f32,
    blocks_ptr: *const TritBlock256,
    num_blocks: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for b in 0..num_blocks {
        let block = &*blocks_ptr.add(b);
        let block_x_ptr = x_ptr.add(b * 256);

        // 256 trits = 32 bytes = 8 chunks of 4 bytes
        for c in 0..8 {
            let x_offset = c * 32;
            let b_offset = c * 4;

            let b_pos0 = *block.pos.as_ptr().add(b_offset);
            let b_neg0 = *block.neg.as_ptr().add(b_offset);
            let x0 = _mm256_loadu_ps(block_x_ptr.add(x_offset));
            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(x0, get_mask_avx2(b_pos0)));
            acc0 = _mm256_sub_ps(acc0, _mm256_and_ps(x0, get_mask_avx2(b_neg0)));

            let b_pos1 = *block.pos.as_ptr().add(b_offset + 1);
            let b_neg1 = *block.neg.as_ptr().add(b_offset + 1);
            let x1 = _mm256_loadu_ps(block_x_ptr.add(x_offset + 8));
            acc1 = _mm256_add_ps(acc1, _mm256_and_ps(x1, get_mask_avx2(b_pos1)));
            acc1 = _mm256_sub_ps(acc1, _mm256_and_ps(x1, get_mask_avx2(b_neg1)));

            let b_pos2 = *block.pos.as_ptr().add(b_offset + 2);
            let b_neg2 = *block.neg.as_ptr().add(b_offset + 2);
            let x2 = _mm256_loadu_ps(block_x_ptr.add(x_offset + 16));
            acc2 = _mm256_add_ps(acc2, _mm256_and_ps(x2, get_mask_avx2(b_pos2)));
            acc2 = _mm256_sub_ps(acc2, _mm256_and_ps(x2, get_mask_avx2(b_neg2)));

            let b_pos3 = *block.pos.as_ptr().add(b_offset + 3);
            let b_neg3 = *block.neg.as_ptr().add(b_offset + 3);
            let x3 = _mm256_loadu_ps(block_x_ptr.add(x_offset + 24));
            acc3 = _mm256_add_ps(acc3, _mm256_and_ps(x3, get_mask_avx2(b_pos3)));
            acc3 = _mm256_sub_ps(acc3, _mm256_and_ps(x3, get_mask_avx2(b_neg3)));
        }
    }

    let total_acc = _mm256_add_ps(
        _mm256_add_ps(acc0, acc1),
        _mm256_add_ps(acc2, acc3),
    );

    hsum_avx2(total_acc)
}

/// Fallback naive scalar dot product for verification and benchmarking baseline.
pub fn dot_ternary_scalar_naive(x: &[f32], w: &[i8]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..x.len() {
        let weight = w[i];
        if weight == 1 {
            sum += x[i];
        } else if weight == -1 {
            sum -= x[i];
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packing::{pack_ternary_weights, pack_into_blocks256};

    #[test]
    fn test_dot_product_equivalence() {
        let k = 512;
        let mut x = vec![0.0f32; k];
        let mut w = vec![0i8; k];

        for i in 0..k {
            x[i] = ((i as f32) * 0.1).sin();
            w[i] = match (i * 3 + 1) % 5 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }

        let ref_sum = dot_ternary_scalar_naive(&x, &w);

        let mut pos_bits = vec![0u8; k / 8];
        let mut neg_bits = vec![0u8; k / 8];
        pack_ternary_weights(&w, &mut pos_bits, &mut neg_bits);

        let avx_sum = unsafe {
            dot_ternary_row_avx2(x.as_ptr(), pos_bits.as_ptr(), neg_bits.as_ptr(), k)
        };

        let diff = (ref_sum - avx_sum).abs();
        assert!(diff < 1e-4, "Ref: {}, AVX: {}, diff: {}", ref_sum, avx_sum, diff);

        // Test 4-row kernel
        let mut out4 = [0.0f32; 4];
        unsafe {
            dot_ternary_row4_avx2(
                x.as_ptr(),
                pos_bits.as_ptr(), neg_bits.as_ptr(),
                pos_bits.as_ptr(), neg_bits.as_ptr(),
                pos_bits.as_ptr(), neg_bits.as_ptr(),
                pos_bits.as_ptr(), neg_bits.as_ptr(),
                k,
                &mut out4,
            );
        }
        for o in out4.iter() {
            assert!((ref_sum - *o).abs() < 1e-4);
        }

        // Test TritBlock256 kernel
        let mut blocks = vec![TritBlock256::default(); k / 256];
        pack_into_blocks256(&w, 1, k, &mut blocks);
        let block_sum = unsafe {
            dot_ternary_block256_row_avx2(x.as_ptr(), blocks.as_ptr(), k / 256)
        };
        assert!((ref_sum - block_sum).abs() < 1e-4);
    }
}
