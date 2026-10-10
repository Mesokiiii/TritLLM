//! Bit-packing and bit-plane representation for ternary weights {-1, 0, +1}.
//!
//! Dual Bit-Plane format:
//! - W_pos[k] = 1 if W[k] == +1, else 0
//! - W_neg[k] = 1 if W[k] == -1, else 0
//! - W[k] == 0 is implicitly encoded when both bits are 0.
//!
//! Interleaved 64-byte block format:
//! - TritBlock256 contains 256 trits: 32 bytes W_pos and 32 bytes W_neg.
//! - Exactly 64 bytes = 1 CPU cache line on Intel Raptor Lake.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// 64-byte cache-line aligned block of 256 ternary weights.
/// 32 bytes (256 bits) for positive weights (+1)
/// 32 bytes (256 bits) for negative weights (-1)
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TritBlock256 {
    pub pos: [u8; 32],
    pub neg: [u8; 32],
}

impl Default for TritBlock256 {
    fn default() -> Self {
        Self {
            pos: [0u8; 32],
            neg: [0u8; 32],
        }
    }
}

/// Pack a slice of signed ternary values {-1, 0, 1} into separate pos and neg bitplanes.
///
/// Both `pos_bits` and `neg_bits` must have length at least `(src.len() + 7) / 8`.
pub fn pack_ternary_scalar(src: &[i8], pos_bits: &mut [u8], neg_bits: &mut [u8]) {
    pos_bits.fill(0);
    neg_bits.fill(0);

    for (i, &w) in src.iter().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if w == 1 {
            pos_bits[byte_idx] |= 1 << bit_idx;
        } else if w == -1 {
            neg_bits[byte_idx] |= 1 << bit_idx;
        }
    }
}

/// Unpack pos and neg bitplanes back to dense `i8` values {-1, 0, 1}.
pub fn unpack_ternary_scalar(pos_bits: &[u8], neg_bits: &[u8], dst: &mut [i8]) {
    for (i, d) in dst.iter_mut().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_pos = (pos_bits[byte_idx] >> bit_idx) & 1;
        let is_neg = (neg_bits[byte_idx] >> bit_idx) & 1;
        *d = if is_pos != 0 {
            1
        } else if is_neg != 0 {
            -1
        } else {
            0
        };
    }
}

/// AVX2-accelerated packing of 32 trits into a 32-bit pos mask and a 32-bit neg mask.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn pack_32_trits_avx2(src_ptr: *const i8) -> (u32, u32) {
    let v = _mm256_loadu_si256(src_ptr as *const __m256i);
    let one = _mm256_set1_epi8(1);
    let minus_one = _mm256_set1_epi8(-1);

    let cmp_pos = _mm256_cmpeq_epi8(v, one);
    let cmp_neg = _mm256_cmpeq_epi8(v, minus_one);

    let mask_pos = _mm256_movemask_epi8(cmp_pos) as u32;
    let mask_neg = _mm256_movemask_epi8(cmp_neg) as u32;

    (mask_pos, mask_neg)
}

/// High-speed AVX2 packing of dense ternary weights into dual bitplanes.
#[cfg(target_arch = "x86_64")]
pub fn pack_ternary_weights(src: &[i8], pos_bits: &mut [u8], neg_bits: &mut [u8]) {
    pos_bits.fill(0);
    neg_bits.fill(0);

    let len = src.len();
    let chunks_32 = len / 32;

    unsafe {
        for c in 0..chunks_32 {
            let offset_trits = c * 32;
            let offset_bytes = c * 4;

            let (m_pos, m_neg) = pack_32_trits_avx2(src.as_ptr().add(offset_trits));

            let p_dst = pos_bits.as_mut_ptr().add(offset_bytes) as *mut u32;
            let n_dst = neg_bits.as_mut_ptr().add(offset_bytes) as *mut u32;

            core::ptr::write_unaligned(p_dst, m_pos);
            core::ptr::write_unaligned(n_dst, m_neg);
        }
    }

    // Scalar tail
    let rem_start = chunks_32 * 32;
    for i in rem_start..len {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let w = src[i];
        if w == 1 {
            pos_bits[byte_idx] |= 1 << bit_idx;
        } else if w == -1 {
            neg_bits[byte_idx] |= 1 << bit_idx;
        }
    }
}

/// Pack dense ternary matrix [N, K] into 64-byte cache-line aligned blocks.
///
/// K must be a multiple of 256 for perfect alignment.
pub fn pack_into_blocks256(src: &[i8], n: usize, k: usize, blocks: &mut [TritBlock256]) {
    assert_eq!(src.len(), n * k);
    let blocks_per_row = (k + 255) / 256;
    assert!(blocks.len() >= n * blocks_per_row);

    for row in 0..n {
        let row_src = &src[row * k..(row + 1) * k];
        let row_blocks = &mut blocks[row * blocks_per_row..(row + 1) * blocks_per_row];

        for b in 0..blocks_per_row {
            let mut block = TritBlock256::default();
            let start = b * 256;
            let end = (start + 256).min(k);
            let count = end - start;

            if count == 256 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    for chunk in 0..8 {
                        let c_trits = start + chunk * 32;
                        let c_bytes = chunk * 4;
                        let (m_pos, m_neg) = pack_32_trits_avx2(row_src.as_ptr().add(c_trits));
                        let p_dst = block.pos.as_mut_ptr().add(c_bytes) as *mut u32;
                        let n_dst = block.neg.as_mut_ptr().add(c_bytes) as *mut u32;
                        core::ptr::write_unaligned(p_dst, m_pos);
                        core::ptr::write_unaligned(n_dst, m_neg);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    pack_ternary_scalar(&row_src[start..end], &mut block.pos, &mut block.neg);
                }
            } else {
                pack_ternary_scalar(&row_src[start..end], &mut block.pos, &mut block.neg);
            }

            row_blocks[b] = block;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packing_roundtrip() {
        let mut original = vec![0i8; 1024];
        for i in 0..1024 {
            original[i] = match i % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            };
        }

        let mut pos_bits = vec![0u8; 1024 / 8];
        let mut neg_bits = vec![0u8; 1024 / 8];

        pack_ternary_weights(&original, &mut pos_bits, &mut neg_bits);

        let mut reconstructed = vec![0i8; 1024];
        unpack_ternary_scalar(&pos_bits, &neg_bits, &mut reconstructed);

        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_blocks256_packing() {
        let n = 2;
        let k = 512;
        let mut src = vec![0i8; n * k];
        for (i, v) in src.iter_mut().enumerate() {
            *v = match (i * 7) % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }

        let mut blocks = vec![TritBlock256::default(); n * (k / 256)];
        pack_into_blocks256(&src, n, k, &mut blocks);

        // Verify first block
        let mut recon = vec![0i8; 256];
        unpack_ternary_scalar(&blocks[0].pos, &blocks[0].neg, &mut recon);
        assert_eq!(&src[0..256], &recon[..]);

        // Verify second block
        let mut recon2 = vec![0i8; 256];
        unpack_ternary_scalar(&blocks[1].pos, &blocks[1].neg, &mut recon2);
        assert_eq!(&src[256..512], &recon2[..]);
    }
}
