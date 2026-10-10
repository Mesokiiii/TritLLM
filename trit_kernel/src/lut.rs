//! Look-Up Table (LUT) for expanding 8-bit ternary bitmasks into 256-bit AVX2 SIMD masks.
//! 
//! An 8-bit byte encodes 8 trits (+1 or -1 presence).
//! For each byte in [0, 255], this LUT maps each bit to a 32-bit all-ones (0xFFFFFFFF)
//! or all-zeros (0x00000000) mask.
//! Total table size: 256 * 8 * 4 bytes = 8192 bytes (8 KB), fitting 100% inside
//! the L1 Data Cache (48 KB on Intel Raptor Lake P-cores, 32 KB on E-cores).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[repr(C, align(64))]
pub struct MaskLut {
    pub masks: [f32; 256 * 8],
}

impl MaskLut {
    pub const fn init() -> Self {
        let mut masks = [0.0f32; 256 * 8];
        let mut byte = 0usize;
        while byte < 256 {
            let mut bit = 0usize;
            while bit < 8 {
                let mask_val = if (byte & (1 << bit)) != 0 {
                    0xFFFFFFFFu32
                } else {
                    0x00000000u32
                };
                masks[byte * 8 + bit] = f32::from_bits(mask_val);
                bit += 1;
            }
            byte += 1;
        }
        Self { masks }
    }
}

pub static MASK_LUT: MaskLut = MaskLut::init();

/// Retrieve a 256-bit AVX2 mask for an 8-bit mask byte
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn get_mask_avx2(byte: u8) -> __m256 {
    let ptr = MASK_LUT.masks.as_ptr().add((byte as usize) * 8);
    _mm256_load_ps(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_values() {
        // Byte 0: all zeros
        for i in 0..8 {
            assert_eq!(MASK_LUT.masks[0 * 8 + i].to_bits(), 0x00000000);
        }
        // Byte 1: bit 0 is 1, others 0
        assert_eq!(MASK_LUT.masks[1 * 8 + 0].to_bits(), 0xFFFFFFFF);
        for i in 1..8 {
            assert_eq!(MASK_LUT.masks[1 * 8 + i].to_bits(), 0x00000000);
        }
        // Byte 0xFF: all ones
        for i in 0..8 {
            assert_eq!(MASK_LUT.masks[255 * 8 + i].to_bits(), 0xFFFFFFFF);
        }
        // Byte 0b00000101 = 5 (bits 0 and 2 are 1)
        assert_eq!(MASK_LUT.masks[5 * 8 + 0].to_bits(), 0xFFFFFFFF);
        assert_eq!(MASK_LUT.masks[5 * 8 + 1].to_bits(), 0x00000000);
        assert_eq!(MASK_LUT.masks[5 * 8 + 2].to_bits(), 0xFFFFFFFF);
        assert_eq!(MASK_LUT.masks[5 * 8 + 3].to_bits(), 0x00000000);
    }
}
