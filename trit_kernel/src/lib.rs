//! # trit_kernel
//!
//! Ultra-high-performance Addition-Only Ternary GEMM & GEMV Kernel for x86_64 AVX2/FMA.
//!
//! Designed for BitNet b1.58 and ternary neural networks (weights in {-1, 0, +1}).
//! Eliminates all floating-point multiplications from weight-matrix accumulation passes,
//! executing 100% of ternary dot-products via AVX2 vector masking, additions (`_mm256_add_ps`),
//! and subtractions (`_mm256_sub_ps`).
//!
//! Features:
//! - Dual Bit-Plane Representation (2 bits/weight: W_pos and W_neg).
//! - 64-Byte Cache-Line Aligned Interleaved Blocks (`TritBlock256`).
//! - Multi-level cache blocking optimized for Intel Raptor Lake architecture (i5-13420H).
//! - Multi-threaded scheduling via Rayon work-stealing pool.
//! - C ABI exports for direct zero-overhead calling from Python (`ctypes`, `cffi`, PyTorch).

pub mod lut;
pub mod packing;
pub mod simd_avx2;
pub mod gemv;
pub mod gemm;
pub mod c_abi;

// Re-exports
pub use packing::{
    TritBlock256,
    pack_ternary_weights,
    pack_ternary_scalar,
    unpack_ternary_scalar,
    pack_into_blocks256,
};
pub use gemv::{
    trit_gemv_single_thread,
    trit_gemv_parallel,
    trit_gemv_blocks_parallel,
};
pub use gemm::{
    trit_gemm_single_thread,
    trit_gemm_parallel,
    trit_gemm_blocks_parallel,
    trit_gemm_naive_scalar,
};

/// High-level ergonomic container for a packed ternary weight matrix.
#[derive(Clone, Debug)]
pub struct TritMatrix {
    pub rows: usize,
    pub cols: usize,
    pub pos_bits: Vec<u8>,
    pub neg_bits: Vec<u8>,
    pub scale: f32,
}

impl TritMatrix {
    /// Create a packed ternary matrix from a row-major dense slice of values in {-1, 0, 1}.
    pub fn from_dense(src: &[i8], rows: usize, cols: usize, scale: f32) -> Self {
        assert_eq!(src.len(), rows * cols);
        let bytes_per_row = (cols + 7) / 8;
        let mut pos_bits = vec![0u8; rows * bytes_per_row];
        let mut neg_bits = vec![0u8; rows * bytes_per_row];

        pack_ternary_weights(src, &mut pos_bits, &mut neg_bits);

        Self {
            rows,
            cols,
            pos_bits,
            neg_bits,
            scale,
        }
    }

    /// Multiply matrix by activation vector x: y = scale * (W * x).
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        trit_gemv_parallel(x, &self.pos_bits, &self.neg_bits, y, self.rows, self.cols, self.scale);
    }

    /// Multiply matrix by activation matrix X [M, K]: Y = scale * (X * W^T).
    pub fn gemm(&self, x: &[f32], y: &mut [f32], m: usize) {
        assert_eq!(x.len(), m * self.cols);
        assert_eq!(y.len(), m * self.rows);
        trit_gemm_parallel(x, &self.pos_bits, &self.neg_bits, y, m, self.rows, self.cols, self.scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_level_api() {
        let rows = 64;
        let cols = 128;
        let dense: Vec<i8> = (0..rows * cols)
            .map(|i| match (i * 11) % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();

        let mat = TritMatrix::from_dense(&dense, rows, cols, 1.0);

        let x = vec![1.0f32; cols];
        let mut y = vec![0.0f32; rows];
        mat.gemv(&x, &mut y);

        // Verify first row against dense
        let mut expected0 = 0.0f32;
        for j in 0..cols {
            expected0 += dense[j] as f32;
        }
        assert!((y[0] - expected0).abs() < 1e-4);
    }
}
