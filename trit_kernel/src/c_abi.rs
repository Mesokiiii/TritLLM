//! C ABI foreign function interface for linking with Python (`ctypes`, `cffi`, PyTorch C++ extension).
//!
//! All functions follow standard C calling conventions (`extern "C"`) and are protected
//! against unwinding panics across FFI boundaries.

use std::panic::catch_unwind;
use crate::packing::{pack_ternary_weights, unpack_ternary_scalar, pack_into_blocks256, TritBlock256};
use crate::gemv::{trit_gemv_single_thread, trit_gemv_parallel};
use crate::gemm::{trit_gemm_single_thread, trit_gemm_parallel, trit_gemm_blocks_parallel, trit_gemm_naive_scalar};

/// Pack dense ternary matrix [N, K] with values {-1, 0, 1} into dual bitplanes.
///
/// Output buffers `pos_out` and `neg_out` must have at least `N * ((K + 7) / 8)` bytes.
#[no_mangle]
pub unsafe extern "C" fn trit_pack_weights(
    weights_ptr: *const i8,
    n: usize,
    k: usize,
    pos_out: *mut u8,
    neg_out: *mut u8,
) -> i32 {
    if weights_ptr.is_null() || pos_out.is_null() || neg_out.is_null() || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let src = std::slice::from_raw_parts(weights_ptr, n * k);
        let bytes_per_row = (k + 7) / 8;
        let pos_slice = std::slice::from_raw_parts_mut(pos_out, n * bytes_per_row);
        let neg_slice = std::slice::from_raw_parts_mut(neg_out, n * bytes_per_row);
        pack_ternary_weights(src, pos_slice, neg_slice);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Unpack dual bitplanes back to dense `i8` ternary matrix [N, K].
#[no_mangle]
pub unsafe extern "C" fn trit_unpack_weights(
    pos_in: *const u8,
    neg_in: *const u8,
    n: usize,
    k: usize,
    weights_out: *mut i8,
) -> i32 {
    if pos_in.is_null() || neg_in.is_null() || weights_out.is_null() || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let bytes_per_row = (k + 7) / 8;
        let pos_slice = std::slice::from_raw_parts(pos_in, n * bytes_per_row);
        let neg_slice = std::slice::from_raw_parts(neg_in, n * bytes_per_row);
        let dst = std::slice::from_raw_parts_mut(weights_out, n * k);
        unpack_ternary_scalar(pos_slice, neg_slice, dst);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Pack dense ternary matrix [N, K] into 64-byte `TritBlock256` array.
#[no_mangle]
pub unsafe extern "C" fn trit_pack_blocks256(
    weights_ptr: *const i8,
    n: usize,
    k: usize,
    blocks_out: *mut u8,
) -> i32 {
    if weights_ptr.is_null() || blocks_out.is_null() || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let src = std::slice::from_raw_parts(weights_ptr, n * k);
        let blocks_per_row = (k + 255) / 256;
        let blocks_slice = std::slice::from_raw_parts_mut(
            blocks_out as *mut TritBlock256,
            n * blocks_per_row,
        );
        pack_into_blocks256(src, n, k, blocks_slice);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Single-threaded Addition-Only GEMV: y = scale * (W * x)
#[no_mangle]
pub unsafe extern "C" fn trit_gemv(
    x_ptr: *const f32,
    pos_ptr: *const u8,
    neg_ptr: *const u8,
    y_ptr: *mut f32,
    n: usize,
    k: usize,
    scale: f32,
) -> i32 {
    if x_ptr.is_null() || pos_ptr.is_null() || neg_ptr.is_null() || y_ptr.is_null() || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let x = std::slice::from_raw_parts(x_ptr, k);
        let bytes_per_row = (k + 7) / 8;
        let pos_bits = std::slice::from_raw_parts(pos_ptr, n * bytes_per_row);
        let neg_bits = std::slice::from_raw_parts(neg_ptr, n * bytes_per_row);
        let y = std::slice::from_raw_parts_mut(y_ptr, n);
        trit_gemv_single_thread(x, pos_bits, neg_bits, y, n, k, scale);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Multi-threaded Addition-Only GEMV using Rayon.
#[no_mangle]
pub unsafe extern "C" fn trit_gemv_parallel_c(
    x_ptr: *const f32,
    pos_ptr: *const u8,
    neg_ptr: *const u8,
    y_ptr: *mut f32,
    n: usize,
    k: usize,
    scale: f32,
) -> i32 {
    if x_ptr.is_null() || pos_ptr.is_null() || neg_ptr.is_null() || y_ptr.is_null() || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let x = std::slice::from_raw_parts(x_ptr, k);
        let bytes_per_row = (k + 7) / 8;
        let pos_bits = std::slice::from_raw_parts(pos_ptr, n * bytes_per_row);
        let neg_bits = std::slice::from_raw_parts(neg_ptr, n * bytes_per_row);
        let y = std::slice::from_raw_parts_mut(y_ptr, n);
        trit_gemv_parallel(x, pos_bits, neg_bits, y, n, k, scale);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Single-threaded Addition-Only GEMM: Y = scale * (X * W^T)
#[no_mangle]
pub unsafe extern "C" fn trit_gemm(
    x_ptr: *const f32,
    pos_ptr: *const u8,
    neg_ptr: *const u8,
    y_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) -> i32 {
    if x_ptr.is_null() || pos_ptr.is_null() || neg_ptr.is_null() || y_ptr.is_null() || m == 0 || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let x = std::slice::from_raw_parts(x_ptr, m * k);
        let bytes_per_row = (k + 7) / 8;
        let pos_bits = std::slice::from_raw_parts(pos_ptr, n * bytes_per_row);
        let neg_bits = std::slice::from_raw_parts(neg_ptr, n * bytes_per_row);
        let y = std::slice::from_raw_parts_mut(y_ptr, m * n);
        trit_gemm_single_thread(x, pos_bits, neg_bits, y, m, n, k, scale);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Multi-threaded Cache-Blocked Addition-Only GEMM using Rayon.
#[no_mangle]
pub unsafe extern "C" fn trit_gemm_parallel_c(
    x_ptr: *const f32,
    pos_ptr: *const u8,
    neg_ptr: *const u8,
    y_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) -> i32 {
    if x_ptr.is_null() || pos_ptr.is_null() || neg_ptr.is_null() || y_ptr.is_null() || m == 0 || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let x = std::slice::from_raw_parts(x_ptr, m * k);
        let bytes_per_row = (k + 7) / 8;
        let pos_bits = std::slice::from_raw_parts(pos_ptr, n * bytes_per_row);
        let neg_bits = std::slice::from_raw_parts(neg_ptr, n * bytes_per_row);
        let y = std::slice::from_raw_parts_mut(y_ptr, m * n);
        trit_gemm_parallel(x, pos_bits, neg_bits, y, m, n, k, scale);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Multi-threaded GEMM with interleaved 64-byte `TritBlock256`.
#[no_mangle]
pub unsafe extern "C" fn trit_gemm_blocks_parallel_c(
    x_ptr: *const f32,
    blocks_ptr: *const u8,
    y_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) -> i32 {
    if x_ptr.is_null() || blocks_ptr.is_null() || y_ptr.is_null() || m == 0 || n == 0 || k == 0 {
        return -1;
    }
    let res = catch_unwind(|| {
        let x = std::slice::from_raw_parts(x_ptr, m * k);
        let blocks_per_row = (k + 255) / 256;
        let blocks = std::slice::from_raw_parts(blocks_ptr as *const TritBlock256, n * blocks_per_row);
        let y = std::slice::from_raw_parts_mut(y_ptr, m * n);
        trit_gemm_blocks_parallel(x, blocks, y, m, n, k, scale);
    });
    if res.is_ok() { 0 } else { -2 }
}

/// Benchmark comparison helper executable via C ABI.
///
/// Runs naive scalar vs AVX2 single-thread vs AVX2 parallel and writes times in milliseconds.
#[no_mangle]
pub unsafe extern "C" fn trit_benchmark_compare(
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
    naive_ms: *mut f64,
    single_ms: *mut f64,
    parallel_ms: *mut f64,
) -> i32 {
    let res = catch_unwind(|| {
        let mut x = vec![0.5f32; m * k];
        let mut w = vec![0i8; n * k];
        for i in 0..(n * k) {
            w[i] = match (i * 7) % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }
        for i in 0..(m * k) {
            x[i] = ((i as f32) * 0.01).sin();
        }

        let mut pos_bits = vec![0u8; n * ((k + 7) / 8)];
        let mut neg_bits = vec![0u8; n * ((k + 7) / 8)];
        pack_ternary_weights(&w, &mut pos_bits, &mut neg_bits);

        let mut y = vec![0.0f32; m * n];

        // 1. Naive scalar benchmark (limited iterations to avoid taking too long)
        let naive_iters = (iters / 4).max(1);
        let t0 = std::time::Instant::now();
        for _ in 0..naive_iters {
            trit_gemm_naive_scalar(&x, &w, &mut y, m, n, k, 1.0);
        }
        let naive_dur = t0.elapsed().as_secs_f64() * 1000.0 / (naive_iters as f64);
        if !naive_ms.is_null() {
            *naive_ms = naive_dur;
        }

        // 2. Single-threaded AVX2
        // Warmup
        trit_gemm_single_thread(&x, &pos_bits, &neg_bits, &mut y, m, n, k, 1.0);
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            trit_gemm_single_thread(&x, &pos_bits, &neg_bits, &mut y, m, n, k, 1.0);
        }
        let single_dur = t1.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
        if !single_ms.is_null() {
            *single_ms = single_dur;
        }

        // 3. Multi-threaded Rayon AVX2
        // Warmup
        trit_gemm_parallel(&x, &pos_bits, &neg_bits, &mut y, m, n, k, 1.0);
        let t2 = std::time::Instant::now();
        for _ in 0..iters {
            trit_gemm_parallel(&x, &pos_bits, &neg_bits, &mut y, m, n, k, 1.0);
        }
        let parallel_dur = t2.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
        if !parallel_ms.is_null() {
            *parallel_ms = parallel_dur;
        }
    });

    if res.is_ok() { 0 } else { -1 }
}
