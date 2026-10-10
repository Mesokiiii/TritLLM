"""
Python ctypes wrapper and integration test for trit_kernel.dll
Exposes bit-packed ternary linear layers with AVX2 addition-only inference.
"""

import ctypes
import os
import sys
import time
import numpy as np
import torch

# Locate the compiled DLL
DLL_PATH = os.path.join(os.path.dirname(__file__), "target", "release", "trit_kernel.dll")
if not os.path.exists(DLL_PATH):
    raise FileNotFoundError(f"Cannot find compiled DLL at {DLL_PATH}")

lib = ctypes.CDLL(DLL_PATH)

# Function signatures
# trit_pack_weights(weights_ptr: *const i8, n: usize, k: usize, pos_out: *mut u8, neg_out: *mut u8) -> i32
lib.trit_pack_weights.argtypes = [
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_char_p,
]
lib.trit_pack_weights.restype = ctypes.c_int32

# trit_unpack_weights(pos_in: *const u8, neg_in: *const u8, n: usize, k: usize, weights_out: *mut i8) -> i32
lib.trit_unpack_weights.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
]
lib.trit_unpack_weights.restype = ctypes.c_int32

# trit_gemv_parallel_c(x_ptr: *const f32, pos_ptr: *const u8, neg_ptr: *const u8, y_ptr: *mut f32, n: usize, k: usize, scale: f32) -> i32
lib.trit_gemv_parallel_c.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_float,
]
lib.trit_gemv_parallel_c.restype = ctypes.c_int32

# trit_gemm_parallel_c(x_ptr: *const f32, pos_ptr: *const u8, neg_ptr: *const u8, y_ptr: *mut f32, m: usize, n: usize, k: usize, scale: f32) -> i32
lib.trit_gemm_parallel_c.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_float,
]
lib.trit_gemm_parallel_c.restype = ctypes.c_int32


class TritLinearPy:
    """High-performance Python wrapper for Rust AVX2 Addition-Only Linear Layer."""

    def __init__(self, in_features: int, out_features: int, scale: float = 1.0):
        self.in_features = in_features
        self.out_features = out_features
        self.scale = float(scale)
        self.bytes_per_row = (in_features + 7) // 8
        self.pos_bits = np.zeros((out_features, self.bytes_per_row), dtype=np.uint8)
        self.neg_bits = np.zeros((out_features, self.bytes_per_row), dtype=np.uint8)

    def load_dense_weights(self, weights: np.ndarray):
        """Weights must be a numpy array of shape [out_features, in_features] with values {-1, 0, 1}."""
        assert weights.shape == (self.out_features, self.in_features)
        weights_i8 = np.ascontiguousarray(weights, dtype=np.int8)

        ret = lib.trit_pack_weights(
            weights_i8.ctypes.data_as(ctypes.c_char_p),
            self.out_features,
            self.in_features,
            self.pos_bits.ctypes.data_as(ctypes.c_char_p),
            self.neg_bits.ctypes.data_as(ctypes.c_char_p),
        )
        if ret != 0:
            raise RuntimeError(f"trit_pack_weights failed with code {ret}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: computes y = scale * (x @ W^T). Supports 1D [K] or 2D [M, K]."""
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            assert x.shape[0] == self.in_features
            y = np.zeros(self.out_features, dtype=np.float32)
            ret = lib.trit_gemv_parallel_c(
                x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.pos_bits.ctypes.data_as(ctypes.c_char_p),
                self.neg_bits.ctypes.data_as(ctypes.c_char_p),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.out_features,
                self.in_features,
                ctypes.c_float(self.scale),
            )
            if ret != 0:
                raise RuntimeError(f"trit_gemv_parallel_c failed with code {ret}")
            return y
        elif x.ndim == 2:
            m, k = x.shape
            assert k == self.in_features
            y = np.zeros((m, self.out_features), dtype=np.float32)
            ret = lib.trit_gemm_parallel_c(
                x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.pos_bits.ctypes.data_as(ctypes.c_char_p),
                self.neg_bits.ctypes.data_as(ctypes.c_char_p),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                m,
                self.out_features,
                self.in_features,
                ctypes.c_float(self.scale),
            )
            if ret != 0:
                raise RuntimeError(f"trit_gemm_parallel_c failed with code {ret}")
            return y
        else:
            raise ValueError("Input must be 1D or 2D")


def test_python_integration():
    print("Testing trit_kernel.dll from Python...")
    print(f"Loaded library from: {DLL_PATH}")

    m, n, k = 16, 2048, 2048
    print(f"Creating test layer with M={m}, N={n}, K={k}")

    # Generate random ternary weights {-1, 0, 1}
    np.random.seed(42)
    w_dense = np.random.choice([-1, 0, 1], size=(n, k)).astype(np.int8)
    x = np.random.randn(m, k).astype(np.float32)

    layer = TritLinearPy(in_features=k, out_features=n, scale=1.0)
    layer.load_dense_weights(w_dense)

    # Reference computation using PyTorch
    torch_w = torch.from_numpy(w_dense.astype(np.float32))
    torch_x = torch.from_numpy(x)
    torch_y = torch.matmul(torch_x, torch_w.t()).numpy()

    # Fast AVX2 Addition-Only Kernel computation
    y_fast = layer.forward(x)

    max_err = np.max(np.abs(torch_y - y_fast))
    mean_err = np.mean(np.abs(torch_y - y_fast))
    print(f"Max absolute discrepancy vs PyTorch FP32 matmul: {max_err:.2e}")
    print(f"Mean absolute discrepancy: {mean_err:.2e}")
    assert max_err < 1e-4, f"Discrepancy too large: {max_err}"
    print("Numerical verification PASSED successfully!")

    # Speed Benchmark from Python
    iters = 50
    print(f"\nBenchmarking Python invocation over {iters} iterations (Batch M={m})...")

    # PyTorch CPU matmul
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(torch_x, torch_w.t())
    dur_torch = (time.perf_counter() - t0) / iters

    # trit_kernel AVX2
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = layer.forward(x)
    dur_trit = (time.perf_counter() - t0) / iters

    gops = (2.0 * m * n * k) / (dur_trit * 1e9)
    print(f"PyTorch CPU (FP32 matmul):   {dur_torch * 1000.0:.2f} ms")
    print(f"trit_kernel AVX2 (Addition): {dur_trit * 1000.0:.2f} ms | Throughput: {gops:.2f} GOPS")
    print(f"Speedup vs PyTorch CPU:      {dur_torch / dur_trit:.2f}x")

    # Now test GEMV (M=1 Decode Token Generation)
    print(f"\nBenchmarking LLM Decode Phase (M=1, N={n}, K={k}) over 100 iterations...")
    x1 = np.random.randn(k).astype(np.float32)
    torch_x1 = torch.from_numpy(x1)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(torch_w, torch_x1)
    dur_torch1 = (time.perf_counter() - t0) / 100

    t0 = time.perf_counter()
    for _ in range(100):
        _ = layer.forward(x1)
    dur_trit1 = (time.perf_counter() - t0) / 100

    gops1 = (2.0 * n * k) / (dur_trit1 * 1e9)
    print(f"PyTorch CPU Decode (M=1):   {dur_torch1 * 1000.0:.3f} ms")
    print(f"trit_kernel AVX2 Decode:    {dur_trit1 * 1000.0:.3f} ms | Throughput: {gops1:.2f} GOPS")
    print(f"Speedup in Decode Phase:    {dur_torch1 / dur_trit1:.2f}x")


if __name__ == "__main__":
    test_python_integration()
