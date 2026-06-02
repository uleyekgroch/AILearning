/**
 * @file cuda_fp16_utils.cuh
 * @brief FP16 Mixed Precision Utilities for RTX 4060 Ada (sm_89)
 *
 * Provides:
 * - FP16/FP32 conversion utilities
 * - half2 vectorized operations
 * - Tensor Core GEMM via cublasGemmEx with CUDA_R_16F
 * - GpuBufferPool for memory reuse (reduce cudaMalloc overhead)
 * - FP16 availability detection (sm_70+)
 *
 * Design:
 * - cublasGemmEx with CUDA_R_16F data type + CUBLAS_COMPUTE_32F accumulation
 *   for accuracy (FP32 accumulate preserves precision)
 * - half2 operations for 2x throughput on elementwise kernels
 * - GpuBufferPool avoids repeated cudaMalloc/free in hot loops
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ai_learning::core {

// ── FP16 Availability Detection ───────────────────────────────────

/// Query whether the current GPU supports FP16 Tensor Core operations.
/// Requires sm_70+ (Volta and later). Thread-safe, caches result.
inline auto cuda_fp16_available() -> bool {
    static bool cached = false;
    static bool result = false;
    static std::once_flag flag;
    std::call_once(flag, [] {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            int sm_version = prop.major * 10 + prop.minor;
            result = (sm_version >= 70);
            cached = true;
            if (result) {
                std::cout << "[CUDA FP16] Tensor Core FP16 supported on "
                          << prop.name << " (SM " << prop.major << "."
                          << prop.minor << ")\n";
            }
        }
    });
    return result;
}

// ── FP16 <-> FP32 Conversion ──────────────────────────────────────

/// Convert FP32 host array to FP16 device buffer.
/// @param host_fp32  Source FP32 data on host
/// @param count      Number of elements
/// @param d_fp16     Destination FP16 device pointer (must be pre-allocated)
/// @param stream     CUDA stream
inline void fp32_to_fp16_device(const float* host_fp32, size_t count,
                                 half* d_fp16, cudaStream_t stream) {
    // Stage: allocate temp FP32 on device, copy, convert on GPU
    float* d_fp32 = nullptr;
    cudaMalloc(&d_fp32, count * sizeof(float));
    cudaMemcpyAsync(d_fp32, host_fp32, count * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Launch conversion kernel (1:1 mapping, each thread converts one element)
    int threads = 256;
    int blocks = (static_cast<int>(count) + threads - 1) / threads;
    // Use CUDA's built-in __float2half operator
    // We need a small kernel for this
    auto kernel = [] __device__(const float* in, half* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = __float2half(in[idx]);
        }
    };
    // Lambda kernels not directly launchable; use explicit kernel below
    // Instead, do CPU-side conversion for simplicity and correctness
    std::vector<half> h_fp16(count);
    for (size_t i = 0; i < count; ++i) {
        h_fp16[i] = __float2half(host_fp32[i]);
    }
    cudaMemcpyAsync(d_fp16, h_fp16.data(), count * sizeof(half),
                    cudaMemcpyHostToDevice, stream);
    cudaFree(d_fp32);
}

/// Convert FP16 device buffer to FP32 host array.
/// @param d_fp16     Source FP16 data on device
/// @param count      Number of elements
/// @param host_fp32  Destination FP32 data on host
/// @param stream     CUDA stream
inline void fp16_to_fp32_host(const half* d_fp16, size_t count,
                               float* host_fp32, cudaStream_t stream) {
    std::vector<half> h_fp16(count);
    cudaMemcpyAsync(h_fp16.data(), d_fp16, count * sizeof(half),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (size_t i = 0; i < count; ++i) {
        host_fp32[i] = __half2float(h_fp16[i]);
    }
}

// ── FP16 Batch Conversion Kernels ─────────────────────────────────

/// CUDA kernel: convert FP32 device array to FP16 device array
__global__ void kernel_fp32_to_fp16(const float* in, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

/// CUDA kernel: convert FP16 device array to FP32 device array
__global__ void kernel_fp16_to_fp32(const half* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// ── half2 Vectorized Elementwise Operations ───────────────────────

/// half2 vectorized addition: out[i:i+2] = a[i:i+2] + b[i:i+2]
__global__ void kernel_half2_add(const half2* a, const half2* b,
                                  half2* out, int n_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pairs) {
        out[idx] = __hadd2(a[idx], b[idx]);
    }
}

/// half2 vectorized multiply: out[i:i+2] = a[i:i+2] * b[i:i+2]
__global__ void kernel_half2_mul(const half2* a, const half2* b,
                                  half2* out, int n_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pairs) {
        out[idx] = __hmul2(a[idx], b[idx]);
    }
}

/// half2 vectorized scale: out[i:i+2] = a[i:i+2] * s
__global__ void kernel_half2_scale(const half2* a, half2 s,
                                    half2* out, int n_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pairs) {
        out[idx] = __hmul2(a[idx], s);
    }
}

// ── Tensor Core GEMM Helper ───────────────────────────────────────

/**
 * Perform FP16 GEMM using Tensor Cores via cublasGemmEx.
 *
 * Computes: C = alpha * A * B + beta * C
 * Where A, B, C are in FP16, but accumulation is FP32 for accuracy.
 *
 * @param cublas     cuBLAS handle
 * @param stream     CUDA stream
 * @param m          rows of A / rows of C
 * @param n          cols of B / cols of C
 * @param k          cols of A / rows of B
 * @param d_A        FP16 device buffer for A (m x k, row-major will be handled)
 * @param d_B        FP16 device buffer for B (k x n, row-major will be handled)
 * @param d_C        FP16 device buffer for C (m x n, output)
 * @param alpha_fp32 scalar alpha in FP32
 * @param beta_fp32  scalar beta in FP32
 * @return true if GEMM succeeded
 *
 * Note: Inputs are interpreted as column-major by cuBLAS. For row-major
 * data, we use CUBLAS_OP_T transpositions to get correct results:
 *   row-major C(m,n) = A(m,k) * B(k,n)
 *   cublas: C^T(n,m) = B^T(n,k) * A^T(k,m)
 *   => cublasGemmEx(OP_N, OP_N, n, m, k, B, k, A, k, C, n)
 */
inline auto cublas_fp16_gemm(cublasHandle_t cublas, cudaStream_t stream,
                              int m, int n, int k,
                              const half* d_A, const half* d_B, half* d_C,
                              float alpha_fp32 = 1.0f,
                              float beta_fp32 = 0.0f) -> bool {
    // Set math mode to allow Tensor Cores
    cublasSetStream(cublas, stream);
    cublasMath_t prev_math;
    cublasGetMathMode(cublas, &prev_math);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

    // cuBLAS uses column-major. Our data is row-major.
    // Row-major: C(m,n) = A(m,k) * B(k,n)
    // Col-major equivalent: C^T(n,m) = B^T(n,k) * A^T(k,m)
    // So: gemm(N, N, n, m, k, B, k, A, k, C, n)
    // But actually, simpler: use CUBLAS_OP_T for both A and B:
    // gemm(OP_T, OP_T, m, n, k, A, k, B, n, C, n) -- this treats A as
    // row-major naturally. Let me use the standard approach:
    //
    // For row-major A(m,k) stored as A[k][m] in col-major = A^T
    // For row-major B(k,n) stored as B[n][k] in col-major = B^T
    // C(m,n) row-major stored as C[n][m] in col-major = C^T
    //
    // C^T = (A*B)^T = B^T * A^T
    // cublasGemmEx: C_col = alpha * op(A_col) * op(B_col) + beta * C_col
    // We want: C^T = B^T * A^T = op(B_row) * op(A_row)
    // => lda = n (B row-major cols), ldb = k (A row-major cols)
    // => C = n x m, lda for B = n, ldb for A = k, ldc = n

    const half alpha_h = __float2half(alpha_fp32);
    const half beta_h = __float2half(beta_fp32);

    // Use FP32 accumulation for accuracy
    cublasStatus_t status = cublasGemmEx(
        cublas,
        CUBLAS_OP_T,       // op(A): transpose row-major A
        CUBLAS_OP_T,       // op(B): transpose row-major B
        m,                  // rows of op(A) = rows of result
        n,                  // cols of op(B) = cols of result
        k,                  // cols of op(A) = rows of op(B)
        &alpha_h,
        d_A, CUDA_R_16F, k,    // A is m*k row-major => lda=k
        d_B, CUDA_R_16F, n,    // B is k*n row-major => ldb=n
        &beta_h,
        d_C, CUDA_R_16F, n,    // C is m*n row-major => ldc=n
        CUBLAS_COMPUTE_32F,    // FP32 accumulation
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use Tensor Cores
    );

    cublasSetMathMode(cublas, prev_math);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[CUDA FP16] cublasGemmEx failed: " << status << "\n";
        return false;
    }
    return true;
}

// ── GPU Buffer Pool ───────────────────────────────────────────────

/**
 * GpuBufferPool: Reuse GPU memory allocations to reduce cudaMalloc overhead.
 *
 * Typical usage in hot loops:
 *   auto* buf = pool.get(1024 * sizeof(half));
 *   // ... use buf->ptr() ...
 *   pool.release(buf);
 *
 * Thread-safe. Internally caches freed blocks by size bucket.
 */
class GpuBufferPool {
public:
    /// Get a buffer of at least `bytes` bytes. Allocates if none cached.
    auto get(size_t bytes) -> void* {
        std::lock_guard<std::mutex> lock(mutex_);
        // Round up to 256-byte buckets for better reuse
        size_t bucket = (bytes + 255) & ~size_t(255);
        auto& pool = free_buffers_[bucket];
        if (!pool.empty()) {
            void* ptr = pool.back();
            pool.pop_back();
            return ptr;
        }
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bucket);
        if (err != cudaSuccess) {
            std::cerr << "[GpuBufferPool] cudaMalloc failed for "
                      << bucket << " bytes\n";
            return nullptr;
        }
        return ptr;
    }

    /// Return a buffer to the pool for reuse.
    void release(void* ptr, size_t bytes) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t bucket = (bytes + 255) & ~size_t(255);
        free_buffers_[bucket].push_back(ptr);
    }

    /// Free all cached buffers. Call at shutdown.
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [bucket, pool] : free_buffers_) {
            for (void* ptr : pool) {
                cudaFree(ptr);
            }
        }
        free_buffers_.clear();
    }

    ~GpuBufferPool() {
        clear();
    }

private:
    std::mutex mutex_;
    std::unordered_map<size_t, std::vector<void*>> free_buffers_;
};

}  // namespace ai_learning::core
