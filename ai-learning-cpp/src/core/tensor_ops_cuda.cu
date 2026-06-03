/**
 * @file tensor_ops_cuda.cu
 * @brief CUDA 加速的 tensor 运算 — GPU 实现（FP32 + FP16 Tensor Core）
 *
 * 加速 PredictiveCodingEngine 的核心矩阵运算：
 * - mat_vec / mat_vec_bias: 矩阵×向量+偏置 (FP32 cuBLAS Sgemv)
 * - mat_vec_fp16: 矩阵×向量 FP16 Tensor Core (cublasGemmEx)
 * - vec_mat: 向量×矩阵（转置传播）
 * - mat_add_outer: 外积权重更新
 * - tensor_relu / tensor_relu_deriv / tensor_clamp: 激活函数
 *
 * 使用 cuBLAS Sgemv + 自定义 CUDA kernel 实现高效运算。
 * FP16 路径使用 cublasGemmEx + CUBLAS_COMPUTE_32F 保证精度。
 */

#include "ai_learning/core/tensor_ops.hpp"
#include "ai_learning/core/cuda_fp16_utils.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

namespace ai_learning::core {

// ── CUDA 资源管理（懒加载单例）────────────────────────────────────

struct CudaContext {
    cublasHandle_t cublas{nullptr};
    cudaStream_t stream{nullptr};
    bool initialized{false};
    bool available{false};
};

static CudaContext g_cuda;
static std::once_flag g_cuda_init_flag;
static GpuBufferPool g_buffer_pool;

static void init_cuda_context() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] No GPU available: " << cudaGetErrorString(err) << "\n";
        g_cuda.available = false;
        g_cuda.initialized = true;
        return;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    err = cudaStreamCreate(&g_cuda.stream);
    if (err != cudaSuccess) {
        g_cuda.available = false;
        g_cuda.initialized = true;
        return;
    }

    cublasStatus_t cb_err = cublasCreate(&g_cuda.cublas);
    if (cb_err != CUBLAS_STATUS_SUCCESS) {
        g_cuda.available = false;
        g_cuda.initialized = true;
        return;
    }

    cublasSetStream(g_cuda.cublas, g_cuda.stream);

    g_cuda.available = true;
    g_cuda.initialized = true;

    std::cout << "[CUDA] Initialized on " << prop.name
              << " (SM " << prop.major << "." << prop.minor
              << ", " << prop.totalGlobalMem / (1024 * 1024) << " MB)\n";
}

auto cuda_available() -> bool {
    std::call_once(g_cuda_init_flag, init_cuda_context);
    return g_cuda.available;
}

// ── GPU 内存辅助 ────────────────────────────────────────────────

class GpuBuffer {
public:
    explicit GpuBuffer(size_t bytes) : size_(bytes) {
        cudaMalloc(&ptr_, bytes);
    }
    ~GpuBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    void upload(const float* host, size_t bytes) {
        cudaMemcpyAsync(ptr_, host, bytes, cudaMemcpyHostToDevice, g_cuda.stream);
    }

    void download(float* host, size_t bytes) const {
        cudaMemcpyAsync(host, ptr_, bytes, cudaMemcpyDeviceToHost, g_cuda.stream);
        cudaStreamSynchronize(g_cuda.stream);
    }

    auto ptr() -> float* { return ptr_; }
    auto ptr() const -> const float* { return ptr_; }

private:
    float* ptr_{nullptr};
    size_t size_;
};

// ── CUDA Kernel 定义（必须在 host 函数之前）──────────────────────

static constexpr int BLOCK_SIZE = 256;
static int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

__global__ void kernel_relu(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f;
    }
}

__global__ void kernel_relu_deriv(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

__global__ void kernel_clamp(const float* in, float* out, int n, float lo, float hi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(fmaxf(in[idx], lo), hi);
    }
}

__global__ void kernel_sub(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void kernel_mul(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_scale(const float* a, float s, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * s;
    }
}

// 向量 × 矩阵: vec(cols) × mat(cols×rows, stride=rows) → result(rows)
__global__ void kernel_vec_mat(const float* vec, int cols, int rows,
                                const float* mat, float* result) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += vec[c] * mat[c * rows + r];
        }
        result[r] = sum;
    }
}

// 外积更新（row-major）: mat(a_size × b_size) += lr * a * b^T
__global__ void kernel_mat_add_outer(float* mat, int a_size, int b_size,
                                      float lr, const float* a, const float* b) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row (a index)
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col (b index)
    if (i < a_size && j < b_size) {
        mat[i * b_size + j] += lr * a[i] * b[j];
    }
}

// ── 矩阵运算 ──────────────────────────────────────────────────────

auto cuda_mat_vec(const std::vector<float>& mat,
                  int rows, int cols,
                  const Tensor& vec) -> Tensor {
    assert(static_cast<int>(mat.size()) == rows * cols);
    assert(static_cast<int>(vec.size()) == cols);

    Tensor result(rows, 0.0f);

    float alpha = 1.0f;
    float beta = 0.0f;

    GpuBuffer d_mat(mat.size() * sizeof(float));
    GpuBuffer d_vec(vec.size() * sizeof(float));
    GpuBuffer d_result(rows * sizeof(float));

    d_mat.upload(mat.data(), mat.size() * sizeof(float));
    d_vec.upload(vec.data(), vec.size() * sizeof(float));
    cudaMemsetAsync(d_result.ptr(), 0, rows * sizeof(float), g_cuda.stream);

    // mat is row-major (rows × cols): mat[r * cols + c]
    // cuBLAS col-major interpretation: A is (cols × rows)^T
    // result = mat * vec = Sgemv(OP_T, m=cols, n=rows, ...)
    cublasSgemv(g_cuda.cublas,
                CUBLAS_OP_T,
                cols, rows,
                &alpha,
                d_mat.ptr(), cols,
                d_vec.ptr(), 1,
                &beta,
                d_result.ptr(), 1);

    d_result.download(result.data(), rows * sizeof(float));
    return result;
}

auto cuda_vec_mat(const Tensor& vec,
                  const std::vector<float>& mat,
                  int cols, int rows) -> Tensor {
    Tensor result(rows, 0.0f);

    GpuBuffer d_vec(vec.size() * sizeof(float));
    GpuBuffer d_mat(mat.size() * sizeof(float));
    GpuBuffer d_result(rows * sizeof(float));

    d_vec.upload(vec.data(), vec.size() * sizeof(float));
    d_mat.upload(mat.data(), mat.size() * sizeof(float));
    cudaMemsetAsync(d_result.ptr(), 0, rows * sizeof(float), g_cuda.stream);

    kernel_vec_mat<<<grid_size(rows), BLOCK_SIZE, 0, g_cuda.stream>>>(
        d_vec.ptr(), cols, rows, d_mat.ptr(), d_result.ptr());

    d_result.download(result.data(), rows * sizeof(float));
    return result;
}

void cuda_mat_add_outer(std::vector<float>& mat,
                         float lr,
                         const Tensor& a,
                         const Tensor& b) {
    auto a_size = static_cast<int>(a.size());
    auto b_size = static_cast<int>(b.size());

    GpuBuffer d_mat(mat.size() * sizeof(float));
    GpuBuffer d_a(a.size() * sizeof(float));
    GpuBuffer d_b(b.size() * sizeof(float));

    d_mat.upload(mat.data(), mat.size() * sizeof(float));
    d_a.upload(a.data(), a.size() * sizeof(float));
    d_b.upload(b.data(), b.size() * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((b_size + 15) / 16, (a_size + 15) / 16);
    kernel_mat_add_outer<<<grid, block, 0, g_cuda.stream>>>(
        d_mat.ptr(), a_size, b_size, lr, d_a.ptr(), d_b.ptr());

    d_mat.download(mat.data(), mat.size() * sizeof(float));
}

auto cuda_mat_vec_bias(const std::vector<float>& mat,
                        int rows, int cols,
                        const Tensor& vec,
                        const Tensor& bias) -> Tensor {
    auto result = cuda_mat_vec(mat, rows, cols, vec);
    for (int i = 0; i < rows; ++i) {
        result[i] += bias[i];
    }
    return result;
}

// ── FP16 Tensor Core mat_vec ──────────────────────────────────────

/**
 * FP16 矩阵×向量 — 使用 Tensor Core cublasGemmEx 加速
 *
 * 适用场景：计算密集型操作（大矩阵乘法、embedding lookup）
 * 知识查询等精度敏感场景应继续使用 cuda_mat_vec (FP32)
 *
 * 实现策略：
 * 1. FP32 host → FP16 device（利用 GpuBufferPool 减少 malloc）
 * 2. cublasGemmEx with CUDA_R_16F + CUBLAS_COMPUTE_32F (FP32 accumulate)
 * 3. FP16 device → FP32 host
 */
auto cuda_mat_vec_fp16(const std::vector<float>& mat,
                        int rows, int cols,
                        const Tensor& vec) -> Tensor {
    assert(static_cast<int>(mat.size()) == rows * cols);
    assert(static_cast<int>(vec.size()) == cols);

    // Fall back to FP32 if FP16 not supported
    if (!cuda_fp16_available()) {
        return cuda_mat_vec(mat, rows, cols, vec);
    }

    Tensor result(rows, 0.0f);

    size_t mat_bytes = static_cast<size_t>(rows * cols) * sizeof(half);
    size_t vec_bytes = static_cast<size_t>(cols) * sizeof(half);
    size_t res_bytes = static_cast<size_t>(rows) * sizeof(half);

    // Allocate FP16 device buffers from pool
    half* d_mat = static_cast<half*>(g_buffer_pool.get(mat_bytes));
    half* d_vec = static_cast<half*>(g_buffer_pool.get(vec_bytes));
    half* d_res = static_cast<half*>(g_buffer_pool.get(res_bytes));

    if (!d_mat || !d_vec || !d_res) {
        // Pool allocation failed, release what we got and fallback
        if (d_mat) g_buffer_pool.release(d_mat, mat_bytes);
        if (d_vec) g_buffer_pool.release(d_vec, vec_bytes);
        if (d_res) g_buffer_pool.release(d_res, res_bytes);
        return cuda_mat_vec(mat, rows, cols, vec);
    }

    // Convert FP32 host data to FP16 on host, then upload
    // mat: rows*cols elements
    std::vector<half> h_mat(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        h_mat[i] = __float2half(mat[i]);
    }
    // vec: cols elements
    std::vector<half> h_vec(cols);
    for (int i = 0; i < cols; ++i) {
        h_vec[i] = __float2half(vec[i]);
    }

    cudaMemcpyAsync(d_mat, h_mat.data(), mat_bytes,
                    cudaMemcpyHostToDevice, g_cuda.stream);
    cudaMemcpyAsync(d_vec, h_vec.data(), vec_bytes,
                    cudaMemcpyHostToDevice, g_cuda.stream);
    cudaMemsetAsync(d_res, 0, res_bytes, g_cuda.stream);

    // Treat mat×vec as a GEMM:
    //   mat is (rows × cols), vec is (cols × 1), result is (rows × 1)
    //   C(m,1) = A(m,k) * B(k,1)
    //   cublasGemmEx with row-major: use OP_T trick
    bool ok = cublas_fp16_gemm(
        g_cuda.cublas, g_cuda.stream,
        rows, 1, cols,
        d_mat, d_vec, d_res,
        1.0f, 0.0f);

    if (!ok) {
        // GEMM failed, release and fallback
        g_buffer_pool.release(d_mat, mat_bytes);
        g_buffer_pool.release(d_vec, vec_bytes);
        g_buffer_pool.release(d_res, res_bytes);
        return cuda_mat_vec(mat, rows, cols, vec);
    }

    // Download result: FP16 → FP32
    std::vector<half> h_res(rows);
    cudaMemcpyAsync(h_res.data(), d_res, res_bytes,
                    cudaMemcpyDeviceToHost, g_cuda.stream);
    cudaStreamSynchronize(g_cuda.stream);

    for (int i = 0; i < rows; ++i) {
        result[i] = __half2float(h_res[i]);
    }

    // Return buffers to pool
    g_buffer_pool.release(d_mat, mat_bytes);
    g_buffer_pool.release(d_vec, vec_bytes);
    g_buffer_pool.release(d_res, res_bytes);

    return result;
}

// ── 向量运算 host 函数 ──────────────────────────────────────────

auto cuda_tensor_relu(const Tensor& a) -> Tensor {
    auto n = static_cast<int>(a.size());
    Tensor result(n);
    GpuBuffer d_a(n * sizeof(float));
    GpuBuffer d_r(n * sizeof(float));
    d_a.upload(a.data(), n * sizeof(float));
    kernel_relu<<<grid_size(n), BLOCK_SIZE, 0, g_cuda.stream>>>(
        d_a.ptr(), d_r.ptr(), n);
    d_r.download(result.data(), n * sizeof(float));
    return result;
}

auto cuda_tensor_relu_deriv(const Tensor& a) -> Tensor {
    auto n = static_cast<int>(a.size());
    Tensor result(n);
    GpuBuffer d_a(n * sizeof(float));
    GpuBuffer d_r(n * sizeof(float));
    d_a.upload(a.data(), n * sizeof(float));
    kernel_relu_deriv<<<grid_size(n), BLOCK_SIZE, 0, g_cuda.stream>>>(
        d_a.ptr(), d_r.ptr(), n);
    d_r.download(result.data(), n * sizeof(float));
    return result;
}

auto cuda_tensor_clamp(const Tensor& a, float lo, float hi) -> Tensor {
    auto n = static_cast<int>(a.size());
    Tensor result(n);
    GpuBuffer d_a(n * sizeof(float));
    GpuBuffer d_r(n * sizeof(float));
    d_a.upload(a.data(), n * sizeof(float));
    kernel_clamp<<<grid_size(n), BLOCK_SIZE, 0, g_cuda.stream>>>(
        d_a.ptr(), d_r.ptr(), n, lo, hi);
    d_r.download(result.data(), n * sizeof(float));
    return result;
}

auto cuda_tensor_sub(const Tensor& a, const Tensor& b) -> Tensor {
    auto n = static_cast<int>(a.size());
    Tensor result(n);
    GpuBuffer d_a(n * sizeof(float));
    GpuBuffer d_b(n * sizeof(float));
    GpuBuffer d_r(n * sizeof(float));
    d_a.upload(a.data(), n * sizeof(float));
    d_b.upload(b.data(), n * sizeof(float));
    kernel_sub<<<grid_size(n), BLOCK_SIZE, 0, g_cuda.stream>>>(
        d_a.ptr(), d_b.ptr(), d_r.ptr(), n);
    d_r.download(result.data(), n * sizeof(float));
    return result;
}

auto cuda_tensor_mul(const Tensor& a, const Tensor& b) -> Tensor {
    auto n = static_cast<int>(a.size());
    Tensor result(n);
    GpuBuffer d_a(n * sizeof(float));
    GpuBuffer d_b(n * sizeof(float));
    GpuBuffer d_r(n * sizeof(float));
    d_a.upload(a.data(), n * sizeof(float));
    d_b.upload(b.data(), n * sizeof(float));
    kernel_mul<<<grid_size(n), BLOCK_SIZE, 0, g_cuda.stream>>>(
        d_a.ptr(), d_b.ptr(), d_r.ptr(), n);
    d_r.download(result.data(), n * sizeof(float));
    return result;
}

auto cuda_tensor_scale(const Tensor& a, float s) -> Tensor {
    auto n = static_cast<int>(a.size());
    Tensor result(n);
    GpuBuffer d_a(n * sizeof(float));
    GpuBuffer d_r(n * sizeof(float));
    d_a.upload(a.data(), n * sizeof(float));
    kernel_scale<<<grid_size(n), BLOCK_SIZE, 0, g_cuda.stream>>>(
        d_a.ptr(), s, d_r.ptr(), n);
    d_r.download(result.data(), n * sizeof(float));
    return result;
}

}  // namespace ai_learning::core
