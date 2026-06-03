/**
 * @file flash_attention_cuda.cu
 * @brief Flash Attention 2 (Dao, 2023) — Tiled GPU Attention Implementation
 *
 * Memory-efficient scaled dot-product attention using tiled computation
 * and online softmax. Adapted for RTX 4060 8GB (sm_89).
 *
 * Algorithm (Flash Attention 2, per query row):
 *   Initialize: O = 0, l = 0, m = -inf
 *   For each KV block j (Bc rows):
 *     1. S_j = Q * K_j^T * scale        [global reads, no N^2 storage]
 *     2. block_max = rowmax(S_j)
 *     3. new_max = max(m, block_max)
 *     4. correction = exp(m - new_max)
 *     5. O = O * correction + exp(S_j - new_max) * V_j
 *     6. l = l * correction + sum(exp(S_j - new_max))
 *     7. m = new_max
 *   O = O / l
 *
 * Memory: O(N * d) instead of O(N^2).
 *
 * Implementation strategy:
 *   - One CUDA thread per query row (avoids synchronization overhead)
 *   - Q row cached in registers, K/V read directly from global memory
 *   - Bc (KV block size) is a template parameter for compile-time unrolling
 *   - FP32 throughout for numerical stability (FP16 Q*K^T deferred to v2)
 *   - Handles seq_len not divisible by Bc (partial last block)
 */

#include "ai_learning/core/flash_attention_cuda.cuh"
#include "ai_learning/core/cuda_fp16_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>

namespace ai_learning::core {

// ── CUDA Context ─────────────────────────────────────────────────────

struct FlashCudaContext {
    cudaStream_t stream{nullptr};
    bool initialized{false};
    bool available{false};
};

static FlashCudaContext g_flash_cuda;
static std::once_flag g_flash_init_flag;

static void init_flash_cuda() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        g_flash_cuda.available = false;
        g_flash_cuda.initialized = true;
        return;
    }

    err = cudaStreamCreate(&g_flash_cuda.stream);
    if (err != cudaSuccess) {
        g_flash_cuda.available = false;
        g_flash_cuda.initialized = true;
        return;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    g_flash_cuda.available = true;
    g_flash_cuda.initialized = true;

    std::cout << "[Flash Attention] CUDA ready on " << prop.name
              << " (SM " << prop.major << "." << prop.minor
              << ", " << prop.totalGlobalMem / (1024 * 1024) << " MB)\n";
}

static auto flash_cuda_available() -> bool {
    std::call_once(g_flash_init_flag, init_flash_cuda);
    return g_flash_cuda.available;
}

// ── GPU Buffer Helper ────────────────────────────────────────────────

class FlashGpuBuffer {
public:
    explicit FlashGpuBuffer(size_t bytes) : size_(bytes) {
        if (cudaMalloc(&ptr_, bytes) != cudaSuccess) {
            ptr_ = nullptr;
        }
    }
    ~FlashGpuBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    FlashGpuBuffer(const FlashGpuBuffer&) = delete;
    FlashGpuBuffer& operator=(const FlashGpuBuffer&) = delete;

    void upload(const void* host, size_t bytes, cudaStream_t stream) {
        cudaMemcpyAsync(ptr_, host, bytes, cudaMemcpyHostToDevice, stream);
    }

    void download(void* host, size_t bytes, cudaStream_t stream) const {
        cudaMemcpyAsync(host, ptr_, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    auto ptr() -> void* { return ptr_; }
    auto valid() const -> bool { return ptr_ != nullptr; }

private:
    void* ptr_{nullptr};
    size_t size_;
};

// ── Flash Attention FP32 Kernel ──────────────────────────────────────
//
// One thread per query row. Each thread:
//   1. Loads its Q row into registers
//   2. Iterates over K/V in blocks of Bc rows
//   3. Uses online softmax (running max/sum) for numerical stability
//   4. Accumulates weighted V in registers
//   5. Writes normalized output at the end
//
// Template parameter Bc controls the KV block size (64 or 128).
// Max supported dim = 256 (fits in registers per thread on sm_89).

template<int Bc>
__global__ void kernel_flash_attention(
    const float* __restrict__ Q,      // [seq_len, dim] row-major
    const float* __restrict__ K,      // [seq_len, dim] row-major
    const float* __restrict__ V,      // [seq_len, dim] row-major
    float* __restrict__ O,            // [seq_len, dim] row-major output
    const int seq_len,
    const int dim,
    const float scale)
{
    // One thread per query row
    const int q_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_row >= seq_len) return;

    // Load Q row into registers (dim <= 256)
    float q[256];
    #pragma unroll 4
    for (int d = 0; d < dim; ++d) {
        q[d] = Q[q_row * dim + d];
    }

    // Online softmax accumulators
    float o[256];          // Output accumulator
    float row_l = 0.0f;    // Running sum of exp(scores - max)
    float row_m = -FLT_MAX; // Running max of attention scores

    #pragma unroll 4
    for (int d = 0; d < dim; ++d) {
        o[d] = 0.0f;
    }

    // Iterate over K/V blocks
    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_start = kv_block * Bc;
        const int kv_end = min(kv_start + Bc, seq_len);
        const int kv_rows = kv_end - kv_start;

        // Step 1: Compute attention scores Q[q_row] dot K[kv_start..kv_end]
        float scores[128]; // Bc <= 128
        float block_max = -FLT_MAX;

        for (int kv = 0; kv < kv_rows; ++kv) {
            float dot = 0.0f;
            const float* k_row = K + (kv_start + kv) * dim;
            #pragma unroll 4
            for (int d = 0; d < dim; ++d) {
                dot += q[d] * k_row[d];
            }
            scores[kv] = dot * scale;
            block_max = fmaxf(block_max, scores[kv]);
        }

        // Step 2: Online softmax update
        // new_max = max(row_m, block_max)
        float new_max = fmaxf(row_m, block_max);
        // Correction: rescale previous accumulations
        float correction = expf(row_m - new_max);

        // Rescale existing output and sum
        #pragma unroll 4
        for (int d = 0; d < dim; ++d) {
            o[d] *= correction;
        }
        float new_l = row_l * correction;

        // Step 3: Compute exp(scores - new_max) and accumulate weighted V
        for (int kv = 0; kv < kv_rows; ++kv) {
            float p = expf(scores[kv] - new_max);
            new_l += p;
            const float* v_row = V + (kv_start + kv) * dim;
            #pragma unroll 4
            for (int d = 0; d < dim; ++d) {
                o[d] += p * v_row[d];
            }
        }

        row_m = new_max;
        row_l = new_l;
    }

    // Step 4: Final normalization O = O / l
    float inv_l = (row_l > 1e-8f) ? (1.0f / row_l) : 0.0f;
    #pragma unroll 4
    for (int d = 0; d < dim; ++d) {
        O[q_row * dim + d] = o[d] * inv_l;
    }
}

// ── CPU Fallback Implementation ──────────────────────────────────────

auto cpu_flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    float scale) -> std::vector<float>
{
    auto total = static_cast<size_t>(seq_len) * dim;
    std::vector<float> O(total, 0.0f);

    for (int i = 0; i < seq_len; ++i) {
        // Compute attention scores for row i
        std::vector<float> scores(seq_len);
        float max_score = -std::numeric_limits<float>::max();

        for (int j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += Q[i * dim + d] * K[j * dim + d];
            }
            scores[j] = dot * scale;
            max_score = std::max(max_score, scores[j]);
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        float inv_sum = (sum_exp > 1e-8f) ? (1.0f / sum_exp) : 0.0f;

        // Weighted sum of V
        for (int j = 0; j < seq_len; ++j) {
            float weight = scores[j] * inv_sum;
            for (int d = 0; d < dim; ++d) {
                O[i * dim + d] += weight * V[j * dim + d];
            }
        }
    }

    return O;
}

// ── Auto-dispatch Wrapper ────────────────────────────────────────────

auto flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config) -> FlashAttentionResult
{
    auto scale = config.scale > 0.0f
        ? config.scale
        : 1.0f / std::sqrt(static_cast<float>(dim));

    // Small inputs: CPU is faster (avoids GPU latency)
    if (seq_len < FLASH_ATTENTION_CUDA_THRESHOLD || !flash_cuda_available()) {
        FlashAttentionResult result;
        result.output = cpu_flash_attention(Q, K, V, seq_len, dim, scale);
        result.used_cuda = false;
        return result;
    }

    return cuda_flash_attention(Q, K, V, seq_len, dim, config);
}

// ── CUDA Flash Attention Wrapper ─────────────────────────────────────

auto cuda_flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config) -> FlashAttentionResult
{
    FlashAttentionResult result;
    result.output.resize(static_cast<size_t>(seq_len) * dim, 0.0f);

    if (!flash_cuda_available()) {
        auto scale = config.scale > 0.0f
            ? config.scale
            : 1.0f / std::sqrt(static_cast<float>(dim));
        result.output = cpu_flash_attention(Q, K, V, seq_len, dim, scale);
        result.used_cuda = false;
        return result;
    }

    auto scale = config.scale > 0.0f
        ? config.scale
        : 1.0f / std::sqrt(static_cast<float>(dim));

    // Validate dim is within supported range
    assert(dim <= 256 && "Flash Attention supports dim <= 256");
    assert(dim > 0 && seq_len > 0 && "Invalid dimensions");

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device buffers
    size_t mat_bytes = static_cast<size_t>(seq_len) * dim * sizeof(float);

    FlashGpuBuffer d_Q(mat_bytes);
    FlashGpuBuffer d_K(mat_bytes);
    FlashGpuBuffer d_V(mat_bytes);
    FlashGpuBuffer d_O(mat_bytes);

    if (!d_Q.valid() || !d_K.valid() || !d_V.valid() || !d_O.valid()) {
        result.output = cpu_flash_attention(Q, K, V, seq_len, dim, scale);
        result.used_cuda = false;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return result;
    }

    // Upload Q, K, V to device
    d_Q.upload(Q.data(), mat_bytes, g_flash_cuda.stream);
    d_K.upload(K.data(), mat_bytes, g_flash_cuda.stream);
    d_V.upload(V.data(), mat_bytes, g_flash_cuda.stream);

    // Launch configuration:
    //   threads_per_block = 64 (one thread per query row, keeps register usage
    //                           manageable with 256-dim local arrays)
    //   num_blocks = ceil(seq_len / 64)
    constexpr int THREADS = 64;
    const int num_blocks = (seq_len + THREADS - 1) / THREADS;

    cudaEventRecord(start, g_flash_cuda.stream);

    // Dispatch based on KV block size (template parameter)
    switch (config.block_size) {
    case 128:
        kernel_flash_attention<128><<<num_blocks, THREADS, 0, g_flash_cuda.stream>>>(
            static_cast<const float*>(d_Q.ptr()),
            static_cast<const float*>(d_K.ptr()),
            static_cast<const float*>(d_V.ptr()),
            static_cast<float*>(d_O.ptr()),
            seq_len, dim, scale);
        break;
    case 64:
    default:
        kernel_flash_attention<64><<<num_blocks, THREADS, 0, g_flash_cuda.stream>>>(
            static_cast<const float*>(d_Q.ptr()),
            static_cast<const float*>(d_K.ptr()),
            static_cast<const float*>(d_V.ptr()),
            static_cast<float*>(d_O.ptr()),
            seq_len, dim, scale);
        break;
    }

    cudaEventRecord(stop, g_flash_cuda.stream);

    // Download result
    d_O.download(result.output.data(), mat_bytes, g_flash_cuda.stream);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.elapsed_ms, start, stop);
    result.used_cuda = true;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

// ── Batched Flash Attention ──────────────────────────────────────────

auto cuda_flash_attention_batch(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int batch,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config) -> FlashAttentionResult
{
    FlashAttentionResult result;
    auto total_elems = static_cast<size_t>(batch) * seq_len * dim;
    result.output.resize(total_elems, 0.0f);

    auto scale = config.scale > 0.0f
        ? config.scale
        : 1.0f / std::sqrt(static_cast<float>(dim));

    if (!flash_cuda_available()) {
        for (int b = 0; b < batch; ++b) {
            size_t off = static_cast<size_t>(b) * seq_len * dim;
            std::vector<float> q_b(Q.begin() + off, Q.begin() + off + seq_len * dim);
            std::vector<float> k_b(K.begin() + off, K.begin() + off + seq_len * dim);
            std::vector<float> v_b(V.begin() + off, V.begin() + off + seq_len * dim);
            auto out = cpu_flash_attention(q_b, k_b, v_b, seq_len, dim, scale);
            std::copy(out.begin(), out.end(), result.output.begin() + off);
        }
        result.used_cuda = false;
        return result;
    }

    // Process each batch item (could be parallelized with streams in future)
    for (int b = 0; b < batch; ++b) {
        size_t off = static_cast<size_t>(b) * seq_len * dim;
        std::vector<float> q_b(Q.begin() + off, Q.begin() + off + seq_len * dim);
        std::vector<float> k_b(K.begin() + off, K.begin() + off + seq_len * dim);
        std::vector<float> v_b(V.begin() + off, V.begin() + off + seq_len * dim);

        auto batch_result = cuda_flash_attention(q_b, k_b, v_b,
            seq_len, dim, config);
        std::copy(batch_result.output.begin(), batch_result.output.end(),
                  result.output.begin() + off);
        result.elapsed_ms += batch_result.elapsed_ms;
    }

    result.used_cuda = true;
    return result;
}

}  // namespace ai_learning::core
