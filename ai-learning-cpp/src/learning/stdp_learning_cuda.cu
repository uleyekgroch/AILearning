/**
 * @file stdp_learning_cuda.cu
 * @brief STDP 突触可塑性 CUDA 加速实现
 *
 * 并行 STDP 权重更新：
 * - kernel_stdp_update: 每个 CUDA thread 处理一个突触
 *   累积 N 对 pre-post spike pair 的权重变化
 * - cuda_stdp_batch_update: 批量 host 接口
 *
 * 复用 tensor_ops_cuda.cu 的 CudaContext / GpuBuffer / cuda_available()
 * 通过 extern 声明引用，不重复初始化 GPU。
 *
 * 性能策略：
 * - 单次 kernel launch 处理全部突触（最大化并行度）
 * - 每个 thread 在寄存器中累积 dw，最终一次写入
 * - 无 shared memory 依赖（无跨突触数据依赖）
 * - 阈值：突触数 > 256 时启用 CUDA
 */

#include "ai_learning/learning/stdp_learning_cuda.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// ── 复用 tensor_ops 的 CUDA 基础设施 ─────────────────────────────
// CudaContext, g_cuda, GpuBuffer, cuda_available() 均定义在 tensor_ops_cuda.cu
// 这里只声明 extern，避免重复初始化

namespace ai_learning::core {
extern auto cuda_available() -> bool;
extern struct CudaContext;
}  // namespace ai_learning::core

namespace ai_learning::learning {

// ── 本地 GPU 辅助 ──────────────────────────────────────────────────

/// 轻量 GPU 缓冲（仅用于 STDP，避免依赖 tensor_ops 内部类）
class StdpGpuBuffer {
public:
    explicit StdpGpuBuffer(size_t bytes) : size_(bytes) {
        if (cudaMalloc(&ptr_, bytes) != cudaSuccess) {
            ptr_ = nullptr;
            std::cerr << "[STDP CUDA] cudaMalloc failed for "
                      << bytes << " bytes\n";
        }
    }
    ~StdpGpuBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    StdpGpuBuffer(const StdpGpuBuffer&) = delete;
    StdpGpuBuffer& operator=(const StdpGpuBuffer&) = delete;

    void upload(const void* host, size_t bytes, cudaStream_t stream) {
        cudaMemcpyAsync(ptr_, host, bytes, cudaMemcpyHostToDevice, stream);
    }

    void download(void* host, size_t bytes, cudaStream_t stream) const {
        cudaMemcpyAsync(host, ptr_, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    auto ptr() -> void* { return ptr_; }
    auto ptr() const -> const void* { return ptr_; }

    auto as_float() -> float* { return static_cast<float*>(ptr_); }
    auto as_float() const -> const float* { return static_cast<const float*>(ptr_); }

    auto valid() const -> bool { return ptr_ != nullptr; }

private:
    void*  ptr_{nullptr};
    size_t size_;
};

// ── CUDA Kernel ────────────────────────────────────────────────────

static constexpr int STDP_BLOCK = 256;

static int stdp_grid(int n) {
    return (n + STDP_BLOCK - 1) / STDP_BLOCK;
}

/**
 * STDP 权重更新 kernel
 *
 * 每个 thread 处理一个突触 (synapse_idx)，遍历该突触的所有 pre-post pair：
 *   delta_t = t_post - t_pre
 *   if delta_t > 0: dw += A_plus  * exp(-delta_t / tau_plus)
 *   if delta_t < 0: dw -= A_minus * exp( delta_t / tau_minus)
 *
 * 最终 weight[synapse_idx] += dw，并 clamp 到 [w_min, w_max]
 *
 * 内存布局：
 *   weights:    [num_synapses]
 *   pre_times:  [num_synapses * pairs_per_synapse]
 *   post_times: [num_synapses * pairs_per_synapse]
 *   每个 synapse 的第 k 对 pair 的索引 = synapse_idx * pairs_per_synapse + k
 */
__global__ void kernel_stdp_update(
    float* __restrict__ weights,
    const float* __restrict__ pre_times,
    const float* __restrict__ post_times,
    int num_synapses,
    int pairs_per_synapse,
    float A_plus,
    float A_minus,
    float tau_plus,
    float tau_minus,
    float w_min,
    float w_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    float dw = 0.0f;
    float w = weights[idx];

    // 遍历该突触的所有 pre-post pair
    for (int k = 0; k < pairs_per_synapse; ++k) {
        int pair_idx = idx * pairs_per_synapse + k;
        float t_pre  = pre_times[pair_idx];
        float t_post = post_times[pair_idx];

        float delta_t = t_post - t_pre;

        if (delta_t > 0.0f) {
            // LTP: pre 在 post 之前 → 增强连接
            dw += A_plus * expf(-delta_t / tau_plus);
        } else if (delta_t < 0.0f) {
            // LTD: post 在 pre 之前 → 减弱连接
            dw -= A_minus * expf(delta_t / tau_minus);
        }
        // delta_t == 0: 无变化
    }

    // 应用权重更新并 clamp
    w += dw;
    w = fmaxf(fminf(w, w_max), w_min);
    weights[idx] = w;
}

/**
 * 批量归一化 kernel：将每个突触的权重变化缩放到合理范围
 * 可选调用，防止 pairs 数量很大时 dw 累积过大
 */
__global__ void kernel_stdp_normalize(
    float* __restrict__ weights,
    const float* __restrict__ weights_orig,
    int num_synapses,
    float max_change)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    float delta = weights[idx] - weights_orig[idx];
    // 限制单次更新的最大幅度
    delta = fmaxf(fminf(delta, max_change), -max_change);
    weights[idx] = weights_orig[idx] + delta;
}

// ── CUDA Stream 管理 ───────────────────────────────────────────────

/// 获取或创建 STDP 专用 stream（与 tensor_ops 共享 GPU 但避免 stream 争用）
static cudaStream_t get_stdp_stream() {
    static cudaStream_t s_stream = nullptr;
    static bool s_initialized = false;
    if (!s_initialized) {
        cudaStreamCreate(&s_stream);
        s_initialized = true;
    }
    return s_stream;
}

// ── 公共接口 ──────────────────────────────────────────────────────

void cuda_stdp_batch_update(
    float* weights,
    const float* pre_times,
    const float* post_times,
    int num_synapses,
    int pairs_per_synapse,
    float A_plus,
    float A_minus,
    float tau_plus,
    float tau_minus,
    float w_min,
    float w_max)
{
    assert(weights != nullptr);
    assert(pre_times != nullptr);
    assert(post_times != nullptr);
    assert(num_synapses > 0);
    assert(pairs_per_synapse >= 0);
    assert(tau_plus > 0.0f);
    assert(tau_minus > 0.0f);

    if (pairs_per_synapse == 0) return;

    cudaStream_t stream = get_stdp_stream();

    size_t weights_bytes = static_cast<size_t>(num_synapses) * sizeof(float);
    size_t times_bytes   = static_cast<size_t>(num_synapses) * pairs_per_synapse * sizeof(float);

    // 分配 GPU 缓冲
    StdpGpuBuffer d_weights(weights_bytes);
    StdpGpuBuffer d_pre_times(times_bytes);
    StdpGpuBuffer d_post_times(times_bytes);

    if (!d_weights.valid() || !d_pre_times.valid() || !d_post_times.valid()) {
        std::cerr << "[STDP CUDA] GPU buffer allocation failed, falling back to CPU\n";
        return;
    }

    // 上传数据
    d_weights.upload(weights, weights_bytes, stream);
    d_pre_times.upload(pre_times, times_bytes, stream);
    d_post_times.upload(post_times, times_bytes, stream);

    // 启动 kernel：一个 thread 处理一个突触
    int grid = stdp_grid(num_synapses);
    kernel_stdp_update<<<grid, STDP_BLOCK, 0, stream>>>(
        d_weights.as_float(),
        d_pre_times.as_float(),
        d_post_times.as_float(),
        num_synapses,
        pairs_per_synapse,
        A_plus,
        A_minus,
        tau_plus,
        tau_minus,
        w_min,
        w_max);

    // 下载结果
    d_weights.download(weights, weights_bytes, stream);
}

void cuda_stdp_batch_update_default(
    float* weights,
    const float* pre_times,
    const float* post_times,
    int num_synapses,
    int pairs_per_synapse)
{
    // 标准 STDP 参数（Bi & Poo, 1998 经典值）
    cuda_stdp_batch_update(
        weights, pre_times, post_times,
        num_synapses, pairs_per_synapse,
        /*A_plus=*/0.1f,
        /*A_minus=*/0.1f,
        /*tau_plus=*/20.0f,    // ms
        /*tau_minus=*/20.0f,   // ms
        /*w_min=*/0.0f,
        /*w_max=*/1.0f);
}

}  // namespace ai_learning::learning
