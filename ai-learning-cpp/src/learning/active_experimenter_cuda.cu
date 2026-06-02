/**
 * @file active_experimenter_cuda.cu
 * @brief Bayesian Hypothesis Update CUDA 加速实现
 *
 * 并行贝叶斯后验更新：
 * - kernel_bayesian_update: 每个 CUDA thread 处理一个假设
 *   计算后验 = 先验 * 似然，然后两遍重归一化
 * - kernel_bayesian_normalize: 两遍归一化的第二遍（除以总和）
 * - kernel_information_gain: 计算每个假设的期望信息增益
 * - cuda_bayesian_batch_update: 批量 host 接口
 *
 * 复用 tensor_ops_cuda.cu 的 CudaContext / cuda_available()
 * 通过 extern 声明引用，不重复初始化 GPU。
 *
 * 性能策略：
 * - 单次 kernel launch 处理全部假设（最大化并行度）
 * - 每个 thread 在寄存器中计算后验
 * - 两遍归一化：第一遍 block 归约求和，第二遍 thread 除法
 * - FP32 存储（贝叶斯更新对精度敏感）
 * - 阈值：假设数 > CUDA_BAYESIAN_THRESHOLD 时启用 CUDA
 */

#include "ai_learning/learning/active_experimenter_cuda.cuh"
#include "ai_learning/learning/active_experimenter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

// ── 复用 tensor_ops 的 CUDA 基础设施 ─────────────────────────────
namespace ai_learning::core {
extern auto cuda_available() -> bool;
}  // namespace ai_learning::core

namespace ai_learning::learning {

// ── GPU Buffer Helper ──────────────────────────────────────────────

/// 轻量 GPU 缓冲（仅用于 Bayesian update）
class BayesianGpuBuffer {
public:
    explicit BayesianGpuBuffer(size_t bytes) : size_(bytes) {
        if (cudaMalloc(&ptr_, bytes) != cudaSuccess) {
            ptr_ = nullptr;
            std::cerr << "[Bayesian CUDA] cudaMalloc failed for "
                      << bytes << " bytes\n";
        }
    }
    ~BayesianGpuBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    BayesianGpuBuffer(const BayesianGpuBuffer&) = delete;
    BayesianGpuBuffer& operator=(const BayesianGpuBuffer&) = delete;

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
    auto as_int() -> int* { return static_cast<int*>(ptr_); }
    auto valid() const -> bool { return ptr_ != nullptr; }

private:
    void*  ptr_{nullptr};
    size_t size_;
};

// ── CUDA Constants ─────────────────────────────────────────────────

static constexpr int BAYES_BLOCK = 256;

static int bayes_grid(int n) {
    return (n + BAYES_BLOCK - 1) / BAYES_BLOCK;
}

// ── CUDA Stream Management ─────────────────────────────────────────

static cudaStream_t get_bayesian_stream() {
    static cudaStream_t s_stream = nullptr;
    static bool s_initialized = false;
    if (!s_initialized) {
        cudaStreamCreate(&s_stream);
        s_initialized = true;
    }
    return s_stream;
}

// ── Device-side GpuHypothesis (POD, GPU-friendly) ──────────────────

struct DeviceHypothesis {
    float prior;
    float posterior;
    float likelihood;
    float information_value;
};

struct DeviceEvidence {
    int   type;   // 0=binary, 1=continuous
    float strength;
};

// ── Kernel 1: Bayesian Posterior Compute (Pass 1) ──────────────────

/**
 * kernel_bayesian_update — 计算未归一化的后验概率。
 *
 * 每个 thread 处理一个假设：
 *   二元证据:
 *     likelihood = strength (支持) 或 (1 - strength) (反对)
 *     实际使用 CPU 端预计算的 likelihood 值
 *   连续证据:
 *     likelihood 直接使用传入值
 *
 *   unnormalized_posterior[i] = prior[i] * likelihood[i]
 *
 * 同时计算 block 级部分和用于归一化。
 *
 * @param d_hypotheses  假设数组 [num_hypotheses]，含 prior + likelihood
 * @param d_unnormalized 输出未归一化后验 [num_hypotheses]
 * @param d_block_sums  输出每个 block 的部分和 [num_blocks]
 * @param num_hypotheses 假设数量
 */
__global__ void kernel_bayesian_update(
    const DeviceHypothesis* __restrict__ d_hypotheses,
    float*                  __restrict__ d_unnormalized,
    float*                  __restrict__ d_block_sums,
    int num_hypotheses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;

    if (idx < num_hypotheses) {
        // posterior = prior * likelihood
        float prior = d_hypotheses[idx].prior;
        float likelihood = d_hypotheses[idx].likelihood;

        // 数值安全: 确保 prior 和 likelihood 非负
        prior = fmaxf(prior, 1e-30f);
        likelihood = fmaxf(likelihood, 1e-30f);

        val = prior * likelihood;
        d_unnormalized[idx] = val;
    }

    // Block 级归约求和（用于全局归一化）
    __shared__ float s_sum[BAYES_BLOCK];

    s_sum[threadIdx.x] = val;
    __syncthreads();

    // 归约: 每轮减半
    for (int stride = BAYES_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // block 内 thread 0 写入部分和
    if (threadIdx.x == 0) {
        d_block_sums[blockIdx.x] = s_sum[0];
    }
}

// ── Kernel 1b: Final Normalization ─────────────────────────────────

/**
 * kernel_bayesian_normalize — 用全局总和归一化后验概率。
 *
 * 在 kernel_bayesian_update 之后调用，
 * d_total_sum 已由 CPU 端从 d_block_sums 求和后写回。
 *
 * posterior[i] = unnormalized[i] / total_sum
 *
 * 同时写入 d_hypotheses[i].posterior 和 d_hypotheses[i].information_value。
 *
 * @param d_unnormalized  未归一化后验 [num_hypotheses]
 * @param d_total_sum     全局总和（标量，由 CPU 从 block_sums 归约）
 * @param d_posteriors    输出归一化后验 [num_hypotheses]
 * @param d_info_values   输出信息价值 [num_hypotheses]
 * @param d_priors        先验概率 [num_hypotheses]
 * @param num_hypotheses  假设数量
 */
__global__ void kernel_bayesian_normalize(
    const float* __restrict__ d_unnormalized,
    const float* __restrict__ d_total_sum,
    float*       __restrict__ d_posteriors,
    float*       __restrict__ d_info_values,
    const float* __restrict__ d_priors,
    int num_hypotheses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hypotheses) return;

    float total = *d_total_sum;

    // 数值安全: 防止除以零
    float posterior = (total > 1e-30f)
        ? d_unnormalized[idx] / total
        : 1.0f / __uint2float_rn(num_hypotheses);  // 均匀分布回退

    // Clamp 到 [epsilon, 1-epsilon] 避免 log(0)
    posterior = fmaxf(fminf(posterior, 1.0f - 1e-7f), 1e-7f);

    d_posteriors[idx] = posterior;

    // 信息价值 = KL 散度的近似 = posterior * log(posterior / prior)
    float prior = fmaxf(d_priors[idx], 1e-7f);
    d_info_values[idx] = posterior * logf(posterior / prior);
}

// ── Kernel 2: Information Gain ─────────────────────────────────────

/**
 * kernel_information_gain — 计算每个假设的期望信息增益。
 *
 * 对于每个假设，计算:
 *   IG(h) = H(prior) - E[H(posterior)]
 * 其中 H(p) 是二元熵:
 *   H(p) = -p*log(p) - (1-p)*log(1-p)
 *
 * 对于候选实验的信息增益，需要计算在两种结果下（支持/反对）
 * 的后验熵的加权平均:
 *   E[H(posterior)] = P(support) * H(posterior|support)
 *                    + P(reject) * H(posterior|reject)
 *
 * 简化版: 直接用当前后验计算熵的变化
 *   IG(h) = H(prior(h)) - H(posterior(h))
 *
 * @param d_priors     先验概率 [num_hypotheses]
 * @param d_posteriors 后验概率 [num_hypotheses]
 * @param d_ig_output  输出信息增益 [num_hypotheses]
 * @param num_hypotheses 假设数量
 */
__global__ void kernel_information_gain(
    const float* __restrict__ d_priors,
    const float* __restrict__ d_posteriors,
    float*       __restrict__ d_ig_output,
    int num_hypotheses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hypotheses) return;

    float prior = d_priors[idx];
    float posterior = d_posteriors[idx];

    // 数值安全
    prior = fmaxf(fminf(prior, 1.0f - 1e-7f), 1e-7f);
    posterior = fmaxf(fminf(posterior, 1.0f - 1e-7f), 1e-7f);

    // 二元熵: H(p) = -p*log(p) - (1-p)*log(1-p)
    float h_prior =
        -prior * logf(prior) - (1.0f - prior) * logf(1.0f - prior);
    float h_posterior =
        -posterior * logf(posterior) - (1.0f - posterior) * logf(1.0f - posterior);

    // 信息增益 = 先验熵 - 后验熵（减少的不确定性）
    d_ig_output[idx] = fmaxf(h_prior - h_posterior, 0.0f);
}

// ── Kernel 3: Experiment Expected Information Gain ─────────────────

/**
 * kernel_experiment_gain — 计算候选实验的期望信息增益。
 *
 * 每个实验覆盖一组假设。一个 CUDA thread 处理一个实验:
 *   total_gain[exp] = sum(IG(h)) for h in experiment_hypotheses[exp]
 *
 * @param d_info_gains       每个假设的信息增益 [num_hypotheses]
 * @param d_exp_hyp_indices  实验对应的假设索引（展平）
 * @param d_exp_hyp_offsets  每个实验在 d_exp_hyp_indices 中的起始偏移 [num_experiments]
 * @param d_exp_hyp_counts   每个实验包含的假设数量 [num_experiments]
 * @param d_exp_gains        输出每个实验的期望信息增益 [num_experiments]
 * @param num_experiments    实验数量
 */
__global__ void kernel_experiment_gain(
    const float* __restrict__ d_info_gains,
    const int*   __restrict__ d_exp_hyp_indices,
    const int*   __restrict__ d_exp_hyp_offsets,
    const int*   __restrict__ d_exp_hyp_counts,
    float*       __restrict__ d_exp_gains,
    int num_experiments)
{
    int exp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (exp_idx >= num_experiments) return;

    int offset = d_exp_hyp_offsets[exp_idx];
    int count  = d_exp_hyp_counts[exp_idx];

    float total_gain = 0.0f;
    for (int k = 0; k < count; ++k) {
        int hyp_idx = d_exp_hyp_indices[offset + k];
        total_gain += d_info_gains[hyp_idx];
    }

    // 平均信息增益（除以假设数量）
    d_exp_gains[exp_idx] = (count > 0) ? total_gain / __int2float_rn(count) : 0.0f;
}

// ── Public: CUDA Bayesian Batch Update ─────────────────────────────

auto cuda_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult
{
    int num_hypotheses = static_cast<int>(input.hypotheses.size());
    assert(num_hypotheses > 0);
    assert(static_cast<int>(input.evidence.size()) == num_hypotheses);

    BayesianUpdateResult result;
    result.num_hypotheses = num_hypotheses;
    result.posteriors.resize(num_hypotheses, 0.0f);
    result.information_gains.resize(num_hypotheses, 0.0f);

    cudaStream_t stream = get_bayesian_stream();

    // ── 准备 device-side hypothesis 数据 ─────────────────────────
    std::vector<DeviceHypothesis> dev_hyps(num_hypotheses);
    std::vector<float> priors(num_hypotheses);

    for (int i = 0; i < num_hypotheses; ++i) {
        dev_hyps[i].prior = input.hypotheses[i].prior;
        dev_hyps[i].posterior = input.hypotheses[i].posterior;
        dev_hyps[i].likelihood = input.hypotheses[i].likelihood;
        dev_hyps[i].information_value = input.hypotheses[i].information_value;
        priors[i] = input.hypotheses[i].prior;
    }

    // ── 分配 GPU 缓冲 ─────────────────────────────────────────────
    size_t hyp_bytes   = num_hypotheses * sizeof(DeviceHypothesis);
    size_t float_bytes = num_hypotheses * sizeof(float);

    int num_blocks = bayes_grid(num_hypotheses);
    size_t block_sums_bytes = num_blocks * sizeof(float);

    BayesianGpuBuffer d_hypotheses(hyp_bytes);
    BayesianGpuBuffer d_unnormalized(float_bytes);
    BayesianGpuBuffer d_block_sums(block_sums_bytes);
    BayesianGpuBuffer d_total_sum(sizeof(float));
    BayesianGpuBuffer d_posteriors(float_bytes);
    BayesianGpuBuffer d_info_values(float_bytes);
    BayesianGpuBuffer d_priors(float_bytes);
    BayesianGpuBuffer d_ig_output(float_bytes);

    if (!d_hypotheses.valid() || !d_unnormalized.valid() ||
        !d_block_sums.valid() || !d_total_sum.valid() ||
        !d_posteriors.valid() || !d_info_values.valid() ||
        !d_priors.valid() || !d_ig_output.valid()) {
        std::cerr << "[Bayesian CUDA] GPU buffer allocation failed, "
                  << "falling back to CPU\n";
        return cpu_bayesian_batch_update(input);
    }

    // ── 上传数据 ──────────────────────────────────────────────────
    d_hypotheses.upload(dev_hyps.data(), hyp_bytes, stream);
    d_priors.upload(priors.data(), float_bytes, stream);

    // ── Kernel 1: 计算未归一化后验 + block 部分和 ─────────────────
    kernel_bayesian_update<<<num_blocks, BAYES_BLOCK, 0, stream>>>(
        static_cast<const DeviceHypothesis*>(d_hypotheses.ptr()),
        d_unnormalized.as_float(),
        d_block_sums.as_float(),
        num_hypotheses);

    // ── CPU 端归约 block 部分和 → 全局总和 ────────────────────────
    std::vector<float> block_sums_host(num_blocks);
    d_block_sums.download(block_sums_host.data(), block_sums_bytes, stream);

    float total_sum = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        total_sum += block_sums_host[b];
    }
    d_total_sum.upload(&total_sum, sizeof(float), stream);

    // ── Kernel 1b: 归一化后验 ─────────────────────────────────────
    kernel_bayesian_normalize<<<num_blocks, BAYES_BLOCK, 0, stream>>>(
        d_unnormalized.as_float(),
        d_total_sum.as_float(),
        d_posteriors.as_float(),
        d_info_values.as_float(),
        d_priors.as_float(),
        num_hypotheses);

    // ── Kernel 2: 计算信息增益 ────────────────────────────────────
    kernel_information_gain<<<num_blocks, BAYES_BLOCK, 0, stream>>>(
        d_priors.as_float(),
        d_posteriors.as_float(),
        d_ig_output.as_float(),
        num_hypotheses);

    // ── Kernel 3: 计算实验期望信息增益（可选）─────────────────────
    int num_experiments = static_cast<int>(input.experiment_hypothesis_counts.size());
    if (num_experiments > 0 && !input.experiment_hypothesis_indices.empty()) {
        // 构建偏移数组
        std::vector<int> exp_offsets(num_experiments);
        int offset = 0;
        for (int e = 0; e < num_experiments; ++e) {
            exp_offsets[e] = offset;
            offset += input.experiment_hypothesis_counts[e];
        }

        size_t indices_bytes = input.experiment_hypothesis_indices.size() * sizeof(int);
        size_t offsets_bytes  = num_experiments * sizeof(int);
        size_t counts_bytes   = num_experiments * sizeof(int);
        size_t gains_bytes    = num_experiments * sizeof(float);

        BayesianGpuBuffer d_exp_indices(indices_bytes);
        BayesianGpuBuffer d_exp_offsets(offsets_bytes);
        BayesianGpuBuffer d_exp_counts(counts_bytes);
        BayesianGpuBuffer d_exp_gains(gains_bytes);

        if (d_exp_indices.valid() && d_exp_offsets.valid() &&
            d_exp_counts.valid() && d_exp_gains.valid()) {
            d_exp_indices.upload(input.experiment_hypothesis_indices.data(),
                                 indices_bytes, stream);
            d_exp_offsets.upload(exp_offsets.data(), offsets_bytes, stream);
            d_exp_counts.upload(input.experiment_hypothesis_counts.data(),
                                counts_bytes, stream);

            int exp_grid = bayes_grid(num_experiments);
            kernel_experiment_gain<<<exp_grid, BAYES_BLOCK, 0, stream>>>(
                d_ig_output.as_float(),
                d_exp_indices.as_int(),
                d_exp_offsets.as_int(),
                d_exp_counts.as_int(),
                d_exp_gains.as_float(),
                num_experiments);

            result.experiment_expected_gains.resize(num_experiments);
            d_exp_gains.download(result.experiment_expected_gains.data(),
                                 gains_bytes, stream);
        }
    }

    // ── 下载结果 ──────────────────────────────────────────────────
    d_posteriors.download(result.posteriors.data(), float_bytes, stream);
    d_ig_output.download(result.information_gains.data(), float_bytes, stream);

    return result;
}

// ── CPU Fallback ───────────────────────────────────────────────────

namespace {

/// 二元熵计算
float binary_entropy(float p) {
    p = std::max(std::min(p, 1.0f - 1e-7f), 1e-7f);
    return -p * std::log(p) - (1.0f - p) * std::log(1.0f - p);
}

}  // anonymous namespace

auto cpu_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult
{
    int n = static_cast<int>(input.hypotheses.size());

    BayesianUpdateResult result;
    result.num_hypotheses = n;
    result.posteriors.resize(n, 0.0f);
    result.information_gains.resize(n, 0.0f);

    // Pass 1: 计算未归一化后验
    std::vector<float> unnormalized(n);
    float total_sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        float prior = std::max(input.hypotheses[i].prior, 1e-30f);
        float likelihood = std::max(input.hypotheses[i].likelihood, 1e-30f);
        unnormalized[i] = prior * likelihood;
        total_sum += unnormalized[i];
    }

    // Pass 2: 归一化 + 信息增益
    for (int i = 0; i < n; ++i) {
        float posterior = (total_sum > 1e-30f)
            ? unnormalized[i] / total_sum
            : 1.0f / n;
        posterior = std::max(std::min(posterior, 1.0f - 1e-7f), 1e-7f);
        result.posteriors[i] = posterior;

        float prior = std::max(input.hypotheses[i].prior, 1e-7f);
        result.information_gains[i] =
            std::max(binary_entropy(prior) - binary_entropy(posterior), 0.0f);
    }

    // 计算实验期望信息增益
    int num_experiments = static_cast<int>(input.experiment_hypothesis_counts.size());
    if (num_experiments > 0) {
        result.experiment_expected_gains.resize(num_experiments);
        int offset = 0;
        for (int e = 0; e < num_experiments; ++e) {
            int count = input.experiment_hypothesis_counts[e];
            float total_gain = 0.0f;
            for (int k = 0; k < count; ++k) {
                int hyp_idx = input.experiment_hypothesis_indices[offset + k];
                total_gain += result.information_gains[hyp_idx];
            }
            result.experiment_expected_gains[e] =
                (count > 0) ? total_gain / count : 0.0f;
            offset += count;
        }
    }

    return result;
}

// ── CPU/GPU Auto-Dispatch ──────────────────────────────────────────

auto dispatch_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult
{
    int n = static_cast<int>(input.hypotheses.size());

    if (n >= CUDA_BAYESIAN_THRESHOLD && ai_learning::core::cuda_available()) {
        return cuda_bayesian_batch_update(input);
    }
    return cpu_bayesian_batch_update(input);
}

// ── Build Input from ActiveExperimenter Hypotheses ──────────────────

auto build_bayesian_input(
    const std::vector<Hypothesis>& hypotheses,
    const std::vector<GpuEvidence>& evidence) -> BayesianUpdateInput
{
    BayesianUpdateInput input;
    int n = static_cast<int>(hypotheses.size());
    input.hypotheses.resize(n);
    input.evidence.resize(n);

    for (int i = 0; i < n; ++i) {
        const auto& hyp = hypotheses[i];
        input.hypotheses[i].prior = static_cast<float>(hyp.prior_confidence);
        input.hypotheses[i].posterior = static_cast<float>(hyp.posterior_confidence);
        input.hypotheses[i].likelihood = 0.5f;  // 默认似然
        input.hypotheses[i].information_value = static_cast<float>(hyp.information_value);

        if (i < static_cast<int>(evidence.size())) {
            input.evidence[i] = evidence[i];

            // 根据证据类型计算似然
            float strength = evidence[i].strength;
            if (evidence[i].type == EvidenceType::Binary) {
                // 二元证据: strength=1 表示支持, strength=0 表示反对
                // 似然 = strength * 0.8 + (1 - strength) * 0.2
                // 支持时似然高，反对时似然低
                input.hypotheses[i].likelihood = strength * 0.8f + (1.0f - strength) * 0.2f;
            } else {
                // 连续证据: 直接使用 strength 作为似然
                input.hypotheses[i].likelihood = strength;
            }
        }
    }

    return input;
}

}  // namespace ai_learning::learning
