/**
 * @file stdp_learning.hpp
 * @brief STDP 赫布学习 — 时序依赖的突触可塑性
 *
 * STDP (Spike-Timing-Dependent Plasticity)：
 * - 先激活的神经元 → 后激活的神经元：增强连接 (LTP)
 * - 反向时序：减弱连接 (LTD)
 *
 * 用于快速关联推理和 think() 中的概念联想。
 * 支持 CUDA 加速批量权重更新（编译时自动检测）。
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <utility>

namespace ai_learning::core {
// 由 tensor_ops_cuda.cu 或 stub 提供
auto cuda_available() -> bool;
}  // namespace ai_learning::core

namespace ai_learning::learning {

struct STDPConnection {
    std::string pre;
    std::string post;
    double      weight = 0.0;
};

// ── CUDA 加速函数声明（由 stdp_learning_cuda.cu 或 stub 提供）────

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
    float w_max);

void cuda_stdp_batch_update_default(
    float* weights,
    const float* pre_times,
    const float* post_times,
    int num_synapses,
    int pairs_per_synapse);

// ── CPU/GPU 分发批量 STDP 更新 ─────────────────────────────────────

/**
 * 批量 STDP 权重更新 — 自动 CPU/GPU 分发
 *
 * STDP 规则：
 *   delta_t = t_post - t_pre
 *   delta_t > 0: dw = A_plus  * exp(-delta_t / tau_plus)   (LTP)
 *   delta_t < 0: dw = -A_minus * exp( delta_t / tau_minus)  (LTD)
 *
 * @param weights          突触权重 [num_synapses]，原地更新
 * @param pre_times        pre 脉冲时间 [num_synapses * pairs_per_synapse]
 * @param post_times       post 脉冲时间 [num_synapses * pairs_per_synapse]
 * @param num_synapses     突触数量
 * @param pairs_per_synapse 每个突触的 pre-post 时序对数量
 * @param A_plus           LTP 振幅
 * @param A_minus          LTD 振幅
 * @param tau_plus         LTP 时间常数 (ms)
 * @param tau_minus        LTD 时间常数 (ms)
 * @param w_min            权重下限
 * @param w_max            权重上限
 */
inline void stdp_batch_update(
    std::vector<float>& weights,
    const std::vector<float>& pre_times,
    const std::vector<float>& post_times,
    int num_synapses,
    int pairs_per_synapse,
    float A_plus = 0.1f,
    float A_minus = 0.1f,
    float tau_plus = 20.0f,
    float tau_minus = 20.0f,
    float w_min = 0.0f,
    float w_max = 1.0f)
{
    // 小数据走 CPU（避免 PCIe 开销）
    if (num_synapses < 256 || pairs_per_synapse == 0 || !ai_learning::core::cuda_available()) {
        for (int s = 0; s < num_synapses; ++s) {
            float dw = 0.0f;
            for (int k = 0; k < pairs_per_synapse; ++k) {
                int idx = s * pairs_per_synapse + k;
                float delta_t = post_times[idx] - pre_times[idx];
                if (delta_t > 0.0f) {
                    dw += A_plus * std::exp(-delta_t / tau_plus);
                } else if (delta_t < 0.0f) {
                    dw -= A_minus * std::exp(delta_t / tau_minus);
                }
            }
            weights[s] += dw;
            weights[s] = std::clamp(weights[s], w_min, w_max);
        }
        return;
    }

    // GPU 路径
    cuda_stdp_batch_update(
        weights.data(), pre_times.data(), post_times.data(),
        num_synapses, pairs_per_synapse,
        A_plus, A_minus, tau_plus, tau_minus, w_min, w_max);
}

class STDP {
public:
    explicit STDP(double lr = 0.01, double tau = 10.0, double decay = 0.95);

    /// STDP 强化/弱化：timing_delta > 0 表示 pre 先于 post (LTP)
    void strengthen(const std::string& pre, const std::string& post,
                    double timing_delta);

    /// 获取连接权重
    [[nodiscard]] auto get_weight(const std::string& pre,
                                  const std::string& post) const -> double;

    /// 获取与某实体有强连接的所有实体
    [[nodiscard]] auto get_strong_connections(
        const std::string& entity,
        double threshold = 0.2) const -> std::vector<std::pair<std::string, double>>;

    /// 全局衰减
    void decay_all();

    /// 统计
    [[nodiscard]] auto connection_count() const -> size_t;

private:
    double lr_;
    double tau_;
    double decay_;

    // (pre, post) → weight
    std::map<std::pair<std::string, std::string>, double> connections_;
};

}  // namespace ai_learning::learning
