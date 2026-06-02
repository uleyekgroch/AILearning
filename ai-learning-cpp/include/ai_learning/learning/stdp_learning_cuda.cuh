/**
 * @file stdp_learning_cuda.cuh
 * @brief STDP (Spike-Timing-Dependent Plasticity) CUDA 加速内核
 *
 * 提供并行 STDP 突触权重更新：
 * - kernel_stdp_update: 每个 CUDA thread 处理一个突触，累积多对 pre-post 时序差
 * - cuda_stdp_batch_update: 批量 host 接口，一次 kernel launch 更新全部突触
 *
 * STDP 规则：
 *   delta_t = t_post - t_pre
 *   delta_t > 0 (pre 先于 post): dw = A_plus  * exp(-delta_t / tau_plus)   (LTP)
 *   delta_t < 0 (post 先于 pre): dw = -A_minus * exp( delta_t / tau_minus)  (LTD)
 *
 * 不使用 FP16 — STDP 处理标量时序值，FP32 精度足够。
 */

#pragma once

#include <vector>

namespace ai_learning::learning {

// ── CUDA 函数声明（由 stdp_learning_cuda.cu 提供）──────────────────

/// 批量 STDP 权重更新 — GPU 加速版本
///
/// @param weights      突触权重数组 [num_synapses]，原地更新
/// @param pre_times    pre 脉冲时间，展平为 [num_synapses * pairs_per_synapse]
/// @param post_times   post 脉冲时间，展平为 [num_synapses * pairs_per_synapse]
/// @param num_synapses 突触数量
/// @param pairs_per_synapse  每个突触的 pre-post 时序对数量
/// @param A_plus       LTP 振幅系数
/// @param A_minus      LTD 振幅系数
/// @param tau_plus     LTP 时间常数 (ms)
/// @param tau_minus    LTD 时间常数 (ms)
/// @param w_min        权重下限
/// @param w_max        权重上限
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

/// 简化版：使用默认 STDP 参数批量更新
void cuda_stdp_batch_update_default(
    float* weights,
    const float* pre_times,
    const float* post_times,
    int num_synapses,
    int pairs_per_synapse);

}  // namespace ai_learning::learning
