/**
 * @file active_experimenter_cuda.cuh
 * @brief Bayesian Hypothesis Update CUDA 加速内核
 *
 * GPU 加速主动实验设计中的贝叶斯推理：
 * - kernel_bayesian_update: 并行后验更新，每个 CUDA thread 处理一个假设
 * - kernel_information_gain: 计算候选实验的期望信息增益
 * - cuda_bayesian_batch_update: 批量 host 接口 (upload -> kernels -> download)
 *
 * 核心算法：
 *   1. 后验更新: posterior[i] = prior[i] * likelihood[i] / evidence
 *   2. 重归一化: 两遍扫描 — 先求和，再除以总和
 *   3. 信息增益: H(prior) - E[H(posterior)]，二元熵 H(p) = -p*log(p) - (1-p)*log(1-p)
 *
 * 使用 FP32 存储概率值（贝叶斯更新对精度敏感）。
 * 批量处理：同时更新 100+ 假设。
 *
 * 设计参考：
 *   - stdp_learning_cuda.cuh（GPU 缓冲、stream 管理模式）
 *   - analogical_transfer_cuda.cuh（CPU/GPU 自动分发模式）
 */

#pragma once

#include <vector>

namespace ai_learning::learning {

// ── GPU Bayesian Data Types ────────────────────────────────────────

/**
 * GpuHypothesis — 单个假设在 GPU 上的紧凑表示。
 *
 * 只保留贝叶斯更新所需的数值字段，
 * 字符串字段（id, statement, domain）留在 CPU 端。
 */
struct GpuHypothesis {
    float prior;           ///< 先验置信度
    float posterior;       ///< 后验置信度（kernel 输出）
    float likelihood;      ///< 似然值（由 CPU 端根据证据计算后传入）
    float information_value; ///< 信息价值（kernel 输出）
};

/**
 * EvidenceType — 证据类型枚举。
 */
enum class EvidenceType : int {
    Binary = 0,     ///< 二元证据：支持/反对
    Continuous = 1  ///< 连续证据：实数值 [0, 1]
};

/**
 * GpuEvidence — GPU 端证据数据。
 *
 * 二元证据: strength = 1.0 (支持) 或 0.0 (反对)
 * 连续证据: strength = [0.0, 1.0] 区间的实数值
 */
struct GpuEvidence {
    EvidenceType type;   ///< 证据类型
    float strength;      ///< 证据强度
};

/**
 * BayesianUpdateInput — 从 CPU 端准备的 GPU 更新输入数据。
 */
struct BayesianUpdateInput {
    /// 假设数据（prior + likelihood）
    std::vector<GpuHypothesis> hypotheses;

    /// 证据数据（每个假设对应一个证据）
    std::vector<GpuEvidence> evidence;

    /// 候选实验的假设索引（用于信息增益计算）
    /// 每个实验对应一组待测试的假设索引
    std::vector<int> experiment_hypothesis_indices;

    /// 每个实验包含的假设数量
    std::vector<int> experiment_hypothesis_counts;

    /// 假设间独立性因子（用于多元贝叶斯更新，默认 1.0）
    float independence_factor = 1.0f;
};

/**
 * BayesianUpdateResult — GPU 更新输出。
 */
struct BayesianUpdateResult {
    /// 更新后的后验置信度（与输入假设一一对应）
    std::vector<float> posteriors;

    /// 每个假设的信息增益（与输入假设一一对应）
    std::vector<float> information_gains;

    /// 每个候选实验的期望信息增益
    std::vector<float> experiment_expected_gains;

    /// 假设数量
    int num_hypotheses;
};

// ── CUDA Function Declarations ──────────────────────────────────────

/**
 * GPU 加速贝叶斯批量更新 — 完整流程。
 *
 * 1. 上传假设先验和似然数据到 GPU
 * 2. 运行 kernel_bayesian_update 计算后验并重归一化
 * 3. 运行 kernel_information_gain 计算信息增益
 * 4. 下载结果
 *
 * @param input  CPU 端准备的输入数据
 * @return BayesianUpdateResult 更新结果
 */
auto cuda_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult;

/**
 * CPU 回退：纯 CPU 实现相同算法。
 * 用于小规模数据或无 CUDA 环境。
 *
 * @param input  CPU 端准备的输入数据
 * @return BayesianUpdateResult 更新结果
 */
auto cpu_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult;

// ── CPU/GPU Auto-Dispatch ───────────────────────────────────────────

/// 假设数量超过此阈值时启用 CUDA
inline constexpr int CUDA_BAYESIAN_THRESHOLD = 64;

/**
 * 自动分发贝叶斯更新：大规模用 GPU，小规模用 CPU。
 *
 * @param input  CPU 端准备的输入数据
 * @return BayesianUpdateResult 更新结果
 */
auto dispatch_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult;

// ── Utility: Build Input from ActiveExperimenter ────────────────────

struct Hypothesis;  // forward decl from active_experimenter.hpp

/**
 * 从 ActiveExperimenter 的假设列表构建 GPU 更新输入。
 *
 * @param hypotheses  活跃假设列表
 * @param evidence    对应的证据列表（支持/反对 + 强度）
 * @return BayesianUpdateInput GPU 输入数据
 */
auto build_bayesian_input(
    const std::vector<Hypothesis>& hypotheses,
    const std::vector<GpuEvidence>& evidence) -> BayesianUpdateInput;

}  // namespace ai_learning::learning
