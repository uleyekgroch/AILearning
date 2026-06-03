/**
 * @file active_experimenter_cuda_stub.cpp
 * @brief Active Experimenter CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有 GPU 函数退回到 CPU 实现，确保纯 CPU 构建正常工作。
 * CUDA 函数声明在 active_experimenter_cuda.cuh 中，分发逻辑通过
 * cuda_available() == false 自动回退到 CPU 路径。
 */

#include "ai_learning/learning/active_experimenter_cuda.cuh"
#include "ai_learning/learning/active_experimenter.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ai_learning::learning {

namespace {

/// 二元熵计算
float binary_entropy(float p) {
    p = std::max(std::min(p, 1.0f - 1e-7f), 1e-7f);
    return -p * std::log(p) - (1.0f - p) * std::log(1.0f - p);
}

}  // anonymous namespace

// ── CUDA function stubs ─────────────────────────────────────────────

auto cuda_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult
{
    // stub: delegate to CPU
    return cpu_bayesian_batch_update(input);
}

auto cpu_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult
{
    int n = static_cast<int>(input.hypotheses.size());

    BayesianUpdateResult result;
    result.num_hypotheses = n;
    result.posteriors.resize(n, 0.0f);
    result.information_gains.resize(n, 0.0f);

    // Pass 1: compute unnormalized posteriors
    std::vector<float> unnormalized(n);
    float total_sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        float prior = std::max(input.hypotheses[i].prior, 1e-30f);
        float likelihood = std::max(input.hypotheses[i].likelihood, 1e-30f);
        unnormalized[i] = prior * likelihood;
        total_sum += unnormalized[i];
    }

    // Pass 2: normalize + information gain
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

    // Compute experiment expected information gains
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

// ── CPU/GPU Auto-Dispatch (CPU-only) ────────────────────────────────

auto dispatch_bayesian_batch_update(const BayesianUpdateInput& input)
    -> BayesianUpdateResult
{
    // stub: always CPU
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
        input.hypotheses[i].likelihood = 0.5f;
        input.hypotheses[i].information_value = static_cast<float>(hyp.information_value);

        if (i < static_cast<int>(evidence.size())) {
            input.evidence[i] = evidence[i];

            float strength = evidence[i].strength;
            if (evidence[i].type == EvidenceType::Binary) {
                input.hypotheses[i].likelihood =
                    strength * 0.8f + (1.0f - strength) * 0.2f;
            } else {
                input.hypotheses[i].likelihood = strength;
            }
        }
    }

    return input;
}

}  // namespace ai_learning::learning
