/**
 * @file flash_attention_cuda_stub.cpp
 * @brief Flash Attention CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有 CUDA 函数返回 CPU fallback 结果，确保纯 CPU 构建正常工作。
 */

#include "ai_learning/core/flash_attention_cuda.cuh"

#include <cmath>
#include <limits>
#include <vector>

namespace ai_learning::core {

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
        std::vector<float> scores(seq_len);
        float max_score = -std::numeric_limits<float>::max();

        for (int j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += Q[i * dim + d] * K[j * dim + d];
            }
            scores[j] = dot * scale;
            if (scores[j] > max_score) {
                max_score = scores[j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        float inv_sum = (sum_exp > 1e-8f) ? (1.0f / sum_exp) : 0.0f;

        for (int j = 0; j < seq_len; ++j) {
            float weight = scores[j] * inv_sum;
            for (int d = 0; d < dim; ++d) {
                O[i * dim + d] += weight * V[j * dim + d];
            }
        }
    }

    return O;
}

auto cuda_flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config) -> FlashAttentionResult
{
    FlashAttentionResult result;
    auto scale = config.scale > 0.0f
        ? config.scale
        : 1.0f / std::sqrt(static_cast<float>(dim));
    result.output = cpu_flash_attention(Q, K, V, seq_len, dim, scale);
    result.used_cuda = false;
    return result;
}

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

auto flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config) -> FlashAttentionResult
{
    FlashAttentionResult result;
    auto scale = config.scale > 0.0f
        ? config.scale
        : 1.0f / std::sqrt(static_cast<float>(dim));
    result.output = cpu_flash_attention(Q, K, V, seq_len, dim, scale);
    result.used_cuda = false;
    return result;
}

}  // namespace ai_learning::core
