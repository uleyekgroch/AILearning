/**
 * @file flash_attention_cuda.cuh
 * @brief Flash Attention 2 (Dao, 2023) — Tiled GPU Attention Kernel
 *
 * Memory-efficient scaled dot-product attention:
 *   O = softmax(Q * K^T / sqrt(d)) * V
 *
 * Key design choices (adapted for RTX 4060 8GB, sm_89):
 *   - Tiled computation: load Q_block and K_block into shared memory
 *   - Online softmax: running max and sum for numerical stability
 *   - FP16 for Q*K^T (Tensor Core), FP32 for softmax/accumulation
 *   - O(N*d) memory instead of O(N^2) standard attention
 *
 * Integration points:
 *   - embedding_trainer: attention-weighted context scoring
 *   - activation_spread: attention-based message passing
 *
 * Reference: "FlashAttention-2: Faster Attention with Better Parallelism
 *             and Work Partitioning" (Dao, 2023)
 */

#pragma once

#include <vector>

namespace ai_learning::core {

// ── Configuration ────────────────────────────────────────────────────

/// Flash Attention configuration parameters
struct FlashAttentionConfig {
    int block_size = 64;      ///< Tile size (64 or 128, tune for RTX 4060)
    int max_seq_len = 4096;   ///< Maximum sequence length
    int max_dim = 256;        ///< Maximum head dimension
    float scale = 0.0f;       ///< Attention scale (0 => auto = 1/sqrt(dim))
    bool use_fp16 = true;     ///< Use FP16 for Q*K^T (requires sm_70+)
};

/// Flash Attention result
struct FlashAttentionResult {
    std::vector<float> output;  ///< [seq_len * dim] attention output
    bool used_cuda = false;     ///< Whether CUDA was used
    float elapsed_ms = 0.0f;   ///< Execution time in milliseconds
};

// ── CUDA Flash Attention ─────────────────────────────────────────────

/**
 * CUDA Flash Attention — tiled, memory-efficient attention computation.
 *
 * Computes: O = softmax(Q * K^T / sqrt(d)) * V
 * Without materializing the full N x N attention matrix.
 *
 * @param Q  Query matrix [seq_len * dim], row-major FP32
 * @param K  Key matrix   [seq_len * dim], row-major FP32
 * @param V  Value matrix [seq_len * dim], row-major FP32
 * @param seq_len  Sequence length N
 * @param dim      Head dimension d
 * @param config   Configuration (block size, scale, etc.)
 * @return FlashAttentionResult with output [seq_len * dim]
 */
auto cuda_flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config = FlashAttentionConfig{}
) -> FlashAttentionResult;

/**
 * CUDA Flash Attention — batched version for multiple attention heads.
 *
 * @param Q       [batch * seq_len * dim], row-major
 * @param K       [batch * seq_len * dim], row-major
 * @param V       [batch * seq_len * dim], row-major
 * @param batch   Number of attention heads / batch size
 * @param seq_len Sequence length
 * @param dim     Head dimension
 * @param config  Configuration
 * @return FlashAttentionResult with output [batch * seq_len * dim]
 */
auto cuda_flash_attention_batch(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int batch,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config = FlashAttentionConfig{}
) -> FlashAttentionResult;

/**
 * CPU fallback — standard attention for small inputs or non-CUDA builds.
 * Used internally when CUDA is unavailable or seq_len is below threshold.
 */
auto cpu_flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    float scale
) -> std::vector<float>;

/**
 * Auto-dispatch wrapper: selects CUDA or CPU based on availability and size.
 * CUDA is used when available and seq_len >= CUDA_THRESHOLD.
 */
auto flash_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len,
    int dim,
    const FlashAttentionConfig& config = FlashAttentionConfig{}
) -> FlashAttentionResult;

/// Minimum sequence length to use CUDA (below this, CPU is faster)
inline constexpr int FLASH_ATTENTION_CUDA_THRESHOLD = 64;

}  // namespace ai_learning::core
