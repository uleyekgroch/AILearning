/**
 * @file embedding_trainer_cuda.cuh
 * @brief Embedding Trainer CUDA 加速内核 — SGNS 批量训练
 *
 * GPU 加速 Skip-gram with Negative Sampling (SGNS) 训练：
 * - kernel_sgns_train: 每个 CUDA thread 处理一个 (center, context) 训练对
 * - kernel_apply_grad: 将累积梯度应用到中心词向量
 * - train_epoch_cuda_dispatch: 完整 epoch 训练接口
 *
 * 核心算法：
 *   对每对 (center, context):
 *     1. 正样本: 最大化 sigma(v_context . v_center)
 *     2. 负采样: 最大化 sigma(-v_neg . v_center)
 *     3. SGD:    v_center += lr * (label - sigma(score)) * v_other
 *
 * 设计参考: stdp_learning_cuda.cuh, analogical_transfer_cuda.cuh
 */

#pragma once

#include <random>
#include <vector>

namespace ai_learning::learning {

// ── CUDA 函数声明（由 embedding_trainer_cuda.cu 提供）──────────────

/// GPU 加速的 SGNS epoch 训练
///
/// 由 EmbeddingTrainer::train() 在 CUDA 可用时调用。
/// 完整执行一个 epoch 的训练，处理所有 (center, context) 对。
///
/// @param W_in             中心词向量矩阵 [vocab * dim]，原地更新
/// @param W_out            上下文词向量矩阵 [vocab * dim]，原地更新
/// @param corpus           训练语料（token index 序列，-1 为文档分隔）
/// @param neg_table        负采样表
/// @param embedding_dim    嵌入维度
/// @param neg_samples      每个正样本的负采样数
/// @param window_size      上下文窗口大小
/// @param epoch            当前 epoch（用于进度报告）
/// @param base_lr          基础学习率
/// @param total_loss       [out] 总 loss
/// @param rng              随机数生成器（用于窗口大小和负采样）
/// @return 处理的训练对数量
auto train_epoch_cuda_dispatch(
    std::vector<float>& W_in,
    std::vector<float>& W_out,
    const std::vector<int>& corpus,
    const std::vector<int>& neg_table,
    int embedding_dim,
    int neg_samples,
    int window_size,
    int epoch,
    double base_lr,
    double& total_loss,
    std::mt19937& rng) -> long long;

}  // namespace ai_learning::learning
