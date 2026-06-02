/**
 * @file activation_spread_cuda.cuh
 * @brief Activation Spreading CUDA 加速内核 — 知识图谱上的联想推理
 *
 * 在 CSR (Compressed Sparse Row) 格式的知识图谱上执行激活扩散，
 * 本质上是稀疏矩阵-向量乘法 (SpMV)。
 *
 * 提供：
 * - kernel_activation_spread: 单轮激活扩散（SpMV + decay + threshold）
 * - cuda_activation_spread_multi_hop: 多轮扩散
 * - cuda_activation_spread_batch: 批量种子节点同时扩散
 * - CsrGraph: CSR 格式图数据结构（CPU 构建，GPU 计算）
 * - cpu/gpu 自动分发包装器
 *
 * 优化策略：
 * - CSR 格式避免邻接矩阵的稀疏零元素存储
 * - 每个 CUDA thread 处理一个节点的所有入边
 * - FP16 激活值可选（通过 cuda_fp16_utils.cuh）
 * - AtomicAdd 用于并发边贡献（batch 模式）
 */

#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::reasoning {

// ── CSR Graph Format ──────────────────────────────────────────────────

/**
 * CSR (Compressed Sparse Row) 格式的图结构。
 *
 * 用于高效 GPU SpMV:
 *   node i 的邻居 = col_indices[row_ptr[i] .. row_ptr[i+1])
 *   对应边权重   = edge_weights[row_ptr[i] .. row_ptr[i+1])
 *
 * CPU 端构建，一次性上传到 GPU。
 */
struct CsrGraph {
    int num_nodes;                        ///< 节点总数
    int num_edges;                        ///< 有向边总数
    std::vector<int>   row_ptr;           ///< [num_nodes + 1] 行偏移
    std::vector<int>   col_indices;       ///< [num_edges] 列索引（邻居节点 ID）
    std::vector<float> edge_weights;      ///< [num_edges] 边权重（confidence）
    std::vector<std::string> node_id_map; ///< [num_nodes] 节点 ID → 原始 entity_id

    /// 从知识图谱构建 CSR（调用者需传入邻接数据）
    static auto build(
        int num_nodes,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_indices,
        const std::vector<float>& edge_weights,
        std::vector<std::string> node_id_map) -> CsrGraph;
};

// ── 激活扩散结果 ────────────────────────────────────────────────────────

struct SpreadResult {
    std::vector<float> activations;       ///< [num_nodes] 各节点激活值
    int hops_completed;                   ///< 实际扩散轮数
};

struct BatchSpreadResult {
    std::vector<std::vector<float>> activations; ///< [num_queries][num_nodes]
    int hops_completed;
};

// ── CUDA 函数声明（由 activation_spread_cuda.cu 提供）─────────────────

/**
 * 多轮激活扩散 — GPU 加速版本
 *
 * @param graph        CSR 格式图（已上传到 GPU 或在函数内上传）
 * @param seed_activations  种子激活值 [num_nodes]，仅种子节点非零
 * @param decay_rate   衰减因子（0.0~1.0），每轮乘以此值
 * @param threshold    激活阈值，低于此值的节点不参与下一轮扩散
 * @param num_hops     扩散轮数
 * @return SpreadResult 包含最终激活值和实际轮数
 */
auto cuda_activation_spread_multi_hop(
    const CsrGraph& graph,
    const std::vector<float>& seed_activations,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult;

/**
 * 批量种子节点同时扩散 — GPU 加速版本
 *
 * 同时处理多个查询，利用 GPU 并行性。
 * 本质上是 batch SpMV: 多个稀疏向量同时与同一个稀疏矩阵相乘。
 *
 * @param graph        CSR 格式图
 * @param batch_seeds  [num_queries][num_nodes] 每个查询的种子激活
 * @param decay_rate   衰减因子
 * @param threshold    激活阈值
 * @param num_hops     扩散轮数
 * @return BatchSpreadResult
 */
auto cuda_activation_spread_batch(
    const CsrGraph& graph,
    const std::vector<std::vector<float>>& batch_seeds,
    float decay_rate,
    float threshold,
    int num_hops) -> BatchSpreadResult;

// ── CPU/GPU 自动分发 ────────────────────────────────────────────────────

/**
 * 多轮激活扩散 — 自动选择 CPU 或 GPU
 *
 * 当 CUDA 可用且节点数 > CUDA_THRESHOLD 时使用 GPU，否则 CPU。
 */
auto activation_spread_dispatch(
    const CsrGraph& graph,
    const std::vector<float>& seed_activations,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult;

/**
 * 构建 CSR 图并执行激活扩散的便捷函数
 *
 * @param adjacency    邻接表: node_id → [(neighbor_id, weight), ...]
 * @param node_ids     有序节点 ID 列表（确定 CSR 行顺序）
 * @param seed_entity_ids  种子实体 ID 列表
 * @param seed_values  种子激活值列表（与 seed_entity_ids 一一对应）
 * @param decay_rate   衰减因子
 * @param threshold    激活阈值
 * @param num_hops     扩散轮数
 * @return SpreadResult 最终激活值（按 node_ids 顺序）
 */
auto activation_spread_from_adjacency(
    const std::map<std::string, std::vector<std::pair<std::string, float>>>& adjacency,
    const std::vector<std::string>& node_ids,
    const std::vector<std::string>& seed_entity_ids,
    const std::vector<float>& seed_values,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult;

/// CPU 阈值：节点数超过此值时启用 CUDA
inline constexpr int CUDA_SPREAD_THRESHOLD = 256;

}  // namespace ai_learning::reasoning
