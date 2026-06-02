/**
 * @file analogical_transfer_cuda.cuh
 * @brief Analogical Transfer Alignment CUDA 加速内核
 *
 * GPU 加速类比迁移中的 N^2 概念对齐：
 * - kernel_jaccard_batch: 并行计算所有 source-target 概念对的 Jaccard 相似度
 * - kernel_greedy_alignment: 基于相似度矩阵的贪心匹配
 * - cuda_analogical_align: 一站式 host 接口 (upload -> kernels -> download)
 *
 * 核心算法：
 *   1. 属性哈希化：字符串属性 → uint32_t hash (CPU 端预处理)
 *   2. Jaccard 相似度：|intersection| / |union|，每个 CUDA thread 一对
 *   3. 贪心对齐：每个 source 概念选最佳未分配 target (atomicCAS 保证互斥)
 *
 * 设计参考：
 *   - Gentner 结构映射理论 (SME, 1983)
 *   - 项目模式参考: stdp_learning_cuda.cuh, knowledge_graph_cuda.cuh
 */

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::learning {

// ── GPU Alignment Data Types ────────────────────────────────────────

/**
 * GpuConceptSet — 单个概念在 GPU 上的紧凑表示。
 *
 * 属性被哈希为 uint32_t，连续存储。
 * 所有概念的哈希拼接在一个大数组中，通过 offset/size 定位。
 */
struct GpuConceptSet {
    int      concept_idx;    ///< 概念在 source/target 数组中的索引
    int      hash_offset;    ///< 该概念属性哈希在全局数组中的起始偏移
    int      hash_count;     ///< 该概念的属性哈希数量
    uint32_t domain_tag;     ///< 领域标签哈希（用于过滤）
};

/**
 * AlignmentInput — 从 CPU 端准备的 GPU 对齐输入数据。
 *
 * 调用者将 ConceptDescriptor 的属性字符串哈希化后填入此结构，
 * 传递给 cuda_analogical_align()。
 */
struct AlignmentInput {
    /// source 侧概念集合
    std::vector<GpuConceptSet> source_concepts;

    /// target 侧概念集合
    std::vector<GpuConceptSet> target_concepts;

    /// 所有 source 概念的属性哈希（连续拼接）
    std::vector<uint32_t> source_hashes;

    /// 所有 target 概念的属性哈希（连续拼接）
    std::vector<uint32_t> target_hashes;

    /// source 概念的原始 ID（索引对齐 source_concepts）
    std::vector<std::string> source_ids;

    /// target 概念的原始 ID（索引对齐 target_concepts）
    std::vector<std::string> target_ids;

    /// 对齐分数阈值（低于此值的映射被过滤）
    float min_alignment_score = 0.3f;
};

/**
 * AlignmentPair — 一对对齐结果。
 */
struct AlignmentPair {
    int   source_idx;     ///< source 概念索引
    int   target_idx;     ///< target 概念索引
    float jaccard_score;  ///< Jaccard 相似度 [0, 1]
};

/**
 * AlignmentResult — GPU 对齐输出。
 */
struct AlignmentResult {
    /// 找到的对齐对（按 jaccard_score 降序）
    std::vector<AlignmentPair> pairs;

    /// N_source x N_target 的完整相似度矩阵（FP32，行优先）
    /// similarity_matrix[s * N_target + t] = Jaccard(source[s], target[t])
    std::vector<float> similarity_matrix;

    int n_source;  ///< source 概念数
    int n_target;  ///< target 概念数
};

// ── CUDA Function Declarations ──────────────────────────────────────

/**
 * GPU 加速类比对齐 — 完整流程。
 *
 * 1. 上传概念哈希数据到 GPU
 * 2. 运行 kernel_jaccard_batch 计算所有 N_source x N_target 对的 Jaccard
 * 3. 运行 kernel_greedy_alignment 执行贪心匹配
 * 4. 下载结果
 *
 * @param input  CPU 端准备的输入数据
 * @return AlignmentResult 对齐结果
 */
auto cuda_analogical_align(const AlignmentInput& input) -> AlignmentResult;

/**
 * CPU 回退：纯 CPU 实现相同算法。
 * 用于小规模数据或无 CUDA 环境。
 *
 * @param input  CPU 端准备的输入数据
 * @return AlignmentResult 对齐结果
 */
auto cpu_analogical_align(const AlignmentInput& input) -> AlignmentResult;

// ── CPU/GPU Auto-Dispatch ───────────────────────────────────────────

/// source * target 概念对数超过此阈值时启用 CUDA
inline constexpr int CUDA_ALIGNMENT_THRESHOLD = 256;

/**
 * 自动分发类比对齐：大规模用 GPU，小规模用 CPU。
 *
 * @param input  CPU 端准备的输入数据
 * @return AlignmentResult 对齐结果
 */
auto dispatch_analogical_align(const AlignmentInput& input) -> AlignmentResult;

// ── Utility: Build AlignmentInput from ConceptDescriptors ───────────

struct ConceptDescriptor;  // forward decl from analogical_transfer.hpp

/**
 * 从 ConceptDescriptor 列表构建 AlignmentInput。
 *
 * 将属性字符串哈希为 uint32_t，准备 GPU 所需的数据布局。
 *
 * @param source  source 侧概念描述符列表
 * @param target  target 侧概念描述符列表
 * @param min_score 最低对齐分数阈值
 * @return AlignmentInput GPU 输入数据
 */
auto build_alignment_input(
    const std::vector<ConceptDescriptor>& source,
    const std::vector<ConceptDescriptor>& target,
    float min_score = 0.3f) -> AlignmentInput;

}  // namespace ai_learning::learning
