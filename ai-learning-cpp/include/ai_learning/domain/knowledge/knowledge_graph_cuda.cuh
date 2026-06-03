/**
 * @file knowledge_graph_cuda.cuh
 * @brief Knowledge Graph Query CUDA 加速内核 — CSR 格式批量图查询
 *
 * 在 GPU 上加速知识图谱的三大核心查询操作：
 * - 批量邻居查找 (cuda_graph_find_neighbors)
 * - BFS 最短路径   (cuda_graph_shortest_path)
 * - 实体属性搜索   (cuda_graph_entity_search)
 *
 * 设计要点：
 * - CSR (Compressed Sparse Row) 格式存储图结构，GPU 友好的连续内存访问
 * - cuda_graph_upload() 一次性上传图数据到 GPU，后续查询复用
 * - 批量查询并行：多个查询同时在不同 CUDA thread 上执行
 * - L2 cache persistence hint 提升热点图数据的访问速度
 * - CPU/GPU 自动分发：小规模图退回 CPU 实现
 *
 * 与 activation_spread_cuda 复用相同的 CSR 基础设施和 GPU buffer 管理模式。
 */

#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {

// ── GPU Graph Data (Opaque Handle) ───────────────────────────────────

/**
 * GpuGraphHandle — GPU 上驻留的图数据句柄。
 *
 * 用户通过 cuda_graph_upload() 创建，在多次查询间复用，
 * 最后通过 cuda_graph_release() 释放。
 *
 * 内部持有 CSR 邻接表、实体属性数组和关系类型数组的 GPU 副本。
 * 不透明设计：调用者无需了解 CUDA 细节。
 */
struct GpuGraphHandle;

// ── CSR Graph Build Data ─────────────────────────────────────────────

/**
 * CsrGraphData — 从 KnowledgeGraph 聚合根提取的 CSR 格式构建数据。
 *
 * CPU 端准备，传递给 cuda_graph_upload() 上传到 GPU。
 * CSR 行对应 source 实体，列对应 target 实体（出边方向）。
 *
 * row_ptr[i..i+1) 给出实体 i 的所有出边在 col_indices 中的范围。
 * edge_relation_types[e] 给出边 e 的关系类型编号。
 */
struct CsrGraphData {
    int num_nodes;                           ///< 实体（节点）总数
    int num_edges;                           ///< 关系（有向边）总数
    std::vector<int>   row_ptr;              ///< [num_nodes + 1] 行偏移
    std::vector<int>   col_indices;          ///< [num_edges] 目标节点索引
    std::vector<int>   edge_relation_types;  ///< [num_edges] 关系类型编号
    std::vector<float> edge_confidences;     ///< [num_edges] 边置信度

    /// 实体属性：每种属性存储为 [num_nodes] 数组。
    /// key = 属性名, value = 属性值字符串。
    /// GPU 上仅支持整数化的属性比较。
    std::map<std::string, std::vector<int>> int_properties;

    /// 实体类型编号 [num_nodes]，从 entity.type() 字符串映射而来
    std::vector<int> entity_type_codes;

    /// 节点 ID 映射 [num_nodes]：索引 → 原始 entity_id
    std::vector<std::string> node_id_map;

    /// 反向映射：entity_id → 节点索引
    std::map<std::string, int> id_to_index;
};

// ── Query Result Types ───────────────────────────────────────────────

/**
 * NeighborQueryResult — 单个邻居查询的结果。
 */
struct NeighborQueryResult {
    std::vector<int>    neighbor_indices;  ///< 邻居节点索引列表
    std::vector<float>  edge_confidences;  ///< 对应边置信度
    std::vector<int>    relation_types;    ///< 对应关系类型编号
};

/**
 * BatchNeighborResult — 批量邻居查询结果。
 */
struct BatchNeighborResult {
    std::vector<NeighborQueryResult> results;  ///< [num_queries] 每个查询的结果
};

/**
 * PathQueryResult — 单条路径查询结果。
 */
struct PathQueryResult {
    std::vector<int> path_indices;  ///< 路径上节点索引序列
    int              path_length;   ///< 路径长度（边数），-1 表示未找到
    bool             found;         ///< 是否找到路径
};

/**
 * BatchPathResult — 批量路径查询结果。
 */
struct BatchPathResult {
    std::vector<PathQueryResult> results;  ///< [num_queries] 每个查询的结果
};

/**
 * EntitySearchResult — 单次实体搜索结果。
 */
struct EntitySearchResult {
    std::vector<int> matched_indices;  ///< 匹配的实体索引列表
};

/**
 * BatchEntitySearchResult — 批量实体搜索结果。
 */
struct BatchEntitySearchResult {
    std::vector<EntitySearchResult> results;  ///< [num_queries] 每个查询的结果
};

// ── Query Request Types ──────────────────────────────────────────────

/**
 * NeighborQuery — 单个邻居查询请求。
 */
struct NeighborQuery {
    int         source_index;     ///< 源实体节点索引
    int         relation_filter;  ///< 关系类型过滤（-1 表示不过滤）
    int         max_neighbors;    ///< 最多返回邻居数（0 = 全部）
};

/**
 * PathQuery — 单个路径查询请求。
 */
struct PathQuery {
    int source_index;  ///< 起点
    int target_index;  ///< 终点
    int max_depth;     ///< 最大搜索深度
};

/**
 * EntitySearchQuery — 单个实体搜索请求。
 */
struct EntitySearchQuery {
    int entity_type_code;              ///< 实体类型编号（-1 = 任意类型）
    int property_key_index;            ///< 属性键索引（-1 = 不匹配属性）
    int property_value;                ///< 属性值（整数化后）
};

// ── CUDA Graph Lifecycle ─────────────────────────────────────────────

/**
 * 上传知识图谱到 GPU，返回可复用的句柄。
 *
 * @param data CSR 格式图数据（CPU 端准备）
 * @return GpuGraphHandle* GPU 句柄，nullptr 表示上传失败
 */
auto cuda_graph_upload(const CsrGraphData& data) -> GpuGraphHandle*;

/**
 * 释放 GPU 图数据句柄。
 *
 * @param handle 要释放的句柄，释放后置 nullptr
 */
void cuda_graph_release(GpuGraphHandle*& handle);

/**
 * 检查 GPU 图句柄是否有效。
 */
auto cuda_graph_handle_valid(const GpuGraphHandle* handle) -> bool;

// ── CUDA Batch Query Functions ───────────────────────────────────────

/**
 * 批量并行邻居查找。
 *
 * 对每个查询，在 CSR 中查找 source_index 的所有出边邻居。
 * 如果 relation_filter >= 0，仅返回匹配的关系类型。
 * 多个查询在 GPU 上并行执行。
 *
 * @param handle  GPU 图句柄
 * @param queries 查询列表
 * @return BatchNeighborResult 每个查询的邻居结果
 */
auto cuda_graph_find_neighbors(
    const GpuGraphHandle* handle,
    const std::vector<NeighborQuery>& queries) -> BatchNeighborResult;

/**
 * 批量 BFS 最短路径查找。
 *
 * 对每个 (source, target) 对，执行并行 BFS。
 * 使用 frontier-based 并行 BFS：每轮扩展所有活跃查询的 frontier。
 *
 * @param handle  GPU 图句柄
 * @param queries 查询列表
 * @return BatchPathResult 每个查询的路径结果
 */
auto cuda_graph_shortest_path(
    const GpuGraphHandle* handle,
    const std::vector<PathQuery>& queries) -> BatchPathResult;

/**
 * 批量并行实体属性搜索。
 *
 * 对每个查询条件，扫描所有节点并返回匹配的实体索引。
 * 支持按实体类型和整数化属性值过滤。
 *
 * @param handle  GPU 图句柄
 * @param queries 查询列表
 * @return BatchEntitySearchResult 每个查询的匹配结果
 */
auto cuda_graph_entity_search(
    const GpuGraphHandle* handle,
    const std::vector<EntitySearchQuery>& queries) -> BatchEntitySearchResult;

// ── CPU Fallback Functions ───────────────────────────────────────────

/**
 * CPU 邻居查找（用于小图或无 CUDA 的回退）。
 */
auto cpu_graph_find_neighbors(
    const CsrGraphData& data,
    const std::vector<NeighborQuery>& queries) -> BatchNeighborResult;

/**
 * CPU BFS 最短路径（用于小图或无 CUDA 的回退）。
 */
auto cpu_graph_shortest_path(
    const CsrGraphData& data,
    const std::vector<PathQuery>& queries) -> BatchPathResult;

/**
 * CPU 实体搜索（用于小图或无 CUDA 的回退）。
 */
auto cpu_graph_entity_search(
    const CsrGraphData& data,
    const std::vector<EntitySearchQuery>& queries) -> BatchEntitySearchResult;

// ── CPU/GPU Auto-Dispatch ────────────────────────────────────────────

/// 节点数超过此阈值时启用 CUDA
inline constexpr int CUDA_GRAPH_THRESHOLD = 256;

/**
 * 从 KnowledgeGraph 聚合根构建 CsrGraphData。
 *
 * 遍历 KnowledgeGraph 的实体和关系，构建 CSR 格式数据。
 * 实体属性字符串被整数化以便 GPU 比较。
 *
 * @param kg KnowledgeGraph 聚合根
 * @return CsrGraphData CSR 格式构建数据
 */
class KnowledgeGraph;  // forward decl
auto build_csr_graph_data(const KnowledgeGraph& kg) -> CsrGraphData;

/**
 * 自动分发邻居查询：大图用 GPU，小图用 CPU。
 */
auto dispatch_graph_find_neighbors(
    const CsrGraphData& data,
    const GpuGraphHandle* gpu_handle,
    const std::vector<NeighborQuery>& queries) -> BatchNeighborResult;

/**
 * 自动分发路径查询：大图用 GPU，小图用 CPU。
 */
auto dispatch_graph_shortest_path(
    const CsrGraphData& data,
    const GpuGraphHandle* gpu_handle,
    const std::vector<PathQuery>& queries) -> BatchPathResult;

/**
 * 自动分发实体搜索：大图用 GPU，小图用 CPU。
 */
auto dispatch_graph_entity_search(
    const CsrGraphData& data,
    const GpuGraphHandle* gpu_handle,
    const std::vector<EntitySearchQuery>& queries) -> BatchEntitySearchResult;

}  // namespace ai_learning::domain::knowledge
