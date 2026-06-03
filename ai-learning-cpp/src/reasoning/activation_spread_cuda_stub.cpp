/**
 * @file activation_spread_cuda_stub.cpp
 * @brief Activation Spread CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有 GPU 函数不执行操作，确保纯 CPU 构建正常工作。
 * CPU/GPU 自动分发通过 cuda_available() == false 自动回退 CPU 路径。
 * CsrGraph::build 和 CPU fallback 实现在 .cu 文件中，
 * 纯 CPU 构建时这些函数通过 activation_spread_dispatch 的 CPU 路径提供。
 *
 * 此 stub 仅提供 .cuh 中声明的 CUDA 接口符号，避免链接错误。
 */

#include "ai_learning/reasoning/activation_spread_cuda.cuh"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace ai_learning::reasoning {

auto cuda_activation_spread_multi_hop(
    const CsrGraph&,
    const std::vector<float>&,
    float,
    float,
    int) -> SpreadResult
{
    // stub: no-op, dispatch falls back to CPU via cuda_available() check
    return {};
}

auto cuda_activation_spread_batch(
    const CsrGraph&,
    const std::vector<std::vector<float>>&,
    float,
    float,
    int) -> BatchSpreadResult
{
    // stub: no-op
    return {};
}

// ── CPU fallback 和 CSR 构建需要在此 stub 中也提供 ──────────────────────
// CsrGraph::build, cpu_activation_spread_multi_hop,
// activation_spread_dispatch, activation_spread_from_adjacency
// 在 .cu 文件中实现。无 CUDA 时这些符号由 .cu stub 的 CPU 路径提供。
// 但实际上 .cu 文件不参与无 CUDA 构建，所以这些需要在 stub 中实现。

auto activation_spread_dispatch(
    const CsrGraph& graph,
    const std::vector<float>& seed_activations,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult
{
    // 纯 CPU 路径
    int N = graph.num_nodes;

    SpreadResult result;
    result.hops_completed = 0;
    result.activations = seed_activations;

    if (graph.num_edges == 0) return result;

    std::vector<float> act_next(N, 0.0f);

    for (int hop = 0; hop < num_hops; ++hop) {
        std::fill(act_next.begin(), act_next.end(), 0.0f);

        for (int i = 0; i < N; ++i) {
            int start = graph.row_ptr[i];
            int end   = graph.row_ptr[i + 1];

            float sum = 0.0f;
            for (int e = start; e < end; ++e) {
                int neighbor = graph.col_indices[e];
                float weight = graph.edge_weights[e];
                sum += result.activations[neighbor] * weight;
            }

            float new_act = sum * decay_rate;
            if (new_act < threshold) {
                new_act = 0.0f;
            }
            act_next[i] = (result.activations[i] > new_act)
                ? result.activations[i] : new_act;
        }

        result.activations = act_next;
        result.hops_completed = hop + 1;
    }

    return result;
}

auto activation_spread_from_adjacency(
    const std::map<std::string, std::vector<std::pair<std::string, float>>>& adjacency,
    const std::vector<std::string>& node_ids,
    const std::vector<std::string>& seed_entity_ids,
    const std::vector<float>& seed_values,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult
{
    int N = static_cast<int>(node_ids.size());

    // entity_id → node_index
    std::unordered_map<std::string, int> id_to_idx;
    for (int i = 0; i < N; ++i) {
        id_to_idx[node_ids[i]] = i;
    }

    // 构建入边 CSR
    std::vector<int> in_degree(N, 0);
    for (const auto& [src_id, neighbors] : adjacency) {
        auto src_it = id_to_idx.find(src_id);
        if (src_it == id_to_idx.end()) continue;
        for (const auto& [dst_id, weight] : neighbors) {
            auto dst_it = id_to_idx.find(dst_id);
            if (dst_it == id_to_idx.end()) continue;
            in_degree[dst_it->second]++;
        }
    }

    std::vector<int> row_ptr(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + in_degree[i];
    }

    std::vector<int> col_indices(row_ptr[N]);
    std::vector<float> edge_weights(row_ptr[N]);
    std::vector<int> current_pos(N, 0);

    for (const auto& [src_id, neighbors] : adjacency) {
        auto src_it = id_to_idx.find(src_id);
        if (src_it == id_to_idx.end()) continue;
        int src_idx = src_it->second;
        for (const auto& [dst_id, w] : neighbors) {
            auto dst_it = id_to_idx.find(dst_id);
            if (dst_it == id_to_idx.end()) continue;
            int dst_idx = dst_it->second;
            int pos = row_ptr[dst_idx] + current_pos[dst_idx];
            col_indices[pos] = src_idx;
            edge_weights[pos] = w;
            current_pos[dst_idx]++;
        }
    }

    auto graph = CsrGraph::build(
        N, row_ptr, col_indices, edge_weights,
        std::vector<std::string>(node_ids.begin(), node_ids.end()));

    std::vector<float> seed_activations(N, 0.0f);
    for (size_t i = 0; i < seed_entity_ids.size(); ++i) {
        auto it = id_to_idx.find(seed_entity_ids[i]);
        if (it != id_to_idx.end()) {
            seed_activations[it->second] = seed_values[i];
        }
    }

    return activation_spread_dispatch(
        graph, seed_activations, decay_rate, threshold, num_hops);
}

auto CsrGraph::build(
    int num_nodes,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_indices,
    const std::vector<float>& edge_weights,
    std::vector<std::string> node_id_map) -> CsrGraph
{
    CsrGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = static_cast<int>(col_indices.size());
    g.row_ptr = row_ptr;
    g.col_indices = col_indices;
    g.edge_weights = edge_weights;
    g.node_id_map = std::move(node_id_map);
    return g;
}

}  // namespace ai_learning::reasoning
