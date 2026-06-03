/**
 * @file knowledge_graph_cuda_stub.cpp
 * @brief Knowledge Graph CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有 GPU 函数退回到 CPU 实现，确保纯 CPU 构建正常工作。
 * CUDA 函数声明在 knowledge_graph_cuda.cuh 中，分发逻辑通过
 * cuda_available() == false 自动回退到 CPU 路径。
 */

#include "ai_learning/domain/knowledge/knowledge_graph_cuda.cuh"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_learning::domain::knowledge {

// ── CPU Fallback: Neighbor Query ────────────────────────────────────

auto cpu_graph_find_neighbors(
    const CsrGraphData& data,
    const std::vector<NeighborQuery>& queries) -> BatchNeighborResult
{
    BatchNeighborResult result;
    result.results.resize(queries.size());

    for (size_t q = 0; q < queries.size(); ++q) {
        const auto& query = queries[q];
        int src = query.source_index;

        if (src < 0 || src >= data.num_nodes) continue;

        int start = data.row_ptr[src];
        int end   = data.row_ptr[src + 1];

        for (int e = start; e < end; ++e) {
            if (query.relation_filter >= 0 &&
                data.edge_relation_types[e] != query.relation_filter) {
                continue;
            }
            result.results[q].neighbor_indices.push_back(data.col_indices[e]);
            result.results[q].edge_confidences.push_back(data.edge_confidences[e]);
            result.results[q].relation_types.push_back(data.edge_relation_types[e]);

            if (query.max_neighbors > 0 &&
                static_cast<int>(result.results[q].neighbor_indices.size()) >= query.max_neighbors) {
                break;
            }
        }
    }

    return result;
}

// ── CPU Fallback: BFS Shortest Path ─────────────────────────────────

auto cpu_graph_shortest_path(
    const CsrGraphData& data,
    const std::vector<PathQuery>& queries) -> BatchPathResult
{
    BatchPathResult result;
    result.results.resize(queries.size());

    for (size_t q = 0; q < queries.size(); ++q) {
        int src = queries[q].source_index;
        int tgt = queries[q].target_index;
        int max_depth = queries[q].max_depth;

        if (src == tgt) {
            result.results[q].path_indices = {src};
            result.results[q].path_length = 0;
            result.results[q].found = true;
            continue;
        }

        if (src < 0 || src >= data.num_nodes || tgt < 0 || tgt >= data.num_nodes) {
            result.results[q].found = false;
            result.results[q].path_length = -1;
            continue;
        }

        // BFS
        std::vector<int> parent(data.num_nodes, -1);
        std::vector<bool> visited(data.num_nodes, false);
        std::queue<int> queue;

        visited[src] = true;
        queue.push(src);

        bool found = false;
        int depth = 0;

        while (!queue.empty() && depth < max_depth && !found) {
            int level_size = static_cast<int>(queue.size());
            for (int i = 0; i < level_size && !found; ++i) {
                int cur = queue.front();
                queue.pop();

                int start = data.row_ptr[cur];
                int end   = data.row_ptr[cur + 1];

                for (int e = start; e < end; ++e) {
                    int next = data.col_indices[e];
                    if (visited[next]) continue;

                    visited[next] = true;
                    parent[next] = cur;

                    if (next == tgt) {
                        found = true;
                        break;
                    }
                    queue.push(next);
                }
            }
            depth++;
        }

        if (found) {
            // Backtrack path
            std::vector<int> path;
            int cur = tgt;
            while (cur >= 0) {
                path.push_back(cur);
                cur = parent[cur];
            }
            std::reverse(path.begin(), path.end());
            result.results[q].path_indices = path;
            result.results[q].path_length = static_cast<int>(path.size()) - 1;
            result.results[q].found = true;
        } else {
            result.results[q].found = false;
            result.results[q].path_length = -1;
        }
    }

    return result;
}

// ── CPU Fallback: Entity Search ─────────────────────────────────────

auto cpu_graph_entity_search(
    const CsrGraphData& data,
    const std::vector<EntitySearchQuery>& queries) -> BatchEntitySearchResult
{
    BatchEntitySearchResult result;
    result.results.resize(queries.size());

    for (size_t q = 0; q < queries.size(); ++q) {
        const auto& query = queries[q];

        for (int node = 0; node < data.num_nodes; ++node) {
            // Type filter
            if (query.entity_type_code >= 0 &&
                data.entity_type_codes[node] != query.entity_type_code) {
                continue;
            }

            // Property filter
            if (query.property_key_index >= 0) {
                // Check if the property exists and matches
                int prop_idx = 0;
                bool prop_matched = false;
                for (const auto& [key, values] : data.int_properties) {
                    if (prop_idx == query.property_key_index) {
                        if (node < static_cast<int>(values.size()) &&
                            values[node] == query.property_value) {
                            prop_matched = true;
                        }
                        break;
                    }
                    ++prop_idx;
                }
                if (!prop_matched) continue;
            }

            result.results[q].matched_indices.push_back(node);
        }
    }

    return result;
}

// ── GpuGraphHandle stubs ────────────────────────────────────────────

auto cuda_graph_upload(const CsrGraphData&) -> GpuGraphHandle* {
    // stub: no GPU available
    return nullptr;
}

void cuda_graph_release(GpuGraphHandle*&) {
    // stub: no-op
}

auto cuda_graph_handle_valid(const GpuGraphHandle*) -> bool {
    return false;
}

auto cuda_graph_find_neighbors(
    const GpuGraphHandle*,
    const std::vector<NeighborQuery>&) -> BatchNeighborResult
{
    return {};
}

auto cuda_graph_shortest_path(
    const GpuGraphHandle*,
    const std::vector<PathQuery>&) -> BatchPathResult
{
    return {};
}

auto cuda_graph_entity_search(
    const GpuGraphHandle*,
    const std::vector<EntitySearchQuery>&) -> BatchEntitySearchResult
{
    return {};
}

// ── CPU/GPU Auto-Dispatch (CPU-only) ────────────────────────────────

auto dispatch_graph_find_neighbors(
    const CsrGraphData& data,
    const GpuGraphHandle*,
    const std::vector<NeighborQuery>& queries) -> BatchNeighborResult
{
    return cpu_graph_find_neighbors(data, queries);
}

auto dispatch_graph_shortest_path(
    const CsrGraphData& data,
    const GpuGraphHandle*,
    const std::vector<PathQuery>& queries) -> BatchPathResult
{
    return cpu_graph_shortest_path(data, queries);
}

auto dispatch_graph_entity_search(
    const CsrGraphData& data,
    const GpuGraphHandle*,
    const std::vector<EntitySearchQuery>& queries) -> BatchEntitySearchResult
{
    return cpu_graph_entity_search(data, queries);
}

// ── Build CSR Graph Data from KnowledgeGraph ────────────────────────

auto build_csr_graph_data(const KnowledgeGraph& kg) -> CsrGraphData {
    CsrGraphData data;

    auto all_ids = kg.get_all_entity_ids();
    int N = static_cast<int>(all_ids.size());
    data.num_nodes = N;
    data.node_id_map = all_ids;

    // Build entity_id -> index map
    for (int i = 0; i < N; ++i) {
        data.id_to_index[all_ids[i]] = i;
    }

    // Build entity type codes
    std::map<std::string, int> type_to_code;
    int next_code = 0;
    data.entity_type_codes.resize(N, -1);
    for (int i = 0; i < N; ++i) {
        auto entity = kg.get_entity(all_ids[i]);
        if (entity) {
            const std::string& type = entity->get().type();
            auto it = type_to_code.find(type);
            if (it == type_to_code.end()) {
                type_to_code[type] = next_code++;
            }
            data.entity_type_codes[i] = type_to_code[type];
        }
    }

    // Count outgoing edges for CSR row_ptr
    std::vector<int> out_degree(N, 0);
    for (int i = 0; i < N; ++i) {
        auto relations = kg.get_relations_of(all_ids[i], "out");
        out_degree[i] = static_cast<int>(relations.size());
    }

    data.row_ptr.resize(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        data.row_ptr[i + 1] = data.row_ptr[i] + out_degree[i];
    }

    int E = data.row_ptr[N];
    data.num_edges = E;
    data.col_indices.resize(E);
    data.edge_relation_types.resize(E);
    data.edge_confidences.resize(E, 1.0f);

    // Map relation types to integer codes
    std::map<std::string, int> rel_type_to_code;
    int rel_code = 0;

    std::vector<int> current_pos(N, 0);
    for (int i = 0; i < N; ++i) {
        auto relations = kg.get_relations_of(all_ids[i], "out");
        for (const auto& rel_ref : relations) {
            const auto& rel = rel_ref.get();
            auto tgt_it = data.id_to_index.find(rel.target_id());
            if (tgt_it == data.id_to_index.end()) continue;

            const std::string& rtype = rel.type();
            auto rit = rel_type_to_code.find(rtype);
            if (rit == rel_type_to_code.end()) {
                rel_type_to_code[rtype] = rel_code++;
            }

            int pos = data.row_ptr[i] + current_pos[i];
            data.col_indices[pos] = tgt_it->second;
            data.edge_relation_types[pos] = rel_type_to_code[rtype];
            data.edge_confidences[pos] = static_cast<float>(rel.confidence());
            current_pos[i]++;
        }
    }

    return data;
}

}  // namespace ai_learning::domain::knowledge
