/**
 * @file knowledge_graph_cpu.cpp
 * @brief Knowledge Graph Query CPU 回退实现 + CSR 构建逻辑
 *
 * 包含：
 * - cpu_graph_find_neighbors:   CPU 邻居查找
 * - cpu_graph_shortest_path:    CPU BFS 最短路径
 * - cpu_graph_entity_search:    CPU 实体属性搜索
 * - build_csr_graph_data:       从 KnowledgeGraph 聚合根构建 CSR 数据
 *
 * 这些函数在 GPU 不可用或图规模较小时作为回退路径。
 * build_csr_graph_data 无论 CPU/GPU 都需要使用。
 */

#include "ai_learning/domain/knowledge/knowledge_graph_cuda.cuh"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

#include <algorithm>
#include <map>
#include <queue>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {

// ── CPU Fallbacks ────────────────────────────────────────────────────

auto cpu_graph_find_neighbors(
    const CsrGraphData& data,
    const std::vector<NeighborQuery>& queries) -> BatchNeighborResult
{
    BatchNeighborResult result;
    result.results.resize(queries.size());

    for (size_t q = 0; q < queries.size(); ++q) {
        int src = queries[q].source_index;
        int rel = queries[q].relation_filter;
        int max_n = queries[q].max_neighbors;
        int start = data.row_ptr[src];
        int end   = data.row_ptr[src + 1];

        auto& r = result.results[q];
        for (int e = start; e < end; ++e) {
            if (rel >= 0 && data.edge_relation_types[e] != rel) continue;
            r.neighbor_indices.push_back(data.col_indices[e]);
            r.edge_confidences.push_back(data.edge_confidences[e]);
            r.relation_types.push_back(data.edge_relation_types[e]);
            if (max_n > 0 && static_cast<int>(r.neighbor_indices.size()) >= max_n) break;
        }
    }
    return result;
}

auto cpu_graph_shortest_path(
    const CsrGraphData& data,
    const std::vector<PathQuery>& queries) -> BatchPathResult
{
    BatchPathResult result;
    result.results.resize(queries.size());

    for (size_t q = 0; q < queries.size(); ++q) {
        int src = queries[q].source_index;
        int tgt = queries[q].target_index;
        int max_d = queries[q].max_depth;

        if (src == tgt) {
            result.results[q].path_indices = {src};
            result.results[q].path_length = 0;
            result.results[q].found = true;
            continue;
        }

        // BFS
        std::vector<int> parent(data.num_nodes, -1);
        std::vector<bool> visited(data.num_nodes, false);
        std::queue<int> frontier;

        visited[src] = true;
        parent[src] = -2;  // root marker
        frontier.push(src);

        bool found = false;
        int depth = 0;

        while (!frontier.empty() && depth < max_d && !found) {
            int level_size = static_cast<int>(frontier.size());
            for (int i = 0; i < level_size && !found; ++i) {
                int node = frontier.front();
                frontier.pop();

                int start = data.row_ptr[node];
                int end   = data.row_ptr[node + 1];
                for (int e = start; e < end; ++e) {
                    int neighbor = data.col_indices[e];
                    if (visited[neighbor]) continue;
                    visited[neighbor] = true;
                    parent[neighbor] = node;
                    frontier.push(neighbor);

                    if (neighbor == tgt) {
                        found = true;
                        break;
                    }
                }
            }
            ++depth;
        }

        auto& r = result.results[q];
        if (found) {
            int cur = tgt;
            while (cur >= 0) {
                r.path_indices.push_back(cur);
                int p = parent[cur];
                if (p == -2) break;
                cur = p;
            }
            std::reverse(r.path_indices.begin(), r.path_indices.end());
            r.path_length = static_cast<int>(r.path_indices.size()) - 1;
            r.found = true;
        } else {
            r.found = false;
            r.path_length = -1;
        }
    }
    return result;
}

auto cpu_graph_entity_search(
    const CsrGraphData& data,
    const std::vector<EntitySearchQuery>& queries) -> BatchEntitySearchResult
{
    BatchEntitySearchResult result;
    result.results.resize(queries.size());

    for (size_t q = 0; q < queries.size(); ++q) {
        int type_filter = queries[q].entity_type_code;
        int prop_key    = queries[q].property_key_index;
        int prop_val    = queries[q].property_value;

        auto& r = result.results[q];
        for (int node = 0; node < data.num_nodes; ++node) {
            if (type_filter >= 0 && data.entity_type_codes[node] != type_filter) continue;

            if (prop_key >= 0) {
                // 查找属性值
                int prop_idx = 0;
                bool found_prop = false;
                for (const auto& [key, values] : data.int_properties) {
                    if (prop_idx == prop_key) {
                        if (node < static_cast<int>(values.size()) &&
                            values[node] == prop_val) {
                            found_prop = true;
                        }
                        break;
                    }
                    ++prop_idx;
                }
                if (!found_prop) continue;
            }

            r.matched_indices.push_back(node);
        }
    }
    return result;
}

// ── build_csr_graph_data ─────────────────────────────────────────────

auto build_csr_graph_data(const KnowledgeGraph& kg) -> CsrGraphData {
    CsrGraphData data;

    // 收集所有实体 ID 并建立索引
    auto all_ids = kg.get_all_entity_ids();
    data.num_nodes = static_cast<int>(all_ids.size());
    data.node_id_map = all_ids;

    data.id_to_index.clear();
    for (int i = 0; i < data.num_nodes; ++i) {
        data.id_to_index[all_ids[i]] = i;
    }

    // 收集关系类型，建立字符串 -> 整数映射
    std::map<std::string, int> relation_type_map;
    int next_rel_type = 0;

    auto all_relations = kg.save_state().relations;

    // 先扫描一遍建立关系类型映射
    for (const auto& rel : all_relations) {
        auto it = relation_type_map.find(rel.type());
        if (it == relation_type_map.end()) {
            relation_type_map[rel.type()] = next_rel_type++;
        }
    }

    // 收集实体类型映射
    std::map<std::string, int> entity_type_map;
    int next_entity_type = 0;
    data.entity_type_codes.resize(data.num_nodes, -1);

    for (int i = 0; i < data.num_nodes; ++i) {
        auto entity = kg.get_entity(all_ids[i]);
        if (entity) {
            const auto& type = entity->get().type();
            auto it = entity_type_map.find(type);
            if (it == entity_type_map.end()) {
                entity_type_map[type] = next_entity_type++;
            }
            data.entity_type_codes[i] = entity_type_map[type];

            // 收集整数属性
            const auto& props = entity->get().properties();
            for (const auto& [key, value] : props) {
                try {
                    int int_val = std::stoi(value);
                    if (data.int_properties.find(key) == data.int_properties.end()) {
                        data.int_properties[key] = std::vector<int>(data.num_nodes, 0);
                    }
                    data.int_properties[key][i] = int_val;
                } catch (...) {
                    // 非整数属性跳过
                }
            }
        }
    }

    // 构建 CSR（出边方向）
    std::vector<int> out_degree(data.num_nodes, 0);
    data.num_edges = static_cast<int>(all_relations.size());

    for (const auto& rel : all_relations) {
        auto src_it = data.id_to_index.find(rel.source_id());
        auto tgt_it = data.id_to_index.find(rel.target_id());
        if (src_it != data.id_to_index.end() && tgt_it != data.id_to_index.end()) {
            out_degree[src_it->second]++;
        }
    }

    data.row_ptr.resize(data.num_nodes + 1, 0);
    data.row_ptr[0] = 0;
    for (int i = 0; i < data.num_nodes; ++i) {
        data.row_ptr[i + 1] = data.row_ptr[i] + out_degree[i];
    }

    data.col_indices.resize(data.num_edges);
    data.edge_relation_types.resize(data.num_edges);
    data.edge_confidences.resize(data.num_edges);

    // 填充边数据
    std::vector<int> current_pos(data.num_nodes, 0);
    for (const auto& rel : all_relations) {
        auto src_it = data.id_to_index.find(rel.source_id());
        auto tgt_it = data.id_to_index.find(rel.target_id());
        if (src_it == data.id_to_index.end() || tgt_it == data.id_to_index.end()) continue;

        int src = src_it->second;
        int tgt = tgt_it->second;
        int pos = data.row_ptr[src] + current_pos[src];

        data.col_indices[pos] = tgt;
        data.edge_relation_types[pos] = relation_type_map[rel.type()];
        data.edge_confidences[pos] = static_cast<float>(rel.confidence());
        current_pos[src]++;
    }

    return data;
}

// ── Auto-Dispatch ────────────────────────────────────────────────────

auto dispatch_graph_find_neighbors(
    const CsrGraphData& data,
    const GpuGraphHandle* gpu_handle,
    const std::vector<NeighborQuery>& queries) -> BatchNeighborResult
{
    if (data.num_nodes >= CUDA_GRAPH_THRESHOLD &&
        gpu_handle && cuda_graph_handle_valid(gpu_handle)) {
        return cuda_graph_find_neighbors(gpu_handle, queries);
    }
    return cpu_graph_find_neighbors(data, queries);
}

auto dispatch_graph_shortest_path(
    const CsrGraphData& data,
    const GpuGraphHandle* gpu_handle,
    const std::vector<PathQuery>& queries) -> BatchPathResult
{
    if (data.num_nodes >= CUDA_GRAPH_THRESHOLD &&
        gpu_handle && cuda_graph_handle_valid(gpu_handle)) {
        return cuda_graph_shortest_path(gpu_handle, queries);
    }
    return cpu_graph_shortest_path(data, queries);
}

auto dispatch_graph_entity_search(
    const CsrGraphData& data,
    const GpuGraphHandle* gpu_handle,
    const std::vector<EntitySearchQuery>& queries) -> BatchEntitySearchResult
{
    if (data.num_nodes >= CUDA_GRAPH_THRESHOLD &&
        gpu_handle && cuda_graph_handle_valid(gpu_handle)) {
        return cuda_graph_entity_search(gpu_handle, queries);
    }
    return cpu_graph_entity_search(data, queries);
}

}  // namespace ai_learning::domain::knowledge
