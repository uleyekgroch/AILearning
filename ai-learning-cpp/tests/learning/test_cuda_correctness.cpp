/**
 * @file test_cuda_correctness.cpp
 * @brief CUDA 模块正确性验证 — CPU/GPU 结果一致性
 *
 * 验证 Phase 8 的 6 个 CUDA 算法 + Flash Attention + Embedding:
 *   1. tensor_ops (mat_vec/vec_mat/mat_vec_bias)
 *   2. STDP batch update
 *   3. Activation Spread (CSR SpMV)
 *   4. Knowledge Graph (CSR batch queries)
 *   5. Analogical Transfer (Jaccard + greedy alignment)
 *   6. Bayesian Update (posterior + info gain)
 *   7. Flash Attention (bonus)
 *   8. Embedding Trainer CUDA dispatch (bonus)
 *
 * Strategy: CPU reference vs actual; SKIP if no GPU.
 * Tags: [cuda], [correctness], [benchmark]
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/core/tensor_ops.hpp"
#include "ai_learning/core/flash_attention_cuda.cuh"
#include "ai_learning/learning/stdp_learning.hpp"
#include "ai_learning/reasoning/activation_spread_cuda.cuh"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph_cuda.cuh"
#include "ai_learning/learning/analogical_transfer.hpp"
#include "ai_learning/learning/analogical_transfer_cuda.cuh"
#include "ai_learning/learning/active_experimenter.hpp"
#include "ai_learning/learning/active_experimenter_cuda.cuh"
#include "ai_learning/learning/embedding_trainer_cuda.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ai_learning::core;
using namespace ai_learning::learning;
using namespace ai_learning::reasoning;
namespace kg = ai_learning::domain::knowledge;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Helpers ──────────────────────────────────────────────────────

namespace {

/// 确定性随机数生成器
auto make_rng() -> std::mt19937 {
    return std::mt19937(42);
}

/// 生成随机矩阵 (flat row-major)
auto random_matrix(std::mt19937& rng, int rows, int cols,
                   float lo = -1.0f, float hi = 1.0f)
    -> std::vector<float>
{
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> mat(rows * cols);
    for (auto& v : mat) v = dist(rng);
    return mat;
}

/// 生成随机向量
auto random_vector(std::mt19937& rng, int n,
                   float lo = -1.0f, float hi = 1.0f)
    -> std::vector<float>
{
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> vec(n);
    for (auto& v : vec) v = dist(rng);
    return vec;
}

/// CPU 参考实现：矩阵 x 向量
auto ref_mat_vec(const std::vector<float>& mat,
                 int rows, int cols,
                 const std::vector<float>& vec) -> std::vector<float>
{
    std::vector<float> result(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result[r] += mat[r * cols + c] * vec[c];
        }
    }
    return result;
}

/// CPU 参考实现：向量 x 矩阵
auto ref_vec_mat(const std::vector<float>& vec,
                 const std::vector<float>& mat,
                 int cols, int rows) -> std::vector<float>
{
    std::vector<float> result(rows, 0.0f);
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            result[r] += vec[c] * mat[c * rows + r];
        }
    }
    return result;
}

/// CPU 参考实现：mat_vec_bias
auto ref_mat_vec_bias(const std::vector<float>& mat,
                       int rows, int cols,
                       const std::vector<float>& vec,
                       const std::vector<float>& bias) -> std::vector<float>
{
    auto result = ref_mat_vec(mat, rows, cols, vec);
    for (int i = 0; i < rows; ++i) result[i] += bias[i];
    return result;
}

/// CPU 参考：STDP 批量更新
auto ref_stdp_batch_update(
    const std::vector<float>& weights_in,
    const std::vector<float>& pre_times,
    const std::vector<float>& post_times,
    int num_synapses,
    int pairs_per_synapse,
    float A_plus, float A_minus,
    float tau_plus, float tau_minus,
    float w_min, float w_max) -> std::vector<float>
{
    auto weights = weights_in;
    for (int s = 0; s < num_synapses; ++s) {
        float dw = 0.0f;
        for (int k = 0; k < pairs_per_synapse; ++k) {
            int idx = s * pairs_per_synapse + k;
            float delta_t = post_times[idx] - pre_times[idx];
            if (delta_t > 0.0f) {
                dw += A_plus * std::exp(-delta_t / tau_plus);
            } else if (delta_t < 0.0f) {
                dw -= A_minus * std::exp(delta_t / tau_minus);
            }
        }
        weights[s] += dw;
        weights[s] = std::clamp(weights[s], w_min, w_max);
    }
    return weights;
}

/// 最大绝对误差
auto max_abs_error(const std::vector<float>& a,
                   const std::vector<float>& b) -> float {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

}  // anonymous namespace

// ══════════════════════════════════════════════════════════════════
// 1. CUDA 可用性检测
// ══════════════════════════════════════════════════════════════════

TEST_CASE("CUDA availability check", "[cuda][correctness]") {
    bool avail = cuda_available();
    bool fp16  = cuda_fp16_available();

    // 记录状态，不要求必须可用（WSL/CPU 构建可能没有 GPU）
    INFO("cuda_available() = " << avail);
    INFO("cuda_fp16_available() = " << fp16);

    // FP16 只有在 CUDA 可用时才可能为 true
    if (!avail) {
        CHECK_FALSE(fp16);
    }
}

// ══════════════════════════════════════════════════════════════════
// 2. Tensor Ops — mat_vec / vec_mat / mat_vec_bias
// ══════════════════════════════════════════════════════════════════

TEST_CASE("Tensor ops: mat_vec / vec_mat / mat_vec_bias", "[cuda][correctness]") {
    auto rng = make_rng();

    SECTION("identity matrix") {
        std::vector<float> I = {1,0,0, 0,1,0, 0,0,1};
        Tensor v = {3.0f, 4.0f, 5.0f};
        auto result = mat_vec(I, 3, 3, v);
        REQUIRE_THAT(result[0], WithinAbs(3.0, 1e-5));
        REQUIRE_THAT(result[1], WithinAbs(4.0, 1e-5));
        REQUIRE_THAT(result[2], WithinAbs(5.0, 1e-5));
    }

    SECTION("random mat_vec") {
        int rows = 256, cols = 128;
        auto mat = random_matrix(rng, rows, cols);
        auto vec = random_vector(rng, cols);
        auto result = mat_vec(mat, rows, cols, vec);
        auto ref = ref_mat_vec(mat, rows, cols, vec);
        REQUIRE_THAT(max_abs_error(result, ref), WithinAbs(0.0, 1e-3));
    }

    SECTION("random vec_mat") {
        int cols = 64, rows = 32;
        auto vec = random_vector(rng, cols);
        auto mat = random_matrix(rng, cols, rows);
        auto result = vec_mat(vec, mat, cols, rows);
        auto ref = ref_vec_mat(vec, mat, cols, rows);
        REQUIRE_THAT(max_abs_error(result, ref), WithinAbs(0.0, 1e-4));
    }

    SECTION("random mat_vec_bias") {
        int rows = 64, cols = 64;
        auto mat = random_matrix(rng, rows, cols);
        auto vec = random_vector(rng, cols);
        auto bias = random_vector(rng, rows);
        auto result = mat_vec_bias(mat, rows, cols, vec, bias);
        auto ref = ref_mat_vec_bias(mat, rows, cols, vec, bias);
        REQUIRE_THAT(max_abs_error(result, ref), WithinAbs(0.0, 1e-4));
    }
}

// ══════════════════════════════════════════════════════════════════
// 3. STDP batch update
// ══════════════════════════════════════════════════════════════════

TEST_CASE("STDP: batch update consistency", "[cuda][correctness]") {
    auto rng = make_rng();
    std::uniform_real_distribution<float> wdist(0.3f, 0.7f);
    std::uniform_real_distribution<float> tdist(0.0f, 50.0f);

    SECTION("random LTP+LTD") {
        int N = 100, pairs = 3;
        std::vector<float> w(N), pre(N*pairs), post(N*pairs);
        for (auto& v : w) v = wdist(rng);
        for (auto& v : pre) v = tdist(rng);
        for (auto& v : post) v = tdist(rng);
        auto ref = ref_stdp_batch_update(w, pre, post, N, pairs, 0.1f, 0.1f, 20.f, 20.f, 0.f, 1.f);
        auto test = w;
        stdp_batch_update(test, pre, post, N, pairs, 0.1f, 0.1f, 20.f, 20.f, 0.f, 1.f);
        REQUIRE_THAT(max_abs_error(ref, test), WithinAbs(0.0, 1e-5));
    }

    SECTION("all LTP") {
        int N = 50, pairs = 2;
        std::vector<float> w(N, 0.5f), pre(N*pairs), post(N*pairs);
        for (int i = 0; i < N*pairs; ++i) { pre[i] = float(i); post[i] = pre[i]+10.f; }
        auto ref = ref_stdp_batch_update(w, pre, post, N, pairs, .1f,.1f,20.f,20.f,0.f,1.f);
        auto test = w;
        stdp_batch_update(test, pre, post, N, pairs);
        for (int i = 0; i < N; ++i) CHECK(test[i] >= 0.5f);
        REQUIRE_THAT(max_abs_error(ref, test), WithinAbs(0.0, 1e-5));
    }

    SECTION("clamping") {
        int N = 10;
        std::vector<float> w(N, 0.99f), pre(N, 0.f), post(N, 1.f);
        auto ref = ref_stdp_batch_update(w, pre, post, N, 1, 0.5f,.1f,20.f,20.f,0.f,1.f);
        auto test = w;
        stdp_batch_update(test, pre, post, N, 1, 0.5f,.1f,20.f,20.f,0.f,1.f);
        for (int i = 0; i < N; ++i) CHECK(test[i] <= 1.0f);
        REQUIRE_THAT(max_abs_error(ref, test), WithinAbs(0.0, 1e-5));
    }
}

// ══════════════════════════════════════════════════════════════════
// 4. Activation Spread — CSR 图上扩散
// ══════════════════════════════════════════════════════════════════

TEST_CASE("Activation Spread: CSR dispatch consistency", "[cuda][correctness]") {
    auto rng = make_rng();
    using Adj = std::map<std::string, std::vector<std::pair<std::string, float>>>;

    SECTION("simple chain 0→1→2→3") {
        Adj adj;
        adj["0"] = {{"1", 1.0f}};
        adj["1"] = {{"2", 1.0f}};
        adj["2"] = {{"3", 1.0f}};
        auto result = activation_spread_from_adjacency(
            adj, {"0","1","2","3"}, {"0"}, {1.0f}, 0.7f, 0.01f, 3);
        REQUIRE_THAT(result.activations[0], WithinAbs(1.0, 1e-5));
        REQUIRE_THAT(result.activations[1], WithinAbs(0.7, 1e-5));
        REQUIRE_THAT(result.activations[2], WithinAbs(0.49, 1e-4));
        CHECK(result.hops_completed == 3);
    }

    SECTION("branched 0→{1,2}") {
        Adj adj;
        adj["0"] = {{"1", 0.8f}, {"2", 0.6f}};
        auto result = activation_spread_from_adjacency(
            adj, {"0","1","2"}, {"0"}, {1.0f}, 1.0f, 0.01f, 1);
        REQUIRE_THAT(result.activations[1], WithinAbs(0.8, 1e-4));
        REQUIRE_THAT(result.activations[2], WithinAbs(0.6, 1e-4));
    }

    SECTION("random graph") {
        int N = 50;
        Adj adj;
        std::vector<std::string> ids;
        for (int i = 0; i < N; ++i) ids.push_back(std::to_string(i));
        std::uniform_real_distribution<float> wdist(0.1f, 1.0f);
        std::uniform_int_distribution<int> tdist(0, N - 1);
        for (int i = 0; i < N; ++i) {
            for (int e = 0; e < 2 + (i % 2); ++e) {
                int t = tdist(rng);
                if (t != i) adj[ids[i]].emplace_back(ids[t], wdist(rng));
            }
        }
        auto result = activation_spread_from_adjacency(
            adj, ids, {"0","1"}, {1.0f, 0.8f}, 0.7f, 0.01f, 3);
        CHECK(result.activations[0] >= 1.0f - 1e-5f);
        CHECK(result.hops_completed == 3);
        CHECK(static_cast<int>(result.activations.size()) == N);
    }
}

// ══════════════════════════════════════════════════════════════════
// 5. Knowledge Graph — CSR 批量查询
// ══════════════════════════════════════════════════════════════════

TEST_CASE("Knowledge Graph: neighbor query consistency", "[cuda][correctness]") {
    // 构建简单 CSR 图: 4 节点, 5 条边
    kg::CsrGraphData data;
    data.num_nodes = 4;
    data.num_edges = 5;
    data.row_ptr = {0, 2, 4, 5, 5};
    data.col_indices = {1, 2, 0, 3, 1};
    data.edge_relation_types = {0, 1, 0, 2, 1};
    data.edge_confidences = {0.9f, 0.7f, 0.8f, 0.6f, 0.5f};
    data.entity_type_codes = {0, 0, 1, 1};
    data.node_id_map = {"A", "B", "C", "D"};
    data.id_to_index = {{"A",0}, {"B",1}, {"C",2}, {"D",3}};

    SECTION("all neighbors of node 0") {
        std::vector<kg::NeighborQuery> queries = {{0, -1, 0}};
        auto cpu = kg::cpu_graph_find_neighbors(data, queries);

        REQUIRE(cpu.results.size() == 1);
        REQUIRE(cpu.results[0].neighbor_indices.size() == 2);
        CHECK(cpu.results[0].neighbor_indices[0] == 1);
        CHECK(cpu.results[0].neighbor_indices[1] == 2);
    }

    SECTION("filtered by relation type") {
        std::vector<kg::NeighborQuery> queries = {{0, 0, 0}};
        auto cpu = kg::cpu_graph_find_neighbors(data, queries);

        REQUIRE(cpu.results.size() == 1);
        REQUIRE(cpu.results[0].neighbor_indices.size() == 1);
        CHECK(cpu.results[0].neighbor_indices[0] == 1);
        CHECK(cpu.results[0].relation_types[0] == 0);
    }

    SECTION("max neighbors limit") {
        std::vector<kg::NeighborQuery> queries = {{0, -1, 1}};
        auto cpu = kg::cpu_graph_find_neighbors(data, queries);

        REQUIRE(cpu.results[0].neighbor_indices.size() == 1);
    }

    SECTION("dispatch matches CPU") {
        std::vector<kg::NeighborQuery> queries = {{0, -1, 0}, {1, -1, 0}};
        auto cpu = kg::cpu_graph_find_neighbors(data, queries);
        auto disp = kg::dispatch_graph_find_neighbors(data, nullptr, queries);

        REQUIRE(cpu.results.size() == disp.results.size());
        for (size_t q = 0; q < cpu.results.size(); ++q) {
            REQUIRE(cpu.results[q].neighbor_indices ==
                    disp.results[q].neighbor_indices);
        }
    }
}

TEST_CASE("Knowledge Graph: shortest path consistency", "[cuda][correctness]") {
    kg::CsrGraphData data;
    data.num_nodes = 5;
    data.num_edges = 6;
    data.row_ptr = {0, 2, 3, 4, 6, 6};
    data.col_indices = {1, 2, 3, 4, 0, 2};
    data.edge_relation_types = {0, 0, 0, 0, 0, 0};
    data.edge_confidences = {1,1,1,1,1,1};
    data.entity_type_codes = {0,0,0,0,0};
    data.node_id_map = {"0","1","2","3","4"};
    data.id_to_index = {{"0",0},{"1",1},{"2",2},{"3",3},{"4",4}};

    SECTION("path 0 -> 3") {
        std::vector<kg::PathQuery> queries = {{0, 3, 5}};
        auto cpu = kg::cpu_graph_shortest_path(data, queries);

        REQUIRE(cpu.results.size() == 1);
        CHECK(cpu.results[0].found);
        CHECK(cpu.results[0].path_length >= 1);
        CHECK(cpu.results[0].path_indices.front() == 0);
        CHECK(cpu.results[0].path_indices.back() == 3);
    }

    SECTION("self path") {
        std::vector<kg::PathQuery> queries = {{2, 2, 5}};
        auto cpu = kg::cpu_graph_shortest_path(data, queries);

        CHECK(cpu.results[0].found);
        CHECK(cpu.results[0].path_length == 0);
    }

    SECTION("unreachable") {
        std::vector<kg::PathQuery> queries = {{4, 0, 2}};
        auto cpu = kg::cpu_graph_shortest_path(data, queries);

        // 节点 4 没有出边，0 不可达
        CHECK_FALSE(cpu.results[0].found);
    }

    SECTION("dispatch matches CPU") {
        std::vector<kg::PathQuery> queries = {{0, 3, 5}, {0, 4, 5}};
        auto cpu = kg::cpu_graph_shortest_path(data, queries);
        auto disp = kg::dispatch_graph_shortest_path(data, nullptr, queries);

        REQUIRE(cpu.results.size() == disp.results.size());
        for (size_t q = 0; q < cpu.results.size(); ++q) {
            CHECK(cpu.results[q].found == disp.results[q].found);
            if (cpu.results[q].found) {
                CHECK(cpu.results[q].path_length == disp.results[q].path_length);
            }
        }
    }
}

TEST_CASE("Knowledge Graph: entity search consistency", "[cuda][correctness]") {
    kg::CsrGraphData data;
    data.num_nodes = 4;
    data.num_edges = 0;
    data.row_ptr = {0, 0, 0, 0, 0};
    data.col_indices = {};
    data.edge_relation_types = {};
    data.edge_confidences = {};
    data.entity_type_codes = {0, 0, 1, 1};
    data.node_id_map = {"cat", "dog", "apple", "orange"};
    data.id_to_index = {{"cat",0}, {"dog",1}, {"apple",2}, {"orange",3}};

    SECTION("search by entity type") {
        std::vector<kg::EntitySearchQuery> queries;
        queries.push_back({0, -1, 0});  // type_code = 0 (animal)
        auto cpu = kg::cpu_graph_entity_search(data, queries);

        REQUIRE(cpu.results.size() == 1);
        REQUIRE(cpu.results[0].matched_indices.size() == 2);
        CHECK(cpu.results[0].matched_indices[0] == 0);
        CHECK(cpu.results[0].matched_indices[1] == 1);
    }

    SECTION("dispatch matches CPU") {
        std::vector<kg::EntitySearchQuery> queries = {{1, -1, 0}};
        auto cpu = kg::cpu_graph_entity_search(data, queries);
        auto disp = kg::dispatch_graph_entity_search(data, nullptr, queries);

        REQUIRE(cpu.results.size() == disp.results.size());
        CHECK(cpu.results[0].matched_indices == disp.results[0].matched_indices);
    }
}

// ══════════════════════════════════════════════════════════════════
// 6. Analogical Transfer — Jaccard + greedy alignment
// ══════════════════════════════════════════════════════════════════

TEST_CASE("Analogical: CPU alignment consistency", "[cuda][correctness]") {
    SECTION("identical concepts") {
        std::vector<ConceptDescriptor> source = {
            {"water", "physics", {"liquid", "fluid", "wet"}, {}, {}}
        };
        std::vector<ConceptDescriptor> target = {
            {"current", "electricity", {"liquid", "fluid", "wet"}, {}, {}}
        };

        auto input = build_alignment_input(source, target);
        auto result = cpu_analogical_align(input);

        // 相同属性 → Jaccard = 1.0
        REQUIRE(result.n_source == 1);
        REQUIRE(result.n_target == 1);
        REQUIRE_THAT(result.similarity_matrix[0], WithinAbs(1.0, 1e-5));
        REQUIRE(result.pairs.size() == 1);
        CHECK(result.pairs[0].jaccard_score >= 0.3f);
    }

    SECTION("no overlap") {
        std::vector<ConceptDescriptor> source = {
            {"a", "domain1", {"x", "y"}, {}, {}}
        };
        std::vector<ConceptDescriptor> target = {
            {"b", "domain2", {"p", "q"}, {}, {}}
        };

        auto input = build_alignment_input(source, target);
        auto result = cpu_analogical_align(input);

        // 完全不同属性 → Jaccard = 0.0
        REQUIRE_THAT(result.similarity_matrix[0], WithinAbs(0.0, 1e-5));
        CHECK(result.pairs.empty());
    }

    SECTION("partial overlap") {
        std::vector<ConceptDescriptor> source = {
            {"s1", "d1", {"a", "b", "c"}, {}, {}},
            {"s2", "d1", {"a", "d", "e"}, {}, {}}
        };
        std::vector<ConceptDescriptor> target = {
            {"t1", "d2", {"a", "b", "f"}, {}, {}},
            {"t2", "d2", {"a", "d", "g"}, {}, {}}
        };

        auto input = build_alignment_input(source, target);
        auto cpu_result = cpu_analogical_align(input);
        auto disp_result = dispatch_analogical_align(input);

        // CPU 和 dispatch 结果应一致
        REQUIRE(cpu_result.n_source == disp_result.n_source);
        REQUIRE(cpu_result.n_target == disp_result.n_target);
        REQUIRE(cpu_result.similarity_matrix.size() ==
                disp_result.similarity_matrix.size());

        float err = max_abs_error(cpu_result.similarity_matrix,
                                  disp_result.similarity_matrix);
        REQUIRE_THAT(err, WithinAbs(0.0, 1e-5));
    }

    SECTION("greedy alignment avoids duplicate matching") {
        std::vector<ConceptDescriptor> source = {
            {"s1", "d1", {"a", "b"}, {}, {}},
            {"s2", "d1", {"a", "b"}, {}, {}}
        };
        std::vector<ConceptDescriptor> target = {
            {"t1", "d2", {"a", "b"}, {}, {}}
        };

        auto input = build_alignment_input(source, target);
        auto result = cpu_analogical_align(input);

        // 两个 source 概念都匹配 t1，但贪心只允许一对一
        int matched_targets = 0;
        for (const auto& p : result.pairs) {
            if (p.target_idx == 0) matched_targets++;
        }
        CHECK(matched_targets <= 1);
    }
}

// ══════════════════════════════════════════════════════════════════
// 7. Bayesian Hypothesis Update
// ══════════════════════════════════════════════════════════════════

TEST_CASE("Bayesian: CPU update consistency", "[cuda][correctness]") {
    SECTION("uniform priors with strong evidence") {
        int n = 4;
        BayesianUpdateInput input;
        input.hypotheses.resize(n);
        input.evidence.resize(n);
        for (int i = 0; i < n; ++i) {
            input.hypotheses[i].prior = 0.25f;
            input.hypotheses[i].likelihood = (i == 0) ? 0.9f : 0.033f;
            input.evidence[i] = {EvidenceType::Continuous, 0.5f};
        }
        auto cpu = cpu_bayesian_batch_update(input);
        float max_post = *std::max_element(cpu.posteriors.begin(), cpu.posteriors.end());
        CHECK(cpu.posteriors[0] == max_post);
        float sum = std::accumulate(cpu.posteriors.begin(), cpu.posteriors.end(), 0.0f);
        REQUIRE_THAT(sum, WithinAbs(1.0, 1e-4));
    }

    SECTION("dispatch matches CPU") {
        int n = 10;
        BayesianUpdateInput input;
        input.hypotheses.resize(n);
        input.evidence.resize(n);
        for (int i = 0; i < n; ++i) {
            input.hypotheses[i].prior = 0.1f;
            input.hypotheses[i].likelihood = 0.05f * (i + 1);
            input.evidence[i] = {EvidenceType::Binary, 1.0f};
        }
        auto cpu = cpu_bayesian_batch_update(input);
        auto disp = dispatch_bayesian_batch_update(input);
        REQUIRE_THAT(max_abs_error(cpu.posteriors, disp.posteriors), WithinAbs(0.0, 1e-5));
    }

    SECTION("info gain non-negative") {
        int n = 5;
        BayesianUpdateInput input;
        input.hypotheses.resize(n);
        input.evidence.resize(n);
        for (int i = 0; i < n; ++i) {
            input.hypotheses[i].prior = 0.2f;
            input.hypotheses[i].likelihood = 0.5f;
            input.evidence[i] = {EvidenceType::Binary, 1.0f};
        }
        auto result = cpu_bayesian_batch_update(input);
        for (int i = 0; i < n; ++i) CHECK(result.information_gains[i] >= -1e-6f);
    }

    SECTION("build_bayesian_input") {
        std::vector<Hypothesis> hyps = {
            {"h1", "s1", "physics", 0.6, 0.5, {}, {}, false, false, 0.3},
            {"h2", "s2", "physics", 0.4, 0.3, {}, {}, false, false, 0.2}
        };
        std::vector<GpuEvidence> evidence = {{EvidenceType::Binary, 1.0f}, {EvidenceType::Binary, 0.0f}};
        auto input = build_bayesian_input(hyps, evidence);
        REQUIRE(input.hypotheses.size() == 2);
        REQUIRE_THAT(input.hypotheses[0].likelihood, WithinAbs(0.8, 1e-5));
        REQUIRE_THAT(input.hypotheses[1].likelihood, WithinAbs(0.2, 1e-5));
    }
}

// ══════════════════════════════════════════════════════════════════
// 8. Flash Attention — CPU reference
// ══════════════════════════════════════════════════════════════════

TEST_CASE("Flash Attention: CPU and auto-dispatch", "[cuda][correctness]") {
    auto rng = make_rng();

    SECTION("identity Q=K=V") {
        int seq = 4, dim = 4;
        std::vector<float> I(seq * dim, 0.0f);
        for (int i = 0; i < seq; ++i) I[i * dim + i] = 1.0f;
        float scale = 1.0f / std::sqrt(static_cast<float>(dim));
        auto result = cpu_flash_attention(I, I, I, seq, dim, scale);
        REQUIRE(static_cast<int>(result.size()) == seq * dim);
        for (int i = 0; i < seq; ++i) {
            float row_sum = 0.0f;
            for (int d = 0; d < dim; ++d) row_sum += result[i * dim + d];
            REQUIRE_THAT(row_sum, WithinAbs(1.0, 0.1));
        }
    }

    SECTION("auto dispatch vs CPU") {
        int seq = 8, dim = 16;
        auto Q = random_matrix(rng, seq, dim);
        auto K = random_matrix(rng, seq, dim);
        auto V = random_matrix(rng, seq, dim);
        FlashAttentionConfig cfg;
        cfg.scale = 1.0f / std::sqrt(static_cast<float>(dim));
        auto result = flash_attention(Q, K, V, seq, dim, cfg);
        auto ref = cpu_flash_attention(Q, K, V, seq, dim, cfg.scale);
        REQUIRE(result.output.size() == ref.size());
        REQUIRE_THAT(max_abs_error(result.output, ref), WithinAbs(0.0, 1e-4));
    }

    SECTION("batch vs single") {
        int batch = 2, seq = 4, dim = 8;
        auto Q = random_matrix(rng, batch * seq, dim);
        auto K = random_matrix(rng, batch * seq, dim);
        auto V = random_matrix(rng, batch * seq, dim);
        FlashAttentionConfig cfg;
        cfg.scale = 1.0f / std::sqrt(static_cast<float>(dim));
        auto bresult = cuda_flash_attention_batch(Q, K, V, batch, seq, dim, cfg);
        for (int b = 0; b < batch; ++b) {
            size_t off = static_cast<size_t>(b) * seq * dim;
            auto single = cuda_flash_attention(
                {Q.begin()+off, Q.begin()+off+seq*dim},
                {K.begin()+off, K.begin()+off+seq*dim},
                {V.begin()+off, V.begin()+off+seq*dim}, seq, dim, cfg);
            for (int i = 0; i < seq * dim; ++i) {
                REQUIRE_THAT(std::abs(bresult.output[off+i] - single.output[i]),
                             WithinAbs(0.0, 1e-5));
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// 9. Embedding Trainer — train_epoch_cuda_dispatch stub
// ══════════════════════════════════════════════════════════════════

TEST_CASE("Embedding: CUDA dispatch stub returns 0 on CPU build",
          "[cuda][correctness]") {
    // 在纯 CPU 构建中，train_epoch_cuda_dispatch 应返回 0
    // (实际训练走 CPU 路径)
    std::vector<float> W_in, W_out;
    std::vector<int> corpus, neg_table;
    double loss = 0.0;
    auto rng = make_rng();

    auto count = ai_learning::learning::train_epoch_cuda_dispatch(
        W_in, W_out, corpus, neg_table,
        128, 5, 5, 1, 0.025, loss, rng);

    if (!cuda_available()) {
        CHECK(count == 0);
    }
}

// ══════════════════════════════════════════════════════════════════
// 10. CSR build from KnowledgeGraph
// ══════════════════════════════════════════════════════════════════

TEST_CASE("CSR build from KnowledgeGraph", "[cuda][correctness]") {
    SECTION("empty graph") {
        kg::KnowledgeGraph kg;
        auto data = kg::build_csr_graph_data(kg);
        CHECK(data.num_nodes == 0);
        CHECK(data.num_edges == 0);
        CHECK(data.row_ptr.size() == 1);
    }

    SECTION("populated graph") {
        kg::KnowledgeGraph kg;
        kg.add_entity(kg::Entity("cat", "animal"));
        kg.add_entity(kg::Entity("dog", "animal"));
        kg.add_entity(kg::Entity("fish", "animal"));
        kg.add_relation(kg::Relation("cat", "dog", "chases", 0.9));
        kg.add_relation(kg::Relation("dog", "cat", "chases", 0.8));
        auto data = kg::build_csr_graph_data(kg);
        CHECK(data.num_nodes == 3);
        CHECK(data.num_edges == 2);
        for (size_t i = 1; i < data.row_ptr.size(); ++i)
            CHECK(data.row_ptr[i] >= data.row_ptr[i - 1]);
    }
}

// ══════════════════════════════════════════════════════════════════
// 11. Performance Benchmarks (informational, not gating)
// ══════════════════════════════════════════════════════════════════

TEST_CASE("CUDA vs CPU performance: mat_vec", "[cuda][benchmark]") {
    if (!cuda_available()) SKIP("CUDA not available");

    auto rng = make_rng();
    int rows = 1024, cols = 1024;
    auto mat = random_matrix(rng, rows, cols);
    auto vec = random_vector(rng, cols);

    // Warm up
    (void)mat_vec(mat, rows, cols, vec);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; ++i) (void)mat_vec(mat, rows, cols, vec);
    auto t1 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 50;

    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; ++i) (void)ref_mat_vec(mat, rows, cols, vec);
    auto t3 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / 50;

    INFO("mat_vec 1024x1024: CPU " << cpu_ms << " ms, GPU " << gpu_ms << " ms");
    INFO("Speedup: " << cpu_ms / gpu_ms << "x");

    auto gpu_result = mat_vec(mat, rows, cols, vec);
    auto cpu_ref = ref_mat_vec(mat, rows, cols, vec);
    float err = max_abs_error(gpu_result, cpu_ref);
    REQUIRE_THAT(err, WithinAbs(0.0, 1e-3));
}

TEST_CASE("CUDA vs CPU performance: STDP + Bayesian", "[cuda][benchmark]") {
    if (!cuda_available()) SKIP("CUDA not available");

    auto rng = make_rng();

    // STDP benchmark
    int N = 10000, pairs = 4;
    std::vector<float> weights(N, 0.5f), pre(N * pairs), post(N * pairs);
    std::uniform_real_distribution<float> dist(0.0f, 50.0f);
    for (auto& v : pre) v = dist(rng);
    for (auto& v : post) v = dist(rng);

    auto w_copy = weights;
    stdp_batch_update(w_copy, pre, post, N, pairs);
    auto ref = ref_stdp_batch_update(weights, pre, post, N, pairs,
                                      0.1f, 0.1f, 20.0f, 20.0f, 0.0f, 1.0f);
    REQUIRE_THAT(max_abs_error(ref, w_copy), WithinAbs(0.0, 1e-3));

    // Bayesian benchmark
    int n = 1000;
    BayesianUpdateInput input;
    input.hypotheses.resize(n);
    input.evidence.resize(n);
    for (int i = 0; i < n; ++i) {
        input.hypotheses[i].prior = 0.5f;
        input.hypotheses[i].likelihood = 0.01f * (i + 1);
        input.evidence[i] = {EvidenceType::Continuous,
                             static_cast<float>(i) / n};
    }
    auto cpu = cpu_bayesian_batch_update(input);
    auto disp = dispatch_bayesian_batch_update(input);
    REQUIRE_THAT(max_abs_error(cpu.posteriors, disp.posteriors), WithinAbs(0.0, 1e-3));
}
