/**
 * @file analogical_transfer_cuda_stub.cpp
 * @brief Analogical Transfer CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有函数退回到 CPU 实现，确保纯 CPU 构建正常工作。
 * CUDA 函数声明在 analogical_transfer_cuda.cuh 中，分发逻辑通过
 * cuda_available() == false 自动回退到 CPU 路径。
 */

#include "ai_learning/learning/analogical_transfer_cuda.cuh"
#include "ai_learning/learning/analogical_transfer.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

namespace ai_learning::learning {

namespace {

/// FNV-1a 32-bit hash
auto fnv1a_hash(const std::string& s) -> uint32_t {
    uint32_t h = 2166136261u;
    for (char c : s) {
        h ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
        h *= 16777619u;
    }
    return h;
}

auto cpu_jaccard(const std::vector<uint32_t>& a,
                 const std::vector<uint32_t>& b) -> float {
    if (a.empty() && b.empty()) return 0.0f;
    int intersection = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { intersection++; i++; j++; }
        else if (a[i] < b[j]) { i++; }
        else { j++; }
    }
    int union_size = static_cast<int>(a.size() + b.size()) - intersection;
    return union_size > 0 ? static_cast<float>(intersection) / union_size : 0.0f;
}

}  // anonymous namespace

auto cuda_analogical_align(const AlignmentInput& input) -> AlignmentResult {
    // stub: no GPU available, delegate to CPU
    return cpu_analogical_align(input);
}

auto cpu_analogical_align(const AlignmentInput& input) -> AlignmentResult {
    int n_source = static_cast<int>(input.source_concepts.size());
    int n_target = static_cast<int>(input.target_concepts.size());

    AlignmentResult result;
    result.n_source = n_source;
    result.n_target = n_target;
    result.similarity_matrix.resize(n_source * n_target, 0.0f);

    auto get_hashes = [](const GpuConceptSet& c,
                         const std::vector<uint32_t>& pool)
        -> std::vector<uint32_t> {
        std::vector<uint32_t> h(
            pool.begin() + c.hash_offset,
            pool.begin() + c.hash_offset + c.hash_count);
        std::sort(h.begin(), h.end());
        return h;
    };

    // N^2 Jaccard
    for (int s = 0; s < n_source; ++s) {
        auto src_hashes = get_hashes(input.source_concepts[s], input.source_hashes);
        for (int t = 0; t < n_target; ++t) {
            auto tgt_hashes = get_hashes(input.target_concepts[t], input.target_hashes);
            float sim = cpu_jaccard(src_hashes, tgt_hashes);
            result.similarity_matrix[s * n_target + t] = sim;
        }
    }

    // 贪心匹配
    struct ScoredPair { int s, t; float score; };
    std::vector<ScoredPair> all_pairs;
    all_pairs.reserve(n_source * n_target);
    for (int s = 0; s < n_source; ++s) {
        for (int t = 0; t < n_target; ++t) {
            all_pairs.push_back({s, t, result.similarity_matrix[s * n_target + t]});
        }
    }
    std::sort(all_pairs.begin(), all_pairs.end(),
              [](const ScoredPair& a, const ScoredPair& b) {
                  return a.score > b.score;
              });

    std::vector<bool> source_matched(n_source, false);
    std::vector<bool> target_used(n_target, false);
    for (const auto& p : all_pairs) {
        if (source_matched[p.s] || target_used[p.t]) continue;
        if (p.score < input.min_alignment_score) break;
        source_matched[p.s] = true;
        target_used[p.t] = true;
        result.pairs.push_back({p.s, p.t, p.score});
    }

    return result;
}

auto dispatch_analogical_align(const AlignmentInput& input) -> AlignmentResult {
    // stub: always CPU
    return cpu_analogical_align(input);
}

auto build_alignment_input(
    const std::vector<ConceptDescriptor>& source,
    const std::vector<ConceptDescriptor>& target,
    float min_score) -> AlignmentInput
{
    AlignmentInput input;
    input.min_alignment_score = min_score;

    auto hash_concept = [](const ConceptDescriptor& cd,
                          std::vector<GpuConceptSet>& concepts,
                          std::vector<uint32_t>& hashes,
                          std::vector<std::string>& ids) {
        int offset = static_cast<int>(hashes.size());
        std::set<std::string> unique_attrs(cd.attributes.begin(), cd.attributes.end());
        std::vector<uint32_t> attr_hashes;
        attr_hashes.reserve(unique_attrs.size());
        for (const auto& attr : unique_attrs) {
            attr_hashes.push_back(fnv1a_hash(attr));
        }
        std::sort(attr_hashes.begin(), attr_hashes.end());
        attr_hashes.erase(
            std::unique(attr_hashes.begin(), attr_hashes.end()),
            attr_hashes.end());

        int count = static_cast<int>(attr_hashes.size());
        concepts.push_back({
            static_cast<int>(concepts.size()),
            offset,
            count,
            fnv1a_hash(cd.domain)
        });
        hashes.insert(hashes.end(), attr_hashes.begin(), attr_hashes.end());
        ids.push_back(cd.id);
    };

    for (const auto& cd : source) {
        hash_concept(cd, input.source_concepts, input.source_hashes, input.source_ids);
    }
    for (const auto& cd : target) {
        hash_concept(cd, input.target_concepts, input.target_hashes, input.target_ids);
    }

    return input;
}

}  // namespace ai_learning::learning
