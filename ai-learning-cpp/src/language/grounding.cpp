/**
 * @file grounding.cpp
 * @brief 符号接地模块实现
 */

#include "ai_learning/language/grounding.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ai_learning::language {

GroundingModule::GroundingModule(int obs_dim) : obs_dim_(obs_dim) {}

auto GroundingModule::ground_from_perception(
    const std::vector<float>& observation) -> int {
    int best_cluster = -1;
    float best_distance = std::numeric_limits<float>::max();

    // 找最近聚类
    for (const auto& [id, cluster] : clusters_) {
        float dist = euclidean_distance(observation, cluster.centroid);
        if (dist < best_distance) {
            best_distance = dist;
            best_cluster = id;
        }
    }

    // 距离阈值：超过 0.5 创建新聚类
    constexpr float kThreshold = 0.5F;
    if (best_distance > kThreshold || best_cluster < 0) {
        int new_id = cluster_counter_++;
        clusters_[new_id] = {observation, 1};
        return new_id;
    }

    // 更新已有聚类
    auto& cluster = clusters_[best_cluster];
    cluster.count++;
    update_centroid(cluster.centroid, observation, cluster.count);
    return best_cluster;
}

void GroundingModule::ground_from_social(const std::string& symbol,
                                         const std::vector<float>& referent,
                                         const std::string& context) {
    int cluster_id = ground_from_perception(referent);

    auto it = symbol_mappings_.find(symbol);
    if (it == symbol_mappings_.end()) {
        SymbolMapping mapping;
        mapping.referent_clusters = {cluster_id};
        mapping.confidence = 0.5F;
        if (!context.empty()) {
            mapping.contexts = {context};
        }
        mapping.usage_count = 1;
        symbol_mappings_[symbol] = mapping;
    } else {
        auto& mapping = it->second;
        // 添加新聚类（如果不重复）
        if (std::find(mapping.referent_clusters.begin(),
                      mapping.referent_clusters.end(),
                      cluster_id) == mapping.referent_clusters.end()) {
            mapping.referent_clusters.push_back(cluster_id);
        }
        mapping.usage_count++;
        mapping.confidence = std::min(1.0F, mapping.confidence + 0.1F);
        if (!context.empty() &&
            std::find(mapping.contexts.begin(), mapping.contexts.end(),
                      context) == mapping.contexts.end()) {
            mapping.contexts.push_back(context);
        }
    }
}

auto GroundingModule::get_symbol_meaning(
    const std::string& symbol) const -> std::optional<SymbolMapping> {
    auto it = symbol_mappings_.find(symbol);
    if (it != symbol_mappings_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto GroundingModule::get_grounded_symbols() const -> std::vector<std::string> {
    std::vector<std::string> symbols;
    symbols.reserve(symbol_mappings_.size());
    for (const auto& [sym, _] : symbol_mappings_) {
        symbols.push_back(sym);
    }
    return symbols;
}

auto GroundingModule::find_similar_concepts(
    const std::vector<float>& observation,
    int top_k) const -> std::vector<SimilarConcept> {
    std::vector<SimilarConcept> results;
    results.reserve(clusters_.size());

    for (const auto& [id, cluster] : clusters_) {
        float dist = euclidean_distance(observation, cluster.centroid);
        float similarity = 1.0F / (1.0F + dist);
        results.push_back({id, similarity});
    }

    // 降序排列
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.similarity > b.similarity;
              });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }
    return results;
}

auto GroundingModule::euclidean_distance(const std::vector<float>& a,
                                         const std::vector<float>& b)
    -> float {
    float sum = 0.0F;
    auto len = std::min(a.size(), b.size());
    for (size_t i = 0; i < len; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void GroundingModule::update_centroid(std::vector<float>& centroid,
                                      const std::vector<float>& new_obs,
                                      int new_count) {
    // 增量均值: new_centroid = old_centroid * (n-1)/n + new_obs / n
    float ratio = static_cast<float>(new_count - 1) /
                  static_cast<float>(new_count);
    float inv_n = 1.0F / static_cast<float>(new_count);
    auto len = std::min(centroid.size(), new_obs.size());
    for (size_t i = 0; i < len; ++i) {
        centroid[i] = centroid[i] * ratio + new_obs[i] * inv_n;
    }
}

}  // namespace ai_learning::language
