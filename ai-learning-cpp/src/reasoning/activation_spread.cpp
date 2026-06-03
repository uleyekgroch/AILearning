/**
 * @file activation_spread.cpp
 * @brief 激活扩散推理实现
 */

#include "ai_learning/reasoning/activation_spread.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

#include <algorithm>
#include <queue>
#include <set>
#include <utility>

namespace ai_learning::reasoning {

ActivationSpread::ActivationSpread(
    const domain::knowledge::KnowledgeGraph& graph,
    double decay_rate,
    double threshold,
    int max_depth)
    : graph_(graph),
      decay_rate_(decay_rate),
      threshold_(threshold),
      max_depth_(max_depth) {}

auto ActivationSpread::spread(const std::vector<std::string>& seed_entities,
                              int depth) const -> std::vector<ActivatedNode> {
    clear();
    int effective_depth = (depth > 0) ? depth : max_depth_;

    // BFS 队列
    struct QueueEntry {
        std::string entity;
        double      activation;
        int         hop;
    };
    std::queue<QueueEntry> queue;
    std::set<std::string>  visited;

    // 初始化种子节点
    for (const auto& seed : seed_entities) {
        queue.push(QueueEntry{seed, 1.0, 0});
        activations_[seed] = 1.0;
        visited.insert(seed);
    }

    // BFS 扩散
    while (!queue.empty()) {
        QueueEntry entry = queue.front();
        queue.pop();
        const auto& entity = entry.entity;
        double activation = entry.activation;
        int hop = entry.hop;

        if (hop >= effective_depth) continue;

        // 沿出边扩散
        auto relations = graph_.get_relations_of(entity, "out");
        double child_activation = activation * decay_rate_;

        if (child_activation < threshold_) continue;

        for (const auto& rel_ref : relations) {
            const auto& rel = rel_ref.get();
            const auto& target = rel.target_id();
            double existing = activations_.count(target) ? activations_[target] : 0.0;
            double new_act = std::max(existing, child_activation);
            activations_[target] = new_act;

            if (visited.find(target) == visited.end()) {
                visited.insert(target);
                queue.push(QueueEntry{target, child_activation, hop + 1});
            }
        }

        // 沿入边扩散
        auto incoming = graph_.get_relations_of(entity, "in");
        double in_activation = activation * decay_rate_ * 0.8;  // 入边略弱
        if (in_activation >= threshold_) {
            for (const auto& rel_ref : incoming) {
                const auto& rel = rel_ref.get();
                const auto& source = rel.source_id();
                double existing = activations_.count(source) ? activations_[source] : 0.0;
                double new_act = std::max(existing, in_activation);
                activations_[source] = new_act;

                if (visited.find(source) == visited.end()) {
                    visited.insert(source);
                    queue.push(QueueEntry{source, in_activation, hop + 1});
                }
            }
        }
    }

    // 收集结果（按激活度排序，排除种子）
    std::vector<ActivatedNode> results;
    std::set<std::string> seed_set(seed_entities.begin(), seed_entities.end());

    for (const auto& [eid, act] : activations_) {
        if (seed_set.count(eid)) continue;
        if (act >= threshold_) {
            results.push_back(ActivatedNode{eid, act, 0});
        }
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.activation > b.activation; });

    return results;
}

void ActivationSpread::set_decay_rate(double rate) { decay_rate_ = rate; }
void ActivationSpread::set_threshold(double threshold) { threshold_ = threshold; }

auto ActivationSpread::get_activation(const std::string& entity_id) const
    -> double {
    auto it = activations_.find(entity_id);
    return (it != activations_.end()) ? it->second : 0.0;
}

void ActivationSpread::clear() const { activations_.clear(); }

}  // namespace ai_learning::reasoning
