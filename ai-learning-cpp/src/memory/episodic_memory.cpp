/**
 * @file episodic_memory.cpp
 * @brief 情景记忆实现
 */

#include "ai_learning/memory/episodic_memory.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ai_learning::memory {

EpisodicMemory::EpisodicMemory(size_t capacity, double decay_factor,
                               double removal_threshold)
    : capacity_(capacity),
      decay_factor_(decay_factor),
      removal_threshold_(removal_threshold) {}

void EpisodicMemory::store(const std::vector<float>& representation,
                           const std::map<std::string, std::string>& metadata) {
    domain::MemoryItem item;
    item.memory_id      = "ep_" + std::to_string(next_id_++);
    item.representation = representation;
    item.metadata       = metadata;
    item.strength       = 1.0;
    item.importance     = 0.5;
    item.created_at     = next_id_;
    item.last_accessed  = next_id_;

    items_.push_back(std::move(item));

    // FIFO 容量限制
    while (items_.size() > capacity_) {
        items_.erase(items_.begin());
    }
}

auto EpisodicMemory::retrieve(const std::vector<float>& cue, int k) const
    -> std::vector<domain::MemoryItem> {
    if (items_.empty() || k <= 0) return {};

    // 计算相似度并排序
    std::vector<std::pair<double, size_t>> scored;
    scored.reserve(items_.size());
    for (size_t i = 0; i < items_.size(); ++i) {
        double sim = cosine_similarity(cue, items_[i].representation);
        scored.emplace_back(sim * items_[i].strength, i);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<domain::MemoryItem> results;
    int count = std::min(k, static_cast<int>(scored.size()));
    for (int i = 0; i < count; ++i) {
        results.push_back(items_[scored[i].second]);
    }
    return results;
}

auto EpisodicMemory::consolidate() -> std::map<std::string, double> {
    int removed = 0;
    int boosted = 0;

    // 衰减所有强度
    for (auto& item : items_) {
        item.strength *= decay_factor_;
    }

    // 移除过弱记忆
    auto it = std::remove_if(items_.begin(), items_.end(),
                             [this](const auto& item) {
                                 return item.strength < removal_threshold_;
                             });
    removed = static_cast<int>(items_.end() - it);
    items_.erase(it, items_.end());

    // 强化最近访问的记忆
    for (auto& item : items_) {
        if (item.last_accessed > next_id_ - 10) {
            item.strength = std::min(item.strength + 0.1, 1.0);
            ++boosted;
        }
    }

    return {{"removed", removed}, {"boosted", boosted}, {"remaining", static_cast<double>(items_.size())}};
}

auto EpisodicMemory::get_recent(size_t n) const
    -> std::vector<domain::MemoryItem> {
    std::vector<domain::MemoryItem> result;
    size_t start = items_.size() > n ? items_.size() - n : 0;
    for (size_t i = start; i < items_.size(); ++i) {
        result.push_back(items_[i]);
    }
    return result;
}

auto EpisodicMemory::size() const -> size_t { return items_.size(); }
void EpisodicMemory::clear() { items_.clear(); }

auto EpisodicMemory::cosine_similarity(const std::vector<float>& a,
                                       const std::vector<float>& b) -> double {
    if (a.size() != b.size() || a.empty()) return 0.0;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot    += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a < 1e-12 || norm_b < 1e-12) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

}  // namespace ai_learning::memory
