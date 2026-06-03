/**
 * @file consolidation.cpp
 * @brief 睡眠巩固实现
 */

#include "ai_learning/learning/consolidation.hpp"

#include <algorithm>
#include <cmath>

namespace ai_learning::learning {

// ── CorticalMemory ──────────────────────────────────────────────

void CorticalMemory::store(const HippocampalEntry& entry, int source_id) {
    // 如果已经存在，增强而不是重复存储
    auto it = source_index_.find(source_id);
    if (it != source_index_.end()) {
        auto& existing = entries_[it->second];
        existing.strength = std::min(1.0F, existing.strength + 0.1F);
        return;
    }

    CorticalEntry cortical;
    cortical.entities = entry.entities;
    cortical.relations = entry.relations;
    cortical.context = entry.context;
    cortical.strength = std::min(1.0F, entry.strength * 1.2F);
    cortical.source_id = source_id;

    source_index_[source_id] = entries_.size();
    entries_.push_back(std::move(cortical));
}

auto CorticalMemory::retrieve(int top_k) const -> std::vector<CorticalEntry> {
    std::vector<CorticalEntry> sorted = entries_;
    std::partial_sort(
        sorted.begin(),
        sorted.begin() + std::min(top_k, static_cast<int>(sorted.size())),
        sorted.end(),
        [](const auto& a, const auto& b) { return a.strength > b.strength; });

    if (static_cast<int>(sorted.size()) > top_k) {
        sorted.resize(static_cast<size_t>(top_k));
    }
    return sorted;
}

void CorticalMemory::decay(float rate) {
    for (auto& e : entries_) {
        e.strength = std::max(0.0F, e.strength - rate);
    }
}

void CorticalMemory::strengthen(int source_id, float amount) {
    auto it = source_index_.find(source_id);
    if (it != source_index_.end()) {
        entries_[it->second].strength =
            std::min(1.0F, entries_[it->second].strength + amount);
    }
}

auto CorticalMemory::size() const -> int {
    return static_cast<int>(entries_.size());
}

// ── SleepConsolidation ──────────────────────────────────────────

SleepConsolidation::SleepConsolidation(HippocampalMemory& hippocampal,
                                        CorticalMemory& cortical)
    : hippocampal_(hippocampal), cortical_(cortical) {}

auto SleepConsolidation::sleep() -> ConsolidationReport {
    ConsolidationReport report;

    // 1. 获取海马中最强的记忆，巩固到皮层
    auto strongest = hippocampal_.get_strongest(5);
    for (const auto& entry : strongest) {
        cortical_.store(entry, entry.timestamp);
        ++report.memories_consolidated;
    }

    // 2. 遗忘海马中弱记忆
    report.memories_forgotten = hippocampal_.forget_weak(forget_threshold_);

    // 3. 衰减皮层记忆
    cortical_.decay(decay_rate_);

    // 4. 衰减海马记忆
    hippocampal_.decay(0.05F);

    // 5. 计算皮层平均强度
    auto cortical_entries = cortical_.retrieve(cortical_.size());
    if (!cortical_entries.empty()) {
        float total = 0.0F;
        for (const auto& e : cortical_entries) {
            total += e.strength;
        }
        report.avg_strength = total / static_cast<float>(cortical_entries.size());
    }

    return report;
}

void SleepConsolidation::set_forget_threshold(float threshold) {
    forget_threshold_ = threshold;
}

void SleepConsolidation::set_decay_rate(float rate) {
    decay_rate_ = rate;
}

}  // namespace ai_learning::learning
