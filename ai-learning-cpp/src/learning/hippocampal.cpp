/**
 * @file hippocampal.cpp
 * @brief 海马记忆实现
 */

#include "ai_learning/learning/hippocampal.hpp"

#include <algorithm>
#include <cmath>

namespace ai_learning::learning {

HippocampalMemory::HippocampalMemory(int capacity) : capacity_(capacity) {
    entries_.reserve(static_cast<size_t>(capacity));
}

auto HippocampalMemory::encode(const std::vector<std::string>& entities,
                                const std::vector<std::string>& relations,
                                const std::string& context) -> int {
    HippocampalEntry entry;
    entry.entities = entities;
    entry.relations = relations;
    entry.context = context;
    entry.strength = 1.0F;
    entry.timestamp = next_id_;

    int id = next_id_;
    ++next_id_;

    // 容量满时移除最旧的
    if (static_cast<int>(entries_.size()) >= capacity_) {
        entries_.erase(entries_.begin());
        rebuild_index();
    }

    entries_.push_back(std::move(entry));

    // 更新实体索引
    for (const auto& e : entities) {
        entity_index_[e].push_back(id);
    }

    return id;
}

auto HippocampalMemory::recall(const std::string& entity, int top_k) const
    -> std::vector<HippocampalEntry> {
    auto it = entity_index_.find(entity);
    if (it == entity_index_.end()) return {};

    std::vector<HippocampalEntry> results;
    for (int idx : it->second) {
        // 查找对应条目
        for (const auto& e : entries_) {
            if (e.timestamp == idx) {
                results.push_back(e);
                break;
            }
        }
        if (static_cast<int>(results.size()) >= top_k) break;
    }

    // 按强度降序排列
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.strength > b.strength;
              });
    return results;
}

auto HippocampalMemory::get_strongest(int n) const
    -> std::vector<HippocampalEntry> {
    std::vector<HippocampalEntry> sorted = entries_;
    std::partial_sort(sorted.begin(),
                      sorted.begin() + std::min(n, static_cast<int>(sorted.size())),
                      sorted.end(),
                      [](const auto& a, const auto& b) {
                          return a.strength > b.strength;
                      });
    if (static_cast<int>(sorted.size()) > n) {
        sorted.resize(static_cast<size_t>(n));
    }
    return sorted;
}

void HippocampalMemory::decay(float rate) {
    for (auto& e : entries_) {
        e.strength = std::max(0.0F, e.strength - rate);
    }
}

auto HippocampalMemory::forget_weak(float threshold) -> int {
    int forgotten = 0;
    auto it = std::remove_if(entries_.begin(), entries_.end(),
                             [threshold, &forgotten](const auto& e) {
                                 if (e.strength < threshold) {
                                     ++forgotten;
                                     return true;
                                 }
                                 return false;
                             });
    entries_.erase(it, entries_.end());
    if (forgotten > 0) rebuild_index();
    return forgotten;
}

void HippocampalMemory::strengthen(int entry_id, float amount) {
    for (auto& e : entries_) {
        if (e.timestamp == entry_id) {
            e.strength = std::min(1.0F, e.strength + amount);
            return;
        }
    }
}

auto HippocampalMemory::size() const -> int {
    return static_cast<int>(entries_.size());
}

auto HippocampalMemory::capacity() const -> int {
    return capacity_;
}

void HippocampalMemory::rebuild_index() {
    entity_index_.clear();
    for (const auto& e : entries_) {
        for (const auto& ent : e.entities) {
            entity_index_[ent].push_back(e.timestamp);
        }
    }
}

}  // namespace ai_learning::learning
