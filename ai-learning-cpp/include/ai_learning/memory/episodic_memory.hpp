/**
 * @file episodic_memory.hpp
 * @brief 情景记忆 — 经验存储与检索
 *
 * 实现三层记忆中的情景记忆层：
 * - 存储经验（观察向量 + 元数据）
 * - 按余弦相似度检索
 * - 巩固：衰减弱记忆、强化重要记忆
 */
#pragma once

#include "ai_learning/domain/memory/i_memory.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ai_learning::memory {

class EpisodicMemory : public domain::IMemory {
public:
    explicit EpisodicMemory(size_t capacity = 5000,
                            double decay_factor = 0.9,
                            double removal_threshold = 0.1);

    /// IMemory 接口
    void store(const std::vector<float>& representation,
               const std::map<std::string, std::string>& metadata) override;

    auto retrieve(const std::vector<float>& cue, int k = 5) const
        -> std::vector<domain::MemoryItem> override;

    auto consolidate() -> std::map<std::string, double> override;

    /// 获取最近 N 条记忆
    [[nodiscard]] auto get_recent(size_t n) const
        -> std::vector<domain::MemoryItem>;

    /// 辅助
    [[nodiscard]] auto size() const -> size_t;
    void clear();

private:
    static auto cosine_similarity(const std::vector<float>& a,
                                  const std::vector<float>& b) -> double;

    size_t                           capacity_;
    double                           decay_factor_;
    double                           removal_threshold_;
    std::vector<domain::MemoryItem>  items_;
    int64_t                          next_id_ = 0;
};

}  // namespace ai_learning::memory
