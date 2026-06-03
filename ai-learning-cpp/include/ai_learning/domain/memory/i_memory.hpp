/**
 * @file i_memory.hpp
 * @brief 记忆领域接口 — 三层记忆系统的统一协议
 *
 * DDD 限界上下文：记忆
 */
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <optional>

namespace ai_learning::domain {

/// 记忆条目（值对象）
struct MemoryItem {
    std::string              memory_id;
    std::vector<float>       representation;
    std::map<std::string, std::string> metadata;
    double                   strength     = 1.0;
    double                   importance   = 0.5;
    int64_t                  created_at   = 0;
    int64_t                  last_accessed = 0;
};

class IMemory {
public:
    virtual ~IMemory() = default;

    /// 存储经验（表示向量 + 元数据）
    virtual void store(const std::vector<float>& representation,
                       const std::map<std::string, std::string>& metadata) = 0;

    /// 按线索检索最相似的 k 条经验
    virtual auto retrieve(const std::vector<float>& cue, int k = 5) const
        -> std::vector<MemoryItem> = 0;

    /// 巩固记忆（模拟睡眠），返回巩固报告
    virtual auto consolidate() -> std::map<std::string, double> = 0;
};

}  // namespace ai_learning::domain
