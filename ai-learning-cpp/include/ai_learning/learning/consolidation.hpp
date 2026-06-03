#pragma once
/**
 * @file consolidation.hpp
 * @brief 睡眠巩固 — 海马到皮层的记忆转移
 *
 * 模拟生物睡眠巩固：
 * 1. 重放海马中的强记忆
 * 2. 将它们存储到皮层（长期存储）
 * 3. 衰减弱记忆并遗忘
 * 4. 皮层记忆也会缓慢衰减
 */

#include "ai_learning/learning/hippocampal.hpp"

#include <map>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 皮层记忆条目（长期存储）
struct CorticalEntry {
    std::vector<std::string> entities;
    std::vector<std::string> relations;
    std::string context;
    float strength = 0.0F;
    int source_id = -1;  ///< 来源的海马条目 ID
};

/// 巩固报告
struct ConsolidationReport {
    int memories_consolidated = 0;  ///< 巩固数
    int memories_forgotten = 0;     ///< 遗忘数
    float avg_strength = 0.0F;     ///< 平均皮层强度
};

/// 皮层记忆 — 长期存储
class CorticalMemory {
public:
    /// 存储一条记忆到皮层
    void store(const HippocampalEntry& entry, int source_id);

    /// 检索皮层记忆，按强度降序，最多 top_k 条
    [[nodiscard]] auto retrieve(int top_k = 10) const
        -> std::vector<CorticalEntry>;

    /// 衰减所有皮层记忆
    void decay(float rate = 0.01F);

    /// 增强指定来源的记忆
    void strengthen(int source_id, float amount = 0.1F);

    /// 当前皮层记忆数
    [[nodiscard]] auto size() const -> int;

private:
    std::vector<CorticalEntry> entries_;
    std::map<int, size_t> source_index_;  ///< source_id → entries_ 索引
};

/// 睡眠巩固系统
class SleepConsolidation {
public:
    explicit SleepConsolidation(HippocampalMemory& hippocampal,
                                CorticalMemory& cortical);

    /// 执行一轮睡眠巩固
    [[nodiscard]] auto sleep() -> ConsolidationReport;

    /// 设置巩固参数
    void set_forget_threshold(float threshold);
    void set_decay_rate(float rate);

private:
    HippocampalMemory& hippocampal_;
    CorticalMemory& cortical_;
    float forget_threshold_ = 0.3F;
    float decay_rate_ = 0.01F;
};

}  // namespace ai_learning::learning
