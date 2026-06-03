#pragma once
/**
 * @file hippocampal.hpp
 * @brief 海马记忆 — 快速单次学习，容量有限
 *
 * 模拟生物海马体：
 * 1. 单次暴露即可编码
 * 2. 容量有限（FIFO 淘汰）
 * 3. 实体索引加速检索
 */

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 单条海马记忆
struct HippocampalEntry {
    std::vector<std::string> entities;   ///< 相关实体
    std::vector<std::string> relations;  ///< 相关关系
    std::string context;                 ///< 上下文
    float strength = 1.0F;              ///< 记忆强度 [0,1]
    int timestamp = 0;                   ///< 时间戳（全局序号）
};

/// 海马记忆系统 — 快速单次学习
class HippocampalMemory {
public:
    explicit HippocampalMemory(int capacity = 100);

    /// 编码一条记忆（单次学习），返回条目 ID
    [[nodiscard]] auto encode(const std::vector<std::string>& entities,
                              const std::vector<std::string>& relations,
                              const std::string& context) -> int;

    /// 按实体检索相关记忆，最多 top_k 条
    [[nodiscard]] auto recall(const std::string& entity,
                              int top_k = 5) const
        -> std::vector<HippocampalEntry>;

    /// 获取最强的 n 条记忆
    [[nodiscard]] auto get_strongest(int n = 3) const
        -> std::vector<HippocampalEntry>;

    /// 衰减所有记忆强度
    void decay(float rate = 0.05F);

    /// 遗忘强度低于阈值的记忆，返回被遗忘的数量
    [[nodiscard]] auto forget_weak(float threshold = 0.3F) -> int;

    /// 增强指定条目强度
    void strengthen(int entry_id, float amount = 0.1F);

    /// 当前记忆数
    [[nodiscard]] auto size() const -> int;

    /// 容量
    [[nodiscard]] auto capacity() const -> int;

private:
    int capacity_;
    std::vector<HippocampalEntry> entries_;
    std::map<std::string, std::vector<int>> entity_index_;
    int next_id_ = 0;

    void rebuild_index();
};

}  // namespace ai_learning::learning
