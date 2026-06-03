/**
 * @file goal_manager.hpp
 * @brief 目标管理器 — 分解、规划、追踪目标
 *
 * 三个核心类：
 * - GoalDecomposer: 利用知识图谱将目标分解为子目标
 * - LearningPlanner: 为目标生成学习计划（拓扑排序 + 策略选择）
 * - GoalManager: 目标生命周期管理（创建、追踪、进度传播）
 *
 * 依赖：
 * - KnowledgeGraph（知识图谱查询）
 * - MetacognitionEngine（可选，用于盲区检测增强）
 */
#pragma once

#include "ai_learning/goals/goal.hpp"

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_learning::domain::knowledge {
class KnowledgeGraph;
}

namespace ai_learning::core {
class MetacognitionEngine;
}

namespace ai_learning::goals {

// ── GoalDecomposer ──────────────────────────────────────────────

class GoalDecomposer {
public:
    GoalDecomposer(domain::knowledge::KnowledgeGraph& kg,
                   const core::MetacognitionEngine* meta = nullptr);

    /// 将目标分解为子目标列表
    [[nodiscard]] auto decompose_goal(const Goal& goal) const
        -> std::vector<Goal>;

    /// 识别目标所需知识（置信度 < 0.3 的实体）
    [[nodiscard]] auto identify_required_knowledge(const Goal& goal) const
        -> std::vector<std::string>;

private:
    /// 从文本中贪婪匹配知识图谱实体
    [[nodiscard]] auto extract_entities_(const std::string& text) const
        -> std::vector<std::string>;

    /// 无实体匹配时按标点分割
    [[nodiscard]] auto fallback_decompose_(const std::string& text) const
        -> std::vector<Goal>;

    domain::knowledge::KnowledgeGraph& kg_;
    const core::MetacognitionEngine*   meta_;
};

// ── LearningPlanner ─────────────────────────────────────────────

class LearningPlanner {
public:
    LearningPlanner(domain::knowledge::KnowledgeGraph& kg,
                    const core::MetacognitionEngine* meta = nullptr);

    /// 为目标生成学习计划
    [[nodiscard]] auto plan_learning(const Goal& goal) const
        -> LearningPlan;

private:
    /// 拓扑排序知识缺口（Kahn 算法）
    [[nodiscard]] auto topological_sort_(
        const std::vector<std::string>& gaps) const
        -> std::vector<std::string>;

    /// 估算单个知识点的难度
    [[nodiscard]] auto estimate_difficulty_(const std::string& entity_id) const
        -> double;

    /// 根据难度和缺口数选择策略
    [[nodiscard]] auto choose_strategy_(double difficulty, int gap_count) const
        -> LearningStrategy;

    domain::knowledge::KnowledgeGraph& kg_;
    const core::MetacognitionEngine*   meta_;
};

// ── GoalManager ─────────────────────────────────────────────────

/// 持久化状态
struct ManagerState {
    std::unordered_map<std::string, Goal>         goals;
    std::unordered_map<std::string, LearningPlan>  plans;
    int counter = 0;
};

class GoalManager {
public:
    explicit GoalManager(domain::knowledge::KnowledgeGraph& kg,
                         const core::MetacognitionEngine* meta = nullptr);

    /// 创建新目标（自动分配 ID，自动识别所需知识）
    auto create_goal(const std::string& description,
                     double priority = 0.5) -> Goal;

    /// 分解目标并生成学习计划
    auto decompose_and_plan(const std::string& goal_id)
        -> std::optional<LearningPlan>;

    /// 更新目标进度（自动状态转换 + 父目标传播）
    void update_progress(const std::string& goal_id, double progress);

    /// 获取下一个待执行的学习步骤
    [[nodiscard]] auto get_next_action() const
        -> std::optional<LearningStep>;

    /// 获取活跃目标（Pending + InProgress）
    [[nodiscard]] auto get_active_goals() const
        -> std::vector<const Goal*>;

    /// 检查目标是否已完成
    [[nodiscard]] auto check_completion(const std::string& goal_id) const
        -> bool;

    /// 按 ID 查找目标（未找到返回 nullptr）
    [[nodiscard]] auto get_goal(const std::string& goal_id) const
        -> const Goal*;

    /// 持久化
    [[nodiscard]] auto save_state() const -> ManagerState;
    void load_state(const ManagerState& state);

private:
    /// 向父目标传播进度
    void propagate_progress_(const std::string& parent_id);

    domain::knowledge::KnowledgeGraph& kg_;
    const core::MetacognitionEngine*   meta_;

    std::unordered_map<std::string, Goal>         goals_;
    std::unordered_map<std::string, LearningPlan>  plans_;
    int                                             counter_ = 0;

    GoalDecomposer  decomposer_;
    LearningPlanner planner_;
};

}  // namespace ai_learning::goals
