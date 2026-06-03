/**
 * @file goal.hpp
 * @brief 目标系统数据结构 — 目标、学习步骤、学习计划
 *
 * DDD 值对象：Goal、LearningStep、LearningPlan 不可变语义。
 * GoalManager（goal_manager.hpp）持有可变状态。
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::goals {

/// 目标状态
enum class GoalStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
    Abandoned
};

/// 步骤动作类型
enum class StepAction { Learn, Execute };

/// 学习策略
enum class LearningStrategy {
    Decompose,
    Analogize,
    Practice,
    Explore
};

/// 单个学习步骤
struct LearningStep {
    StepAction      action      = StepAction::Learn;
    std::string     target;
    LearningStrategy strategy   = LearningStrategy::Explore;
    double          difficulty  = 0.0;
    std::string     analogy_from;
    std::string     reason;
    bool            done        = false;
};

/// 目标
struct Goal {
    std::string                 id;
    std::string                 description;
    GoalStatus                  status              = GoalStatus::Pending;
    std::vector<std::string>    sub_goals;
    std::string                 parent_goal;         // empty = no parent
    std::vector<std::string>    required_knowledge;
    double                      priority            = 0.5;
    double                      progress            = 0.0;
    int                         created_step        = 0;

    [[nodiscard]] auto to_map() const -> std::map<std::string, std::string>;
    static auto from_map(const std::map<std::string, std::string>& m) -> Goal;
};

/// 学习计划
struct LearningPlan {
    std::string                 goal_id;
    std::vector<LearningStep>   steps;
    double                      estimated_effort    = 0.0;
};

// ── 枚举转换 ────────────────────────────────────────────────────

[[nodiscard]] auto goal_status_to_string(GoalStatus s) -> std::string;
[[nodiscard]] auto string_to_goal_status(const std::string& s) -> GoalStatus;
[[nodiscard]] auto strategy_to_string(LearningStrategy s) -> std::string;
[[nodiscard]] auto action_to_string(StepAction a) -> std::string;

}  // namespace ai_learning::goals
