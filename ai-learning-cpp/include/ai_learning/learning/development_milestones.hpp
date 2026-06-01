/**
 * @file development_milestones.hpp
 * @brief 发展里程碑系统 — 追踪学习进展，形成正反馈循环
 *
 * 参考：
 *   - Piaget's Stages of Cognitive Development (1952)
 *   - Self-Efficacy Theory (Bandura, 1977)
 *   - Flow Theory (Csikszentmihalyi, 1990): 挑战与技能匹配
 *
 * 核心机制：
 *   1. 里程碑定义 — 可达成的学习目标
 *   2. 自动检测 — 检查是否达成
 *   3. 奖励反馈 — 达成后增强动机引擎
 *   4. 可视化 — 学习进展追踪
 */
#pragma once

#include "ai_learning/learning/skill_tree.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 里程碑
struct Milestone {
    std::string id;                     ///< 唯一标识
    std::string name;                   ///< 显示名称
    std::string description;            ///< 描述
    std::string domain;                 ///< 所属领域
    int level = 0;                      ///< 级别（0=入门, 1=初级, 2=中级, 3=高级, 4=专家）

    /// 达成条件（所有条件必须满足）
    std::map<std::string, double> conditions;  ///< skill_id → 最低掌握度

    /// 奖励值（用于动机引擎）
    double reward = 1.0;

    /// 是否已达成
    bool achieved = false;

    /// 达成时间戳（迭代数）
    int achieved_at = -1;
};

/// 里程碑达成事件
struct MilestoneEvent {
    std::string milestone_id;
    std::string milestone_name;
    double reward = 0.0;
    int iteration = 0;
    std::vector<std::string> skills_mastered;  ///< 此里程碑确认掌握的技能
};

/// 学习进展快照
struct ProgressSnapshot {
    int iteration = 0;                  ///< 当前迭代
    double total_progress = 0.0;        ///< 总进度 0~1
    int milestones_achieved = 0;        ///< 已达成里程碑数
    int milestones_total = 0;           ///< 总里程碑数
    std::string current_level;          ///< 当前等级描述
    std::map<std::string, double> domain_progress;  ///< 各领域进度
};

/// 发展里程碑系统
class DevelopmentMilestones {
public:
    DevelopmentMilestones();

    // ── 里程碑管理 ────────────────────────────────────────

    /// 添加里程碑
    void add_milestone(const Milestone& milestone);

    /// 获取里程碑
    [[nodiscard]] auto get_milestone(const std::string& id) const
        -> std::optional<Milestone>;

    /// 获取所有里程碑
    [[nodiscard]] auto all_milestones() const
        -> const std::map<std::string, Milestone>& {
        return milestones_;
    }

    // ── 检查达成 ──────────────────────────────────────────

    /// 检查所有里程碑，返回新达成的事件
    auto check_milestones(const SkillTree& skill_tree)
        -> std::vector<MilestoneEvent>;

    /// 检查单个里程碑是否达成
    auto check_milestone(const std::string& id,
                          const SkillTree& skill_tree) const
        -> bool;

    /// 获取下一个未达成的里程碑
    [[nodiscard]] auto next_milestone() const -> std::optional<Milestone>;

    /// 获取下一个最接近达成的里程碑
    [[nodiscard]] auto nearest_milestone(const SkillTree& skill_tree) const
        -> std::optional<Milestone>;

    // ── 进展追踪 ──────────────────────────────────────────

    /// 获取当前进展快照
    [[nodiscard]] auto progress(const SkillTree& skill_tree) const
        -> ProgressSnapshot;

    /// 获取事件历史
    [[nodiscard]] auto event_history() const
        -> const std::vector<MilestoneEvent>& {
        return events_;
    }

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    // ── 预定义里程碑 ──────────────────────────────────────

    /// 初始化编程领域里程碑
    void init_programming_milestones();

    /// 初始化数学领域里程碑
    void init_math_milestones();

    /// 初始化通用认知里程碑
    void init_general_milestones();

private:
    std::map<std::string, Milestone> milestones_;
    std::vector<MilestoneEvent> events_;
    int current_iteration_ = 0;

    /// 计算里程碑的接近度 0~1
    [[nodiscard]] auto milestone_progress_(const Milestone& m,
                                            const SkillTree& skill_tree) const
        -> double;
};

}  // namespace ai_learning::learning
