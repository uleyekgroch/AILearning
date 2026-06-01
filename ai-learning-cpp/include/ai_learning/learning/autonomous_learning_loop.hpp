/**
 * @file autonomous_learning_loop.hpp
 * @brief 自主学习循环 — 串联所有子系统形成自主运行闭环
 *
 * 参考：
 *   - H-GRAIL (Romero 2025): 分层目标驱动架构
 *   - Active Inference (Friston): 自由能最小化闭环
 *   - DreamerV3 (Hafner 2023): 想象→规划→执行→反思
 *
 * 核心闭环（每轮）：
 *   1. 动机产生 → "我想学什么？"
 *   2. 课程规划 → "怎么学？"
 *   3. 学习执行 → "动手学"
 *   4. 反思整合 → "学到了什么？"
 *   5. 动机更新 → 正/负反馈驱动下一轮
 */
#pragma once

#include "ai_learning/learning/intrinsic_motivation.hpp"
#include "ai_learning/learning/skill_tree.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::learning {

// 前向声明
class LearnerRef;

/// 学习计划
struct LearningPlan {
    LearningGoal goal;                   ///< 学习目标
    std::vector<std::string> steps;      ///< 执行步骤描述
    std::string strategy;                ///< 学习策略名
    std::vector<std::string> resources;  ///< 需要的资源
};

/// 学习步骤的结果
struct LearningStepResult {
    int iteration = 0;                   ///< 第几轮
    LearningGoal goal;                    ///< 本轮目标
    LearningPlan plan;                    ///< 本轮计划
    double progress = 0.0;               ///< 进度 0~1
    double motivation_before = 0.0;      ///< 学习前动机
    double motivation_after = 0.0;       ///< 学习后动机
    std::vector<std::string> learned;    ///< 学到的知识
    std::string reflection;              ///< 反思文本
    bool completed = false;              ///< 目标是否完成
};

/// 自主学习循环的整体报告
struct AutonomousLoopReport {
    int total_iterations = 0;            ///< 总迭代次数
    int goals_attempted = 0;             ///< 尝试目标数
    int goals_completed = 0;            ///< 完成目标数
    double total_progress = 0.0;        ///< 总进度
    double avg_motivation = 0.0;        ///< 平均动机
    double elapsed_ms = 0.0;            ///< 耗时
    std::vector<LearningStepResult> history;  ///< 历史
};

/// 学习策略接口 — 策略模式
class ILearningStrategy {
public:
    virtual ~ILearningStrategy() = default;

    /// 执行学习步骤
    virtual auto execute(const LearningGoal& goal,
                         const LearningPlan& plan)
        -> LearningOutcome = 0;

    /// 策略名称
    [[nodiscard]] virtual auto name() const -> std::string = 0;
};

/// 资源提供者接口 — 让系统能主动获取学习材料
class ILearningResourceProvider {
public:
    virtual ~ILearningResourceProvider() = default;

    /// 搜索学习资源
    virtual auto search(const std::string& topic)
        -> std::vector<std::string> = 0;

    /// 获取资源内容
    virtual auto get_content(const std::string& resource_id)
        -> std::string = 0;
};

/// 自主学习循环配置
struct AutonomousLoopConfig {
    int max_iterations = 100;            ///< 最大迭代次数
    double motivation_threshold = 0.1;   ///< 低于此值停止
    int consolidation_interval = 10;     ///< 每 N 轮巩固一次
    bool verbose = false;                ///< 是否打印进度
};

/// 自主学习循环 — 核心整合模块
class AutonomousLearningLoop {
public:
    AutonomousLearningLoop(IntrinsicMotivationEngine& motivation,
                           SkillTree& skill_tree);

    // ── 执行接口 ──────────────────────────────────────────

    /// 运行自主学习循环
    /// @param config 循环配置
    /// @param known_topics 当前已知主题（从知识图谱获取）
    /// @param strategy 学习策略实现
    /// @param resource_provider 学习资源提供者（可选）
    auto run(const AutonomousLoopConfig& config,
             const std::vector<std::string>& known_topics,
             ILearningStrategy& strategy,
             ILearningResourceProvider* resource_provider = nullptr)
        -> AutonomousLoopReport;

    /// 执行单轮学习
    auto one_iteration(const std::vector<std::string>& known_topics,
                       ILearningStrategy& strategy,
                       ILearningResourceProvider* resource_provider = nullptr)
        -> LearningStepResult;

    // ── 查询 ──────────────────────────────────────────────

    /// 获取历史报告
    [[nodiscard]] auto last_report() const
        -> const AutonomousLoopReport& { return report_; }

    /// 获取迭代计数
    [[nodiscard]] auto iteration_count() const -> int {
        return report_.total_iterations;
    }

private:
    IntrinsicMotivationEngine& motivation_;
    SkillTree& skill_tree_;
    AutonomousLoopReport report_;
    double curiosity_signal_ = 0.5;

    // ── 五步闭环 ──────────────────────────────────────────

    /// Phase 1: 动机产生 → 选择学什么
    auto select_goal_(const std::vector<std::string>& known_topics)
        -> LearningGoal;

    /// Phase 2: 课程规划 → 怎么学
    auto plan_learning_(const LearningGoal& goal)
        -> LearningPlan;

    /// Phase 3: 学习执行 → 动手学
    auto execute_learning_(const LearningPlan& plan,
                           ILearningStrategy& strategy)
        -> LearningOutcome;

    /// Phase 4: 反思 → 学到了什么
    auto reflect_(const LearningOutcome& outcome,
                  const LearningGoal& goal) -> std::string;

    /// Phase 5: 整合 → 更新技能/动机
    void integrate_(const LearningOutcome& outcome,
                    const LearningGoal& goal);
};

}  // namespace ai_learning::learning
