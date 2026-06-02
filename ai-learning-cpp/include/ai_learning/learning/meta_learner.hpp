/**
 * @file meta_learner.hpp
 * @brief 元学习 — 学习如何学习
 *
 * 参考：
 *   - MAML (Finn et al., 2017): Model-Agnostic Meta-Learning
 *   - RL² (Duan et al., 2016): Fast Reinforcement Learning via Slow Reinforcement Learning
 *   - Learning to Learn (Thrun & Pratt, 1998)
 *   - Automo-G (ICLR 2025): 自适应梯度优化器
 *   - Snake (Springer, 2024): Snake: A Foundation Model for NL2SQL
 *
 * 核心能力：
 *   1. 学习策略评估 — 追踪每种学习策略在不同情境下的效果
 *   2. 自适应策略选择 — 根据任务特征动态选择最优策略
 *   3. 学习率自适应 — 自动调整学习速率（快速/慢速学习）
 *   4. 跨任务迁移 — 在不同领域间迁移学习经验
 *   5. 自我改进循环 — 用学习经验改进学习过程本身
 *
 * 人类元学习：
 *   新手用"死记硬背"学数学 → 发现"理解原理"更有效 → 切换策略
 *   在简单任务上学的快 → 在困难任务上学的慢 → 自动调整速度
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>
#include <deque>

namespace ai_learning::learning {

/// 学习策略类型
enum class LearningStrategyType {
    kRoteMemorization,     ///< 死记硬背
    kSpacedRepetition,     ///< 间隔重复
    kActiveRecall,         ///< 主动回忆
    kTrialAndError,        ///< 试错法
    kAnalogicalTransfer,   ///< 类比迁移
    kDecomposition,        ///< 分解学习
    kExplanationBased,     ///< 基于解释的学习
    kExploratory,          ///< 探索性学习
    kStructuredPractice,   ///< 结构化练习
};

/// 任务特征描述
struct TaskDescriptor {
    std::string domain;               ///< 任务领域
    std::string task_type;            ///< 任务类型（"memorization"/"reasoning"/"creative"/"procedural"）
    double difficulty = 0.5;          ///< 难度 0~1
    double novelty = 0.5;             ///< 新颖度 0~1
    double urgency = 0.5;            ///< 紧迫度 0~1
    int prior_knowledge_count = 0;    ///< 已有相关知识数量
};

/// 学习经验记录
struct LearningExperience {
    TaskDescriptor task;               ///< 任务描述
    LearningStrategyType strategy_used; ///< 使用的策略
    double performance_before = 0.0;   ///< 学习前表现
    double performance_after = 0.0;    ///< 学习后表现
    double improvement = 0.0;          ///< 提升幅度
    double time_cost = 0.0;            ///< 耗时（相对值）
    int iterations = 0;                ///< 迭代次数
    bool success = false;              ///< 是否达到目标
};

/// 学习率调度
struct LearningRateSchedule {
    double base_rate = 0.1;            ///< 基础学习率
    double current_rate = 0.1;         ///< 当前学习率
    double momentum = 0.0;             ///< 动量
    std::string schedule_type;         ///< "constant"/"decay"/"adaptive"/"cyclic"
    int warmup_steps = 0;             ///< 预热步数
    int step_count = 0;               ///< 步数计数
};

/// 策略效果统计
struct StrategyStats {
    LearningStrategyType strategy;
    int usage_count = 0;
    double total_improvement = 0.0;
    double avg_improvement = 0.0;
    double best_improvement = 0.0;
    double avg_time_cost = 0.0;
    double success_rate = 0.0;
    std::map<std::string, double> domain_affinity;  ///< 各领域亲和度
};

/// 元学习推荐
struct MetaLearningRecommendation {
    LearningStrategyType recommended_strategy;
    double confidence = 0.0;
    double suggested_learning_rate = 0.1;
    std::string rationale;
    std::vector<LearningStrategyType> alternatives;
};

/// 元学习配置
struct MetaLearnerConfig {
    int min_experiences_for_generalization = 5; ///< 泛化所需最少经验
    double improvement_threshold = 0.01;         ///< 显著改进阈值
    double exploration_rate = 0.2;               ///< ε-贪心探索率
    int experience_history_size = 200;           ///< 经验历史大小
};

/// 元学习引擎
class MetaLearner {
public:
    explicit MetaLearner(
        const MetaLearnerConfig& config = MetaLearnerConfig{});

    // ── 经验记录 ──────────────────────────────────────

    /// 记录一次学习经验
    void record_experience(const LearningExperience& experience);

    /// 批量记录
    void record_batch(const std::vector<LearningExperience>& experiences);

    // ── 策略推荐 ──────────────────────────────────────

    /// 根据任务特征推荐最佳策略
    auto recommend_strategy(const TaskDescriptor& task) const
        -> MetaLearningRecommendation;

    /// 获取某个策略的历史统计
    auto get_strategy_stats(LearningStrategyType strategy) const
        -> std::optional<StrategyStats>;

    /// 获取所有策略统计
    [[nodiscard]] auto all_strategy_stats() const
        -> const std::map<LearningStrategyType, StrategyStats>& {
        return strategy_stats_;
    }

    // ── 学习率调度 ──────────────────────────────────────

    /// 获取建议的学习率
    auto suggest_learning_rate(const TaskDescriptor& task) const -> double;

    /// 更新学习率（基于最新表现）
    auto update_learning_rate(double performance_delta) -> double;

    /// 获取当前学习率调度
    [[nodiscard]] auto learning_rate_schedule() const
        -> const LearningRateSchedule& {
        return lr_schedule_;
    }

    // ── 跨任务迁移 ──────────────────────────────────────

    /// 迁移学习经验到新领域
    auto transfer_experience(const std::string& source_domain,
                              const std::string& target_domain)
        -> std::vector<LearningExperience>;

    /// 评估领域间迁移潜力
    auto transfer_potential(const std::string& source_domain,
                             const std::string& target_domain) const -> double;

    // ── 自我改进 ──────────────────────────────────────

    /// 反思学习过程，产生改进建议
    auto reflect() const -> std::vector<std::string>;

    /// 获取学习效率趋势
    [[nodiscard]] auto efficiency_trend() const -> std::vector<double>;

    // ── 查询 ──────────────────────────────────────────

    /// 获取经验历史
    [[nodiscard]] auto experiences() const
        -> const std::deque<LearningExperience>& {
        return experiences_;
    }

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const MetaLearnerConfig& {
        return config_;
    }

private:
    MetaLearnerConfig config_;
    LearningRateSchedule lr_schedule_;

    /// 经验历史
    std::deque<LearningExperience> experiences_;

    /// 各策略统计
    std::map<LearningStrategyType, StrategyStats> strategy_stats_;

    // ── 内部方法 ──────────────────────────────────────

    /// 计算策略-任务匹配分数
    auto strategy_task_match_(LearningStrategyType strategy,
                               const TaskDescriptor& task) const -> double;

    /// 更新策略统计
    void update_strategy_stats_(const LearningExperience& exp);

    /// ε-贪心策略选择
    auto epsilon_greedy_select_(
        const std::vector<std::pair<LearningStrategyType, double>>& scored)
        const -> LearningStrategyType;
};

}  // namespace ai_learning::learning
