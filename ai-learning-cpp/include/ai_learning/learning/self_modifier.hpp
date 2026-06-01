/**
 * @file self_modifier.hpp
 * @brief 自我修改模块 — 让学习体修改自身的学习策略和参数
 *
 * 参考：Darwin Gödel Machine (Sakana AI + UBC, 2025)
 * 核心：AI 读取自己的代码/参数 → 提出修改 → 在 benchmark 上测试 → 保留改进
 *
 * 能力 2：自我修改（Self-Modification）
 * 人类学习会改变自己的认知策略，不是重复做同样的事。
 */
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <variant>

namespace ai_learning::core {

// 前向声明
class Learner;

/// 可修改的参数类型
using ParamValue = std::variant<double, int, std::string>;

/// 参数修改提案
struct ParameterMutation {
    std::string param_name;           ///< 参数名（如 "learning_rate"）
    ParamValue old_value;             ///< 原值
    ParamValue new_value;             ///< 新值
    std::string rationale;            ///< 修改理由
    double expected_improvement = 0.0; ///< 预期提升
};

/// 策略修改提案
struct StrategyMutation {
    std::string strategy_name;        ///< 策略名
    std::string old_strategy;         ///< 旧策略描述
    std::string new_strategy;         ///< 新策略描述
    std::string rationale;            ///< 修改理由
};

/// 修改结果
struct MutationResult {
    bool accepted = false;            ///< 是否接受修改
    double score_before = 0.0;        ///< 修改前得分
    double score_after = 0.0;         ///< 修改后得分
    double improvement = 0.0;         ///< 实际提升
    std::string mutation_summary;     ///< 修改摘要
};

/// 进化历史记录
struct EvolutionRecord {
    int generation = 0;               ///< 进化代数
    std::vector<ParameterMutation> param_mutations;
    std::vector<StrategyMutation> strategy_mutations;
    MutationResult result;
    std::string timestamp;            ///< 时间戳
};

/// 自我修改器配置
struct SelfModifierConfig {
    double improvement_threshold = 0.01; ///< 最低改进阈值（低于此值拒绝修改）
    int max_history = 100;               ///< 最大历史记录数
    double param_mutation_rate = 0.1;    ///< 参数变异率
    double param_mutation_range = 0.5;   ///< 参数变异范围（±50%）
};

/// 适应度函数类型：评估学习体在某个任务上的表现
using FitnessFn = std::function<double(Learner& learner)>;

/// 自我修改器 — 让学习体自主改进学习策略
class SelfModifier {
public:
    explicit SelfModifier(const SelfModifierConfig& config = {});

    /// 执行一轮自我修改进化
    /// @param learner 学习体
    /// @param fitness 适应度评估函数
    /// @return 修改结果
    auto evolve_once(Learner& learner, FitnessFn fitness)
        -> MutationResult;

    /// 执行多轮进化
    auto evolve(Learner& learner, FitnessFn fitness, int generations)
        -> std::vector<MutationResult>;

    /// 生成参数修改提案
    auto propose_param_mutations(const Learner& learner) const
        -> std::vector<ParameterMutation>;

    /// 应用参数修改到学习体
    static auto apply_mutation(Learner& learner,
                                const ParameterMutation& mutation) -> bool;

    /// 回滚修改
    static auto rollback_mutation(Learner& learner,
                                   const ParameterMutation& mutation) -> bool;

    /// 获取进化历史
    [[nodiscard]] auto history() const
        -> const std::vector<EvolutionRecord>& { return history_; }

    /// 获取最佳得分
    [[nodiscard]] auto best_score() const -> double { return best_score_; }

    /// 获取当前代数
    [[nodiscard]] auto generation() const -> int { return generation_; }

private:
    SelfModifierConfig config_;
    std::vector<EvolutionRecord> history_;
    double best_score_ = 0.0;
    int generation_ = 0;

    /// 生成随机参数变异
    auto generate_param_variant_(const std::string& name,
                                  double current_value) const
        -> ParameterMutation;
};

/// 策略选择器 — 根据任务特征选择最佳学习策略
class StrategySelector {
public:
    /// 学习策略枚举
    enum class Strategy {
        kRoteMemorization,    ///< 死记硬背
        kSpacedRepetition,    ///< 间隔重复
        kActiveRecall,        ///< 主动回忆
        kTrialAndError,       ///< 试错法
        kAnalogicalTransfer,  ///< 类比迁移
        kDecomposition,       ///< 分解学习
    };

    /// 根据任务特征推荐策略
    static auto recommend(const std::string& task_type,
                           double current_performance,
                           double difficulty)
        -> Strategy;

    /// 策略转字符串
    static auto to_string(Strategy s) -> std::string;

    /// 记录策略效果（用于未来推荐）
    void record_outcome(Strategy strategy, double performance_before,
                        double performance_after);

    /// 获取策略历史效果
    [[nodiscard]] auto get_strategy_stats() const
        -> const std::map<Strategy, std::pair<double, int>>& {
        return strategy_stats_;
    }

private:
    /// 策略累计效果：{策略 → (累计提升, 使用次数)}
    std::map<Strategy, std::pair<double, int>> strategy_stats_;
};

}  // namespace ai_learning::core
