/**
 * @file self_modifier.cpp
 * @brief 自我修改模块实现 — DGM 风格的参数/策略进化
 *
 * 参考：Darwin Gödel Machine (Sakana AI + UBC, 2025)
 * 核心循环：评估 → 变异 → 测试 → 接受/拒绝
 */

#include "ai_learning/learning/self_modifier.hpp"
#include "ai_learning/core/learner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <sstream>
#include <chrono>

namespace ai_learning::core {

// ── SelfModifier ─────────────────────────────────────────────────

SelfModifier::SelfModifier(const SelfModifierConfig& config)
    : config_(config) {}

auto SelfModifier::evolve_once(Learner& learner, FitnessFn fitness)
    -> MutationResult {
    MutationResult result;

    // 1. 评估当前适应度
    result.score_before = fitness(learner);

    // 2. 生成参数修改提案
    auto proposals = propose_param_mutations(learner);
    if (proposals.empty()) {
        result.accepted = false;
        result.mutation_summary = "无可用变异提案";
        return result;
    }

    // 3. 随机选择一个提案并应用
    static std::mt19937 rng(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> dist(0, static_cast<int>(proposals.size()) - 1);
    auto& chosen = proposals[dist(rng)];

    bool applied = apply_mutation(learner, chosen);
    if (!applied) {
        result.accepted = false;
        result.mutation_summary = "无法应用变异: " + chosen.param_name;
        return result;
    }

    // 4. 重新评估适应度
    result.score_after = fitness(learner);
    result.improvement = result.score_after - result.score_before;

    // 5. 决定是否接受
    if (result.improvement >= config_.improvement_threshold) {
        result.accepted = true;
        result.mutation_summary = "接受: " + chosen.param_name + " (" +
                                   std::to_string(result.improvement) + ")";
        best_score_ = std::max(best_score_, result.score_after);
    } else {
        // 拒绝：回滚
        rollback_mutation(learner, chosen);
        result.accepted = false;
        result.mutation_summary = "拒绝: " + chosen.param_name + " (改进=" +
                                   std::to_string(result.improvement) + ")";
    }

    // 6. 记录历史
    EvolutionRecord record;
    record.generation = generation_;
    record.param_mutations.push_back(chosen);
    record.result = result;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    record.timestamp = std::to_string(static_cast<long long>(time_t));

    history_.push_back(record);
    if (static_cast<int>(history_.size()) > config_.max_history) {
        history_.erase(history_.begin());
    }

    ++generation_;

    return result;
}

auto SelfModifier::evolve(Learner& learner, FitnessFn fitness, int generations)
    -> std::vector<MutationResult> {
    std::vector<MutationResult> results;
    results.reserve(generations);
    for (int i = 0; i < generations; ++i) {
        results.push_back(evolve_once(learner, fitness));
    }
    return results;
}

auto SelfModifier::propose_param_mutations(const Learner& learner) const
    -> std::vector<ParameterMutation> {
    std::vector<ParameterMutation> proposals;

    const auto& cfg = learner.config();

    // 学习率变异
    proposals.push_back(generate_param_variant_("learning_rate", cfg.learning_rate));

    // 推理学习率变异
    proposals.push_back(generate_param_variant_("inference_lr", cfg.inference_lr));

    // 好奇心参数变异
    proposals.push_back(generate_param_variant_("curiosity_alpha", cfg.curiosity_alpha));
    proposals.push_back(generate_param_variant_("curiosity_beta", cfg.curiosity_beta));

    return proposals;
}

auto SelfModifier::apply_mutation(Learner& learner,
                                    const ParameterMutation& mutation) -> bool {
    // 目前通过修改 config 是不可能的（config 是 const），
    // 所以我们通过影响学习行为来"应用"变异。
    // 对于 DGM 风格的真正自我修改，需要更深层的架构支持。
    // 当前实现：通过影响引擎参数来模拟。

    // 注意：真正的 DGM 会修改自己的源代码。
    // 这里我们修改可调参数作为第一步。

    // 暂时返回 true 表示"应用成功"
    // 在未来的迭代中，Learner 需要暴露可变参数接口
    (void)learner;
    (void)mutation;
    return true;
}

auto SelfModifier::rollback_mutation(Learner& learner,
                                       const ParameterMutation& mutation) -> bool {
    (void)learner;
    (void)mutation;
    return true;
}

auto SelfModifier::generate_param_variant_(const std::string& name,
                                             double current_value) const
    -> ParameterMutation {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-config_.param_mutation_range,
                                                 config_.param_mutation_range);

    double delta = dist(rng) * current_value;
    double new_value = current_value + delta;

    // 确保正值
    if (name.find("rate") != std::string::npos ||
        name.find("alpha") != std::string::npos ||
        name.find("beta") != std::string::npos) {
        new_value = std::max(1e-6, new_value);
        new_value = std::min(1.0, new_value);
    }

    std::ostringstream rationale;
    rationale << name << ": " << current_value << " → " << new_value;

    return ParameterMutation{
        name,
        current_value,
        new_value,
        rationale.str(),
        std::abs(delta)
    };
}

// ── StrategySelector ─────────────────────────────────────────────

auto StrategySelector::recommend(const std::string& task_type,
                                   double current_performance,
                                   double difficulty)
    -> Strategy {
    // 基于任务类型、当前表现和难度推荐策略
    if (task_type == "memorization") {
        return Strategy::kSpacedRepetition;
    }

    if (task_type == "exploration") {
        return Strategy::kTrialAndError;
    }

    if (task_type == "understanding") {
        if (current_performance < 0.3) {
            return Strategy::kDecomposition;
        }
        return Strategy::kAnalogicalTransfer;
    }

    if (task_type == "problem_solving") {
        if (difficulty > 0.7) {
            return Strategy::kDecomposition;
        }
        return Strategy::kTrialAndError;
    }

    // 默认：主动回忆
    return Strategy::kActiveRecall;
}

auto StrategySelector::to_string(Strategy s) -> std::string {
    switch (s) {
        case Strategy::kRoteMemorization:   return "死记硬背";
        case Strategy::kSpacedRepetition:   return "间隔重复";
        case Strategy::kActiveRecall:       return "主动回忆";
        case Strategy::kTrialAndError:      return "试错法";
        case Strategy::kAnalogicalTransfer: return "类比迁移";
        case Strategy::kDecomposition:      return "分解学习";
    }
    return "未知";
}

void StrategySelector::record_outcome(Strategy strategy,
                                       double performance_before,
                                       double performance_after) {
    double improvement = performance_after - performance_before;
    auto& [total, count] = strategy_stats_[strategy];
    total += improvement;
    ++count;
}

}  // namespace ai_learning::core
