/**
 * @file meta_learner.cpp
 * @brief 元学习引擎实现
 */

#include "ai_learning/learning/meta_learner.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

namespace ai_learning::learning {

MetaLearner::MetaLearner(const MetaLearnerConfig& config)
    : config_(config) {}

// ── 经验记录 ──────────────────────────────────────────────────

void MetaLearner::record_experience(const LearningExperience& experience)
{
    experiences_.push_back(experience);
    while (static_cast<int>(experiences_.size()) > config_.experience_history_size) {
        experiences_.pop_front();
    }

    update_strategy_stats_(experience);
}

void MetaLearner::record_batch(
    const std::vector<LearningExperience>& experiences)
{
    for (const auto& exp : experiences) {
        record_experience(exp);
    }
}

// ── 策略推荐 ──────────────────────────────────────────────────

auto MetaLearner::recommend_strategy(const TaskDescriptor& task) const
    -> MetaLearningRecommendation
{
    MetaLearningRecommendation rec;
    rec.suggested_learning_rate = suggest_learning_rate(task);

    // 如果经验不足，使用启发式规则
    if (experiences_.size() < static_cast<size_t>(
        config_.min_experiences_for_generalization)) {
        // 启发式：根据任务类型推荐
        if (task.task_type == "memorization") {
            rec.recommended_strategy = LearningStrategyType::kSpacedRepetition;
            rec.rationale = "记忆型任务推荐间隔重复";
        } else if (task.task_type == "reasoning") {
            rec.recommended_strategy = LearningStrategyType::kDecomposition;
            rec.rationale = "推理型任务推荐分解学习";
        } else if (task.task_type == "creative") {
            rec.recommended_strategy = LearningStrategyType::kExploratory;
            rec.rationale = "创造型任务推荐探索性学习";
        } else if (task.task_type == "procedural") {
            rec.recommended_strategy = LearningStrategyType::kStructuredPractice;
            rec.rationale = "程序型任务推荐结构化练习";
        } else if (task.novelty > 0.7) {
            rec.recommended_strategy = LearningStrategyType::kAnalogicalTransfer;
            rec.rationale = "高新颖度任务推荐类比迁移";
        } else {
            rec.recommended_strategy = LearningStrategyType::kActiveRecall;
            rec.rationale = "默认推荐主动回忆";
        }
        rec.confidence = 0.3;
        return rec;
    }

    // 基于经验的推荐：计算每个策略的匹配分数
    std::vector<std::pair<LearningStrategyType, double>> scored;
    for (const auto& [strategy, _] : strategy_stats_) {
        double score = strategy_task_match_(strategy, task);
        scored.emplace_back(strategy, score);
    }

    // 如果没有策略统计，补充所有策略
    if (scored.empty()) {
        for (int i = 0; i <= static_cast<int>(LearningStrategyType::kStructuredPractice); ++i) {
            auto s = static_cast<LearningStrategyType>(i);
            scored.emplace_back(s, strategy_task_match_(s, task));
        }
    }

    // 排序
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // ε-贪心：以 ε 概率探索
    auto selected = epsilon_greedy_select_(scored);

    rec.recommended_strategy = selected;
    rec.confidence = scored.empty() ? 0.0 : scored[0].second;

    // 生成推荐原因
    std::ostringstream oss;
    oss << "基于 " << experiences_.size() << " 次学习经验";
    if (auto it = strategy_stats_.find(selected); it != strategy_stats_.end()) {
        oss << "，该策略平均提升 " << it->second.avg_improvement;
    }
    rec.rationale = oss.str();

    // 备选策略
    for (size_t i = 1; i < std::min(scored.size(), size_t(3)); ++i) {
        rec.alternatives.push_back(scored[i].first);
    }

    return rec;
}

auto MetaLearner::get_strategy_stats(LearningStrategyType strategy) const
    -> std::optional<StrategyStats>
{
    auto it = strategy_stats_.find(strategy);
    if (it != strategy_stats_.end()) return it->second;
    return std::nullopt;
}

// ── 学习率调度 ──────────────────────────────────────────────────

auto MetaLearner::suggest_learning_rate(const TaskDescriptor& task) const
    -> double
{
    double rate = lr_schedule_.base_rate;

    // 高难度 → 低学习率（仔细学）
    rate *= (1.0 - task.difficulty * 0.5);

    // 高新颖度 → 低学习率（小心探索）
    rate *= (1.0 - task.novelty * 0.3);

    // 丰富先验知识 → 高学习率（快速整合）
    if (task.prior_knowledge_count > 5) {
        rate *= 1.2;
    }

    // 紧迫任务 → 高学习率（快速学）
    rate *= (0.8 + task.urgency * 0.4);

    return std::clamp(rate, 0.01, 1.0);
}

auto MetaLearner::update_learning_rate(double performance_delta)
    -> double
{
    lr_schedule_.step_count++;

    // 自适应学习率：表现好 → 加速，表现差 → 减速
    if (performance_delta > 0) {
        lr_schedule_.current_rate = std::min(
            lr_schedule_.current_rate * 1.1, 1.0);
        lr_schedule_.momentum = std::min(lr_schedule_.momentum + 0.05, 0.9);
    } else {
        lr_schedule_.current_rate = std::max(
            lr_schedule_.current_rate * 0.9, 0.01);
        lr_schedule_.momentum = std::max(lr_schedule_.momentum - 0.05, 0.0);
    }

    return lr_schedule_.current_rate;
}

// ── 跨任务迁移 ──────────────────────────────────────────────────

auto MetaLearner::transfer_experience(
    const std::string& source_domain,
    const std::string& target_domain)
    -> std::vector<LearningExperience>
{
    // 找到源领域中效果最好的经验
    std::vector<LearningExperience> best;
    for (const auto& exp : experiences_) {
        if (exp.task.domain == source_domain && exp.success) {
            best.push_back(exp);
        }
    }

    // 按提升排序，取 top 5
    std::sort(best.begin(), best.end(),
        [](const auto& a, const auto& b) {
            return a.improvement > b.improvement;
        });

    if (best.size() > 5) best.resize(5);

    // 将源领域经验"迁移"为目标领域的参考
    for (auto& exp : best) {
        exp.task.domain = target_domain;
        exp.improvement *= transfer_potential(source_domain, target_domain);
    }

    return best;
}

auto MetaLearner::transfer_potential(const std::string& source_domain,
                                      const std::string& target_domain) const
    -> double
{
    // 简单启发式：同领域 1.0，不同领域按经验交集算
    if (source_domain == target_domain) return 1.0;

    // 收集两个领域的策略使用
    std::set<LearningStrategyType> source_strategies, target_strategies;
    for (const auto& exp : experiences_) {
        if (exp.task.domain == source_domain) {
            source_strategies.insert(exp.strategy_used);
        }
        if (exp.task.domain == target_domain) {
            target_strategies.insert(exp.strategy_used);
        }
    }

    // Jaccard 策略相似度
    int intersection = 0;
    for (const auto& s : source_strategies) {
        if (target_strategies.count(s)) intersection++;
    }

    int union_size = static_cast<int>(
        source_strategies.size() + target_strategies.size() - intersection);

    if (union_size == 0) return 0.3;  // 默认中等潜力

    double similarity = static_cast<double>(intersection) / union_size;
    return std::clamp(similarity + 0.2, 0.1, 0.9);  // 最低 10%，最高 90%
}

// ── 自我改进 ──────────────────────────────────────────────────

auto MetaLearner::reflect() const -> std::vector<std::string>
{
    std::vector<std::string> insights;

    if (experiences_.empty()) {
        insights.push_back("尚无学习经验，无法反思");
        return insights;
    }

    // 计算最近 N 次的平均提升
    int recent_count = std::min(static_cast<int>(experiences_.size()), 20);
    double recent_improvement = 0.0;
    double overall_improvement = 0.0;
    int i = 0;
    for (auto it = experiences_.rbegin(); it != experiences_.rend(); ++it, ++i) {
        if (i < recent_count) {
            recent_improvement += it->improvement;
        }
        overall_improvement += it->improvement;
    }
    recent_improvement /= recent_count;
    overall_improvement /= experiences_.size();

    // 趋势分析
    if (recent_improvement > overall_improvement * 1.2) {
        insights.push_back("学习效率在提升 — 策略选择有效");
    } else if (recent_improvement < overall_improvement * 0.8) {
        insights.push_back("学习效率在下降 — 考虑更换策略");
    }

    // 策略分析
    for (const auto& [strategy, stats] : strategy_stats_) {
        if (stats.usage_count >= 3 && stats.success_rate < 0.3) {
            insights.push_back("策略使用次数多但成功率低，考虑减少使用");
        }
        if (stats.usage_count >= 3 && stats.avg_improvement > 0.5) {
            insights.push_back("策略效果显著，可增加使用");
        }
    }

    // 领域分析
    std::map<std::string, double> domain_improvement;
    std::map<std::string, int> domain_count;
    for (const auto& exp : experiences_) {
        domain_improvement[exp.task.domain] += exp.improvement;
        domain_count[exp.task.domain]++;
    }
    for (const auto& [domain, total] : domain_improvement) {
        double domain_avg = total / domain_count[domain];
        if (domain_avg < 0.15) {
            insights.push_back("领域 [" + domain + "] 学习效率低，需要新的学习方法");
        } else if (domain_avg > 0.3) {
            insights.push_back("领域 [" + domain + "] 学习效率良好");
        }
    }

    if (insights.empty()) {
        insights.push_back("学习表现稳定，持续观察中");
    }

    return insights;
}

auto MetaLearner::efficiency_trend() const -> std::vector<double>
{
    std::vector<double> trend;
    int window = 5;
    for (size_t i = 0; i + window <= experiences_.size(); i += window) {
        double sum = 0.0;
        for (size_t j = i; j < i + window; ++j) {
            sum += experiences_[j].improvement;
        }
        trend.push_back(sum / window);
    }
    return trend;
}

// ── 查询 ──────────────────────────────────────────────────────

auto MetaLearner::stats() const -> std::map<std::string, double>
{
    double avg_improvement = 0.0;
    double success_rate = 0.0;
    if (!experiences_.empty()) {
        for (const auto& exp : experiences_) {
            avg_improvement += exp.improvement;
            if (exp.success) success_rate += 1.0;
        }
        avg_improvement /= experiences_.size();
        success_rate /= experiences_.size();
    }

    std::set<std::string> domains;
    for (const auto& e : experiences_) domains.insert(e.task.domain);

    return {
        {"total_experiences", static_cast<double>(experiences_.size())},
        {"strategies_tracked", static_cast<double>(strategy_stats_.size())},
        {"avg_improvement", avg_improvement},
        {"success_rate", success_rate},
        {"current_learning_rate", lr_schedule_.current_rate},
        {"domains_encountered", static_cast<double>(domains.size())},
    };
}

// ── 内部方法 ──────────────────────────────────────────────────

auto MetaLearner::strategy_task_match_(
    LearningStrategyType strategy,
    const TaskDescriptor& task) const -> double
{
    double score = 0.0;

    // 从统计中获取策略效果
    auto it = strategy_stats_.find(strategy);
    if (it != strategy_stats_.end()) {
        score = it->second.avg_improvement;

        // 领域亲和度
        auto domain_it = it->second.domain_affinity.find(task.domain);
        if (domain_it != it->second.domain_affinity.end()) {
            score = score * 0.5 + domain_it->second * 0.5;
        }
    }

    // 启发式规则
    if (task.task_type == "memorization" &&
        strategy == LearningStrategyType::kSpacedRepetition) {
        score += 0.3;
    }
    if (task.task_type == "reasoning" &&
        strategy == LearningStrategyType::kDecomposition) {
        score += 0.3;
    }
    if (task.novelty > 0.7 &&
        strategy == LearningStrategyType::kAnalogicalTransfer) {
        score += 0.2;
    }
    if (task.difficulty > 0.7 &&
        strategy == LearningStrategyType::kStructuredPractice) {
        score += 0.2;
    }

    return score;
}

void MetaLearner::update_strategy_stats_(const LearningExperience& exp)
{
    auto& stats = strategy_stats_[exp.strategy_used];
    stats.strategy = exp.strategy_used;
    stats.usage_count++;
    stats.total_improvement += exp.improvement;
    stats.avg_improvement = stats.total_improvement / stats.usage_count;
    stats.best_improvement = std::max(stats.best_improvement, exp.improvement);
    stats.avg_time_cost = (stats.avg_time_cost * (stats.usage_count - 1) + exp.time_cost)
                          / stats.usage_count;
    if (exp.success) {
        stats.success_rate = (stats.success_rate * (stats.usage_count - 1) + 1.0)
                             / stats.usage_count;
    } else {
        stats.success_rate = stats.success_rate * (stats.usage_count - 1)
                             / stats.usage_count;
    }

    // 更新领域亲和度
    auto& affinity = stats.domain_affinity[exp.task.domain];
    affinity = (affinity * (stats.usage_count - 1) + exp.improvement)
               / stats.usage_count;
}

auto MetaLearner::epsilon_greedy_select_(
    const std::vector<std::pair<LearningStrategyType, double>>& scored) const
    -> LearningStrategyType
{
    if (scored.empty()) return LearningStrategyType::kActiveRecall;

    // ε-贪心
    static std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (dist(gen) < config_.exploration_rate && scored.size() > 1) {
        // 随机选择非最佳策略
        std::uniform_int_distribution<size_t> idx(1, scored.size() - 1);
        return scored[idx(gen)].first;
    }

    return scored[0].first;
}

}  // namespace ai_learning::learning
