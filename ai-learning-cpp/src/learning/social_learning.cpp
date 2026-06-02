/**
 * @file social_learning.cpp
 * @brief 社会性/观察学习引擎实现
 */

#include "ai_learning/learning/social_learning.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <set>

namespace ai_learning::learning {

SocialLearningEngine::SocialLearningEngine(
    const SocialLearningConfig& config)
    : config_(config) {}

// ── 观察建模 ──────────────────────────────────────────────────

auto SocialLearningEngine::observe(const BehaviorObservation& observation)
    -> SocialLearningReport
{
    SocialLearningReport report;
    report.observations_processed = 1;

    // 存储观察
    observations_[observation.domain].push_back(observation);
    total_observations_++;

    // 更新榜样评估
    update_model_assessment(observation.agent_id, observation.outcome_quality);

    // 如果同一领域有足够观察，尝试提取策略
    const auto& domain_obs = observations_[observation.domain];
    if (static_cast<int>(domain_obs.size()) >=
        static_cast<int>(config_.min_observation_for_pattern)) {
        // 检查是否已经提取过
        auto pattern = extract_pattern(observation.domain);
        if (pattern.has_value()) {
            report.learned_strategies.push_back(*pattern);
        }
    }

    return report;
}

auto SocialLearningEngine::observe_batch(
    const std::vector<BehaviorObservation>& observations)
    -> SocialLearningReport
{
    SocialLearningReport report;

    for (const auto& obs : observations) {
        auto sub_report = observe(obs);
        report.observations_processed += sub_report.observations_processed;
        report.learned_strategies.insert(
            report.learned_strategies.end(),
            sub_report.learned_strategies.begin(),
            sub_report.learned_strategies.end());
    }

    return report;
}

auto SocialLearningEngine::extract_pattern(const std::string& domain)
    -> std::optional<StrategyPattern>
{
    auto it = observations_.find(domain);
    if (it == observations_.end()) return std::nullopt;

    const auto& obs = it->second;
    if (static_cast<int>(obs.size()) <
        static_cast<int>(config_.min_observation_for_pattern)) {
        return std::nullopt;
    }

    // 只考虑成功的观察（outcome_quality > 0.5）
    std::vector<BehaviorObservation> successful;
    for (const auto& o : obs) {
        if (o.outcome_quality > 0.5) successful.push_back(o);
    }

    if (successful.size() < config_.min_observation_for_pattern) return std::nullopt;

    StrategyPattern pattern;
    pattern.id = next_id_();
    pattern.domain = domain;
    pattern.observation_count = static_cast<int>(successful.size());

    // 提取共同步骤
    pattern.steps = extract_common_steps_(successful);

    // 提取前置条件
    std::set<std::string> all_preconditions;
    for (const auto& o : successful) {
        for (const auto& pre : o.preconditions) {
            all_preconditions.insert(pre);
        }
    }
    pattern.preconditions.assign(all_preconditions.begin(), all_preconditions.end());

    // 提取预期结果
    std::set<std::string> all_effects;
    for (const auto& o : successful) {
        for (const auto& eff : o.effects) {
            all_effects.insert(eff);
        }
    }
    pattern.expected_outcomes.assign(all_effects.begin(), all_effects.end());

    // 计算成功率
    double total_quality = 0.0;
    std::string most_frequent_agent;
    std::map<std::string, int> agent_counts;
    for (const auto& o : obs) {
        total_quality += o.outcome_quality;
        agent_counts[o.agent_id]++;
    }
    pattern.observed_success_rate = total_quality / obs.size();

    int max_count = 0;
    for (const auto& [agent, count] : agent_counts) {
        if (count > max_count) {
            max_count = count;
            most_frequent_agent = agent;
        }
    }
    pattern.source_agent = most_frequent_agent;

    // 生成名称
    pattern.name = domain + "_learned_pattern";

    // 如果策略已存在且观察次数更多，更新
    strategies_[pattern.id] = pattern;

    return pattern;
}

// ── 榜样管理 ──────────────────────────────────────────────────

auto SocialLearningEngine::evaluate_model(const std::string& agent_id) const
    -> RoleModelProfile
{
    auto it = role_models_.find(agent_id);
    if (it != role_models_.end()) {
        return it->second;
    }

    // 新榜样的默认评估
    RoleModelProfile profile;
    profile.agent_id = agent_id;
    return profile;
}

void SocialLearningEngine::update_model_assessment(
    const std::string& agent_id,
    double outcome_quality)
{
    auto& model = role_models_[agent_id];
    model.agent_id = agent_id;
    model.observations_count++;

    // 滚动更新专业度（指数移动平均）
    double alpha = 0.3;
    model.expertise = model.expertise * (1 - alpha) + outcome_quality * alpha;

    // 可信度随观察次数增加
    model.trustworthiness = std::min(
        0.3 + model.observations_count * 0.05, 1.0);
}

auto SocialLearningEngine::select_role_model(const std::string& domain) const
    -> std::optional<std::string>
{
    std::string best_agent;
    double best_score = 0.0;

    for (const auto& [id, model] : role_models_) {
        // 综合评分 = 专业度权重 × 专业度 + 相似度权重 × 相似度
        double score = config_.expertise_weight * model.expertise +
                       config_.similarity_weight * model.similarity;

        // 检查该榜样是否在目标领域有展示
        bool has_domain_skill = false;
        for (const auto& skill : model.demonstrated_skills) {
            if (skill.find(domain) != std::string::npos) {
                has_domain_skill = true;
                break;
            }
        }

        // 如果有领域特定技能，额外加分
        if (has_domain_skill) score += 0.2;

        if (score > best_score && model.expertise >= config_.expertise_threshold) {
            best_score = score;
            best_agent = id;
        }
    }

    if (!best_agent.empty()) return best_agent;
    return std::nullopt;
}

// ── 策略模仿 ──────────────────────────────────────────────────

auto SocialLearningEngine::imitate(const StrategyPattern& pattern,
                                    const std::string& target_context)
    -> std::vector<std::string>
{
    std::vector<std::string> results;

    // 逐步模仿
    for (const auto& step : pattern.steps) {
        results.push_back("模仿步骤: " + step + " (情境: " + target_context + ")");
    }

    // 评估模仿结果（基于策略的历史成功率）
    if (pattern.observed_success_rate > 0.5) {
        successful_imitations_++;
    } else {
        failed_imitations_++;
    }

    return results;
}

auto SocialLearningEngine::adapt_strategy(
    const StrategyPattern& pattern,
    const std::vector<std::string>& my_capabilities)
    -> StrategyPattern
{
    StrategyPattern adapted = pattern;
    adapted.id = next_id_();
    adapted.name = pattern.name + "_adapted";
    adapted.source_agent = "self";

    // 过滤掉超出自身能力的步骤
    std::vector<std::string> feasible_steps;
    for (const auto& step : pattern.steps) {
        bool can_do = false;
        for (const auto& cap : my_capabilities) {
            if (step.find(cap) != std::string::npos) {
                can_do = true;
                break;
            }
        }
        if (can_do || my_capabilities.empty()) {
            feasible_steps.push_back(step);
        } else {
            feasible_steps.push_back(step + " [需学习]");
        }
    }
    adapted.steps = feasible_steps;

    return adapted;
}

// ── 社会反馈 ──────────────────────────────────────────────────

void SocialLearningEngine::receive_feedback(const SocialFeedback& feedback)
{
    feedback_history_.push_back(feedback);

    // 限制历史大小
    while (static_cast<int>(feedback_history_.size()) > 100) {
        feedback_history_.pop_front();
    }

    // 更新反馈者可信度
    auto& model = role_models_[feedback.from_agent];
    model.agent_id = feedback.from_agent;
    if (feedback.feedback_type == "correction") {
        model.trustworthiness = std::min(model.trustworthiness + 0.05, 1.0);
    }
}

auto SocialLearningEngine::adjust_from_feedback(const std::string& domain)
    -> std::vector<std::string>
{
    std::vector<std::string> adjustments;

    for (const auto& fb : feedback_history_) {
        if (fb.domain == domain || domain.empty()) {
            if (fb.feedback_type == "correction") {
                adjustments.push_back("修正: " + fb.content);
            } else if (fb.feedback_type == "suggestion") {
                adjustments.push_back("建议: " + fb.content);
            }
        }
    }

    return adjustments;
}

// ── 知识传播 ──────────────────────────────────────────────────

auto SocialLearningEngine::share_knowledge(const std::string& domain) const
    -> std::vector<std::string>
{
    std::vector<std::string> knowledge;

    for (const auto& [id, strategy] : strategies_) {
        if (strategy.domain == domain || domain.empty()) {
            std::ostringstream oss;
            oss << "策略 [" << strategy.name << "]: ";
            for (const auto& step : strategy.steps) {
                oss << step << "; ";
            }
            knowledge.push_back(oss.str());
        }
    }

    return knowledge;
}

// ── 查询 ──────────────────────────────────────────────────────

auto SocialLearningEngine::stats() const -> std::map<std::string, double>
{
    return {
        {"total_observations", static_cast<double>(total_observations_)},
        {"successful_imitations", static_cast<double>(successful_imitations_)},
        {"failed_imitations", static_cast<double>(failed_imitations_)},
        {"learned_strategies", static_cast<double>(strategies_.size())},
        {"role_models_count", static_cast<double>(role_models_.size())},
        {"feedback_received", static_cast<double>(feedback_history_.size())},
    };
}

// ── 内部方法 ──────────────────────────────────────────────────

auto SocialLearningEngine::extract_common_steps_(
    const std::vector<BehaviorObservation>& obs) const
    -> std::vector<std::string>
{
    if (obs.empty()) return {};

    // 收集所有动作并找共同模式
    std::map<std::string, int> action_counts;
    for (const auto& o : obs) {
        action_counts[o.action]++;
    }

    // 保留出现频率 >= 50% 的动作作为共同步骤
    std::vector<std::string> common;
    double threshold = obs.size() * 0.5;
    for (const auto& [action, count] : action_counts) {
        if (count >= threshold) {
            common.push_back(action);
        }
    }

    return common;
}

auto SocialLearningEngine::model_credibility_(const std::string& agent_id) const
    -> double
{
    auto it = role_models_.find(agent_id);
    if (it == role_models_.end()) return 0.0;

    return it->second.expertise * 0.6 + it->second.trustworthiness * 0.4;
}

auto SocialLearningEngine::next_id_() -> std::string
{
    return "strategy_" + std::to_string(next_strategy_id_++);
}

}  // namespace ai_learning::learning
