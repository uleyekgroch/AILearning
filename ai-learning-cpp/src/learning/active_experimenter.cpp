/**
 * @file active_experimenter.cpp
 * @brief 主动实验设计引擎实现
 */

#include "ai_learning/learning/active_experimenter.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <set>

namespace ai_learning::learning {

ActiveExperimenter::ActiveExperimenter(const ExperimenterConfig& config)
    : config_(config) {}

// ── 假设管理 ──────────────────────────────────────────────────

auto ActiveExperimenter::generate_hypothesis(
    const std::string& observation,
    const std::string& domain)
    -> Hypothesis
{
    Hypothesis hyp;
    hyp.id = next_hyp_id_();
    hyp.domain = domain;

    // 从观察中归纳假设
    // 简化实现：将观察转化为可测试的陈述
    hyp.statement = "基于观察[" + observation + "]的假设";
    hyp.prior_confidence = 0.5;
    hyp.posterior_confidence = 0.5;
    hyp.supporting_evidence.push_back(observation);
    hyp.information_value = information_value_(hyp);

    hypotheses_[hyp.id] = hyp;

    // 更新探索状态
    exploration_states_[domain].domain = domain;
    exploration_states_[domain].uncertainty = 1.0;

    return hyp;
}

void ActiveExperimenter::register_hypothesis(const Hypothesis& hypothesis)
{
    hypotheses_[hypothesis.id] = hypothesis;

    exploration_states_[hypothesis.domain].domain = hypothesis.domain;
}

auto ActiveExperimenter::hypotheses_in_domain(const std::string& domain) const
    -> std::vector<Hypothesis>
{
    std::vector<Hypothesis> result;
    for (const auto& [id, hyp] : hypotheses_) {
        if (hyp.domain == domain) {
            result.push_back(hyp);
        }
    }
    return result;
}

// ── 实验设计 ──────────────────────────────────────────────────

auto ActiveExperimenter::design_experiment(const std::string& hypothesis_id)
    -> std::optional<ExperimentDesign>
{
    auto it = hypotheses_.find(hypothesis_id);
    if (it == hypotheses_.end()) return std::nullopt;

    const auto& hyp = it->second;
    if (hyp.falsified) return std::nullopt;

    ExperimentDesign design;
    design.id = next_exp_id_();
    design.target_hypothesis = hypothesis_id;
    design.expected_information_gain = expected_information_gain(hypothesis_id);
    design.cost = 1.0 - hyp.posterior_confidence * config_.cost_sensitivity;

    // 设计实验描述和步骤
    std::ostringstream desc;
    desc << "测试假设: " << hyp.statement;
    design.description = desc.str();

    // 实验方法选择：基于假设的置信度
    if (hyp.posterior_confidence > 0.6) {
        design.method = "验证性实验";
        design.steps = {
            "设置验证条件",
            "执行关键测试",
            "检查是否一致",
            "记录结果"
        };
        design.expected_positive = "结果与假设一致，增强置信度";
        design.expected_negative = "结果与假设矛盾，降低置信度";
    } else {
        design.method = "探索性实验";
        design.steps = {
            "收集更多相关数据",
            "测试边界条件",
            "比较正反例",
            "分析模式"
        };
        design.expected_positive = "发现支持假设的证据";
        design.expected_negative = "发现反对假设的证据";
    }

    return design;
}

auto ActiveExperimenter::auto_design_experiment()
    -> std::optional<ExperimentDesign>
{
    // 选择信息价值最高的假设
    std::string best_hyp;
    double best_value = -1.0;

    for (const auto& [id, hyp] : hypotheses_) {
        if (hyp.falsified) continue;
        double value = information_value_(hyp);
        // 加上探索奖励
        value += config_.exploration_bonus *
                 (1.0 - hyp.posterior_confidence);

        if (value > best_value) {
            best_value = value;
            best_hyp = id;
        }
    }

    if (best_hyp.empty()) return std::nullopt;

    return design_experiment(best_hyp);
}

auto ActiveExperimenter::expected_information_gain(
    const std::string& hypothesis_id) const -> double
{
    auto it = hypotheses_.find(hypothesis_id);
    if (it == hypotheses_.end()) return 0.0;

    const auto& hyp = it->second;

    // 信息增益 = 熵的减少
    // 二值熵: H(p) = -p*log(p) - (1-p)*log(1-p)
    double p = hyp.posterior_confidence;
    double entropy = 0.0;
    if (p > 0.001 && p < 0.999) {
        entropy = -p * std::log2(p) - (1 - p) * std::log2(1 - p);
    }

    // 预期实验后的熵（假设完美二分结果）
    double p_after_pos = std::min(p + 0.2, 0.999);
    double p_after_neg = std::max(p - 0.2, 0.001);
    double entropy_pos = 0.0;
    if (p_after_pos > 0.001 && p_after_pos < 0.999) {
        entropy_pos = -p_after_pos * std::log2(p_after_pos) -
                      (1 - p_after_pos) * std::log2(1 - p_after_pos);
    }
    double entropy_neg = 0.0;
    if (p_after_neg > 0.001 && p_after_neg < 0.999) {
        entropy_neg = -p_after_neg * std::log2(p_after_neg) -
                      (1 - p_after_neg) * std::log2(1 - p_after_neg);
    }

    double expected_entropy = p * entropy_pos + (1 - p) * entropy_neg;
    double info_gain = std::max(0.0, entropy - expected_entropy);

    return info_gain;
}

// ── 实验执行 ──────────────────────────────────────────────────

auto ActiveExperimenter::record_result(const ExperimentResult& result)
    -> std::string
{
    results_.push_back(result);

    // 更新假设
    auto it = hypotheses_.find(result.hypothesis_id);
    if (it == hypotheses_.end()) {
        return "未找到对应假设";
    }

    auto& hyp = it->second;
    hyp.tested = true;

    // 贝叶斯更新
    double delta = bayesian_update_(hyp, result.supports_hypothesis);

    // 记录证据
    if (result.supports_hypothesis) {
        hyp.supporting_evidence.push_back(result.observation);
    } else {
        hyp.contradicting_evidence.push_back(result.observation);
    }

    // 检查是否证伪
    if (hyp.posterior_confidence < config_.falsification_threshold) {
        hyp.falsified = true;
    }

    // 更新探索状态
    auto& state = exploration_states_[hyp.domain];
    state.experiments_done++;
    state.total_information_gained += result.information_gain;
    state.uncertainty = domain_uncertainty(hyp.domain);

    // 检查是否可以构建理论
    std::ostringstream analysis;
    analysis << "假设 [" << hyp.statement << "] 置信度变化: "
             << delta << " → 当前 " << hyp.posterior_confidence;
    if (hyp.falsified) {
        analysis << " (已证伪)";
    }

    return analysis.str();
}

auto ActiveExperimenter::analyze_impact(const ExperimentResult& result) const
    -> std::string
{
    auto it = hypotheses_.find(result.hypothesis_id);
    if (it == hypotheses_.end()) return "未知假设";

    const auto& hyp = it->second;
    std::ostringstream oss;

    oss << "实验结果对假设 [" << hyp.statement << "] 的影响:\n";

    if (result.supports_hypothesis) {
        oss << "  支持: +" << result.confidence_delta << " 置信度\n";
    } else {
        oss << "  反对: " << result.confidence_delta << " 置信度\n";
    }

    oss << "  支持证据: " << hyp.supporting_evidence.size() << " 条\n";
    oss << "  反对证据: " << hyp.contradicting_evidence.size() << " 条\n";
    oss << "  信息增益: " << result.information_gain << "\n";
    oss << "  惊讶度: " << result.surprise;

    return oss.str();
}

// ── 理论构建 ──────────────────────────────────────────────────

auto ActiveExperimenter::build_theory(const std::string& domain)
    -> std::optional<Theory>
{
    // 收集该领域已验证的假设
    std::vector<std::string> verified;
    for (const auto& [id, hyp] : hypotheses_) {
        if (hyp.domain == domain &&
            hyp.posterior_confidence >= config_.hypothesis_threshold &&
            !hyp.falsified && hyp.tested) {
            verified.push_back(id);
        }
    }

    if (verified.size() < 2) return std::nullopt;

    // 构建理论
    Theory theory;
    theory.id = next_theory_id_();
    theory.domain = domain;
    theory.verified_hypotheses = verified;

    // 理论名称：由假设数量和领域组成
    theory.name = domain + "_理论_" + std::to_string(verified.size()) + "假设";

    // 计算综合置信度
    double total_conf = 0.0;
    int total_exp = 0;
    int supporting = 0;
    for (const auto& hid : verified) {
        auto it = hypotheses_.find(hid);
        if (it != hypotheses_.end()) {
            total_conf += it->second.posterior_confidence;
            total_exp += static_cast<int>(
                it->second.supporting_evidence.size() +
                it->second.contradicting_evidence.size());
            supporting += static_cast<int>(it->second.supporting_evidence.size());
        }
    }
    theory.confidence = total_conf / verified.size();
    theory.total_experiments = total_exp;
    theory.supporting_experiments = supporting;

    // 生成理论描述
    std::ostringstream desc;
    desc << "在 " << domain << " 领域，基于 " << verified.size()
         << " 个已验证假设构建理论，综合置信度 " << theory.confidence;
    theory.description = desc.str();

    theories_[theory.id] = theory;
    return theory;
}

// ── 探索策略 ──────────────────────────────────────────────────

auto ActiveExperimenter::exploration_state(const std::string& domain) const
    -> ExplorationState
{
    auto it = exploration_states_.find(domain);
    if (it != exploration_states_.end()) return it->second;

    return ExplorationState{domain, 1.0, 0.0, 0, 0.0};
}

auto ActiveExperimenter::recommend_exploration() const
    -> std::optional<std::string>
{
    // 推荐不确定性最高的领域
    std::string best_domain;
    double max_uncertainty = 0.0;

    for (const auto& [domain, state] : exploration_states_) {
        // 综合评分 = 不确定性 × (1 + 探索奖励)
        double score = state.uncertainty *
            (1.0 + config_.exploration_bonus / (1.0 + state.experiments_done));

        if (score > max_uncertainty) {
            max_uncertainty = score;
            best_domain = domain;
        }
    }

    if (!best_domain.empty()) return best_domain;
    return std::nullopt;
}

auto ActiveExperimenter::domain_uncertainty(const std::string& domain) const
    -> double
{
    auto hyps = hypotheses_in_domain(domain);
    if (hyps.empty()) return 1.0;

    // 平均假设不确定性
    double total_entropy = 0.0;
    int count = 0;
    for (const auto& hyp : hyps) {
        if (hyp.falsified) continue;
        double p = hyp.posterior_confidence;
        if (p > 0.001 && p < 0.999) {
            total_entropy += -p * std::log2(p) - (1 - p) * std::log2(1 - p);
        }
        count++;
    }

    return count > 0 ? total_entropy / count : 0.0;
}

// ── 查询 ──────────────────────────────────────────────────────

auto ActiveExperimenter::stats() const -> std::map<std::string, double>
{
    int active = 0, falsified = 0, verified = 0;
    for (const auto& [_, hyp] : hypotheses_) {
        if (hyp.falsified) falsified++;
        else if (hyp.posterior_confidence >= config_.hypothesis_threshold) verified++;
        else active++;
    }

    double total_info = 0.0;
    for (const auto& r : results_) total_info += r.information_gain;

    return {
        {"total_hypotheses", static_cast<double>(hypotheses_.size())},
        {"active_hypotheses", static_cast<double>(active)},
        {"verified_hypotheses", static_cast<double>(verified)},
        {"falsified_hypotheses", static_cast<double>(falsified)},
        {"total_experiments", static_cast<double>(results_.size())},
        {"total_information_gained", total_info},
        {"theories_built", static_cast<double>(theories_.size())},
    };
}

// ── 内部方法 ──────────────────────────────────────────────────

auto ActiveExperimenter::bayesian_update_(
    Hypothesis& hyp,
    bool evidence_supports) const -> double
{
    double prior = hyp.posterior_confidence;

    // 简化的贝叶斯更新
    // likelihood: P(evidence|hypothesis_true) vs P(evidence|hypothesis_false)
    double likelihood_ratio = evidence_supports ? 2.0 : 0.5;

    // P(H|E) = P(E|H) * P(H) / P(E)
    // 简化为: posterior = prior * likelihood / (prior * likelihood + (1-prior) * (1-likelihood_norm))
    double numerator = prior * likelihood_ratio;
    double denominator = numerator + (1 - prior) / likelihood_ratio;

    if (denominator > 0) {
        hyp.posterior_confidence = std::clamp(numerator / denominator, 0.01, 0.99);
    }

    return hyp.posterior_confidence - prior;
}

auto ActiveExperimenter::information_value_(const Hypothesis& hyp) const
    -> double
{
    // 信息价值 = 测试该假设能获得多少信息
    // 二值熵在 p=0.5 时最大
    double p = hyp.posterior_confidence;
    if (p <= 0.001 || p >= 0.999) return 0.0;

    double entropy = -p * std::log2(p) - (1 - p) * std::log2(1 - p);

    // 加上不确定性奖励
    double novelty = 1.0 - std::abs(p - 0.5) * 2;

    return entropy * 0.7 + novelty * 0.3;
}

auto ActiveExperimenter::next_hyp_id_() -> std::string
{
    return "hyp_" + std::to_string(hypothesis_id_counter_++);
}

auto ActiveExperimenter::next_exp_id_() -> std::string
{
    return "exp_" + std::to_string(experiment_id_counter_++);
}

auto ActiveExperimenter::next_theory_id_() -> std::string
{
    return "theory_" + std::to_string(theory_id_counter_++);
}

}  // namespace ai_learning::learning
