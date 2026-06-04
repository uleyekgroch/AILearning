/**
 * @file active_inference.hpp
 * @brief 主动推理引擎 — 自由能原理驱动的感知-行动闭环
 *
 * 理论基础：
 *   - Friston (2025) Active Inference: The Free Energy Principle in Mind, Brain, and Behavior
 *   - Parr, Pezzulo & Friston (2022) Active Inference: The Free Energy Principle
 *   - Spisak & Friston (2026) Self-orthogonalizing attractor networks from FEP
 *
 * 核心公式：
 *   变分自由能 F = E_q[ln q(s) - ln p(o,s)]
 *   预期自由能 G = E_q[ln q(s') - ln p(o',s')]
 *
 * 主动推理闭环：
 *   1. 感知 (Perception):  更新内部信念 q(s) → 最小化变分自由能
 *   2. 规划 (Planning):    评估可能策略的预期自由能 G(π)
 *   3. 行动 (Action):      选择最小化 G 的策略
 *   4. 学习 (Learning):    更新生成模型 p(o,s)
 *
 * 与现有模块的关系：
 *   - WorldModel: 提供生成模型 p(o,s) (因果DAG)
 *   - PredictiveCodingEngine: 提供预测误差 (变分自由能)
 *   - IntrinsicMotivationEngine: 提供 epistemic value (好奇心)
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ai_learning::reasoning {

/// 策略（行动序列）
struct Policy {
    std::string name;                         ///< 策略名称
    std::vector<std::string> actions;          ///< 行动序列
    double expected_free_energy = 0.0;         ///< 预期自由能 G(π)
    double pragmatic_value = 0.0;              ///< 实用价值（达成目标）
    double epistemic_value = 0.0;              ///< 认知价值（减少不确定性）
    double prior_preference = 0.0;             ///< 先验偏好
};

/// 信念状态
struct BeliefState {
    std::string state_id;                      ///< 状态标识
    double probability = 0.0;                  ///< 后验概率 q(s)
    double prior = 0.0;                        ///< 先验概率 p(s)
    double prediction_error = 0.0;             ///< 预测误差
    std::map<std::string, double> features;    ///< 状态特征
};

/// 主动推理配置
struct ActiveInferenceConfig {
    double precision = 1.0;                   ///< 精度 (逆温度)
    double action_selection_temperature = 0.5; ///< Softmax 温度
    double epistemic_weight = 0.5;            ///< 认知价值权重
    double pragmatic_weight = 0.5;            ///< 实用价值权重
    int planning_horizon = 3;                 ///< 规划视界
    double learning_rate = 0.1;               ///< 信念更新率
    bool use_expected_free_energy = true;     ///< 是否使用 EFE
};

/// 主动推理步骤结果
struct AIStepResult {
    BeliefState prior_state;                   ///< 先前的信念
    BeliefState posterior_state;               ///< 更新后的信念
    Policy selected_policy;                    ///< 选择的策略
    double variational_free_energy = 0.0;      ///< 变分自由能
    double expected_free_energy = 0.0;         ///< 预期自由能
    std::string selected_action;               ///< 选择的行动
    bool state_changed = false;                ///< 信念是否显著变化
};

/// 主动推理引擎 — 实现自由能最小化闭环
class ActiveInferenceEngine {
public:
    explicit ActiveInferenceEngine(
        const ActiveInferenceConfig& config = ActiveInferenceConfig{});

    // ═══════════════════════════════════════════════════════════
    // 1. 感知 (Perception): 贝叶斯信念更新
    // ═══════════════════════════════════════════════════════════

    /// 基于观测更新信念状态
    /// @param observation 当前观测
    /// @param predicted   生成模型的预测
    /// @return 后验信念状态
    auto perceive(const std::vector<float>& observation,
                  const std::vector<float>& predicted)
        -> BeliefState;

    /// 计算变分自由能
    /// @param prior 先验信念
    /// @param posterior 后验信念
    /// @param observation 观测
    [[nodiscard]] auto variational_free_energy(
        const BeliefState& prior,
        const BeliefState& posterior,
        const std::vector<float>& observation) const -> double;

    // ═══════════════════════════════════════════════════════════
    // 2. 规划 (Planning): 评估策略的预期自由能
    // ═══════════════════════════════════════════════════════════

    /// 评估策略的预期自由能
    /// @param policy 候选策略
    /// @param current_belief 当前信念状态
    /// @param goal 目标描述
    auto evaluate_policy(const Policy& policy,
                         const BeliefState& current_belief,
                         const std::string& goal = "")
        -> double;

    /// 计算预期自由能 G(π)
    /// G(π) = Epistemic Value + Pragmatic Value + Complexity
    auto expected_free_energy(const Policy& policy,
                              const BeliefState& belief) const -> double;

    /// 计算认知价值（信息增益）
    auto epistemic_value(const Policy& policy,
                         const BeliefState& belief) const -> double;

    /// 计算实用价值（目标达成）
    auto pragmatic_value(const Policy& policy,
                         const std::string& goal) const -> double;

    // ═══════════════════════════════════════════════════════════
    // 3. 行动选择 (Action Selection): Softmax 策略选择
    // ═══════════════════════════════════════════════════════════

    /// 从候选策略中选择最优策略
    /// @param policies 候选策略列表
    /// @param belief 当前信念
    /// @param goal 目标
    auto select_policy(const std::vector<Policy>& policies,
                       const BeliefState& belief,
                       const std::string& goal = "")
        -> Policy;

    /// 生成候选策略（基于世界模型）
    auto generate_policies(const BeliefState& belief,
                           int horizon) const
        -> std::vector<Policy>;

    // ═══════════════════════════════════════════════════════════
    // 4. 学习 (Learning): 更新生成模型
    // ═══════════════════════════════════════════════════════════

    /// 从经验中更新先验
    void update_prior(const std::string& state_id,
                      double evidence,
                      bool positive);

    /// 执行完整的一步主动推理
    auto step(const std::vector<float>& observation,
              const std::vector<float>& prediction,
              const std::vector<Policy>& candidates,
              const std::string& goal = "")
        -> AIStepResult;

    // ═══════════════════════════════════════════════════════════
    // 查询
    // ═══════════════════════════════════════════════════════════

    [[nodiscard]] auto current_belief() const -> const BeliefState& {
        return current_belief_;
    }

    [[nodiscard]] auto config() const -> const ActiveInferenceConfig& {
        return config_;
    }

    [[nodiscard]] auto total_steps() const -> int { return total_steps_; }

    /// 获取信念历史
    [[nodiscard]] auto belief_history() const
        -> const std::vector<BeliefState>& { return belief_history_; }

private:
    ActiveInferenceConfig config_;
    BeliefState current_belief_;
    std::vector<BeliefState> belief_history_;
    int total_steps_ = 0;

    /// 先验注册表 (生成模型的一部分)
    std::map<std::string, double> prior_registry_;

    /// Softmax 归一化
    static auto softmax_(std::vector<double>& scores, double temperature)
        -> std::vector<double>;

    /// KL 散度 D_KL(q||p)
    static auto kl_divergence_(const BeliefState& q,
                               const BeliefState& p) -> double;

    /// 信息增益 I = H(posterior) - H(prior)
    static auto information_gain_(const BeliefState& prior,
                                  const BeliefState& posterior) -> double;
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline ActiveInferenceEngine::ActiveInferenceEngine(
    const ActiveInferenceConfig& config)
    : config_(config) {
    current_belief_.state_id = "initial";
    current_belief_.probability = 1.0;
    current_belief_.prior = 0.5;
}

inline auto ActiveInferenceEngine::perceive(
    const std::vector<float>& observation,
    const std::vector<float>& predicted)
    -> BeliefState {
    // 贝叶斯信念更新: posterior ∝ likelihood × prior
    BeliefState posterior;
    posterior.state_id = "state_" + std::to_string(total_steps_);
    posterior.prior = current_belief_.probability;

    // 计算预测误差 = ||observation - predicted||²
    double error = 0.0;
    size_t n = std::min(observation.size(), predicted.size());
    for (size_t i = 0; i < n; ++i) {
        double diff = observation[i] - predicted[i];
        error += diff * diff;
    }
    error = n > 0 ? error / n : 0.0;
    posterior.prediction_error = error;

    // 精度加权的贝叶斯更新
    double likelihood = std::exp(-0.5 * config_.precision * error);
    double evidence = likelihood * current_belief_.prior;
    double normalization = evidence + (1.0 - likelihood) * (1.0 - current_belief_.prior);
    posterior.probability = normalization > 1e-8
        ? std::clamp(evidence / normalization, 0.001, 0.999)
        : current_belief_.prior;

    belief_history_.push_back(current_belief_);
    current_belief_ = posterior;
    total_steps_++;

    return posterior;
}

inline auto ActiveInferenceEngine::variational_free_energy(
    const BeliefState& prior,
    const BeliefState& posterior,
    const std::vector<float>& /*observation*/) const -> double {
    // F ≈ D_KL(q(s)||p(s)) - ln p(o|s)
    double kl = kl_divergence_(posterior, prior);
    double accuracy = posterior.prediction_error;
    return kl + config_.precision * accuracy;
}

inline auto ActiveInferenceEngine::expected_free_energy(
    const Policy& policy,
    const BeliefState& belief) const -> double {
    // G(π) = E_q[ln q(s') - ln p(o',s')]
    //       ≈ Epistemic Value + Pragmatic Value + Complexity
    double epistemic = epistemic_value(policy, belief);
    double complexity = policy.actions.size() * 0.01;  // 复杂度惩罚
    return -(config_.epistemic_weight * epistemic)
           + complexity;  // 越小越好
}

inline auto ActiveInferenceEngine::epistemic_value(
    const Policy& policy,
    const BeliefState& belief) const -> double {
    // 认知价值 = 预期信息增益
    // 越不确定 → 探索价值越高
    double uncertainty = 1.0 - belief.probability * (1.0 - belief.probability) * 4.0;
    double novelty = policy.actions.size() > 0 ? 1.0 / (1.0 + policy.actions.size()) : 0.5;
    return uncertainty * novelty * config_.epistemic_weight;
}

inline auto ActiveInferenceEngine::pragmatic_value(
    const Policy& policy,
    const std::string& goal) const -> double {
    // 实用价值 = 策略达成目标的可能性
    // 简化：策略中包含与 goal 相关的行动 → 高价值
    if (goal.empty()) return 0.5;
    for (const auto& action : policy.actions) {
        if (action.find(goal) != std::string::npos ||
            goal.find(action) != std::string::npos) {
            return 1.0;
        }
    }
    return 0.1;
}

inline auto ActiveInferenceEngine::evaluate_policy(
    const Policy& policy,
    const BeliefState& current_belief,
    const std::string& goal)
    -> double {
    double efe = expected_free_energy(policy, current_belief);
    double pragmatic = pragmatic_value(policy, goal);
    return -(efe + config_.pragmatic_weight * pragmatic);
}

inline auto ActiveInferenceEngine::select_policy(
    const std::vector<Policy>& policies,
    const BeliefState& belief,
    const std::string& goal)
    -> Policy {
    if (policies.empty()) return Policy{};

    std::vector<double> scores;
    for (const auto& p : policies) {
        scores.push_back(evaluate_policy(p, belief, goal));
    }

    auto probs = softmax_(scores, config_.action_selection_temperature);
    size_t best_idx = std::distance(
        probs.begin(), std::max_element(probs.begin(), probs.end()));

    Policy selected = policies[best_idx];
    selected.expected_free_energy = expected_free_energy(selected, belief);
    selected.pragmatic_value = pragmatic_value(selected, goal);
    selected.epistemic_value = epistemic_value(selected, belief);
    return selected;
}

inline auto ActiveInferenceEngine::generate_policies(
    const BeliefState& /*belief*/, int /*horizon*/) const
    -> std::vector<Policy> {
    std::vector<Policy> policies;

    // 探索策略
    Policy explore;
    explore.name = "explore_unknown";
    explore.actions = {"query", "search", "experiment"};
    policies.push_back(explore);

    // 利用策略
    Policy exploit;
    exploit.name = "exploit_known";
    exploit.actions = {"apply", "refine"};
    policies.push_back(exploit);

    // 观察策略
    Policy observe;
    observe.name = "passive_observe";
    observe.actions = {"observe", "wait"};
    policies.push_back(observe);

    return policies;
}

inline void ActiveInferenceEngine::update_prior(
    const std::string& state_id, double evidence, bool positive) {
    double& prior = prior_registry_[state_id];
    prior = config_.learning_rate * (positive ? evidence : -evidence)
            + (1.0 - config_.learning_rate) * prior;
    prior = std::clamp(prior, 0.0, 1.0);
}

inline auto ActiveInferenceEngine::step(
    const std::vector<float>& observation,
    const std::vector<float>& prediction,
    const std::vector<Policy>& candidates,
    const std::string& goal)
    -> AIStepResult {
    AIStepResult result;
    result.prior_state = current_belief_;

    // 1. 感知
    result.posterior_state = perceive(observation, prediction);

    // 2. 规划 + 选择
    auto policies = candidates.empty()
        ? generate_policies(current_belief_, config_.planning_horizon)
        : candidates;
    result.selected_policy = select_policy(
        policies, current_belief_, goal);

    // 3. 计算自由能
    result.variational_free_energy = variational_free_energy(
        result.prior_state, result.posterior_state, observation);
    result.expected_free_energy = result.selected_policy.expected_free_energy;

    // 4. 选择行动
    result.selected_action = result.selected_policy.actions.empty()
        ? "wait" : result.selected_policy.actions[0];
    result.state_changed = std::abs(
        result.posterior_state.probability - result.prior_state.probability)
        > 0.1;

    return result;
}

inline auto ActiveInferenceEngine::softmax_(
    std::vector<double>& scores, double temperature) -> std::vector<double> {
    if (temperature <= 0.0) temperature = 0.1;
    std::vector<double> probs(scores.size());
    double max_s = *std::max_element(scores.begin(), scores.end());
    double sum = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        probs[i] = std::exp((scores[i] - max_s) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= (sum + 1e-8);
    return probs;
}

inline auto ActiveInferenceEngine::kl_divergence_(
    const BeliefState& q, const BeliefState& p) -> double {
    double qp = std::clamp(q.probability, 1e-8, 1.0 - 1e-8);
    double pp = std::clamp(p.probability, 1e-8, 1.0 - 1e-8);
    return qp * std::log(qp / pp)
           + (1.0 - qp) * std::log((1.0 - qp) / (1.0 - pp));
}

inline auto ActiveInferenceEngine::information_gain_(
    const BeliefState& prior, const BeliefState& posterior) -> double {
    auto entropy = [](double p) {
        if (p <= 0.0 || p >= 1.0) return 0.0;
        return -p * std::log(p) - (1.0 - p) * std::log(1.0 - p);
    };
    return entropy(prior.probability) - entropy(posterior.probability);
}

}  // namespace ai_learning::reasoning