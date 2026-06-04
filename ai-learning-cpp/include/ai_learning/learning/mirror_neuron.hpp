/**
 * @file mirror_neuron.hpp
 * @brief 镜像神经元系统 — 观察即理解，理解即模仿
 *
 * 理论基础：
 *   - Rizzolatti & Craighero (2004) The mirror-neuron system
 *   - Iacoboni (2009) Imitation, empathy, and mirror neurons
 *   - Ramachandran (2011) The Tell-Tale Brain: mirror neurons and civilization
 *
 * 核心机制：
 *   1. 动作观察 (Action Observation): 感知他人行为 → 编码为动作表征
 *   2. 动作理解 (Action Understanding): 映射到自身运动程序
 *   3. 模仿学习 (Imitation Learning): 从观察中学习新技能
 *   4. 意图推断 (Intention Inference): 从动作推断目标
 *   5. 共情映射 (Empathy Mapping): 理解他人情感状态
 *
 * 与现有模块的关系：
 *   - SocialLearning: 高层次社会学习（观察→学习→内化）
 *   - EmotionEngine: 情感状态共享
 *   - SkillTree: 新技能的存储
 *   - KnowledgeGraph: 存储观察到的行为模式
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ai_learning::learning {

/// 动作表征
struct MotorRepresentation {
    std::string action_name;                  ///< 动作名称
    std::string effector;                     ///< 效应器 (hand, eye, voice, ...)
    std::vector<std::string> sub_actions;     ///< 子动作序列
    double activation = 0.0;                  ///< 激活水平 0~1
    int observation_count = 0;               ///< 被观察到的次数
    bool mastered = false;                    ///< 是否已掌握
};

/// 观察到的动作
struct ObservedAction {
    std::string observer_id;                  ///< 观察者
    std::string actor_id;                     ///< 执行者
    std::string action_name;                  ///< 动作名称
    std::string context;                      ///< 发生语境
    std::string inferred_goal;                ///< 推断的目标
    double confidence = 0.0;                  ///< 观察置信度
    int timestamp = 0;                        ///< 时间戳
};

/// 模仿学习结果
struct ImitationResult {
    std::string action_name;                  ///< 模仿的动作
    double similarity = 0.0;                  ///< 相似度 0~1
    bool success = false;                     ///< 模仿是否成功
    int attempts = 0;                         ///< 尝试次数
    std::vector<std::string> errors;          ///< 错误记录
    std::string learned_skill;                ///< 学到的技能
};

/// 意图推断结果
struct IntentionInference {
    std::string action_name;                  ///< 观察到的动作
    std::string inferred_intention;           ///< 推断的意图
    double confidence = 0.0;                  ///< 置信度
    std::vector<std::string> supporting_evidence; ///< 支持证据
    std::string predicted_next_action;        ///< 预测的下一个动作
};

/// 镜像神经元系统配置
struct MirrorNeuronConfig {
    double activation_threshold = 0.3;        ///< 激活阈值
    double imitation_learning_rate = 0.1;     ///< 模仿学习率
    double intention_inference_depth = 3;     ///< 意图推断深度
    int max_observed_actions = 1000;          ///< 最大观察记录
    bool enable_empathy = true;               ///< 启用共情
};

/// 镜像神经元系统 — 观察→理解→模仿
class MirrorNeuronSystem {
public:
    explicit MirrorNeuronSystem(
        const MirrorNeuronConfig& config = MirrorNeuronConfig{});

    // ═══════════════════════════════════════════════════════════
    // 1. 动作观察 (Action Observation)
    // ═══════════════════════════════════════════════════════════

    /// 观察一个动作
    /// @return 激活的运动表征
    auto observe_action(const ObservedAction& action)
        -> MotorRepresentation;

    /// 批量观察（场景理解）
    auto observe_sequence(
        const std::vector<ObservedAction>& sequence)
        -> std::vector<MotorRepresentation>;

    // ═══════════════════════════════════════════════════════════
    // 2. 动作理解 (Action Understanding)
    // ═══════════════════════════════════════════════════════════

    /// 理解动作：映射到内部运动程序
    auto understand(const std::string& action_name)
        -> std::optional<MotorRepresentation>;

    /// 检查是否"理解"某个动作（能映射到自身运动程序）
    [[nodiscard]] auto is_understood(const std::string& action_name) const
        -> bool;

    // ═══════════════════════════════════════════════════════════
    // 3. 模仿学习 (Imitation Learning)
    // ═══════════════════════════════════════════════════════════

    /// 尝试模仿一个动作
    /// @param action_name 要模仿的动作
    /// @param feedback 模仿反馈（可选：成功/失败/修正）
    auto imitate(const std::string& action_name,
                 const std::string& feedback = "")
        -> ImitationResult;

    /// 从多次观察中学习
    auto learn_from_observations(const std::string& action_name)
        -> ImitationResult;

    // ═══════════════════════════════════════════════════════════
    // 4. 意图推断 (Intention Inference)
    // ═══════════════════════════════════════════════════════════

    /// 从动作推断意图
    auto infer_intention(const ObservedAction& action)
        -> IntentionInference;

    /// 从动作序列推断更高层意图
    auto infer_intention_from_sequence(
        const std::vector<ObservedAction>& sequence)
        -> IntentionInference;

    // ═══════════════════════════════════════════════════════════
    // 5. 共情映射 (Empathy Mapping)
    // ═══════════════════════════════════════════════════════════

    /// 通过观察推断他人情感状态
    auto infer_emotional_state(const ObservedAction& action,
                               const std::string& context)
        -> std::map<std::string, double>;  ///< 情感 → 强度

    // ═══════════════════════════════════════════════════════════
    // 查询
    // ═══════════════════════════════════════════════════════════

    [[nodiscard]] auto motor_repertoire() const
        -> const std::map<std::string, MotorRepresentation>& {
        return motor_repertoire_;
    }

    [[nodiscard]] auto mastered_actions() const
        -> std::vector<std::string>;

    [[nodiscard]] auto total_observations() const -> int {
        return observation_count_;
    }

    [[nodiscard]] auto imitation_success_rate() const -> double;

private:
    MirrorNeuronConfig config_;

    /// 运动程序库（"我能做的动作"）
    std::map<std::string, MotorRepresentation> motor_repertoire_;

    /// 观察历史
    std::vector<ObservedAction> observation_history_;
    int observation_count_ = 0;

    /// 模仿统计
    int imitation_attempts_ = 0;
    int imitation_successes_ = 0;

    /// 激活运动表征（镜像响应）
    auto activate_motor_(const std::string& action_name,
                         double strength) -> MotorRepresentation;

    /// 分解动作
    auto decompose_action_(const std::string& action_name) const
        -> std::vector<std::string>;

    /// 评估模仿质量
    auto evaluate_imitation_(const MotorRepresentation& target,
                             const MotorRepresentation& actual) const
        -> double;

    /// 基于语境的意图推断
    auto contextual_intention_(const std::string& action,
                               const std::string& context) const
        -> std::string;
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline MirrorNeuronSystem::MirrorNeuronSystem(
    const MirrorNeuronConfig& config) : config_(config) {}

inline auto MirrorNeuronSystem::observe_action(
    const ObservedAction& action)
    -> MotorRepresentation {
    observation_history_.push_back(action);
    observation_count_++;

    // 激活或创建运动表征
    return activate_motor_(action.action_name, action.confidence);
}

inline auto MirrorNeuronSystem::observe_sequence(
    const std::vector<ObservedAction>& sequence)
    -> std::vector<MotorRepresentation> {
    std::vector<MotorRepresentation> activations;
    for (const auto& act : sequence) {
        activations.push_back(observe_action(act));
    }
    return activations;
}

inline auto MirrorNeuronSystem::understand(
    const std::string& action_name)
    -> std::optional<MotorRepresentation> {
    auto it = motor_repertoire_.find(action_name);
    if (it != motor_repertoire_.end()) {
        return it->second;
    }
    return std::nullopt;
}

inline auto MirrorNeuronSystem::is_understood(
    const std::string& action_name) const -> bool {
    auto it = motor_repertoire_.find(action_name);
    return it != motor_repertoire_.end()
        && it->second.activation > config_.activation_threshold;
}

inline auto MirrorNeuronSystem::imitate(
    const std::string& action_name,
    const std::string& feedback)
    -> ImitationResult {
    ImitationResult result;
    result.action_name = action_name;
    imitation_attempts_++;

    auto target = understand(action_name);
    if (!target) {
        result.errors.push_back("action not understood: " + action_name);
        return result;
    }

    // 尝试分解并执行子动作
    auto sub_actions = decompose_action_(action_name);
    result.attempts = 1;

    // 计算模仿相似度
    auto& motor = motor_repertoire_[action_name];
    motor.observation_count++;
    result.similarity = std::min(1.0,
        motor.activation + config_.imitation_learning_rate);

    // 判断成功
    result.success = result.similarity > config_.activation_threshold;
    if (result.success) {
        motor.mastered = true;
        motor.activation = std::min(1.0,
            motor.activation + config_.imitation_learning_rate * 2.0);
        result.learned_skill = action_name;
        imitation_successes_++;
    }

    if (!feedback.empty() && !result.success) {
        result.errors.push_back("feedback: " + feedback);
    }

    return result;
}

inline auto MirrorNeuronSystem::learn_from_observations(
    const std::string& action_name)
    -> ImitationResult {
    ImitationResult result;
    result.action_name = action_name;

    // 统计该动作被观察的次数
    int obs_count = 0;
    for (const auto& obs : observation_history_) {
        if (obs.action_name == action_name) obs_count++;
    }

    if (obs_count < 3) {
        result.errors.push_back(
            "insufficient observations (" + std::to_string(obs_count) + ")");
        return result;
    }

    // 观察足够 → 尝试模仿
    return imitate(action_name, "observed " + std::to_string(obs_count) + " times");
}

inline auto MirrorNeuronSystem::infer_intention(
    const ObservedAction& action)
    -> IntentionInference {
    IntentionInference inference;
    inference.action_name = action.action_name;
    inference.inferred_intention = contextual_intention_(
        action.action_name, action.context);
    inference.confidence = action.confidence * 0.8;

    // 支持证据
    if (action.inferred_goal != action.action_name) {
        inference.supporting_evidence.push_back(
            "explicit goal: " + action.inferred_goal);
    }
    inference.supporting_evidence.push_back(
        "context: " + action.context);

    // 预测下一步
    auto motor = understand(action.action_name);
    if (motor && !motor->sub_actions.empty()) {
        inference.predicted_next_action = motor->sub_actions[0];
    }

    return inference;
}

inline auto MirrorNeuronSystem::infer_intention_from_sequence(
    const std::vector<ObservedAction>& sequence)
    -> IntentionInference {
    if (sequence.empty()) return IntentionInference{};

    IntentionInference inference;
    inference.action_name = "sequence[" + std::to_string(sequence.size()) + "]";

    // 从序列的最后一个动作推断意图
    const auto& last = sequence.back();
    inference = infer_intention(last);

    // 增强置信度（序列一致性）
    inference.confidence = std::min(1.0, inference.confidence * 1.2);

    return inference;
}

inline auto MirrorNeuronSystem::infer_emotional_state(
    const ObservedAction& action,
    const std::string& /*context*/)
    -> std::map<std::string, double> {
    if (!config_.enable_empathy) return {};

    std::map<std::string, double> emotions;
    // 简单映射：根据动作和语境推断情感
    if (action.action_name.find("help") != std::string::npos) {
        emotions["compassion"] = 0.7;
        emotions["concern"] = 0.5;
    } else if (action.action_name.find("attack") != std::string::npos) {
        emotions["anger"] = 0.8;
        emotions["fear"] = 0.3;
    } else if (action.action_name.find("share") != std::string::npos) {
        emotions["joy"] = 0.6;
        emotions["trust"] = 0.5;
    } else {
        emotions["neutral"] = 0.5;
    }

    return emotions;
}

inline auto MirrorNeuronSystem::mastered_actions() const
    -> std::vector<std::string> {
    std::vector<std::string> mastered;
    for (const auto& [name, rep] : motor_repertoire_) {
        if (rep.mastered) mastered.push_back(name);
    }
    return mastered;
}

inline auto MirrorNeuronSystem::imitation_success_rate() const -> double {
    return imitation_attempts_ > 0
        ? static_cast<double>(imitation_successes_) / imitation_attempts_
        : 0.0;
}

inline auto MirrorNeuronSystem::activate_motor_(
    const std::string& action_name, double strength)
    -> MotorRepresentation {
    auto& motor = motor_repertoire_[action_name];
    motor.action_name = action_name;
    motor.activation = std::min(1.0,
        motor.activation * 0.9 + strength * 0.3);  // 衰减 + 新激活
    motor.observation_count++;
    return motor;
}

inline auto MirrorNeuronSystem::decompose_action_(
    const std::string& action_name) const
    -> std::vector<std::string> {
    auto it = motor_repertoire_.find(action_name);
    if (it != motor_repertoire_.end() && !it->second.sub_actions.empty()) {
        return it->second.sub_actions;
    }
    // 默认分解
    return {"observe_" + action_name, "plan_" + action_name, "execute_" + action_name};
}

inline auto MirrorNeuronSystem::evaluate_imitation_(
    const MotorRepresentation& target,
    const MotorRepresentation& actual) const -> double {
    if (target.sub_actions.empty()) return actual.activation;
    int matches = 0;
    for (const auto& sub : target.sub_actions) {
        if (std::find(actual.sub_actions.begin(),
                      actual.sub_actions.end(), sub)
            != actual.sub_actions.end()) {
            matches++;
        }
    }
    return target.sub_actions.empty()
        ? 0.0 : static_cast<double>(matches) / target.sub_actions.size();
}

inline auto MirrorNeuronSystem::contextual_intention_(
    const std::string& action,
    const std::string& context) const -> std::string {
    // 基于语境推断意图
    if (context.find("problem") != std::string::npos
        || context.find("error") != std::string::npos) {
        return "solve_" + context;
    }
    if (context.find("learn") != std::string::npos) {
        return "acquire_knowledge";
    }
    return "perform_" + action;
}

}  // namespace ai_learning::learning