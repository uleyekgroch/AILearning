/**
 * @file human_like_learning_loop.hpp
 * @brief 仿人类学习完整闭环 — 感知→预测→主动推理→行动→学习→反思
 *
 * 理论基础：
 *   - 自由能原理 (Friston 2006-2025): 所有自适应系统最小化自由能
 *   - 预测加工理论 (Clark 2013): 大脑是预测机器
 *   - 主动推理 (Parr, Pezzulo & Friston 2022): 行动服务于认知
 *   - 内在动机 (Oudeyer 2007): 好奇心 = 学习进度最大化
 *   - 互补学习系统 (McClelland 1995): 海马+皮层双系统
 *   - 发展心理学 (Piaget, Vygotsky): 阶段式认知发展
 *   - 镜像神经元 (Rizzolatti 2004): 观察即学习
 *
 * 完整闭环（仿人类学习的一轮迭代）：
 *
 *   ┌──────────────────────────────────────────────────────────┐
 *   │                 Human-Like Learning Loop                │
 *   │                                                        │
 *   │  1. 感知 (Perceive)                                    │
 *   │     └→ MultiModalEncoder: 视觉/听觉/文本 → 嵌入向量     │
 *   │                                                        │
 *   │  2. 预测 (Predict)                                     │
 *   │     └→ PredictiveCodingEngine: 基于当前信念预测下一步   │
 *   │                                                        │
 *   │  3. 主动推理 (Active Inference)                         │
 *   │     ├→ 信念更新: 观测 vs 预测 → 后验信念               │
 *   │     ├→ 策略评估: 预期自由能 G(π)                        │
 *   │     └→ 行动选择: 最小化 G(π) + 最大化信息增益           │
 *   │                                                        │
 *   │  4. 执行行动 (Act)                                     │
 *   │     ├→ 探索 (Explore): 好奇心驱动 → 信息增益            │
 *   │     ├→ 利用 (Exploit): 技能应用 → 目标达成              │
 *   │     └→ 模仿 (Imitate): 镜像神经元 → 社会学习            │
 *   │                                                        │
 *   │  5. 学习 (Learn)                                       │
 *   │     ├→ 知识图谱更新: 新实体/关系                        │
 *   │     ├→ 预测模型更新: Hebbian 学习                       │
 *   │     └→ 技能树更新: 新技能的注册和关联                   │
 *   │                                                        │
 *   │  6. 记忆巩固 (Consolidate)                              │
 *   │     ├→ 海马编码: 快速存储                                │
 *   │     ├→ 皮层巩固: 慢速整合                                │
 *   │     └→ 经验回放: 防遗忘                                  │
 *   │                                                        │
 *   │  7. 反思 (Reflect)                                     │
 *   │     ├→ 元认知评估: 我学到了什么？哪里还不足？            │
 *   │     ├→ 动机更新: 进步 → 满足, 失败 → 调整策略            │
 *   │     ├→ 发展阶段检查: 是否需要晋升？                      │
 *   │     └→ 自我修改: 优化学习策略和参数                      │
 *   │                                                        │
 *   │  8. 社交交互 (Socialize)                               │
 *   │     ├→ 观察他人: 镜像神经元激活                         │
 *   │     ├→ 模仿学习: 从观察中获取技能                        │
 *   │     └→ 知识分享: 教学相长                               │
 *   └──────────────────────────────────────────────────────────┘
 *
 * 与 LLM 的本质区别：
 *   - LLM: 被动接收数据 → 反向传播 → 统计模式匹配
 *   - 本系统: 主动探索 → 预测误差驱动 → 因果理解
 *   - LLM: 需要海量参数和数据
 *   - 本系统: 少量样本 + 结构化知识
 *   - LLM: 黑盒推理
 *   - 本系统: 可解释因果链
 *   - LLM: 无法持续学习（灾难性遗忘）
 *   - 本系统: EWC + 经验回放 终身学习
 */

#pragma once

#include "ai_learning/reasoning/active_inference.hpp"
#include "ai_learning/learning/intrinsic_motivation.hpp"
#include "ai_learning/learning/mirror_neuron.hpp"
#include "ai_learning/learning/continual_learner.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 学习循环阶段
enum class LearningPhase {
    kPerceive,       ///< 感知
    kPredict,         ///< 预测
    kActivelyInfer,   ///< 主动推理
    kAct,             ///< 行动
    kLearn,           ///< 学习
    kConsolidate,     ///< 巩固
    kReflect,         ///< 反思
    kSocialize,       ///< 社交
};

/// 一轮学习的结果
struct LearningCycleResult {
    int cycle_number = 0;                      ///< 第几轮
    LearningPhase phase = LearningPhase::kPerceive;
    std::string action_taken;                   ///< 采取的行动
    double prediction_error = 0.0;             ///< 预测误差
    double free_energy = 0.0;                  ///< 自由能
    double motivation = 0.0;                   ///< 动机水平
    double knowledge_gain = 0.0;               ///< 知识增量
    std::vector<std::string> new_knowledge;    ///< 新学到的知识
    std::string reflection;                     ///< 反思内容
    bool stage_advanced = false;               ///< 是否晋升阶段
    double elapsed_ms = 0.0;                   ///< 耗时
};

/// 学习循环配置
struct HumanLikeLearningConfig {
    int max_cycles = 1000;                     ///< 最大循环次数
    int consolidate_interval = 10;             ///< 每 N 轮巩固一次
    int reflect_interval = 5;                  ///< 每 N 轮反思一次
    int social_interval = 20;                  ///< 每 N 轮社交一次
    double min_motivation = 0.05;              ///< 低于此值暂停
    double learning_rate = 0.01;               ///< 全局学习率
    bool verbose = false;                      ///< 详细日志
    bool enable_active_inference = true;       ///< 启用主动推理
    bool enable_mirror_neurons = true;         ///< 启用镜像神经元
    bool enable_continual_learning = true;     ///< 启用终身学习
};

/// 学习循环统计
struct LearningLoopStats {
    int total_cycles = 0;
    int actions_taken = 0;
    int knowledge_entities_learned = 0;
    int skills_mastered = 0;
    double avg_prediction_error = 0.0;
    double avg_free_energy = 0.0;
    double avg_motivation = 0.0;
    int stage_advancements = 0;
    double total_elapsed_ms = 0.0;
    int social_observations = 0;
    int imitations_attempted = 0;
};

/// 观察提供者 — 让系统能主动感知世界
class IObservationProvider {
public:
    virtual ~IObservationProvider() = default;
    /// 获取当前观测
    virtual auto observe() -> std::vector<float> = 0;
    /// 执行行动并返回结果
    virtual auto act(const std::string& action) -> std::vector<float> = 0;
    /// 是否有更多观测可用
    [[nodiscard]] virtual auto has_more() const -> bool { return true; }
};

/// 仿人类学习完整闭环
class HumanLikeLearningLoop {
public:
    HumanLikeLearningLoop(
        IntrinsicMotivationEngine& motivation,
        reasoning::ActiveInferenceEngine& active_inference,
        MirrorNeuronSystem& mirror_neurons,
        ContinualLearner& continual_learner);

    // ═══════════════════════════════════════════════════════════
    // 执行接口
    // ═══════════════════════════════════════════════════════════

    /// 运行完整的学习循环
    /// @param config 配置
    /// @param observation_provider 观测提供者
    /// @param known_topics 已知主题
    /// @param mastery_map 各领域掌握度
    /// @param knowledge_callback 学习回调（更新知识图谱/预测引擎）
    auto run(const HumanLikeLearningConfig& config,
             IObservationProvider& observation_provider,
             const std::vector<std::string>& known_topics,
             const std::map<std::string, double>& mastery_map,
             std::function<void(const std::string&, const std::vector<float>&)> knowledge_callback)
        -> LearningLoopStats;

    /// 执行单轮学习
    auto one_cycle(int cycle_number,
                   IObservationProvider& obs_provider,
                   const std::vector<std::string>& known_topics,
                   const std::map<std::string, double>& mastery_map,
                   std::function<void(const std::string&, const std::vector<float>&)> on_learn)
        -> LearningCycleResult;

    // ═══════════════════════════════════════════════════════════
    // 查询
    // ═══════════════════════════════════════════════════════════

    [[nodiscard]] auto stats() const -> const LearningLoopStats& {
        return stats_;
    }

    [[nodiscard]] auto cycle_history() const
        -> const std::vector<LearningCycleResult>& { return history_; }

    /// 获取当前学习阶段
    [[nodiscard]] auto current_phase() const -> LearningPhase {
        return current_phase_;
    }

private:
    IntrinsicMotivationEngine& motivation_;
    reasoning::ActiveInferenceEngine& active_inference_;
    MirrorNeuronSystem& mirror_neurons_;
    ContinualLearner& continual_learner_;

    LearningLoopStats stats_;
    std::vector<LearningCycleResult> history_;
    LearningPhase current_phase_ = LearningPhase::kPerceive;

    double cumulative_error_ = 0.0;
    double cumulative_free_energy_ = 0.0;

    /// 相位名称
    static auto phase_name_(LearningPhase phase) -> const char*;

    /// 计算知识增益
    auto compute_knowledge_gain_(const std::vector<float>& before,
                                 const std::vector<float>& after) const -> double;

    /// 生成反思
    auto generate_reflection_(const LearningCycleResult& result) const
        -> std::string;
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline HumanLikeLearningLoop::HumanLikeLearningLoop(
    IntrinsicMotivationEngine& motivation,
    reasoning::ActiveInferenceEngine& active_inference,
    MirrorNeuronSystem& mirror_neurons,
    ContinualLearner& continual_learner)
    : motivation_(motivation),
      active_inference_(active_inference),
      mirror_neurons_(mirror_neurons),
      continual_learner_(continual_learner) {}

inline auto HumanLikeLearningLoop::run(
    const HumanLikeLearningConfig& config,
    IObservationProvider& obs_provider,
    const std::vector<std::string>& known_topics,
    const std::map<std::string, double>& mastery_map,
    std::function<void(const std::string&, const std::vector<float>&)> on_learn)
    -> LearningLoopStats {

    for (int cycle = 0; cycle < config.max_cycles; ++cycle) {
        if (!obs_provider.has_more()) break;
        if (motivation_.total_drive() < config.min_motivation) {
            // 动机过低：尝试生成新目标
            auto goal = motivation_.generate_goal(
                known_topics, 0.3, mastery_map);
            if (goal.estimated_value < 0.1) break;  // 真的无事可学
        }

        auto result = one_cycle(cycle, obs_provider,
            known_topics, mastery_map, on_learn);
        history_.push_back(result);

        if (config.verbose && cycle % 10 == 0) {
            // 进度日志
        }
    }

    return stats_;
}

inline auto HumanLikeLearningLoop::one_cycle(
    int cycle_number,
    IObservationProvider& obs_provider,
    const std::vector<std::string>& known_topics,
    const std::map<std::string, double>& mastery_map,
    std::function<void(const std::string&, const std::vector<float>&)> on_learn)
    -> LearningCycleResult {

    LearningCycleResult result;
    result.cycle_number = cycle_number;
    stats_.total_cycles = cycle_number + 1;

    // ═══════════════════════════════════════════════════════════
    // Phase 1: 感知
    // ═══════════════════════════════════════════════════════════
    current_phase_ = LearningPhase::kPerceive;
    auto observation = obs_provider.observe();
    result.phase = LearningPhase::kPerceive;

    // ═══════════════════════════════════════════════════════════
    // Phase 2: 预测
    // ═══════════════════════════════════════════════════════════
    current_phase_ = LearningPhase::kPredict;
    std::vector<float> prediction(observation.size(), 0.0f);
    // 使用世界模型的先验生成预测
    auto belief = active_inference_.current_belief();
    for (size_t i = 0; i < prediction.size(); ++i) {
        prediction[i] = static_cast<float>(belief.probability) * 0.5f;
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 3: 主动推理
    // ═══════════════════════════════════════════════════════════
    current_phase_ = LearningPhase::kActivelyInfer;
    auto ai_result = active_inference_.step(
        observation, prediction, {}, "");
    result.prediction_error = ai_result.posterior_state.prediction_error;
    result.free_energy = ai_result.variational_free_energy;
    result.phase = LearningPhase::kActivelyInfer;

    // ═══════════════════════════════════════════════════════════
    // Phase 4: 行动
    // ═══════════════════════════════════════════════════════════
    current_phase_ = LearningPhase::kAct;
    result.action_taken = ai_result.selected_action;
    auto action_outcome = obs_provider.act(result.action_taken);
    stats_.actions_taken++;

    // ═══════════════════════════════════════════════════════════
    // Phase 5: 学习
    // ═══════════════════════════════════════════════════════════
    current_phase_ = LearningPhase::kLearn;
    result.knowledge_gain = compute_knowledge_gain_(observation, action_outcome);
    if (on_learn) {
        on_learn(result.action_taken, action_outcome);
    }
    stats_.knowledge_entities_learned++;

    // 终身学习保护
    if (stats_.total_cycles % 50 == 0) {
        auto conflicts = continual_learner_.prepare_for_new_learning(
            "general", {"cycle_" + std::to_string(cycle_number)});
        for (auto& c : conflicts) {
            continual_learner_.resolve_conflict(c);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 6: 巩固 (每 N 轮)
    // ═══════════════════════════════════════════════════════════
    if (cycle_number % 10 == 0) {
        current_phase_ = LearningPhase::kConsolidate;
        continual_learner_.replay(3);
        result.phase = LearningPhase::kConsolidate;
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 7: 反思 (每 N 轮)
    // ═══════════════════════════════════════════════════════════
    if (cycle_number % 5 == 0) {
        current_phase_ = LearningPhase::kReflect;
        result.reflection = generate_reflection_(result);

        // 动机更新
        LearningOutcome outcome;
        outcome.topic = result.action_taken;
        outcome.progress = result.knowledge_gain;
        outcome.surprise = result.prediction_error;
        outcome.mastery_improved = result.knowledge_gain > 0.1;
        motivation_.update_on_learning(outcome);
        motivation_.decay();
    }
    result.motivation = motivation_.total_drive();
    result.phase = LearningPhase::kReflect;

    // ═══════════════════════════════════════════════════════════
    // Phase 8: 社交 (每 N 轮)
    // ═══════════════════════════════════════════════════════════
    if (cycle_number % 20 == 0) {
        current_phase_ = LearningPhase::kSocialize;
        // 镜像神经元：将行动编码为可观察动作
        ObservedAction self_action;
        self_action.observer_id = "self";
        self_action.actor_id = "self";
        self_action.action_name = result.action_taken;
        self_action.confidence = result.knowledge_gain;
        mirror_neurons_.observe_action(self_action);
        stats_.social_observations++;
    }

    // 累计统计
    cumulative_error_ += result.prediction_error;
    cumulative_free_energy_ += result.free_energy;
    stats_.avg_prediction_error = cumulative_error_ / (cycle_number + 1);
    stats_.avg_free_energy = cumulative_free_energy_ / (cycle_number + 1);
    stats_.avg_motivation = result.motivation;

    return result;
}

inline auto HumanLikeLearningLoop::phase_name_(LearningPhase phase) -> const char* {
    switch (phase) {
        case LearningPhase::kPerceive:      return "感知";
        case LearningPhase::kPredict:        return "预测";
        case LearningPhase::kActivelyInfer:  return "主动推理";
        case LearningPhase::kAct:            return "行动";
        case LearningPhase::kLearn:          return "学习";
        case LearningPhase::kConsolidate:    return "巩固";
        case LearningPhase::kReflect:        return "反思";
        case LearningPhase::kSocialize:      return "社交";
    }
    return "未知";
}

inline auto HumanLikeLearningLoop::compute_knowledge_gain_(
    const std::vector<float>& before,
    const std::vector<float>& after) const -> double {
    // 知识增益 = 新旧状态的差异
    double gain = 0.0;
    size_t n = std::min(before.size(), after.size());
    for (size_t i = 0; i < n; ++i) {
        double diff = after[i] - before[i];
        gain += std::abs(diff);
    }
    return n > 0 ? gain / n : 0.0;
}

inline auto HumanLikeLearningLoop::generate_reflection_(
    const LearningCycleResult& result) const -> std::string {
    if (result.knowledge_gain > 0.3) {
        return "学到了很多关于 " + result.action_taken + " 的知识，预测误差 "
               + std::to_string(result.prediction_error);
    } else if (result.prediction_error > 0.5) {
        return "对 " + result.action_taken + " 的预测不准确，需要更多探索";
    } else {
        return "对 " + result.action_taken + " 已经比较熟悉，可以探索新领域";
    }
}

}  // namespace ai_learning::learning