/**
 * @file emotion_depth.hpp
 * @brief 深层情感系统 — 8+维度情感 + 情感调节
 *
 * 理论基础：
 *   - Ekman (1972) 6种基本情感: joy, sadness, anger, fear, disgust, surprise
 *   - Plutchik (1980) 情感轮: 8种基本情感 + 强度维度
 *   - Russell (1980) 环状模型: valence-arousal 二维空间
 *   - Damasio (1994) Descartes' Error: 情感是决策的基础
 *   - Gross (1998) 情感调节模型: situation→attention→appraisal→response
 *
 * 核心机制：
 *   1. 情感生成 (Emotion Generation): 事件 → 评估 → 情感激活
 *   2. 情感混合 (Emotion Blending): 多种情感共存
 *   3. 情感调节 (Emotion Regulation): 抑制/增强/重评
 *   4. 情感记忆 (Emotional Memory): 情感强度影响记忆编码
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::affect {

/// 基本情感类型 (Ekman + Plutchik)
enum class BasicEmotion {
    kJoy,        ///< 快乐
    kSadness,    ///< 悲伤
    kAnger,      ///< 愤怒
    kFear,       ///< 恐惧
    kDisgust,    ///< 厌恶
    kSurprise,   ///< 惊讶
    kTrust,      ///< 信任
    kAnticipation, ///< 期待
    kContempt,   ///< 轻蔑
    kShame,      ///< 羞耻
    kGuilt,      ///< 内疚
    kPride,      ///< 自豪
    kCuriosity,  ///< 好奇
    kConfusion,  ///< 困惑
    kBoredom,    ///< 无聊
    kFrustration ///< 挫折
};

/// 情感状态
struct EmotionalState {
    std::map<BasicEmotion, double> intensities;  ///< 各情感强度 0~1
    double valence = 0.0;        ///< 效价 -1(negative) ~ +1(positive)
    double arousal = 0.5;        ///< 唤醒度 0~1
    double dominance = 0.5;      ///< 支配感 0~1
    std::string dominant_emotion; ///< 主导情感
    double complexity = 0.0;     ///< 情感复杂度（混合程度）
};

/// 情感事件
struct EmotionalEvent {
    std::string description;      ///< 事件描述
    double significance = 0.5;    ///< 重要性 0~1
    double expectedness = 0.5;   ///< 预期度 0~1（越不预期越强）
    double goal_relevance = 0.5; ///< 目标相关性 0~1
    std::string agent;            ///< 事件主体
    bool is_self_relevant = true; ///< 是否与自我相关
};

/// 情感调节策略
enum class RegulationStrategy {
    kSuppression,     ///< 抑制（不表达）
    kReappraisal,     ///< 重评（换角度思考）
    kDistraction,     ///< 分心
    kAcceptance,      ///< 接纳
    kRumination       ///< 反刍（反复思考，通常是消极的）
};

/// 情感记忆
struct EmotionalMemory {
    std::string event;            ///< 事件
    EmotionalState state;          ///< 当时的情感状态
    double intensity = 0.0;       ///< 记忆强度
    int age = 0;                  ///< 记忆年龄
    std::string lesson;            ///< 情感教训
};

/// 情感深度配置
struct EmotionDepthConfig {
    double emotional_reactivity = 0.5;    ///< 情感反应性 0~1
    double regulation_ability = 0.5;      ///< 调节能力 0~1
    double emotional_decay = 0.05;        ///< 情感自然衰减率
    double mood_inertia = 0.8;            ///< 心境惯性（越高越难改变）
    int emotional_memory_capacity = 100;
};

/// 深层情感系统
class EmotionDepthSystem {
public:
    explicit EmotionDepthSystem(
        const EmotionDepthConfig& config = EmotionDepthConfig{});

    // ═══════════════════════════════════════════════════════════
    // 情感生成
    // ═══════════════════════════════════════════════════════════

    /// 从事件生成情感
    auto appraise(const EmotionalEvent& event) -> EmotionalState;

    /// 基于目标达成的评估
    auto appraise_goal_outcome(const std::string& goal,
                                bool achieved,
                                double effort) -> EmotionalState;

    /// 基于社会交互的评估
    auto appraise_social(const std::string& interaction,
                          double reciprocity,
                          bool positive) -> EmotionalState;

    // ═══════════════════════════════════════════════════════════
    // 情感混合
    // ═══════════════════════════════════════════════════════════

    /// 混合两个情感状态
    auto blend(const EmotionalState& a, const EmotionalState& b,
               double weight_a = 0.5) -> EmotionalState;

    /// 获取主导情感
    [[nodiscard]] auto dominant_emotion(const EmotionalState& state) const
        -> BasicEmotion;

    /// 当前情感复杂度
    [[nodiscard]] auto complexity(const EmotionalState& state) const -> double;

    // ═══════════════════════════════════════════════════════════
    // 情感调节
    // ═══════════════════════════════════════════════════════════

    /// 应用情感调节策略
    auto regulate(const EmotionalState& state,
                  RegulationStrategy strategy) -> EmotionalState;

    /// 情感衰减（时间推移）
    auto decay(const EmotionalState& state, double time_steps = 1.0)
        -> EmotionalState;

    /// 心境更新（慢变化的情感底色）
    void update_mood(const EmotionalState& current);

    // ═══════════════════════════════════════════════════════════
    // 情感记忆
    // ═══════════════════════════════════════════════════════════

    /// 记录情感记忆
    void remember_emotion(const EmotionalMemory& memory);

    /// 检索情感记忆
    auto recall_emotional(const std::string& query) const
        -> std::vector<EmotionalMemory>;

    // ═══════════════════════════════════════════════════════════
    // 查询
    // ═══════════════════════════════════════════════════════════

    [[nodiscard]] auto current_state() const -> const EmotionalState& {
        return current_state_;
    }

    [[nodiscard]] auto mood() const -> const EmotionalState& { return mood_; }

    /// 情感名称
    static auto emotion_name(BasicEmotion e) -> std::string;

    /// 情感基础值 (valence, arousal)
    static auto emotion_base(BasicEmotion e) -> std::pair<double, double>;

private:
    EmotionDepthConfig config_;
    EmotionalState current_state_;
    EmotionalState mood_;              ///< 心境（慢变化底色）
    std::vector<EmotionalMemory> emotional_memories_;

    /// 评估→情感映射
    auto map_appraisal_to_emotions_(const EmotionalEvent& event)
        -> std::map<BasicEmotion, double>;

    /// 更新效价和唤醒度
    void update_valence_arousal_(EmotionalState& state);
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline EmotionDepthSystem::EmotionDepthSystem(
    const EmotionDepthConfig& config) : config_(config) {
    current_state_.intensities[BasicEmotion::kCuriosity] = 0.5;
    current_state_.valence = 0.3;
    current_state_.arousal = 0.5;
    mood_ = current_state_;
}

inline auto EmotionDepthSystem::appraise(const EmotionalEvent& event)
    -> EmotionalState {
    auto intensities = map_appraisal_to_emotions_(event);
    EmotionalState state;
    state.intensities = intensities;
    update_valence_arousal_(state);

    // 与心境混合
    state = blend(state, mood_, 0.3);  // 心境占 30%

    current_state_ = state;
    return state;
}

inline auto EmotionDepthSystem::appraise_goal_outcome(
    const std::string& goal, bool achieved, double effort)
    -> EmotionalState {
    EmotionalEvent event;
    event.description = "目标'" + goal + "'" + (achieved ? "达成" : "未达成");
    event.significance = 0.7;
    event.expectedness = achieved ? 0.6 : 0.3;
    event.goal_relevance = 0.9;
    event.is_self_relevant = true;
    return appraise(event);
}

inline auto EmotionDepthSystem::appraise_social(
    const std::string& interaction, double reciprocity, bool positive)
    -> EmotionalState {
    EmotionalEvent event;
    event.description = interaction;
    event.significance = reciprocity;
    event.expectedness = positive ? 0.5 : 0.3;
    event.goal_relevance = 0.5;
    event.is_self_relevant = true;
    event.agent = "other";
    return appraise(event);
}

inline auto EmotionDepthSystem::blend(
    const EmotionalState& a, const EmotionalState& b,
    double weight_a) -> EmotionalState {
    EmotionalState blended;
    double w_b = 1.0 - weight_a;
    blended.valence = a.valence * weight_a + b.valence * w_b;
    blended.arousal = a.arousal * weight_a + b.arousal * w_b;
    blended.dominance = a.dominance * weight_a + b.dominance * w_b;

    // 合并所有情感维度
    std::set<BasicEmotion> all_emotions;
    for (const auto& [e, _] : a.intensities) all_emotions.insert(e);
    for (const auto& [e, _] : b.intensities) all_emotions.insert(e);

    for (auto e : all_emotions) {
        double ia = a.intensities.count(e) ? a.intensities.at(e) : 0.0;
        double ib = b.intensities.count(e) ? b.intensities.at(e) : 0.0;
        blended.intensities[e] = ia * weight_a + ib * w_b;
    }

    blended.complexity = complexity(blended);
    auto dom = dominant_emotion(blended);
    blended.dominant_emotion = emotion_name(dom);
    return blended;
}

inline auto EmotionDepthSystem::dominant_emotion(
    const EmotionalState& state) const -> BasicEmotion {
    BasicEmotion dominant = BasicEmotion::kCuriosity;
    double max_intensity = 0.0;
    for (const auto& [e, intensity] : state.intensities) {
        if (intensity > max_intensity) {
            max_intensity = intensity;
            dominant = e;
        }
    }
    return dominant;
}

inline auto EmotionDepthSystem::complexity(
    const EmotionalState& state) const -> double {
    // 复杂度 = 激活情感的多样性和均衡性
    int active = 0;
    double sum = 0.0, sum_sq = 0.0;
    for (const auto& [_, intensity] : state.intensities) {
        if (intensity > 0.1) {
            active++;
            sum += intensity;
            sum_sq += intensity * intensity;
        }
    }
    if (active <= 1) return 0.0;
    // 熵: 越均匀越复杂
    double entropy = 0.0;
    for (const auto& [_, intensity] : state.intensities) {
        if (intensity > 0.1) {
            double p = intensity / (sum + 1e-8);
            entropy -= p * std::log(p + 1e-8);
        }
    }
    return std::min(1.0, entropy / std::log(static_cast<double>(active + 1)));
}

inline auto EmotionDepthSystem::regulate(
    const EmotionalState& state, RegulationStrategy strategy)
    -> EmotionalState {
    EmotionalState regulated = state;
    double ability = config_.regulation_ability;

    switch (strategy) {
        case RegulationStrategy::kSuppression:
            for (auto& [_, intensity] : regulated.intensities) {
                intensity *= (1.0 - ability * 0.5);
            }
            break;
        case RegulationStrategy::kReappraisal:
            regulated.valence = std::clamp(
                regulated.valence + ability * 0.3, -1.0, 1.0);  // 正向偏移
            for (auto& [e, intensity] : regulated.intensities) {
                if (e == BasicEmotion::kFear || e == BasicEmotion::kAnger
                    || e == BasicEmotion::kSadness) {
                    intensity *= (1.0 - ability * 0.4);
                }
            }
            break;
        case RegulationStrategy::kDistraction:
            for (auto& [_, intensity] : regulated.intensities) {
                intensity *= (1.0 - ability * 0.6);
            }
            break;
        case RegulationStrategy::kAcceptance:
            // 接纳 = 不加抑制，但降低 arousal
            regulated.arousal *= (1.0 - ability * 0.3);
            break;
        case RegulationStrategy::kRumination:
            // 反刍 = 增强消极情感
            for (auto& [e, intensity] : regulated.intensities) {
                if (e == BasicEmotion::kSadness || e == BasicEmotion::kGuilt
                    || e == BasicEmotion::kShame) {
                    intensity = std::min(1.0, intensity * 1.3);
                }
            }
            break;
    }

    update_valence_arousal_(regulated);
    return regulated;
}

inline auto EmotionDepthSystem::decay(
    const EmotionalState& state, double time_steps)
    -> EmotionalState {
    EmotionalState decayed = state;
    double factor = std::exp(-config_.emotional_decay * time_steps);
    for (auto& [_, intensity] : decayed.intensities) {
        intensity *= factor;
    }
    update_valence_arousal_(decayed);
    return decayed;
}

inline void EmotionDepthSystem::update_mood(const EmotionalState& current) {
    double inertia = config_.mood_inertia;
    mood_ = blend(mood_, current, inertia);
}

inline void EmotionDepthSystem::remember_emotion(
    const EmotionalMemory& memory) {
    emotional_memories_.push_back(memory);
    if (static_cast<int>(emotional_memories_.size())
        > config_.emotional_memory_capacity) {
        emotional_memories_.erase(emotional_memories_.begin());
    }
}

inline auto EmotionDepthSystem::recall_emotional(
    const std::string& query) const
    -> std::vector<EmotionalMemory> {
    std::vector<EmotionalMemory> results;
    for (const auto& mem : emotional_memories_) {
        if (mem.event.find(query) != std::string::npos) {
            results.push_back(mem);
        }
    }
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.intensity > b.intensity; });
    return results;
}

inline auto EmotionDepthSystem::emotion_name(BasicEmotion e) -> std::string {
    switch (e) {
        case BasicEmotion::kJoy:          return "快乐";
        case BasicEmotion::kSadness:      return "悲伤";
        case BasicEmotion::kAnger:        return "愤怒";
        case BasicEmotion::kFear:         return "恐惧";
        case BasicEmotion::kDisgust:      return "厌恶";
        case BasicEmotion::kSurprise:     return "惊讶";
        case BasicEmotion::kTrust:        return "信任";
        case BasicEmotion::kAnticipation: return "期待";
        case BasicEmotion::kContempt:     return "轻蔑";
        case BasicEmotion::kShame:        return "羞耻";
        case BasicEmotion::kGuilt:        return "内疚";
        case BasicEmotion::kPride:        return "自豪";
        case BasicEmotion::kCuriosity:    return "好奇";
        case BasicEmotion::kConfusion:    return "困惑";
        case BasicEmotion::kBoredom:      return "无聊";
        case BasicEmotion::kFrustration:  return "挫折";
    }
    return "未知";
}

inline auto EmotionDepthSystem::emotion_base(BasicEmotion e)
    -> std::pair<double, double> {
    // {valence, arousal}
    switch (e) {
        case BasicEmotion::kJoy:          return { 0.8,  0.7};
        case BasicEmotion::kSadness:      return {-0.7,  0.2};
        case BasicEmotion::kAnger:        return {-0.5,  0.8};
        case BasicEmotion::kFear:         return {-0.6,  0.9};
        case BasicEmotion::kDisgust:      return {-0.4,  0.5};
        case BasicEmotion::kSurprise:     return { 0.0,  0.8};
        case BasicEmotion::kTrust:        return { 0.6,  0.3};
        case BasicEmotion::kAnticipation: return { 0.4,  0.6};
        case BasicEmotion::kContempt:     return {-0.3,  0.4};
        case BasicEmotion::kShame:        return {-0.5,  0.5};
        case BasicEmotion::kGuilt:        return {-0.5,  0.4};
        case BasicEmotion::kPride:        return { 0.7,  0.5};
        case BasicEmotion::kCuriosity:    return { 0.5,  0.6};
        case BasicEmotion::kConfusion:    return {-0.1,  0.6};
        case BasicEmotion::kBoredom:      return {-0.2,  0.1};
        case BasicEmotion::kFrustration:  return {-0.4,  0.7};
    }
    return {0.0, 0.5};
}

inline auto EmotionDepthSystem::map_appraisal_to_emotions_(
    const EmotionalEvent& event) -> std::map<BasicEmotion, double> {
    std::map<BasicEmotion, double> intensities;

    double sig = event.significance * config_.emotional_reactivity;
    double surprise = (1.0 - event.expectedness) * sig;

    if (event.goal_relevance > 0.5 && event.expectedness > 0.5) {
        // 目标达成且预期 → 快乐 + 自豪
        intensities[BasicEmotion::kJoy] = sig * 0.8;
        if (event.is_self_relevant)
            intensities[BasicEmotion::kPride] = sig * 0.6;
    } else if (event.goal_relevance > 0.5 && event.expectedness < 0.3) {
        // 目标未达成且不预期 → 挫折 + 悲伤
        intensities[BasicEmotion::kFrustration] = sig * 0.7;
        intensities[BasicEmotion::kSadness] = sig * 0.4;
    }

    // 惊讶 = 不预期的程度
    intensities[BasicEmotion::kSurprise] = surprise;

    // 好奇心 = 中等重要性 + 中等预期
    if (sig > 0.2 && sig < 0.7 && event.expectedness < 0.6) {
        intensities[BasicEmotion::kCuriosity] = sig * 0.7;
    }

    // 困惑 = 重要性高但无法理解
    if (sig > 0.5 && event.expectedness < 0.2) {
        intensities[BasicEmotion::kConfusion] = sig * 0.6;
    }

    return intensities;
}

inline void EmotionDepthSystem::update_valence_arousal_(
    EmotionalState& state) {
    double valence_sum = 0.0, arousal_sum = 0.0, total = 0.0;
    for (const auto& [e, intensity] : state.intensities) {
        auto [v, a] = emotion_base(e);
        valence_sum += v * intensity;
        arousal_sum += a * intensity;
        total += intensity;
    }
    if (total > 1e-8) {
        state.valence = std::clamp(valence_sum / total, -1.0, 1.0);
        state.arousal = std::clamp(arousal_sum / total, 0.0, 1.0);
    }
}

}  // namespace ai_learning::affect