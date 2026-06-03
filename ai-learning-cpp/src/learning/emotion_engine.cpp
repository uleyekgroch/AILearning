/**
 * @file emotion_engine.cpp
 * @brief 情感驱动学习引擎实现
 */

#include "ai_learning/learning/emotion_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ai_learning::learning {

EmotionEngine::EmotionEngine(const EmotionConfig& config)
    : config_(config)
{
    current_.label = infer_label_(current_.valence, current_.arousal);
}

// ── 情绪状态管理 ──────────────────────────────────────────────

auto EmotionEngine::process_event(const EmotionEvent& event)
    -> EmotionState
{
    // Step 1: 计算奖励预测误差
    auto rpe = compute_rpe(event.expected_outcome, event.actual_outcome);

    // Step 2: 根据事件类型和 RPE 更新效价
    // 正效价 = 好结果，负效价 = 坏结果
    double valence_delta = 0.0;
    if (event.event_type == "success" || event.event_type == "progress") {
        valence_delta = event.magnitude * 0.3;
    } else if (event.event_type == "failure" || event.event_type == "block") {
        valence_delta = -event.magnitude * 0.3;
    } else if (event.event_type == "surprise") {
        valence_delta = rpe.is_positive() ? 0.2 : -0.1;
    }

    // RPE 也影响效价
    valence_delta += rpe.error * 0.2;

    current_.valence = std::clamp(current_.valence + valence_delta, -1.0, 1.0);

    // Step 3: 根据事件强度更新唤醒度
    double arousal_delta = event.magnitude * 0.4;
    // RPE 幅度也增加唤醒（无论正负）
    arousal_delta += rpe.magnitude() * 0.2;

    current_.arousal = std::clamp(current_.arousal + arousal_delta, 0.0, 1.0);

    // Step 4: 更新支配度
    if (event.event_type == "success") {
        current_.dominance = std::min(current_.dominance + 0.1, 1.0);
    } else if (event.event_type == "failure") {
        current_.dominance = std::max(current_.dominance - 0.1, 0.0);
    }

    // Step 5: 推断情绪标签
    current_.label = infer_label_(current_.valence, current_.arousal);

    // ★v2: 填充离散情感强度
    current_.intensities.clear();
    double sig = event.magnitude;
    double surprise = std::abs(rpe.magnitude());
    if (rpe.is_positive() && rpe.magnitude() > 0.3) {
        current_.intensities[BasicEmotion::kJoy] = sig * 0.8;
        current_.intensities[BasicEmotion::kPride] = sig * 0.5;
    } else if (!rpe.is_positive() && rpe.magnitude() > 0.3) {
        current_.intensities[BasicEmotion::kFrustration] = sig * 0.7;
        current_.intensities[BasicEmotion::kSadness] = sig * 0.3;
    }
    current_.intensities[BasicEmotion::kSurprise] = surprise;
    if (sig > 0.2 && sig < 0.7) {
        current_.intensities[BasicEmotion::kCuriosity] = sig * 0.7;
    }
    if (sig > 0.5 && surprise > 0.5) {
        current_.intensities[BasicEmotion::kConfusion] = sig * 0.5;
    }

    // 记录历史
    emotion_history_.push_back(current_);
    while (static_cast<int>(emotion_history_.size()) > config_.emotion_history_size) {
        emotion_history_.pop_front();
    }

    // 记录多巴胺
    dopamine_history_.push_back(rpe);
    while (static_cast<int>(dopamine_history_.size()) > 20) {
        dopamine_history_.pop_front();
    }

    return current_;
}

void EmotionEngine::decay()
{
    // 效价向中性衰减
    if (current_.valence > 0) {
        current_.valence = std::max(0.0, current_.valence - config_.valence_decay);
    } else if (current_.valence < 0) {
        current_.valence = std::min(0.0, current_.valence + config_.valence_decay);
    }

    // 唤醒度向基线衰减
    current_.arousal = std::max(0.0, current_.arousal - config_.arousal_decay);

    // 支配度向 0.5 衰减
    current_.dominance = current_.dominance * (1 - 0.02) + 0.5 * 0.02;

    // 更新标签
    current_.label = infer_label_(current_.valence, current_.arousal);
}

void EmotionEngine::set_state(const EmotionState& state)
{
    current_ = state;
    current_.label = infer_label_(current_.valence, current_.arousal);
    emotion_history_.push_back(current_);
}

// ── 多巴胺信号 ──────────────────────────────────────────────────

auto EmotionEngine::compute_rpe(double expected, double actual)
    -> DopamineSignal
{
    DopamineSignal signal;
    signal.prediction = expected;
    signal.actual = actual;
    signal.error = actual - expected;
    return signal;
}

auto EmotionEngine::dopamine_baseline() const -> double
{
    if (dopamine_history_.empty()) return 0.0;

    double sum = 0.0;
    for (const auto& d : dopamine_history_) {
        sum += d.error;
    }
    return sum / dopamine_history_.size();
}

// ── 记忆调制 ──────────────────────────────────────────────────

auto EmotionEngine::compute_modulation() const -> MemoryModulation
{
    MemoryModulation mod;
    mod.encoding_boost = encoding_boost_(current_.arousal, current_.valence);
    mod.consolidation_boost = consolidation_boost_(current_.arousal);
    mod.retrieval_boost = 1.0 + std::abs(current_.valence) * 0.2;
    mod.forgetting_rate = std::max(0.005, 0.02 - current_.arousal * 0.015);

    mod.reason = "arousal=" + std::to_string(current_.arousal) +
                 ", valence=" + std::to_string(current_.valence) +
                 ", label=" + current_.label;

    return mod;
}

auto EmotionEngine::event_modulation(const EmotionEvent& event) const
    -> MemoryModulation
{
    MemoryModulation mod;
    double event_arousal = event.magnitude;
    double event_valence = event.actual_outcome - event.expected_outcome;

    mod.encoding_boost = encoding_boost_(event_arousal, event_valence);
    mod.consolidation_boost = consolidation_boost_(event_arousal);
    mod.retrieval_boost = 1.0;
    mod.forgetting_rate = std::max(0.005, 0.02 - event_arousal * 0.015);
    mod.reason = "event: " + event.event_type + ", magnitude=" + std::to_string(event.magnitude);

    return mod;
}

// ── 学习策略适配 ──────────────────────────────────────────────

auto EmotionEngine::suggest_strategy() const -> StrategySuggestion
{
    StrategySuggestion suggestion;

    // Yerkes-Dodson: 根据唤醒度推荐策略
    if (current_.arousal > config_.high_arousal_threshold) {
        // 高唤醒：需要冷静
        suggestion.strategy = "systematic_review";
        suggestion.confidence = 0.8;
        suggestion.reason = "高唤醒状态，建议系统性复习以避免冲动错误";
        suggestion.expected_benefit = "减少错误率";
    } else if (current_.arousal < config_.low_arousal_threshold) {
        // 低唤醒：需要刺激
        suggestion.strategy = "exploratory_learning";
        suggestion.confidence = 0.7;
        suggestion.reason = "低唤醒状态（无聊），建议探索性学习以重新激活";
        suggestion.expected_benefit = "提高参与度";
    } else if (current_.valence > 0.3) {
        // 正面情绪：创造性学习
        suggestion.strategy = "creative_exploration";
        suggestion.confidence = 0.7;
        suggestion.reason = "正面情绪，适合创造性探索";
        suggestion.expected_benefit = "发现新知识";
    } else if (current_.valence < -0.3) {
        // 负面情绪：巩固已有知识
        suggestion.strategy = "consolidation";
        suggestion.confidence = 0.8;
        suggestion.reason = "负面情绪，建议巩固已有知识";
        suggestion.expected_benefit = "稳定知识基础";
    } else {
        // 中性：标准学习
        suggestion.strategy = "standard_learning";
        suggestion.confidence = 0.5;
        suggestion.reason = "情绪中性，标准学习策略";
        suggestion.expected_benefit = "稳定进步";
    }

    return suggestion;
}

auto EmotionEngine::needs_regulation() const -> bool
{
    // 极端情绪需要调节
    return current_.arousal > 0.9 || current_.arousal < 0.05 ||
           current_.valence < -0.8 || current_.valence > 0.8;
}

auto EmotionEngine::regulate() -> EmotionState
{
    // 认知重评：将极端情绪向中心调节
    current_.valence *= 0.7;
    current_.arousal = current_.arousal * 0.6 + config_.optimal_arousal * 0.4;
    current_.dominance = current_.dominance * 0.8 + 0.5 * 0.2;
    current_.label = infer_label_(current_.valence, current_.arousal);

    emotion_history_.push_back(current_);
    return current_;
}

// ★v2: 指定策略的情绪调节
auto EmotionEngine::regulate_with(RegulationStrategy strategy) -> EmotionState
{
    switch (strategy) {
        case RegulationStrategy::kSuppression:
            for (auto& [_, intensity] : current_.intensities) {
                intensity *= 0.5;
            }
            current_.arousal *= 0.7;
            break;
        case RegulationStrategy::kReappraisal:
            current_.valence = std::clamp(current_.valence + 0.3, -1.0, 1.0);
            for (auto& [e, intensity] : current_.intensities) {
                if (e == BasicEmotion::kFear || e == BasicEmotion::kAnger
                    || e == BasicEmotion::kSadness) {
                    intensity *= 0.6;
                }
            }
            break;
        case RegulationStrategy::kDistraction:
            for (auto& [_, intensity] : current_.intensities) {
                intensity *= 0.4;
            }
            current_.arousal *= 0.5;
            break;
        case RegulationStrategy::kAcceptance:
            current_.arousal *= 0.8;
            break;
    }
    current_.label = infer_label_(current_.valence, current_.arousal);
    emotion_history_.push_back(current_);
    return current_;
}

// ── 查询 ──────────────────────────────────────────────────────

auto EmotionEngine::performance_prediction() const -> double
{
    return yerkes_dodson_(current_.arousal);
}

auto EmotionEngine::emotion_label() const -> std::string
{
    return current_.label;
}

auto EmotionEngine::stats() const -> std::map<std::string, double>
{
    double avg_valence = 0.0, avg_arousal = 0.0;
    if (!emotion_history_.empty()) {
        for (const auto& e : emotion_history_) {
            avg_valence += e.valence;
            avg_arousal += e.arousal;
        }
        avg_valence /= emotion_history_.size();
        avg_arousal /= emotion_history_.size();
    }

    return {
        {"current_valence", current_.valence},
        {"current_arousal", current_.arousal},
        {"current_dominance", current_.dominance},
        {"current_intensity", current_.intensity()},
        {"avg_valence", avg_valence},
        {"avg_arousal", avg_arousal},
        {"dopamine_baseline", dopamine_baseline()},
        {"performance_prediction", performance_prediction()},
        {"needs_regulation", needs_regulation() ? 1.0 : 0.0},
    };
}

// ── 内部方法 ──────────────────────────────────────────────────

auto EmotionEngine::infer_label_(double valence, double arousal) -> std::string
{
    // 基于效价-唤醒度模型的离散情绪标签
    if (valence > 0.3 && arousal > 0.6) return "excited";
    if (valence > 0.3 && arousal > 0.3) return "confident";
    if (valence > 0.3 && arousal <= 0.3) return "content";

    if (valence < -0.3 && arousal > 0.6) return "frustrated";
    if (valence < -0.3 && arousal > 0.3) return "concerned";
    if (valence < -0.3 && arousal <= 0.3) return "bored";

    if (arousal > 0.7) return "surprised";
    if (arousal < 0.2) return "calm";

    return "curious";
}

auto EmotionEngine::yerkes_dodson_(double arousal) const -> double
{
    // 倒 U 型曲线: performance = -4*(arousal - optimal)^2 + 1
    double diff = arousal - config_.optimal_arousal;
    return std::max(0.0, -4.0 * diff * diff + 1.0);
}

auto EmotionEngine::encoding_boost_(double arousal, double valence) const
    -> double
{
    // 高唤醒 → 更强编码（McGaugh, 2004）
    double boost = 1.0 + arousal * 0.5;
    // 强烈情绪（无论正负）→ 更强编码
    boost += std::abs(valence) * 0.3;
    return boost;
}

auto EmotionEngine::consolidation_boost_(double arousal) const
    -> double
{
    // 中等唤醒最优巩固
    double diff = std::abs(arousal - config_.optimal_arousal);
    return 1.0 + (1.0 - diff) * 0.3;
}

}  // namespace ai_learning::learning
