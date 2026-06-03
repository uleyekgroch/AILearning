/**
 * @file emotion_engine.hpp
 * @brief 情感驱动学习 — 情绪调制记忆编码与学习策略（v2: 离散情感 + 调节策略增强）
 *
 * 参考：
 *   - Ekman (1972) 6种基本情感 + Plutchik (1980) 情感轮
 *   - McGaugh 情绪与记忆巩固 (2004)：情绪唤醒增强记忆编码
 *   - 多巴胺学习信号 (Schultz, 1997)：奖励预测误差 (RPE)
 *   - Yerkes-Dodson 定律 (1908)：唤醒水平与绩效的倒 U 型关系
 *   - Gross (1998) 情绪调节理论：认知重评、表达抑制
 *
 * 核心能力：
 *   1. 情绪状态建模 — 维度模型（效价 × 唤醒度） + ★v2 离散情感强度
 *   2. 记忆调制 — 情绪唤醒越强，记忆编码越深
 *   3. 多巴胺 RPE — 预测误差驱动学习动力
 *   4. 学习策略适配 — 根据情绪状态调整学习方式
 *   5. ★v2 情绪调节策略 — 抑制/重评/分心/接纳
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <deque>

namespace ai_learning::learning {

/// ★v2: 离散基本情感 (Ekman 6 + Plutchik 8 扩展)
enum class BasicEmotion {
    kJoy, kSadness, kAnger, kFear, kDisgust, kSurprise,
    kTrust, kAnticipation, kCuriosity, kConfusion,
    kPride, kShame, kFrustration, kBoredom
};

/// 情感基础值
inline auto emotion_base(BasicEmotion e) -> std::pair<double, double> {
    switch (e) {
        case BasicEmotion::kJoy:          return { 0.8, 0.7};
        case BasicEmotion::kSadness:      return {-0.7, 0.2};
        case BasicEmotion::kAnger:        return {-0.5, 0.8};
        case BasicEmotion::kFear:         return {-0.6, 0.9};
        case BasicEmotion::kDisgust:      return {-0.4, 0.5};
        case BasicEmotion::kSurprise:     return { 0.0, 0.8};
        case BasicEmotion::kTrust:        return { 0.6, 0.3};
        case BasicEmotion::kAnticipation: return { 0.4, 0.6};
        case BasicEmotion::kCuriosity:    return { 0.5, 0.6};
        case BasicEmotion::kConfusion:    return {-0.1, 0.6};
        case BasicEmotion::kPride:        return { 0.7, 0.5};
        case BasicEmotion::kShame:        return {-0.5, 0.5};
        case BasicEmotion::kFrustration:  return {-0.4, 0.7};
        case BasicEmotion::kBoredom:      return {-0.2, 0.1};
    }
    return {0.0, 0.5};
}

inline auto emotion_name(BasicEmotion e) -> std::string {
    switch (e) {
        case BasicEmotion::kJoy:          return "快乐";
        case BasicEmotion::kSadness:      return "悲伤";
        case BasicEmotion::kAnger:        return "愤怒";
        case BasicEmotion::kFear:         return "恐惧";
        case BasicEmotion::kDisgust:      return "厌恶";
        case BasicEmotion::kSurprise:     return "惊讶";
        case BasicEmotion::kTrust:        return "信任";
        case BasicEmotion::kAnticipation: return "期待";
        case BasicEmotion::kCuriosity:    return "好奇";
        case BasicEmotion::kConfusion:    return "困惑";
        case BasicEmotion::kPride:        return "自豪";
        case BasicEmotion::kShame:        return "羞耻";
        case BasicEmotion::kFrustration:  return "挫折";
        case BasicEmotion::kBoredom:      return "无聊";
    }
    return "未知";
}

/// ★v2: 情绪调节策略
enum class RegulationStrategy { kSuppression, kReappraisal, kDistraction, kAcceptance };

/// 情绪维度（Russell 环状模型 + 支配度 + ★v2 离散情感）
struct EmotionState {
    double valence = 0.0;        ///< 效价 -1~1（负→正）
    double arousal = 0.0;        ///< 唤醒度 0~1（平静→兴奋）
    double dominance = 0.5;      ///< 支配度 0~1（被动→主动）

    /// ★v2: 离散情感强度
    std::map<BasicEmotion, double> intensities;

    /// 离散情绪标签
    std::string label;           ///< "curious" / "frustrated" / "confident" / "bored" / "surprised"

    /// 强度（综合评分）
    [[nodiscard]] auto intensity() const -> double {
        return std::abs(valence) * 0.4 + arousal * 0.4 + std::abs(dominance - 0.5) * 0.2;
    }

    /// ★v2: 获取主导情感
    [[nodiscard]] auto dominant_emotion() const -> BasicEmotion {
        BasicEmotion dom = BasicEmotion::kCuriosity;
        double max_i = 0.0;
        for (const auto& [e, i] : intensities) {
            if (i > max_i) { max_i = i; dom = e; }
        }
        return dom;
    }
};

/// 多巴胺奖励预测误差信号
struct DopamineSignal {
    double prediction = 0.0;     ///< 预期奖励
    double actual = 0.0;         ///< 实际奖励
    double error = 0.0;          ///< RPE = actual - prediction

    /// 正 RPE → 意外之喜 → 加深记忆
    /// 负 RPE → 意外之失 → 也加深记忆（但方式不同）
    [[nodiscard]] auto is_positive() const -> bool { return error > 0; }
    [[nodiscard]] auto magnitude() const -> double { return std::abs(error); }
};

/// 记忆调制参数
struct MemoryModulation {
    double encoding_boost = 1.0;    ///< 编码增强因子
    double consolidation_boost = 1.0; ///< 巩固增强因子
    double retrieval_boost = 1.0;    ///< 提取增强因子
    double forgetting_rate = 0.01;   ///< 遗忘速率（高情绪 → 低遗忘）
    std::string reason;              ///< 调制原因描述
};

/// 情绪事件 — 触发情绪变化的学习事件
struct EmotionEvent {
    std::string event_type;         ///< "success"/"failure"/"surprise"/"progress"/"block"
    std::string domain;             ///< 相关领域
    std::string description;        ///< 事件描述
    double magnitude = 0.5;         ///< 事件强度
    double expected_outcome = 0.5;  ///< 预期结果
    double actual_outcome = 0.5;    ///< 实际结果
};

/// 学习策略建议
struct StrategySuggestion {
    std::string strategy;           ///< 建议策略
    double confidence = 0.0;        ///< 建议置信度
    std::string reason;             ///< 原因
    std::string expected_benefit;   ///< 预期收益
};

/// 情感引擎配置
struct EmotionConfig {
    double arousal_decay = 0.05;       ///< 唤醒度自然衰减率
    double valence_decay = 0.03;       ///< 效价自然衰减率
    double dopamine_decay = 0.1;       ///< 多巴胺衰减率
    double high_arousal_threshold = 0.7; ///< 高唤醒阈值
    double low_arousal_threshold = 0.2;  ///< 低唤醒阈值
    double optimal_arousal = 0.5;       ///< Yerkes-Dodson 最佳唤醒度
    int emotion_history_size = 50;      ///< 情绪历史长度
};

/// 情感驱动学习引擎
class EmotionEngine {
public:
    explicit EmotionEngine(
        const EmotionConfig& config = EmotionConfig{});

    // ── 情绪状态管理 ──────────────────────────────────────

    /// 处理一个情绪事件，更新情绪状态
    auto process_event(const EmotionEvent& event) -> EmotionState;

    /// 获取当前情绪状态
    [[nodiscard]] auto current_state() const -> const EmotionState& {
        return current_;
    }

    /// 自然衰减（每轮调用）
    void decay();

    /// 强制设置情绪（用于外部干预）
    void set_state(const EmotionState& state);

    // ── 多巴胺信号 ──────────────────────────────────────

    /// 计算奖励预测误差
    auto compute_rpe(double expected, double actual) -> DopamineSignal;

    /// 获取最近的多巴胺信号
    [[nodiscard]] auto recent_dopamine() const
        -> const std::deque<DopamineSignal>& {
        return dopamine_history_;
    }

    /// 获取平均多巴胺水平
    [[nodiscard]] auto dopamine_baseline() const -> double;

    // ── 记忆调制 ──────────────────────────────────────

    /// 根据当前情绪计算记忆调制参数
    auto compute_modulation() const -> MemoryModulation;

    /// 根据事件强度计算该事件的记忆调制
    auto event_modulation(const EmotionEvent& event) const -> MemoryModulation;

    // ── 学习策略适配 ──────────────────────────────────────

    /// 根据情绪建议学习策略
    auto suggest_strategy() const -> StrategySuggestion;

    /// 判断是否需要情绪调节
    [[nodiscard]] auto needs_regulation() const -> bool;

    /// 执行情绪调节（认知重评）
    auto regulate() -> EmotionState;

    /// ★v2: 指定策略的情绪调节
    auto regulate_with(RegulationStrategy strategy) -> EmotionState;

    // ── 查询 ──────────────────────────────────────────

    /// 获取情绪历史
    [[nodiscard]] auto emotion_history() const
        -> const std::deque<EmotionState>& {
        return emotion_history_;
    }

    /// 获取 Yerkes-Dodson 绩效预测
    [[nodiscard]] auto performance_prediction() const -> double;

    /// 获取情绪标签
    [[nodiscard]] auto emotion_label() const -> std::string;

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const EmotionConfig& {
        return config_;
    }

private:
    EmotionConfig config_;
    EmotionState current_;
    std::deque<EmotionState> emotion_history_;
    std::deque<DopamineSignal> dopamine_history_;

    // ── 内部方法 ──────────────────────────────────────

    /// 推断离散情绪标签
    static auto infer_label_(double valence, double arousal) -> std::string;

    /// 计算 Yerkes-Dodson 倒 U 型曲线
    auto yerkes_dodson_(double arousal) const -> double;

    /// 根据情绪计算编码增强因子
    auto encoding_boost_(double arousal, double valence) const -> double;

    /// 根据情绪计算巩固增强因子
    auto consolidation_boost_(double arousal) const -> double;
};

}  // namespace ai_learning::learning
