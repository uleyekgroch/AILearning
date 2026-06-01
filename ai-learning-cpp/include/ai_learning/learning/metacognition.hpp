/**
 * @file metacognition.hpp
 * @brief 元认知模块 — "知道自己不知道什么"的能力
 *
 * 参考：
 *   - Active Inference (Friston 2025)：自由能 = 不确定性
 *   - Nelson & Narens 元认知模型 (1990)：监控 + 控制
 *   - Self-Evolving Embodied AI (arXiv 2602.04411)
 *
 * 能力 4：元认知（Metacognition）
 * 人类能说"我不懂这个，我需要换个方法学"。
 * 本模块实现：
 *   1. 知识不确定性估计 — "我有多确信？"
 *   2. 盲区检测 — "我不知道什么？"
 *   3. 学习策略选择 — "该怎么学？"
 *   4. 主动信息寻求 — "我需要什么信息？"
 */
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::core {

// 前向声明
class Learner;

/// 知识置信度
struct KnowledgeConfidence {
    std::string topic;                ///< 知识主题
    double confidence = 0.0;          ///< 置信度 0~1
    double uncertainty = 1.0;         ///< 不确定性 0~1
    int times_used = 0;               ///< 使用次数
    int times_correct = 0;            ///< 正确次数
    std::string last_source;          ///< 最后来源
};

/// 知识盲区
struct KnowledgeGap {
    std::string topic;                ///< 缺失的主题
    double urgency = 0.0;             ///< 紧迫性 0~1
    std::string reason;               ///< 为什么需要
    std::vector<std::string> related_known;  ///< 相关已知知识
    std::string suggested_action;     ///< 建议的获取方式
};

/// 学习策略评估
struct StrategyAssessment {
    std::string current_strategy;     ///< 当前策略
    double effectiveness = 0.0;       ///< 效果评分 0~1
    std::string recommended_strategy; ///< 推荐策略
    double expected_gain = 0.0;       ///< 预期提升
    std::string reason;               ///< 原因
};

/// 主动信息需求
struct InformationNeed {
    std::string query;                ///< 需要查询什么
    std::string context;              ///< 上下文
    double priority = 0.0;            ///< 优先级
    std::string source_type;          ///< 建议来源类型：text / experiment / analogy
};

/// 元认知监控报告
struct MetacognitiveReport {
    double overall_confidence = 0.0;  ///< 总体自信度
    double learning_efficiency = 0.0; ///< 学习效率
    std::vector<KnowledgeGap> gaps;   ///< 检测到的盲区
    StrategyAssessment strategy;      ///< 策略评估
    std::vector<InformationNeed> needs;  ///< 信息需求
    std::map<std::string, double> dimension_scores;  ///< 各维度评分
};

/// 元认知引擎
class MetacognitionEngine {
public:
    explicit MetacognitionEngine(double confidence_threshold = 0.5);

    // ── 知识监控（Nelson & Narens 监控层）─────────

    /// 评估对某个主题的置信度
    auto assess_confidence(const std::string& topic) const
        -> KnowledgeConfidence;

    /// 记录知识使用结果（用于校准置信度）
    void record_outcome(const std::string& topic, bool correct);

    /// 批量评估多个主题
    auto assess_batch(const std::vector<std::string>& topics) const
        -> std::vector<KnowledgeConfidence>;

    // ── 盲区检测 ──────────────────────────────────────

    /// 检测知识盲区
    auto detect_gaps(const Learner& learner) const
        -> std::vector<KnowledgeGap>;

    /// 评估主题的覆盖度（0~1）
    auto topic_coverage(const std::string& domain,
                         const std::vector<std::string>& required_topics,
                         const Learner& learner) const
        -> std::map<std::string, double>;

    // ── 学习控制（Nelson & Narens 控制层）─────────

    /// 评估当前学习策略效果
    auto evaluate_strategy(double recent_performance,
                           double performance_trend) const
        -> StrategyAssessment;

    /// 生成学习计划（应该学什么、怎么学）
    auto generate_learning_plan(const std::vector<KnowledgeGap>& gaps) const
        -> std::vector<InformationNeed>;

    // ── 主动信息寻求 ──────────────────────────────────

    /// 决定是否需要更多信息
    auto should_seek_info(const std::string& topic) const
        -> std::optional<InformationNeed>;

    /// 生成信息寻求查询
    auto generate_query(const KnowledgeGap& gap) const
        -> InformationNeed;

    // ── 综合报告 ──────────────────────────────────────

    /// 生成完整元认知报告
    auto generate_report(const Learner& learner) const
        -> MetacognitiveReport;

    /// 获取"我是否知道"的判断
    auto knows_about(const std::string& topic) const
        -> bool;

    // ── 配置 ──────────────────────────────────────────

    /// 设置置信度阈值
    void set_confidence_threshold(double threshold);

    /// 获取置信度阈值
    [[nodiscard]] auto confidence_threshold() const -> double {
        return confidence_threshold_;
    }

    /// 获取所有已记录的置信度
    [[nodiscard]] auto confidence_registry() const
        -> const std::map<std::string, KnowledgeConfidence>& {
        return confidence_registry_;
    }

private:
    double confidence_threshold_;
    mutable std::map<std::string, KnowledgeConfidence> confidence_registry_;

    /// 从经验中校准置信度
    auto calibrate_confidence_(const KnowledgeConfidence& kc) const
        -> double;

    /// 估算主题间关联度
    auto estimate_relatedness_(const std::string& topic_a,
                                const std::string& topic_b,
                                const Learner& learner) const
        -> double;
};

}  // namespace ai_learning::core
