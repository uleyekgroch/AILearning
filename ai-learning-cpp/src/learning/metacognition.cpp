/**
 * @file metacognition.cpp
 * @brief 元认知模块实现 — 知道自己知道什么、不知道什么
 *
 * 参考：
 *   - Nelson & Narens 元认知模型 (1990)：监控 + 控制
 *   - Active Inference (Friston 2025)：自由能 = 不确定性
 *   - Self-Evolving Embodied AI (arXiv 2602.04411)
 */

#include "ai_learning/learning/metacognition.hpp"
#include "ai_learning/core/learner.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace ai_learning::core {

// ── 构造 ──────────────────────────────────────────────────────────

MetacognitionEngine::MetacognitionEngine(double confidence_threshold)
    : confidence_threshold_(confidence_threshold) {}

// ── 知识监控（Nelson & Narens 监控层）─────────────────────────────

auto MetacognitionEngine::assess_confidence(const std::string& topic) const
    -> KnowledgeConfidence {
    auto it = confidence_registry_.find(topic);
    if (it == confidence_registry_.end()) {
        return KnowledgeConfidence{topic, 0.0, 1.0, 0, 0, ""};
    }

    auto& kc = it->second;
    // 校准后的置信度
    kc.confidence = calibrate_confidence_(kc);
    kc.uncertainty = 1.0 - kc.confidence;

    return kc;
}

void MetacognitionEngine::record_outcome(const std::string& topic, bool correct) {
    auto& kc = confidence_registry_[topic];
    kc.topic = topic;
    ++kc.times_used;
    if (correct) ++kc.times_correct;
    kc.confidence = calibrate_confidence_(kc);
    kc.uncertainty = 1.0 - kc.confidence;
}

auto MetacognitionEngine::assess_batch(const std::vector<std::string>& topics) const
    -> std::vector<KnowledgeConfidence> {
    std::vector<KnowledgeConfidence> results;
    results.reserve(topics.size());
    for (const auto& t : topics) {
        results.push_back(assess_confidence(t));
    }
    return results;
}

// ── 盲区检测 ──────────────────────────────────────────────────────

auto MetacognitionEngine::detect_gaps(const Learner& learner) const
    -> std::vector<KnowledgeGap> {
    std::vector<KnowledgeGap> gaps;

    // 核心知识领域（学习的先决条件）
    std::vector<std::string> core_topics = {
        "文本理解", "概念形成", "因果推理", "记忆巩固",
        "模式识别", "知识检索", "逻辑推理", "类比推理"
    };

    auto stats = learner.get_stats();

    // 基于当前知识图谱评估覆盖度
    double entity_count = stats.at("entity_count");
    double relation_count = stats.at("relation_count");
    double learning_progress = stats.at("learning_progress");

    // 知识图谱还很空 → 概念形成是盲区
    if (entity_count < 10) {
        gaps.push_back({
            "概念形成", 0.8,
            "知识图谱实体太少，需要更多概念",
            {"文本理解"},
            "通过更多文本输入积累概念"
        });
    }

    if (relation_count < 5) {
        gaps.push_back({
            "关系理解", 0.7,
            "知识图谱关系太少，需要理解概念间关系",
            {"概念形成"},
            "学习包含关系的文本"
        });
    }

    if (learning_progress < 0.3) {
        gaps.push_back({
            "预测能力", 0.6,
            "预测编码引擎尚未收敛",
            {"经验学习"},
            "增加经验学习轮次"
        });
    }

    // 检查每个核心主题
    for (const auto& topic : core_topics) {
        auto confidence = assess_confidence(topic);
        if (confidence.confidence < confidence_threshold_) {
            // 检查是否已在 gaps 中
            bool already = false;
            for (const auto& g : gaps) {
                if (g.topic == topic) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                gaps.push_back({
                    topic,
                    1.0 - confidence.confidence,
                    "置信度低于阈值 (" + std::to_string(confidence.confidence) + ")",
                    {},
                    "通过针对性学习提高"
                });
            }
        }
    }

    // 按紧迫性排序
    std::sort(gaps.begin(), gaps.end(),
        [](const KnowledgeGap& a, const KnowledgeGap& b) {
            return a.urgency > b.urgency;
        });

    return gaps;
}

auto MetacognitionEngine::topic_coverage(
    const std::string& domain,
    const std::vector<std::string>& required_topics,
    const Learner& learner) const
    -> std::map<std::string, double> {
    (void)domain;

    std::map<std::string, double> coverage;

    for (const auto& topic : required_topics) {
        // 检查知识图谱中是否有相关实体
        bool found_in_kg = false;
        // 简化：通过置信度注册表
        auto confidence = assess_confidence(topic);
        if (confidence.confidence >= confidence_threshold_) {
            coverage[topic] = confidence.confidence;
            found_in_kg = true;
        }

        if (!found_in_kg) {
            // 尝试通过 learner.think 检查
            auto answer = learner.think("什么是" + topic);
            if (answer.find(topic) != std::string::npos &&
                answer != "抱歉，我暂时不知道答案") {
                coverage[topic] = 0.5;  // 有一些知识但不确信
            } else {
                coverage[topic] = 0.0;
            }
        }
    }

    return coverage;
}

// ── 学习控制（Nelson & Narens 控制层）─────────────────────────────

auto MetacognitionEngine::evaluate_strategy(double recent_performance,
                                              double performance_trend) const
    -> StrategyAssessment {
    StrategyAssessment assessment;

    assessment.effectiveness = recent_performance;

    if (recent_performance > 0.8) {
        assessment.current_strategy = "当前策略";
        assessment.recommended_strategy = "保持当前策略";
        assessment.expected_gain = 0.0;
        assessment.reason = "性能良好，无需改变";
    } else if (performance_trend < -0.05) {
        assessment.current_strategy = "当前策略";
        assessment.recommended_strategy = "间隔重复 + 主动回忆";
        assessment.expected_gain = 0.2;
        assessment.reason = "性能下降，需要加强记忆巩固";
    } else if (recent_performance < 0.3) {
        assessment.current_strategy = "当前策略";
        assessment.recommended_strategy = "分解学习 + 试错法";
        assessment.expected_gain = 0.3;
        assessment.reason = "性能很低，建议从基础开始重建";
    } else {
        assessment.current_strategy = "当前策略";
        assessment.recommended_strategy = "类比迁移";
        assessment.expected_gain = 0.1;
        assessment.reason = "性能中等，通过类比加速学习";
    }

    return assessment;
}

auto MetacognitionEngine::generate_learning_plan(
    const std::vector<KnowledgeGap>& gaps) const
    -> std::vector<InformationNeed> {
    std::vector<InformationNeed> plan;
    plan.reserve(gaps.size());

    for (const auto& gap : gaps) {
        InformationNeed need;
        need.query = "学习关于 " + gap.topic + " 的知识";
        need.context = gap.reason;
        need.priority = gap.urgency;

        // 根据紧迫性决定来源类型
        if (gap.urgency > 0.8) {
            need.source_type = "experiment";
        } else if (!gap.related_known.empty()) {
            need.source_type = "analogy";
        } else {
            need.source_type = "text";
        }

        plan.push_back(need);
    }

    // 按优先级排序
    std::sort(plan.begin(), plan.end(),
        [](const InformationNeed& a, const InformationNeed& b) {
            return a.priority > b.priority;
        });

    return plan;
}

// ── 主动信息寻求 ──────────────────────────────────────────────────

auto MetacognitionEngine::should_seek_info(const std::string& topic) const
    -> std::optional<InformationNeed> {
    auto confidence = assess_confidence(topic);

    if (confidence.confidence >= confidence_threshold_) {
        return std::nullopt;  // 已经知道了
    }

    InformationNeed need;
    need.query = "深入了解 " + topic;
    need.context = "当前置信度: " + std::to_string(confidence.confidence);
    need.priority = 1.0 - confidence.confidence;

    if (confidence.times_used == 0) {
        need.source_type = "text";  // 全新主题，先看文本
    } else if (confidence.confidence < 0.3) {
        need.source_type = "experiment";  // 完全不懂，动手做
    } else {
        need.source_type = "analogy";  // 有点了解，用类比
    }

    return need;
}

auto MetacognitionEngine::generate_query(const KnowledgeGap& gap) const
    -> InformationNeed {
    InformationNeed need;
    need.query = gap.suggested_action;
    need.context = gap.reason;
    need.priority = gap.urgency;
    need.source_type = gap.suggested_action.find("实验") != std::string::npos
                           ? "experiment" : "text";
    return need;
}

// ── 综合报告 ──────────────────────────────────────────────────────

auto MetacognitionEngine::generate_report(const Learner& learner) const
    -> MetacognitiveReport {
    MetacognitiveReport report;

    auto stats = learner.get_stats();

    // 总体自信度
    double knowledge_score = stats.at("entity_count") / 100.0;
    double progress_score = stats.at("learning_progress");
    double curiosity_score = stats.at("curiosity");

    report.overall_confidence = std::min(1.0,
        (knowledge_score * 0.3 + progress_score * 0.4 + curiosity_score * 0.3));

    // 学习效率
    double steps = stats.at("total_steps");
    report.learning_efficiency = (steps > 0)
        ? progress_score / (steps / 100.0) : 0.0;
    report.learning_efficiency = std::min(1.0, report.learning_efficiency);

    // 盲区检测
    report.gaps = detect_gaps(learner);

    // 策略评估
    report.strategy = evaluate_strategy(report.overall_confidence, 0.0);

    // 信息需求
    report.needs = generate_learning_plan(report.gaps);

    // 维度评分
    report.dimension_scores = {
        {"knowledge_depth", knowledge_score},
        {"learning_progress", progress_score},
        {"curiosity", curiosity_score},
        {"memory_strength", stats.at("hippocampal_episodes") / 100.0},
        {"consolidation", stats.at("cortical_facts") / 50.0},
    };

    return report;
}

auto MetacognitionEngine::knows_about(const std::string& topic) const
    -> bool {
    auto confidence = assess_confidence(topic);
    return confidence.confidence >= confidence_threshold_;
}

// ── 配置 ──────────────────────────────────────────────────────────

void MetacognitionEngine::set_confidence_threshold(double threshold) {
    confidence_threshold_ = std::clamp(threshold, 0.0, 1.0);
}

// ── Private ───────────────────────────────────────────────────────

auto MetacognitionEngine::calibrate_confidence_(const KnowledgeConfidence& kc) const
    -> double {
    if (kc.times_used == 0) return 0.0;

    // 基础正确率
    double base_rate = static_cast<double>(kc.times_correct) /
                       static_cast<double>(kc.times_used);

    // 使用次数越多，置信度校准越准（快速收敛）
    double calibration = std::min(1.0, kc.times_used / 3.0);

    // 校准后的置信度
    return base_rate * calibration;
}

auto MetacognitionEngine::estimate_relatedness_(
    const std::string& topic_a,
    const std::string& topic_b,
    const Learner& learner) const -> double {
    (void)learner;

    // 简化的语义相似度：基于字符串重叠
    if (topic_a == topic_b) return 1.0;

    // 共同字符数
    int common = 0;
    for (char c : topic_a) {
        if (topic_b.find(c) != std::string::npos) ++common;
    }

    double similarity = static_cast<double>(common) /
                        static_cast<double>(std::max(topic_a.size(), topic_b.size()));

    return similarity;
}

}  // namespace ai_learning::core
