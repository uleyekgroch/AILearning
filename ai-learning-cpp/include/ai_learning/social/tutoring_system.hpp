/**
 * @file tutoring_system.hpp
 * @brief 教学相长系统 — 通过教别人来加深自己的理解
 *
 * 理论基础：
 *   - Protégé Effect (Chase et al., 2009): 教别人能加深自己的理解
 *   - Vygotsky ZPD: 最近发展区 — 在指导下学习
 *   - Reciprocal Teaching (Palincsar & Brown, 1984): 互惠教学
 *   - Learning by Teaching (Biswas et al., 2005): 教中学
 *
 * 核心机制：
 *   1. 教中学 (Learn by Teaching): 向他人解释 → 发现自己理解不足 → 深化
 *   2. 学中教 (Teach while Learning): 刚学会立即教 → 巩固
 *   3. 互惠教学 (Reciprocal): 轮流当老师和学生
 *   4. 脚手架 (Scaffolding): 根据学习者水平调整教学难度
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::social {

/// 教学能力级别
enum class TutoringLevel { kNovice = 0, kApprentice = 1, kJourneyman = 2, kMaster = 3 };

/// 知识单元
struct KnowledgeUnit {
    std::string topic;                ///< 主题
    std::string explanation;          ///< 解释
    double mastery = 0.0;            ///< 掌握度 0~1
    int times_taught = 0;            ///< 教授次数
    int times_learned = 0;           ///< 学习次数
    std::vector<std::string> prerequisites;  ///< 前置知识
    std::vector<std::string> common_misconceptions;  ///< 常见误解
};

/// 学习者模型
struct LearnerModel {
    std::string learner_id;           ///< 学习者ID
    std::map<std::string, double> knowledge;  ///< 各主题掌握度
    double motivation = 0.5;          ///< 学习动机
    double attention = 0.5;           ///< 注意力
    std::string preferred_style;      ///< 偏好学习风格
    std::vector<std::string> strengths;
    std::vector<std::string> weaknesses;
};

/// 教学结果
struct TutoringResult {
    std::string topic;                ///< 教授的主题
    std::string tutor_id;             ///< 教师ID
    std::string student_id;           ///< 学生ID
    double knowledge_gain = 0.0;     ///< 学生的知识增长
    double tutor_gain = 0.0;         ///< 教师的知识增长（教学相长）
    std::string student_feedback;     ///< 学生反馈
    std::string tutor_reflection;     ///< 教师反思
    bool misconceptions_corrected = false;
};

/// 教学相长系统配置
struct TutoringConfig {
    double protégé_effect_gain = 0.15;     ///< 教学相长的增益系数
    double zpd_width = 0.2;                ///< ZPD 宽度
    int max_explanation_attempts = 3;       ///< 最大解释尝试次数
    bool enable_reciprocal = true;          ///< 启用互惠教学
};

/// 教学相长系统
class TutoringSystem {
public:
    explicit TutoringSystem(const TutoringConfig& config = TutoringConfig{});

    // ═══════════════════════════════════════════════════════════
    // 教中学 (Learn by Teaching)
    // ═══════════════════════════════════════════════════════════

    /// 向学习者解释一个主题（教学相长：教的人也会进步）
    auto teach(const KnowledgeUnit& topic,
               const LearnerModel& student,
               const std::string& tutor_id)
        -> TutoringResult;

    /// 准备教学材料（迫使教师深入理解）
    auto prepare_lesson(const std::string& topic,
                        const LearnerModel& student)
        -> KnowledgeUnit;

    // ═══════════════════════════════════════════════════════════
    // 学中教 (Teach while Learning)
    // ═══════════════════════════════════════════════════════════

    /// 刚学会就教给别人（巩固效应）
    auto teach_immediately_after_learning(
        const KnowledgeUnit& just_learned,
        const LearnerModel& peer)
        -> TutoringResult;

    // ═══════════════════════════════════════════════════════════
    // 互惠教学 (Reciprocal Teaching)
    // ═══════════════════════════════════════════════════════════

    /// 互惠教学会话
    auto reciprocal_session(
        const std::string& topic,
        const LearnerModel& learner_a,
        const LearnerModel& learner_b)
        -> std::pair<TutoringResult, TutoringResult>;

    // ═══════════════════════════════════════════════════════════
    // 脚手架 (Scaffolding)
    // ═══════════════════════════════════════════════════════════

    /// 为学习者搭建教学脚手架
    auto scaffold(const KnowledgeUnit& topic,
                  const LearnerModel& student) const
        -> std::vector<std::string>;  ///< 教学步骤

    /// 确定最近发展区
    auto determine_zpd(const LearnerModel& student) const
        -> std::vector<std::string>;  ///< 适合学习的主题

    /// 评估教学难度是否适合学习者
    [[nodiscard]] auto is_in_zpd(const KnowledgeUnit& topic,
                                  const LearnerModel& student) const -> bool;

    // ═══════════════════════════════════════════════════════════
    // 评估
    // ═══════════════════════════════════════════════════════════

    /// 评估教学效果
    auto assess_teaching_effectiveness(const TutoringResult& result) const
        -> double;

    /// 评估教师的教学能力
    [[nodiscard]] auto assess_tutoring_level(
        const std::string& tutor_id) const -> TutoringLevel;

    // ═══════════════════════════════════════════════════════════
    // 查询
    // ═══════════════════════════════════════════════════════════

    [[nodiscard]] auto total_sessions() const -> int {
        return total_sessions_;
    }

    [[nodiscard]] auto protégé_effect_strength() const -> double {
        return config_.protégé_effect_gain;
    }

private:
    TutoringConfig config_;
    int total_sessions_ = 0;

    /// 教师教学记录
    std::map<std::string, std::vector<TutoringResult>> teaching_history_;

    /// 检测常见误解
    auto detect_misconceptions_(const LearnerModel& student,
                                const KnowledgeUnit& topic) const
        -> std::vector<std::string>;

    /// 生成个性化解释
    auto generate_explanation_(const KnowledgeUnit& topic,
                               const LearnerModel& student) const
        -> std::string;
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline TutoringSystem::TutoringSystem(const TutoringConfig& config)
    : config_(config) {}

inline auto TutoringSystem::teach(
    const KnowledgeUnit& topic,
    const LearnerModel& student,
    const std::string& tutor_id)
    -> TutoringResult {
    TutoringResult result;
    result.topic = topic.topic;
    result.tutor_id = tutor_id;
    result.student_id = student.learner_id;

    // 学生是否在 ZPD 内？
    if (!is_in_zpd(topic, student)) {
        result.student_feedback = "太难了，需要先学前置知识";
        result.tutor_reflection = "应该先搭建更好的脚手架";
        return result;
    }

    // 检测并纠正误解
    auto misconceptions = detect_misconceptions_(student, topic);
    result.misconceptions_corrected = !misconceptions.empty();

    // 学生知识增长
    double prior_knowledge = 0.0;
    auto it = student.knowledge.find(topic.topic);
    if (it != student.knowledge.end()) prior_knowledge = it->second;
    result.knowledge_gain = std::min(0.3, (1.0 - prior_knowledge) * 0.4);

    // 教学相长效应：教师也获得知识增益
    result.tutor_gain = result.knowledge_gain * config_.protégé_effect_gain;

    // 反馈和反思
    result.student_feedback = result.knowledge_gain > 0.1
        ? "理解了！谢谢老师" : "还是不太明白...";
    result.tutor_reflection = "教" + topic.topic + "让我发现自己的理解"
        + (result.tutor_gain > 0.05 ? "还不够深入" : "比较扎实");

    teaching_history_[tutor_id].push_back(result);
    total_sessions_++;
    return result;
}

inline auto TutoringSystem::prepare_lesson(
    const std::string& topic,
    const LearnerModel& student)
    -> KnowledgeUnit {
    KnowledgeUnit lesson;
    lesson.topic = topic;
    lesson.explanation = generate_explanation_(lesson, student);
    lesson.prerequisites = scaffold(lesson, student);
    return lesson;
}

inline auto TutoringSystem::teach_immediately_after_learning(
    const KnowledgeUnit& just_learned,
    const LearnerModel& peer)
    -> TutoringResult {
    // 学后即教：增益加倍（巩固效应）
    auto result = teach(just_learned, peer, "self");
    result.tutor_gain *= 2.0;  // 巩固效应翻倍
    result.tutor_reflection += "（刚学会就教，理解更深了）";
    return result;
}

inline auto TutoringSystem::reciprocal_session(
    const std::string& topic,
    const LearnerModel& learner_a,
    const LearnerModel& learner_b)
    -> std::pair<TutoringResult, TutoringResult> {
    KnowledgeUnit unit;
    unit.topic = topic;

    // A 教 B
    auto result_a = teach(unit, learner_b, learner_a.learner_id);
    // B 教 A（互惠）
    auto result_b = teach(unit, learner_a, learner_b.learner_id);

    return {result_a, result_b};
}

inline auto TutoringSystem::scaffold(
    const KnowledgeUnit& topic,
    const LearnerModel& student) const
    -> std::vector<std::string> {
    std::vector<std::string> steps;

    // 1. 检查前置知识
    for (const auto& prereq : topic.prerequisites) {
        auto it = student.knowledge.find(prereq);
        if (it == student.knowledge.end() || it->second < 0.5) {
            steps.push_back("先学" + prereq);
        }
    }

    // 2. 从简单到复杂
    steps.push_back("介绍" + topic.topic + "的基本概念");
    steps.push_back("通过例子演示" + topic.topic);
    steps.push_back("让学习者自己尝试");
    steps.push_back("纠正误解并总结");

    return steps;
}

inline auto TutoringSystem::determine_zpd(
    const LearnerModel& student) const
    -> std::vector<std::string> {
    std::vector<std::string> zpd_topics;
    for (const auto& [topic, mastery] : student.knowledge) {
        // ZPD = 掌握度在 0.3~0.7 之间的主题
        if (mastery > 0.3 && mastery < 0.7) {
            zpd_topics.push_back(topic);
        }
    }
    return zpd_topics;
}

inline auto TutoringSystem::is_in_zpd(
    const KnowledgeUnit& topic,
    const LearnerModel& student) const -> bool {
    auto it = student.knowledge.find(topic.topic);
    double mastery = (it != student.knowledge.end()) ? it->second : 0.0;
    return mastery >= 0.2 && mastery <= 0.8;
}

inline auto TutoringSystem::assess_teaching_effectiveness(
    const TutoringResult& result) const -> double {
    return result.knowledge_gain * 0.6 + result.tutor_gain * 0.4;
}

inline auto TutoringSystem::assess_tutoring_level(
    const std::string& tutor_id) const -> TutoringLevel {
    auto it = teaching_history_.find(tutor_id);
    if (it == teaching_history_.end()) return TutoringLevel::kNovice;

    int sessions = static_cast<int>(it->second.size());
    double avg_effectiveness = 0.0;
    for (const auto& r : it->second) {
        avg_effectiveness += assess_teaching_effectiveness(r);
    }
    avg_effectiveness /= sessions;

    if (sessions > 50 && avg_effectiveness > 0.6) return TutoringLevel::kMaster;
    if (sessions > 20 && avg_effectiveness > 0.4) return TutoringLevel::kJourneyman;
    if (sessions > 5) return TutoringLevel::kApprentice;
    return TutoringLevel::kNovice;
}

inline auto TutoringSystem::detect_misconceptions_(
    const LearnerModel& student,
    const KnowledgeUnit& topic) const
    -> std::vector<std::string> {
    std::vector<std::string> found;
    for (const auto& mc : topic.common_misconceptions) {
        if (student.weaknesses.end() != std::find(
                student.weaknesses.begin(), student.weaknesses.end(), mc)) {
            found.push_back(mc);
        }
    }
    return found;
}

inline auto TutoringSystem::generate_explanation_(
    const KnowledgeUnit& topic,
    const LearnerModel& student) const -> std::string {
    if (student.preferred_style == "visual") {
        return "想象" + topic.topic + "像一个...";
    } else if (student.preferred_style == "hands-on") {
        return "让我们动手试试" + topic.topic;
    }
    return topic.explanation;
}

}  // namespace ai_learning::social