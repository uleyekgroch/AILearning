/**
 * @file social_agent.hpp
 * @brief 社会智能体 — 观察学习、教学、协作交互
 *
 * 实现 ISocialAgent 接口。Agent 通过观察伙伴行为学习，
 * 可以教授其他 Agent 特定主题知识。
 */
#pragma once

#include "ai_learning/domain/social/i_social_agent.hpp"

#include <map>
#include <string>
#include <vector>

// 前向声明
namespace ai_learning::core { class Learner; }

namespace ai_learning::social {

/// 社会交互结果
struct InteractionResult {
    double knowledge_gained = 0.0;
    int    facts_transferred = 0;
    double teaching_effectiveness = 0.0;
};

/// 社会智能体
class SocialAgent : public domain::ISocialAgent {
public:
    /// 构造，绑定到学习者
    explicit SocialAgent(core::Learner& learner);

    /// 与伙伴交互：互相学习
    auto interact(ISocialAgent& partner)
        -> std::map<std::string, double> override;

    /// 观察伙伴行为和结果
    void observe_partner(
        const std::vector<float>& partner_action,
        const std::map<std::string, double>& partner_outcome) override;

    /// 教授其他 Agent 特定主题
    auto teach(ISocialAgent& learner, const std::string& topic) const
        -> std::map<std::string, double> override;

    /// 获取累计交互统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 向 Agent 传授文本知识（供 teach 调用）
    auto receive_knowledge(const std::string& text)
        -> std::map<std::string, double>;

private:
    core::Learner& learner_;
    int interaction_count_ = 0;
    int observation_count_ = 0;
    int teaching_count_ = 0;
    double total_knowledge_gained_ = 0.0;

    /// 社会学习课程内容
    static const std::vector<std::string>& kTeachingTopics();
};

}  // namespace ai_learning::social
