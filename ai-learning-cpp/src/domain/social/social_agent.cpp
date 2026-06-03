/**
 * @file social_agent.cpp
 * @brief 社会智能体实现
 */

#include "ai_learning/domain/social/social_agent.hpp"
#include "ai_learning/core/learner.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace ai_learning::social {

SocialAgent::SocialAgent(core::Learner& learner)
    : learner_(learner) {}

auto SocialAgent::interact(ISocialAgent& partner)
    -> std::map<std::string, double> {
    // 双向知识交换
    // 1. 从伙伴处学习
    auto partner_stats_before = partner.teach(*this, "general");

    // 2. 向伙伴教授
    auto my_teach_result = teach(partner, "general");

    ++interaction_count_;
    double gained = 0.0;
    if (partner_stats_before.contains("facts_transferred")) {
        gained = partner_stats_before.at("facts_transferred") * 0.5;
    }
    total_knowledge_gained_ += gained;

    return {
        {"interactions", static_cast<double>(interaction_count_)},
        {"knowledge_gained", gained},
        {"taught_facts", my_teach_result.contains("facts_transferred")
            ? my_teach_result.at("facts_transferred") : 0.0},
    };
}

void SocialAgent::observe_partner(
    const std::vector<float>& partner_action,
    const std::map<std::string, double>& partner_outcome) {

    ++observation_count_;

    // 社会学习：将伙伴的高奖励行为内化
    if (partner_outcome.contains("reward") &&
        partner_outcome.at("reward") > 0.5) {

        // 将伙伴动作转化为文本描述并学习
        int best_action = 0;
        if (!partner_action.empty()) {
            auto max_it = std::max_element(
                partner_action.begin(), partner_action.end());
            best_action = static_cast<int>(
                std::distance(partner_action.begin(), max_it));
        }

        // 根据动作类型学习
        std::string action_desc;
        switch (best_action) {
            case 0: action_desc = "向上移动是一种有效策略"; break;
            case 1: action_desc = "向下移动是一种有效策略"; break;
            case 2: action_desc = "向左移动是一种有效策略"; break;
            case 3: action_desc = "向右移动是一种有效策略"; break;
            default: action_desc = "探索是一种有效策略"; break;
        }

        auto result = learner_.learn_from_text(action_desc, "social_observation");
        total_knowledge_gained_ += static_cast<double>(result.entities.size()) * 0.1;
    }
}

auto SocialAgent::teach(ISocialAgent& learner_agent,
                         const std::string& topic) const
    -> std::map<std::string, double> {
    // 根据主题选择教学内容
    std::string content;
    if (topic == "general" || topic.empty()) {
        // 通用教学：传授系统已有知识
        auto stats = learner_.get_stats();
        content = "知识是学习的基础";
    } else {
        content = topic + "是重要的知识领域";
    }

    // 通过接口传授
    auto* social_learner = dynamic_cast<SocialAgent*>(&learner_agent);
    if (social_learner != nullptr) {
        auto result = social_learner->receive_knowledge(content);
        return {
            {"facts_transferred", result.contains("entities")
                ? result.at("entities") : 0.0},
            {"teaching_effectiveness", result.contains("score")
                ? result.at("score") : 0.5},
        };
    }

    return {{"facts_transferred", 0.0}, {"teaching_effectiveness", 0.0}};
}

auto SocialAgent::receive_knowledge(const std::string& text)
    -> std::map<std::string, double> {
    auto result = learner_.learn_from_text(text, "social_teaching");

    return {
        {"entities", static_cast<double>(result.entities.size())},
        {"triples", static_cast<double>(result.triples.size())},
        {"score", result.verification_passed ? 1.0 : 0.5},
    };
}

auto SocialAgent::stats() const -> std::map<std::string, double> {
    return {
        {"interactions", static_cast<double>(interaction_count_)},
        {"observations", static_cast<double>(observation_count_)},
        {"teaching_count", static_cast<double>(teaching_count_)},
        {"total_knowledge_gained", total_knowledge_gained_},
    };
}

const std::vector<std::string>& SocialAgent::kTeachingTopics() {
    static const std::vector<std::string> topics = {
        "数学是研究数量和结构的学科",
        "物理是研究自然规律的科学",
        "化学是研究物质变化的学科",
        "生物是研究生命现象的科学",
        "语言是人类交流的工具",
    };
    return topics;
}

}  // namespace ai_learning::social
