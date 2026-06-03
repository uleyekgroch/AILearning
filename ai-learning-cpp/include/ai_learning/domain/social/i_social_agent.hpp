/**
 * @file i_social_agent.hpp
 * @brief 社会领域接口 — 多 Agent 交互协议
 *
 * DDD 限界上下文：社会
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain {

class IEnvironment;  // 前向声明

class ISocialAgent {
public:
    virtual ~ISocialAgent() = default;

    /// 与另一个 Agent 在环境中交互
    virtual auto interact(ISocialAgent& partner)
        -> std::map<std::string, double> = 0;

    /// 观察伙伴的行为和结果（社会学习）
    virtual void observe_partner(
        const std::vector<float>& partner_action,
        const std::map<std::string, double>& partner_outcome) = 0;

    /// 向另一个 Agent 教授特定主题
    virtual auto teach(ISocialAgent& learner, const std::string& topic) const
        -> std::map<std::string, double> = 0;
};

}  // namespace ai_learning::domain
