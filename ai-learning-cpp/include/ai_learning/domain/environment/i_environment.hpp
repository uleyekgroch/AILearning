/**
 * @file i_environment.hpp
 * @brief 环境领域接口 — 统一的环境交互协议
 *
 * DDD 限界上下文：环境
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain {

class IEnvironment {
public:
    virtual ~IEnvironment() = default;

    /// 获取当前观测
    virtual auto observe() const
        -> std::map<std::string, std::vector<float>> = 0;

    /// 执行动作，返回奖励
    virtual auto step(int action)
        -> std::pair<double, bool> = 0;

    /// 重置环境
    virtual auto reset()
        -> std::map<std::string, std::vector<float>> = 0;

    /// 按发展阶段配置环境（可选覆盖）
    virtual void configure_for_stage(const std::string& stage) {
        (void)stage;  // 默认无操作
    }
};

}  // namespace ai_learning::domain
