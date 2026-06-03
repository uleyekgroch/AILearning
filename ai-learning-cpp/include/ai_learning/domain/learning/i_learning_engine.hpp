/**
 * @file i_learning_engine.hpp
 * @brief 学习引擎接口 — 预测编码核心
 *
 * DDD 限界上下文：学习引擎
 */
#pragma once

#include <vector>

namespace ai_learning::domain {

class ILearningEngine {
public:
    virtual ~ILearningEngine() = default;

    /// 预测下一个状态
    virtual auto predict(const std::vector<float>& state) const
        -> std::vector<float> = 0;

    /// 从预测误差中学习，返回预测误差（MSE）
    virtual auto learn(const std::vector<float>& predicted,
                       const std::vector<float>& actual) -> double = 0;

    /// 计算好奇心值（内在奖励 = 预测误差 × 可学习性）
    virtual auto get_curiosity(const std::vector<float>& state) const
        -> double = 0;
};

}  // namespace ai_learning::domain
