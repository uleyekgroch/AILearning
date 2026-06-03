/**
 * @file ipredictive_engine.hpp
 * @brief 预测引擎抽象接口
 *
 * 允许注入不同的预测编码实现（Hebbian/Light/Transformer等）。
 * 核心契约：给定 (obs, action) → 预测 next_obs；
 *           给定 (obs, action, actual) → 学习并返回误差。
 */
#pragma once

#include <string>
#include <vector>

namespace ai_learning::learning {

/// 预测引擎序列化状态（实现无关的扁平结构）
struct PredictiveEngineState {
    std::vector<float> weights;
    std::vector<int> shape;
};

/// 预测引擎抽象接口
class IPredictiveEngine {
public:
    virtual ~IPredictiveEngine() = default;

    /// 前向预测
    virtual auto predict(const std::vector<float>& state,
                         const std::vector<float>& action) const
        -> std::vector<float> = 0;

    /// 从预测误差中学习
    virtual auto learn(const std::vector<float>& obs,
                       const std::vector<float>& action,
                       const std::vector<float>& actual)
        -> double = 0;

    /// 好奇心值（预测误差的平滑度量）
    [[nodiscard]] virtual auto get_curiosity() const -> double = 0;

    /// 学习进度（误差下降趋势）
    [[nodiscard]] virtual auto get_learning_progress() const -> double = 0;

    /// 平均推理步数
    [[nodiscard]] virtual auto get_avg_inference_steps() const -> double = 0;

    /// 状态序列化
    [[nodiscard]] virtual auto save_state() const
        -> PredictiveEngineState = 0;

    /// 状态恢复
    virtual void load_state(const PredictiveEngineState& state) = 0;

    /// 引擎类型标识（用于监控和路由）
    [[nodiscard]] virtual auto engine_type() const -> std::string {
        return "unknown";
    }
};

}  // namespace ai_learning::learning
