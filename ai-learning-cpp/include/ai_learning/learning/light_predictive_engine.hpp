/**
 * @file light_predictive_engine.hpp
 * @brief 轻量级单层预测编码引擎
 *
 * 与完整 PredictiveCodingEngine 的区别：
 * - 仅单层隐藏层（无 hidden2）
 * - 无迭代推理（单次前向+局部误差）
 * - 权重更新仍是 Hebbian 风格但更简单
 *
 * 适用场景：小型网络、快速原型、嵌入式环境。
 */
#pragma once

#include "ai_learning/learning/ipredictive_engine.hpp"
#include "ai_learning/core/types.hpp"

#include <deque>

namespace ai_learning::learning {

/// 轻量预测编码配置
struct LightPredictiveConfig {
    int    obs_dim     = 128;
    int    action_dim  = 8;
    int    hidden_dim  = 64;
    double learning_rate = 0.001;
    double clip_value  = 3.0;
};

/// 轻量级单层预测编码引擎
class LightPredictiveEngine : public IPredictiveEngine {
public:
    explicit LightPredictiveEngine(const LightPredictiveConfig& cfg);

    auto predict(const std::vector<float>& state,
                 const std::vector<float>& action) const
        -> std::vector<float> override;

    auto learn(const std::vector<float>& obs,
               const std::vector<float>& action,
               const std::vector<float>& actual)
        -> double override;

    [[nodiscard]] auto get_curiosity() const -> double override;
    [[nodiscard]] auto get_learning_progress() const -> double override;
    [[nodiscard]] auto get_avg_inference_steps() const -> double override;

    [[nodiscard]] auto save_state() const
        -> PredictiveEngineState override;
    void load_state(const PredictiveEngineState& state) override;
    [[nodiscard]] auto engine_type() const -> std::string override {
        return "light";
    }

private:
    auto encode_action_(const std::vector<float>& action) const
        -> std::vector<float>;

    LightPredictiveConfig cfg_;
    int input_dim_;

    ai_learning::core::Matrix w1_;  // (hidden × input)
    std::vector<float> b1_;
    ai_learning::core::Matrix w2_;  // (obs × hidden)
    std::vector<float> b2_;

    std::deque<float> error_history_;
    std::deque<float> prediction_errors_;
    double learning_progress_ = 0.0;
};

}  // namespace ai_learning::learning
