/**
 * @file mlp_forward_engine.hpp
 * @brief 标准前馈 MLP 预测引擎 — SGD 反向传播基线
 *
 * 与 PredictiveCodingEngine 的区别：
 * - 无迭代推理（单次前向）
 * - 标准反向传播而非 Hebbian 局部更新
 * - 通常更快但可能泛化较差
 *
 * 用于对比：当预测编码的迭代推理开销不值得时，
 * MLPForwardEngine 是更快的替代方案。
 */
#pragma once

#include "ai_learning/learning/ipredictive_engine.hpp"
#include "ai_learning/core/types.hpp"

#include <deque>
#include <random>

namespace ai_learning::learning {

/// MLP 前馈引擎配置
struct MLPForwardConfig {
    int    obs_dim     = 128;
    int    action_dim  = 8;
    int    hidden1_dim = 256;
    int    hidden2_dim = 128;
    double learning_rate = 0.001;
    double clip_value  = 3.0;
};

/// 标准前馈 MLP 预测引擎
class MLPForwardEngine : public IPredictiveEngine {
public:
    explicit MLPForwardEngine(const MLPForwardConfig& cfg);

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
        return "mlp";
    }

private:
    /// 编码动作为 one-hot/连续向量
    auto encode_action_(const std::vector<float>& action) const
        -> std::vector<float>;

    /// 前向传播（返回中间激活，用于反向传播）
    struct ForwardResult {
        std::vector<float> input;
        std::vector<float> z1, h1;
        std::vector<float> z2, h2;
        std::vector<float> output;
    };
    auto forward_(const std::vector<float>& input) const
        -> ForwardResult;

    MLPForwardConfig cfg_;
    int input_dim_;

    // 权重
    ai_learning::core::Matrix w1_;  // (hidden1 × input)
    std::vector<float> b1_;
    ai_learning::core::Matrix w2_;  // (hidden2 × hidden1)
    std::vector<float> b2_;
    ai_learning::core::Matrix w3_;  // (obs × hidden2)
    std::vector<float> b3_;

    // 统计
    std::deque<float> error_history_;
    std::deque<float> prediction_errors_;
    double learning_progress_ = 0.0;
};

}  // namespace ai_learning::learning
