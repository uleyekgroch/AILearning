/**
 * @file predictive_coding_engine.hpp
 * @brief 预测编码学习引擎 — C++20 实现
 *
 * 核心算法（Whittington & Bogacz, 2017）：
 * 1. 前向初始化信念 μ
 * 2. 迭代推理：局部残差 + 反馈连接 → 收敛
 * 3. Hebbian 权重更新：ΔW = η × ε_post × f' × μ_pre^T
 *
 * 对应 Python 版 PredictiveCodingEngine。
 */
#pragma once

#include "ai_learning/learning/ipredictive_engine.hpp"

#include <vector>
#include <deque>

namespace ai_learning::learning {

/// 预测编码引擎配置
struct PredictiveCodingConfig {
    int    obs_dim               = 128;
    int    action_dim            = 8;
    int    hidden1_dim           = 256;
    int    hidden2_dim           = 128;
    double learning_rate         = 0.001;
    double inference_lr          = 0.05;
    int    max_inference_steps   = 50;
    double convergence_threshold = 1e-4;
    double clip_value            = 3.0;
    double curiosity_alpha       = 0.5;
    double curiosity_beta        = 0.5;
};

/// 推理中间结果
struct InferenceResult {
    std::vector<float> input;
    std::vector<float> mu_h1;
    std::vector<float> mu_h2;
    std::vector<float> z1;
    std::vector<float> z2;
    std::vector<float> epsilon_out;
    std::vector<float> epsilon_h1;
    std::vector<float> epsilon_h2;
    int steps_taken = 0;
};

/// 预测编码引擎
///
/// 架构：input(obs+action) → hidden1 → hidden2 → output(obs')
/// 激活：ReLU
/// 学习：预测编码 + 局部 Hebbian 更新
class PredictiveCodingEngine : public IPredictiveEngine {
public:
    explicit PredictiveCodingEngine(const PredictiveCodingConfig& cfg);

    /// 前向传播：预测下一个状态
    auto predict(const std::vector<float>& state,
                 const std::vector<float>& action) const
        -> std::vector<float> override;

    /// 从预测误差中学习（预测编码核心）
    /// @return 预测误差 (MSE)
    auto learn(const std::vector<float>& obs,
               const std::vector<float>& action,
               const std::vector<float>& actual)
        -> double override;

    /// 计算好奇心值
    [[nodiscard]] auto get_curiosity() const -> double override;

    /// 获取学习进度
    [[nodiscard]] auto get_learning_progress() const -> double override;

    /// 获取平均推理步数
    [[nodiscard]] auto get_avg_inference_steps() const -> double override;

    // ── 状态访问 ──
    [[nodiscard]] auto error_history() const
        -> const std::deque<float>& { return error_history_; }

    /// 详细状态（内部结构，保留完整权重/偏置）
    struct DetailedState {
        std::vector<float> w1, b1, w2, b2, w3, b3;
    };
    [[nodiscard]] auto save_detailed_state() const -> DetailedState;
    void load_detailed_state(const DetailedState& state);

    // ── IPredictiveEngine 接口实现 ──
    [[nodiscard]] auto save_state() const
        -> PredictiveEngineState override;
    void load_state(const PredictiveEngineState& state) override;

private:
    /// 编码动作为向量
    auto encode_action_(const std::vector<float>& action) const
        -> std::vector<float>;

    /// 迭代推理收敛
    auto infer_beliefs_(const std::vector<float>& input,
                         const std::vector<float>& actual)
        -> InferenceResult;

    /// Hebbian 权重更新
    void update_weights_(const InferenceResult& result);

    /// 检查收敛
    static auto check_convergence_(const std::vector<float>& prev,
                                    const std::vector<float>& curr,
                                    double threshold) -> double;

    // 网络权重（flat row-major）
    std::vector<float> w1_;  // (input_dim × h1)
    std::vector<float> b1_;  // (h1)
    std::vector<float> w2_;  // (h1 × h2)
    std::vector<float> b2_;  // (h2)
    std::vector<float> w3_;  // (h2 × obs_dim)
    std::vector<float> b3_;  // (obs_dim)

    // 配置
    PredictiveCodingConfig cfg_;
    int input_dim_;

    // 统计
    std::deque<float> error_history_;
    std::deque<float> prediction_errors_;
    std::deque<int> inference_steps_log_;
    double learning_progress_ = 0.0;
};

}  // namespace ai_learning::learning
