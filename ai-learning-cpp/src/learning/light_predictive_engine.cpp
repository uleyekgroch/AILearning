/**
 * @file light_predictive_engine.cpp
 * @brief 轻量级单层预测编码引擎实现
 */

#include "ai_learning/learning/light_predictive_engine.hpp"
#include "ai_learning/core/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace ai_learning::learning {

using namespace ai_learning::core;

// ── 构造 ────────────────────────────────────────────────────────

LightPredictiveEngine::LightPredictiveEngine(const LightPredictiveConfig& cfg)
    : cfg_(cfg),
      input_dim_(cfg.obs_dim + cfg.action_dim),
      w1_(cfg.hidden_dim, input_dim_),
      b1_(cfg.hidden_dim, 0.0f),
      w2_(cfg.obs_dim, cfg.hidden_dim),
      b2_(cfg.obs_dim, 0.0f) {

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto he_init = [&](int fan_in) -> float {
        float std = std::sqrt(2.0f / static_cast<float>(fan_in));
        return dist(rng) * std;
    };

    for (auto& w : w1_.data) w = he_init(input_dim_);
    for (auto& w : w2_.data) w = he_init(cfg.hidden_dim);
}

// ── 前向预测 ────────────────────────────────────────────────────

auto LightPredictiveEngine::predict(
    const std::vector<float>& state,
    const std::vector<float>& action) const
    -> std::vector<float> {
    auto action_vec = encode_action_(action);
    auto input = state;
    for (auto v : action_vec) input.push_back(v);
    if (static_cast<int>(input.size()) < input_dim_) {
        input.resize(input_dim_, 0.0f);
    }

    auto h = tensor_relu(mat_vec_bias(w1_, input, b1_));
    return mat_vec_bias(w2_, h, b2_);
}

// ── 学习（单层 Hebbian）────────────────────────────────────────

auto LightPredictiveEngine::learn(
    const std::vector<float>& obs,
    const std::vector<float>& action,
    const std::vector<float>& actual)
    -> double {
    auto action_vec = encode_action_(action);
    auto input = obs;
    for (auto v : action_vec) input.push_back(v);
    if (static_cast<int>(input.size()) < input_dim_) {
        input.resize(input_dim_, 0.0f);
    }

    // 前向
    auto z1 = mat_vec_bias(w1_, input, b1_);
    auto h = tensor_relu(z1);
    auto pred = mat_vec_bias(w2_, h, b2_);

    // 误差
    auto eps_out = tensor_sub(actual, pred);
    auto error = tensor_mse(eps_out, tensor_zeros(eps_out.size()));
    if (!std::isfinite(error)) error = 10.0f;

    // 反向误差传播到隐藏层
    auto relu_d = tensor_relu_deriv(z1);
    auto feedback = vec_mat(eps_out, w2_.data, w2_.cols, w2_.rows);
    auto eps_h = tensor_mul(feedback, relu_d);

    // Hebbian 更新
    float lr = static_cast<float>(cfg_.learning_rate);
    float wc = static_cast<float>(cfg_.clip_value);

    // W2 += lr * outer(h, eps_out)
    mat_add_outer(w2_.data, lr, h, eps_out);
    for (size_t i = 0; i < b2_.size(); ++i) {
        b2_[i] = std::clamp(b2_[i] + lr * eps_out[i], -wc, wc);
    }
    for (auto& w : w2_.data) w = std::clamp(w, -wc, wc);

    // W1 += lr * outer(input, eps_h)
    mat_add_outer(w1_.data, lr, input, eps_h);
    for (size_t i = 0; i < b1_.size(); ++i) {
        b1_[i] = std::clamp(b1_[i] + lr * eps_h[i], -wc, wc);
    }
    for (auto& w : w1_.data) w = std::clamp(w, -wc, wc);

    // 记录
    error_history_.push_back(error);
    prediction_errors_.push_back(error);
    learning_progress_ = get_learning_progress();

    return error;
}

// ── 统计 ──────────────────────────────────────────────────────

auto LightPredictiveEngine::get_curiosity() const -> double {
    if (prediction_errors_.size() < 2) return 1.0;
    float avg_error = 0.0f;
    auto count = std::min(prediction_errors_.size(), size_t(10));
    auto start = prediction_errors_.end() - count;
    for (auto it = start; it != prediction_errors_.end(); ++it) avg_error += *it;
    avg_error /= static_cast<float>(count);
    return std::min(avg_error, 1.0f);
}

auto LightPredictiveEngine::get_learning_progress() const -> double {
    if (error_history_.size() < 10) return 0.0;
    auto errs = std::vector<float>(error_history_.begin(), error_history_.end());
    auto half = errs.size() / 2;
    auto first_avg = 0.0f, second_avg = 0.0f;
    for (size_t i = 0; i < half; ++i) first_avg += errs[i];
    for (size_t i = half; i < errs.size(); ++i) second_avg += errs[i];
    first_avg /= static_cast<float>(half);
    second_avg /= static_cast<float>(errs.size() - half);
    if (first_avg <= 0.0f) return 0.0;
    return std::clamp(static_cast<double>(first_avg - second_avg) / first_avg,
                      0.0, 1.0);
}

auto LightPredictiveEngine::get_avg_inference_steps() const -> double {
    return 1.0;  // 无迭代推理
}

// ── 序列化 ─────────────────────────────────────────────────────

auto LightPredictiveEngine::save_state() const -> PredictiveEngineState {
    PredictiveEngineState s;
    s.weights.reserve(w1_.size() + b1_.size() + w2_.size() + b2_.size());
    s.weights.insert(s.weights.end(), w1_.data.begin(), w1_.data.end());
    s.weights.insert(s.weights.end(), b1_.begin(), b1_.end());
    s.weights.insert(s.weights.end(), w2_.data.begin(), w2_.data.end());
    s.weights.insert(s.weights.end(), b2_.begin(), b2_.end());
    s.shape = {static_cast<int>(w1_.size()), static_cast<int>(b1_.size()),
               static_cast<int>(w2_.size()), static_cast<int>(b2_.size())};
    return s;
}

void LightPredictiveEngine::load_state(const PredictiveEngineState& state) {
    if (state.shape.size() != 4) return;
    size_t off = 0;
    auto copy = [&](std::vector<float>& dst, int sz) {
        if (off + static_cast<size_t>(sz) <= state.weights.size()) {
            dst.assign(state.weights.begin() + off,
                       state.weights.begin() + off + sz);
        }
        off += sz;
    };
    copy(w1_.data, state.shape[0]);
    copy(b1_, state.shape[1]);
    copy(w2_.data, state.shape[2]);
    copy(b2_, state.shape[3]);
}

// ── 内部 ───────────────────────────────────────────────────────

auto LightPredictiveEngine::encode_action_(
    const std::vector<float>& action) const
    -> std::vector<float> {
    auto result = tensor_zeros(cfg_.action_dim);
    if (action.size() == 1) {
        int idx = static_cast<int>(action[0]);
        if (idx >= 0 && idx < cfg_.action_dim) result[idx] = 1.0f;
    } else if (static_cast<int>(action.size()) >= cfg_.action_dim) {
        for (int i = 0; i < cfg_.action_dim; ++i) result[i] = action[i];
    }
    return result;
}

}  // namespace ai_learning::learning
