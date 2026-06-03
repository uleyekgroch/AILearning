/**
 * @file mlp_forward_engine.cpp
 * @brief 标准前馈 MLP 预测引擎实现 — SGD 反向传播
 */

#include "ai_learning/learning/mlp_forward_engine.hpp"
#include "ai_learning/core/tensor_ops.hpp"

#include <algorithm>
#include <cmath>

namespace ai_learning::learning {

using namespace ai_learning::core;

// ── 构造 ────────────────────────────────────────────────────────

MLPForwardEngine::MLPForwardEngine(const MLPForwardConfig& cfg)
    : cfg_(cfg),
      input_dim_(cfg.obs_dim + cfg.action_dim),
      w1_(cfg.hidden1_dim, input_dim_),
      b1_(cfg.hidden1_dim, 0.0f),
      w2_(cfg.hidden2_dim, cfg.hidden1_dim),
      b2_(cfg.hidden2_dim, 0.0f),
      w3_(cfg.obs_dim, cfg.hidden2_dim),
      b3_(cfg.obs_dim, 0.0f) {

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto he_init = [&](int fan_in) -> float {
        float std = std::sqrt(2.0f / static_cast<float>(fan_in));
        return dist(rng) * std;
    };

    for (auto& w : w1_.data) w = he_init(input_dim_);
    for (auto& w : w2_.data) w = he_init(cfg.hidden1_dim);
    for (auto& w : w3_.data) w = he_init(cfg.hidden2_dim);
}

// ── 前向预测 ────────────────────────────────────────────────────

auto MLPForwardEngine::predict(
    const std::vector<float>& state,
    const std::vector<float>& action) const
    -> std::vector<float> {
    auto action_vec = encode_action_(action);
    auto input = state;
    for (auto v : action_vec) input.push_back(v);
    if (static_cast<int>(input.size()) < input_dim_) {
        input.resize(input_dim_, 0.0f);
    }

    auto res = forward_(input);
    return res.output;
}

auto MLPForwardEngine::forward_(
    const std::vector<float>& input) const
    -> ForwardResult {
    auto z1 = mat_vec_bias(w1_, input, b1_);
    auto h1 = tensor_relu(z1);
    auto z2 = mat_vec_bias(w2_, h1, b2_);
    auto h2 = tensor_relu(z2);
    auto output = mat_vec_bias(w3_, h2, b3_);
    return {input, z1, h1, z2, h2, output};
}

// ── SGD 学习（反向传播）────────────────────────────────────────

auto MLPForwardEngine::learn(
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
    auto f = forward_(input);

    // 输出层误差
    auto delta3 = tensor_sub(f.output, actual);  // ∂L/∂output
    float mse = 0.0f;
    for (auto d : delta3) mse += d * d;
    mse /= static_cast<float>(delta3.size());

    // 反向传播
    // δ2 = (W3^T · δ3) ⊙ relu'(z2)
    // W3 是 row-major (obs_dim × hidden2_dim)
    // vec_mat(vec, mat, cols, rows): result[r] = sum_c vec[c] * mat[c*rows + r]
    // 令 cols=w3_.rows (=obs_dim), rows=w3_.cols (=hidden2_dim)
    // 则 mat[c*rows + r] = w3_.data[c*w3_.cols + r] = W3[c,r] 正确
    auto relu_d2 = tensor_relu_deriv(f.z2);
    auto delta3_w3 = vec_mat(delta3, w3_.data, w3_.rows, w3_.cols);
    auto delta2 = tensor_mul(delta3_w3, relu_d2);

    // δ1 = (W2^T · δ2) ⊙ relu'(z1)
    // W2 是 row-major (hidden2_dim × hidden1_dim)
    // 同理: cols=w2_.rows, rows=w2_.cols
    auto relu_d1 = tensor_relu_deriv(f.z1);
    auto delta2_w2 = vec_mat(delta2, w2_.data, w2_.rows, w2_.cols);
    auto delta1 = tensor_mul(delta2_w2, relu_d1);

    // 权重更新（SGD）
    float lr = static_cast<float>(cfg_.learning_rate);
    float wc = static_cast<float>(cfg_.clip_value);

    // W3 -= lr * outer(δ3, h2), b3 -= lr * δ3
    mat_add_outer(w3_.data, -lr, delta3, f.h2);
    for (size_t i = 0; i < b3_.size(); ++i) {
        b3_[i] = std::clamp(b3_[i] - lr * delta3[i], -wc, wc);
    }
    for (auto& w : w3_.data) w = std::clamp(w, -wc, wc);

    // W2 -= lr * outer(δ2, h1)
    mat_add_outer(w2_.data, -lr, delta2, f.h1);
    for (size_t i = 0; i < b2_.size(); ++i) {
        b2_[i] = std::clamp(b2_[i] - lr * delta2[i], -wc, wc);
    }
    for (auto& w : w2_.data) w = std::clamp(w, -wc, wc);

    // W1 -= lr * outer(δ1, input)
    mat_add_outer(w1_.data, -lr, delta1, f.input);
    for (size_t i = 0; i < b1_.size(); ++i) {
        b1_[i] = std::clamp(b1_[i] - lr * delta1[i], -wc, wc);
    }
    for (auto& w : w1_.data) w = std::clamp(w, -wc, wc);

    // 记录
    error_history_.push_back(mse);
    prediction_errors_.push_back(mse);
    learning_progress_ = get_learning_progress();

    return mse;
}

// ── 统计 ──────────────────────────────────────────────────────

auto MLPForwardEngine::get_curiosity() const -> double {
    if (prediction_errors_.size() < 2) return 1.0;
    float avg_error = 0.0f;
    auto count = std::min(prediction_errors_.size(), size_t(10));
    auto start = prediction_errors_.end() - count;
    for (auto it = start; it != prediction_errors_.end(); ++it) {
        avg_error += *it;
    }
    avg_error /= static_cast<float>(count);
    return std::min(avg_error, 1.0f);
}

auto MLPForwardEngine::get_learning_progress() const -> double {
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

auto MLPForwardEngine::get_avg_inference_steps() const -> double {
    return 1.0;  // MLP 无迭代推理
}

// ── 序列化 ─────────────────────────────────────────────────────

auto MLPForwardEngine::save_state() const -> PredictiveEngineState {
    PredictiveEngineState s;
    s.weights.reserve(w1_.size() + b1_.size() + w2_.size() + b2_.size() +
                        w3_.size() + b3_.size());
    s.weights.insert(s.weights.end(), w1_.data.begin(), w1_.data.end());
    s.weights.insert(s.weights.end(), b1_.begin(), b1_.end());
    s.weights.insert(s.weights.end(), w2_.data.begin(), w2_.data.end());
    s.weights.insert(s.weights.end(), b2_.begin(), b2_.end());
    s.weights.insert(s.weights.end(), w3_.data.begin(), w3_.data.end());
    s.weights.insert(s.weights.end(), b3_.begin(), b3_.end());
    s.shape = {static_cast<int>(w1_.size()), static_cast<int>(b1_.size()),
               static_cast<int>(w2_.size()), static_cast<int>(b2_.size()),
               static_cast<int>(w3_.size()), static_cast<int>(b3_.size())};
    return s;
}

void MLPForwardEngine::load_state(const PredictiveEngineState& state) {
    if (state.shape.size() != 6) return;
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
    copy(w3_.data, state.shape[4]);
    copy(b3_, state.shape[5]);
}

// ── 内部方法 ─────────────────────────────────────────────────────

auto MLPForwardEngine::encode_action_(
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
