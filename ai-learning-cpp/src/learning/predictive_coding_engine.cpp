/**
 * @file predictive_coding_engine.cpp
 * @brief 预测编码学习引擎实现
 */

#include "ai_learning/learning/predictive_coding_engine.hpp"
#include "ai_learning/core/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace ai_learning::learning {

using namespace ai_learning::core;

// ── 构造 ────────────────────────────────────────────────────────

PredictiveCodingEngine::PredictiveCodingEngine(
    const PredictiveCodingConfig& cfg)
    : cfg_(cfg),
      input_dim_(cfg.obs_dim + cfg.action_dim) {
    // He 初始化
    auto h1 = cfg.hidden1_dim;
    auto h2 = cfg.hidden2_dim;
    auto inp = input_dim_;
    auto out = cfg.obs_dim;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto he_init = [&](int fan_in) -> float {
        float std = std::sqrt(2.0f / static_cast<float>(fan_in));
        return dist(rng) * std;
    };

    w1_ = Matrix(h1, inp);
    b1_.resize(h1, 0.0f);
    w2_ = Matrix(h2, h1);
    b2_.resize(h2, 0.0f);
    w3_ = Matrix(out, h2);
    b3_.resize(out, 0.0f);
    
    // 初始化精度加权 (Precision matrices / vectors)
    pi_h1_.resize(h1, 1.0f);
    pi_h2_.resize(h2, 1.0f);
    pi_out_.resize(out, 1.0f);

    for (auto& w : w1_.data) w = he_init(inp);
    for (auto& w : w2_.data) w = he_init(h1);
    for (auto& w : w3_.data) w = he_init(h2);
}

// ── 前向传播 ────────────────────────────────────────────────────

auto PredictiveCodingEngine::predict(
    const std::vector<float>& state,
    const std::vector<float>& action) const
    -> std::vector<float> {
    auto action_vec = encode_action_(action);
    // 拼接 obs + action → input_dim
    auto input = state;
    for (auto v : action_vec) input.push_back(v);
    if (static_cast<int>(input.size()) < input_dim_) {
        input.resize(input_dim_, 0.0f);
    }

    auto z1 = mat_vec_bias(w1_, input, b1_);
    auto h1 = tensor_relu(z1);
    auto z2 = mat_vec_bias(w2_, h1, b2_);
    auto h2 = tensor_relu(z2);
    auto output = mat_vec_bias(w3_, h2, b3_);
    return output;
}

// ── 预测编码学习 ────────────────────────────────────────────────

auto PredictiveCodingEngine::learn(
    const std::vector<float>& obs,
    const std::vector<float>& action,
    const std::vector<float>& actual)
    -> double {
    // 构造输入：拼接 obs + action
    auto action_vec = encode_action_(action);
    auto input = obs;
    for (auto v : action_vec) input.push_back(v);
    if (static_cast<int>(input.size()) < input_dim_) {
        input.resize(input_dim_, 0.0f);
    }

    // 推理收敛
    auto result = infer_beliefs_(input, actual);

    // Hebbian 权重更新
    update_weights_(result);

    // 记录误差
    auto error = tensor_mse(result.epsilon_out,
                            tensor_zeros(result.epsilon_out.size()));
    if (!std::isfinite(error)) error = 10.0f;  // 安全回退

    error_history_.push_back(error);
    prediction_errors_.push_back(error);

    // 更新学习进度
    learning_progress_ = get_learning_progress();

    return error;
}

// ── 好奇心 ──────────────────────────────────────────────────────

auto PredictiveCodingEngine::get_curiosity() const -> double {
    if (prediction_errors_.size() < 2) return 1.0;

    float avg_error = 0.0f;
    auto count = std::min(prediction_errors_.size(), size_t(10));
    auto start = prediction_errors_.end() - count;
    for (auto it = start; it != prediction_errors_.end(); ++it) {
        avg_error += *it;
    }
    avg_error /= static_cast<float>(count);

    auto learnability = std::max(0.0, 1.0 - learning_progress_);
    return cfg_.curiosity_alpha * std::min(avg_error, 1.0f) +
           cfg_.curiosity_beta * learnability;
}

auto PredictiveCodingEngine::get_learning_progress() const -> double {
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

auto PredictiveCodingEngine::get_avg_inference_steps() const -> double {
    if (inference_steps_log_.empty()) return 0.0;
    auto count = std::min(inference_steps_log_.size(), size_t(100));
    auto sum = 0;
    auto start = inference_steps_log_.end() - count;
    for (auto it = start; it != inference_steps_log_.end(); ++it) {
        sum += *it;
    }
    return static_cast<double>(sum) / static_cast<double>(count);
}

// ── 序列化 ──────────────────────────────────────────────────────

auto PredictiveCodingEngine::save_detailed_state() const -> DetailedState {
    DetailedState state;
    state.w1 = w1_.data;
    state.b1 = b1_;
    state.w2 = w2_.data;
    state.b2 = b2_;
    state.w3 = w3_.data;
    state.b3 = b3_;
    return state;
}

void PredictiveCodingEngine::load_detailed_state(const DetailedState& state) {
    w1_ = Matrix(state.w1, cfg_.hidden1_dim, input_dim_);
    b1_ = state.b1;
    w2_ = Matrix(state.w2, cfg_.hidden2_dim, cfg_.hidden1_dim);
    b2_ = state.b2;
    w3_ = Matrix(state.w3, cfg_.obs_dim, cfg_.hidden2_dim);
    b3_ = state.b3;
}

auto PredictiveCodingEngine::save_state() const -> PredictiveEngineState {
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

void PredictiveCodingEngine::load_state(const PredictiveEngineState& state) {
    if (state.shape.size() != 6) return;
    size_t off = 0;
    auto copy_vec = [&](std::vector<float>& dst, int sz) {
        if (off + static_cast<size_t>(sz) <= state.weights.size()) {
            dst.assign(state.weights.begin() + off,
                       state.weights.begin() + off + sz);
        }
        off += sz;
    };
    copy_vec(w1_.data, state.shape[0]);
    copy_vec(b1_, state.shape[1]);
    copy_vec(w2_.data, state.shape[2]);
    copy_vec(b2_, state.shape[3]);
    copy_vec(w3_.data, state.shape[4]);
    copy_vec(b3_, state.shape[5]);
}

// ── 内部方法 ─────────────────────────────────────────────────────

auto PredictiveCodingEngine::encode_action_(
    const std::vector<float>& action) const
    -> std::vector<float> {
    auto result = tensor_zeros(cfg_.action_dim);

    if (action.size() == 1) {
        // 离散动作：one-hot
        int idx = static_cast<int>(action[0]);
        if (idx >= 0 && idx < cfg_.action_dim) {
            result[idx] = 1.0f;
        }
    } else if (static_cast<int>(action.size()) >= cfg_.action_dim) {
        // 连续动作向量
        for (int i = 0; i < cfg_.action_dim; ++i) {
            result[i] = action[i];
        }
    }
    return result;
}

auto PredictiveCodingEngine::infer_beliefs_(
    const std::vector<float>& input,
    const std::vector<float>& actual)
    -> InferenceResult {
    auto cv = static_cast<float>(cfg_.clip_value);

    // 前向初始化
    auto z1 = mat_vec_bias(w1_, input, b1_);
    auto mu_h1 = tensor_relu(z1);
    auto z2 = mat_vec_bias(w2_, mu_h1, b2_);
    auto mu_h2 = tensor_relu(z2);

    // 迭代推理
    int steps = 0;
    auto lr = static_cast<float>(cfg_.inference_lr);

    for (int t = 0; t < cfg_.max_inference_steps; ++t) {
        auto prev_h1 = mu_h1;
        auto prev_h2 = mu_h2;

        z1 = mat_vec_bias(w1_, input, b1_);
        z2 = mat_vec_bias(w2_, mu_h1, b2_);
        auto pred_out = mat_vec_bias(w3_, mu_h2, b3_);

        auto eps_out = tensor_clamp(tensor_sub(actual, pred_out), -cv, cv);
        auto eps_h2 = tensor_clamp(tensor_sub(mu_h2, tensor_relu(z2)), -cv, cv);
        auto eps_h1 = tensor_clamp(tensor_sub(mu_h1, tensor_relu(z1)), -cv, cv);

        // ★ 精度加权 (Precision Weighting): Attention 机制的核心
        // 如果系统对某个维度不确信（方差大，精度小），则该维度的预测误差将被抑制
        eps_out = tensor_mul(eps_out, pi_out_);
        eps_h2 = tensor_mul(eps_h2, pi_h2_);
        eps_h1 = tensor_mul(eps_h1, pi_h1_);

        auto relu_d2 = tensor_relu_deriv(z2);
        auto feedback_h2 = tensor_clamp(
            vec_mat(eps_out, w3_.data, w3_.cols, w3_.rows), -cv, cv);
        feedback_h2 = tensor_mul(feedback_h2, relu_d2);

        auto relu_d1 = tensor_relu_deriv(z1);
        auto e_h2_scaled = tensor_clamp(tensor_mul(eps_h2, relu_d2), -cv, cv);
        auto feedback_h1 = tensor_clamp(
            vec_mat(e_h2_scaled, w2_.data, w2_.cols, w2_.rows), -cv, cv);
        feedback_h1 = tensor_mul(feedback_h1, relu_d1);

        auto update_h1 = tensor_clamp(
            tensor_sub(eps_h1, feedback_h1), -cv, cv);
        // 添加 L1 稀疏惩罚项
        if (cfg_.sparsity_penalty > 0.0) {
            update_h1 = tensor_add(update_h1, tensor_scale(tensor_sign(mu_h1), static_cast<float>(cfg_.sparsity_penalty)));
        }
        mu_h1 = tensor_clamp(tensor_sub(mu_h1, tensor_scale(update_h1, lr)), -cv, cv);

        auto update_h2 = tensor_clamp(
            tensor_sub(eps_h2, feedback_h2), -cv, cv);
        // 添加 L1 稀疏惩罚项
        if (cfg_.sparsity_penalty > 0.0) {
            update_h2 = tensor_add(update_h2, tensor_scale(tensor_sign(mu_h2), static_cast<float>(cfg_.sparsity_penalty)));
        }
        mu_h2 = tensor_clamp(tensor_sub(mu_h2, tensor_scale(update_h2, lr)), -cv, cv);

        steps = t + 1;

        auto change = 0.5 * (check_convergence_(prev_h1, mu_h1, 0) +
                              check_convergence_(prev_h2, mu_h2, 0));
        if (change < cfg_.convergence_threshold) break;
    }

    inference_steps_log_.push_back(steps);

    // 最终残差
    z1 = mat_vec_bias(w1_, input, b1_);
    z2 = mat_vec_bias(w2_, mu_h1, b2_);
    auto pred_out = mat_vec_bias(w3_, mu_h2, b3_);
    auto eps_out = tensor_sub(actual, pred_out);
    auto eps_h2 = tensor_sub(mu_h2, tensor_relu(z2));
    auto eps_h1 = tensor_sub(mu_h1, tensor_relu(z1));

    return {input, mu_h1, mu_h2, z1, z2, eps_out, eps_h1, eps_h2, steps};
}

void PredictiveCodingEngine::update_weights_(const InferenceResult& r) {
    auto lr = static_cast<float>(cfg_.learning_rate);
    auto relu_d2 = tensor_relu_deriv(r.z2);
    auto relu_d1 = tensor_relu_deriv(r.z1);

    auto grad_h2 = tensor_mul(r.epsilon_h2, relu_d2);
    auto grad_h1 = tensor_mul(r.epsilon_h1, relu_d1);

    auto wc = static_cast<float>(cfg_.clip_value);

    // W3 += lr * outer(mu_h2, eps_out)
    mat_add_outer(w3_.data, lr, r.mu_h2, r.epsilon_out);
    for (size_t i = 0; i < b3_.size(); ++i) {
        b3_[i] = std::clamp(b3_[i] + lr * r.epsilon_out[i], -wc, wc);
    }
    for (auto& w : w3_.data) w = std::clamp(w, -wc, wc);

    // W2 += lr * outer(mu_h1, grad_h2)
    mat_add_outer(w2_.data, lr, r.mu_h1, grad_h2);
    for (size_t i = 0; i < b2_.size(); ++i) {
        b2_[i] = std::clamp(b2_[i] + lr * grad_h2[i], -wc, wc);
    }
    for (auto& w : w2_.data) w = std::clamp(w, -wc, wc);

    // W1 += lr * outer(input, grad_h1)
    mat_add_outer(w1_.data, lr, r.input, grad_h1);
    for (size_t i = 0; i < b1_.size(); ++i) {
        b1_[i] = std::clamp(b1_[i] + lr * grad_h1[i], -wc, wc);
    }
    for (auto& w : w1_.data) w = std::clamp(w, -wc, wc);

    // ★ 精度更新 (Precision Update)
    // 根据预测误差的平方，动态更新对各个维度的“置信度”。误差越大，精度越低
    auto update_precision = [this](std::vector<float>& pi, const std::vector<float>& eps) {
        float plr = static_cast<float>(cfg_.precision_lr);
        for (size_t i = 0; i < pi.size(); ++i) {
            // pi = pi - lr * (pi * eps^2 - 1)  (Simplified Gamma update derivation)
            // Or simpler: moving average of inverse variance
            float variance = eps[i] * eps[i] + 1e-4f;
            pi[i] = (1.0f - plr) * pi[i] + plr * (1.0f / variance);
            // Limit precision to avoid explosion
            pi[i] = std::clamp(pi[i], 0.1f, 100.0f);
        }
    };
    
    update_precision(pi_out_, r.epsilon_out);
    update_precision(pi_h2_, r.epsilon_h2);
    update_precision(pi_h1_, r.epsilon_h1);
}

auto PredictiveCodingEngine::check_convergence_(
    const std::vector<float>& prev,
    const std::vector<float>& curr,
    double /*threshold*/) -> double {
    float sum = 0.0f;
    for (size_t i = 0; i < prev.size(); ++i) {
        auto d = prev[i] - curr[i];
        sum += d * d;
    }
    return static_cast<double>(sum) / static_cast<double>(prev.size());
}

}  // namespace ai_learning::learning
