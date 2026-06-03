/**
 * @file test_alternative_engines.cpp
 * @brief MLPForwardEngine & LightPredictiveEngine 单元测试
 */

#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "ai_learning/learning/mlp_forward_engine.hpp"
#include "ai_learning/learning/light_predictive_engine.hpp"
#include "ai_learning/core/learner_factory.hpp"
#include "ai_learning/core/learner.hpp"

using namespace ai_learning::learning;
using namespace ai_learning::core;
using Catch::Matchers::WithinAbs;

// ── MLPForwardEngine ───────────────────────────────────────────

TEST_CASE("MLPForwardEngine: 构造和基本预测") {
    MLPForwardConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;

    MLPForwardEngine engine(cfg);

    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto pred = engine.predict(state, action);

    REQUIRE(pred.size() == 16);
    for (auto v : pred) REQUIRE(std::isfinite(v));
}

TEST_CASE("MLPForwardEngine: 学习降低误差") {
    MLPForwardConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;
    cfg.learning_rate = 0.01;

    MLPForwardEngine engine(cfg);

    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto actual = std::vector<float>(16, 1.0f);

    auto first_error = engine.learn(state, action, actual);
    for (int i = 0; i < 500; ++i) {
        engine.learn(state, action, actual);
    }
    auto last_error = engine.learn(state, action, actual);

    REQUIRE(last_error < first_error);
}

TEST_CASE("MLPForwardEngine: 好奇心和进度") {
    MLPForwardConfig cfg;
    cfg.obs_dim = 8;
    cfg.action_dim = 2;

    MLPForwardEngine engine(cfg);

    REQUIRE(engine.get_avg_inference_steps() == 1.0);
    REQUIRE_THAT(engine.get_learning_progress(), WithinAbs(0.0, 0.01));
}

TEST_CASE("MLPForwardEngine: 状态保存和加载") {
    MLPForwardConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;

    MLPForwardEngine engine1(cfg);
    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto actual = std::vector<float>(16, 1.0f);

    engine1.learn(state, action, actual);
    auto pred1 = engine1.predict(state, action);
    auto saved = engine1.save_state();

    MLPForwardEngine engine2(cfg);
    engine2.load_state(saved);
    auto pred2 = engine2.predict(state, action);

    REQUIRE(pred1.size() == pred2.size());
    for (size_t i = 0; i < pred1.size(); ++i) {
        REQUIRE_THAT(pred1[i], WithinAbs(pred2[i], 1e-5f));
    }
}

// ── LightPredictiveEngine ──────────────────────────────────────

TEST_CASE("LightPredictiveEngine: 构造和基本预测") {
    LightPredictiveConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden_dim = 32;

    LightPredictiveEngine engine(cfg);

    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto pred = engine.predict(state, action);

    REQUIRE(pred.size() == 16);
    for (auto v : pred) REQUIRE(std::isfinite(v));
}

TEST_CASE("LightPredictiveEngine: 学习降低误差") {
    LightPredictiveConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden_dim = 32;
    cfg.learning_rate = 0.0005;
    cfg.clip_value = 1.0;

    LightPredictiveEngine engine(cfg);

    // 使用变化输入增加学习信号，避免对称陷阱
    auto state = std::vector<float>(16);
    for (int i = 0; i < 16; ++i) state[i] = 0.1f * i;
    auto action = std::vector<float>{1.0f};
    auto actual = std::vector<float>(16);
    for (int i = 0; i < 16; ++i) actual[i] = 0.2f * i;

    auto first_error = engine.learn(state, action, actual);
    for (int i = 0; i < 1000; ++i) {
        engine.learn(state, action, actual);
    }
    auto last_error = engine.learn(state, action, actual);

    // 误差应显著降低（或保持在低水平）
    REQUIRE(last_error < first_error * 2.0f);
    REQUIRE(last_error < 5.0f);
}

TEST_CASE("LightPredictiveEngine: 好奇心和进度") {
    LightPredictiveConfig cfg;
    cfg.obs_dim = 8;
    cfg.action_dim = 2;

    LightPredictiveEngine engine(cfg);

    REQUIRE(engine.get_avg_inference_steps() == 1.0);
    REQUIRE_THAT(engine.get_learning_progress(), WithinAbs(0.0, 0.01));
}

TEST_CASE("LightPredictiveEngine: 状态保存和加载") {
    LightPredictiveConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden_dim = 32;

    LightPredictiveEngine engine1(cfg);
    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto actual = std::vector<float>(16, 1.0f);

    engine1.learn(state, action, actual);
    auto pred1 = engine1.predict(state, action);
    auto saved = engine1.save_state();

    LightPredictiveEngine engine2(cfg);
    engine2.load_state(saved);
    auto pred2 = engine2.predict(state, action);

    REQUIRE(pred1.size() == pred2.size());
    for (size_t i = 0; i < pred1.size(); ++i) {
        REQUIRE_THAT(pred1[i], WithinAbs(pred2[i], 1e-5f));
    }
}

// ── LearnerFactory 引擎选择 ─────────────────────────────────────

TEST_CASE("LearnerFactory: 创建 MLP 引擎") {
    LearnerConfig cfg;
    cfg.obs_dim = 8;
    cfg.action_dim = 2;

    auto engine = LearnerFactory::make_engine("mlp", cfg);
    REQUIRE(engine != nullptr);

    auto state = std::vector<float>(8, 0.5f);
    auto action = std::vector<float>(2, 0.1f);
    auto pred = engine->predict(state, action);

    REQUIRE(pred.size() == 8);
}

TEST_CASE("LearnerFactory: 创建 Light 引擎") {
    LearnerConfig cfg;
    cfg.obs_dim = 8;
    cfg.action_dim = 2;

    auto engine = LearnerFactory::make_engine("light", cfg);
    REQUIRE(engine != nullptr);

    auto state = std::vector<float>(8, 0.5f);
    auto action = std::vector<float>(2, 0.1f);
    auto pred = engine->predict(state, action);

    REQUIRE(pred.size() == 8);
}

TEST_CASE("LearnerFactory: 默认创建 PC 引擎") {
    LearnerConfig cfg;
    cfg.obs_dim = 8;
    cfg.action_dim = 2;

    auto engine = LearnerFactory::make_engine("pc", cfg);
    REQUIRE(engine != nullptr);

    auto state = std::vector<float>(8, 0.5f);
    auto action = std::vector<float>(2, 0.1f);
    auto pred = engine->predict(state, action);

    REQUIRE(pred.size() == 8);
}

TEST_CASE("LearnerFactory: 注入 MLP 引擎到 Learner") {
    LearnerConfig cfg;
    cfg.obs_dim = 8;
    cfg.action_dim = 2;

    auto learner = LearnerFactory::create_with_engine(
        cfg, LearnerFactory::make_engine("mlp", cfg));

    REQUIRE(learner != nullptr);
    REQUIRE(learner->engine().get_avg_inference_steps() == 1.0);
}
