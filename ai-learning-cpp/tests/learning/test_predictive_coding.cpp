/**
 * @file test_predictive_coding.cpp
 * @brief PredictiveCodingEngine 单元测试 — TDD
 */

#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "ai_learning/learning/predictive_coding_engine.hpp"

using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;

TEST_CASE("PredictiveCodingEngine: 构造和基本预测") {
    PredictiveCodingConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;

    PredictiveCodingEngine engine(cfg);

    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);

    auto pred = engine.predict(state, action);

    SECTION("输出维度正确") {
        REQUIRE(pred.size() == 16);
    }

    SECTION("输出是有限值") {
        for (auto v : pred) {
            REQUIRE(std::isfinite(v));
        }
    }
}

TEST_CASE("PredictiveCodingEngine: 学习降低误差") {
    PredictiveCodingConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;
    cfg.max_inference_steps = 10;

    PredictiveCodingEngine engine(cfg);

    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto actual = std::vector<float>(16, 1.0f);

    // 学习多次
    auto first_error = engine.learn(state, action, actual);
    for (int i = 0; i < 500; ++i) {
        engine.learn(state, action, actual);
    }
    auto last_error = engine.learn(state, action, actual);

    SECTION("误差应随学习降低") {
        REQUIRE(last_error < first_error);
    }
}

TEST_CASE("PredictiveCodingEngine: 好奇心") {
    PredictiveCodingConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;
    cfg.curiosity_alpha = 0.5;
    cfg.curiosity_beta = 0.5;

    PredictiveCodingEngine engine(cfg);

    SECTION("初始好奇心为 1.0") {
        REQUIRE_THAT(engine.get_curiosity(), WithinAbs(1.0, 0.01));
    }
}

TEST_CASE("PredictiveCodingEngine: 学习进度初始为 0") {
    PredictiveCodingConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;

    PredictiveCodingEngine engine(cfg);
    REQUIRE_THAT(engine.get_learning_progress(), WithinAbs(0.0, 0.01));
}

TEST_CASE("PredictiveCodingEngine: 状态保存和加载") {
    PredictiveCodingConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;

    PredictiveCodingEngine engine1(cfg);
    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto actual = std::vector<float>(16, 1.0f);

    engine1.learn(state, action, actual);
    auto pred1 = engine1.predict(state, action);
    auto saved = engine1.save_state();

    // 加载到新引擎
    PredictiveCodingEngine engine2(cfg);
    engine2.load_state(saved);
    auto pred2 = engine2.predict(state, action);

    SECTION("加载后预测一致") {
        REQUIRE(pred1.size() == pred2.size());
        for (size_t i = 0; i < pred1.size(); ++i) {
            REQUIRE_THAT(pred1[i], WithinAbs(pred2[i], 1e-5f));
        }
    }
}

TEST_CASE("PredictiveCodingEngine: 推理步数记录") {
    PredictiveCodingConfig cfg;
    cfg.obs_dim = 16;
    cfg.action_dim = 4;
    cfg.hidden1_dim = 32;
    cfg.hidden2_dim = 16;
    cfg.max_inference_steps = 50;

    PredictiveCodingEngine engine(cfg);

    auto state = std::vector<float>(16, 0.5f);
    auto action = std::vector<float>(4, 0.1f);
    auto actual = std::vector<float>(16, 1.0f);

    engine.learn(state, action, actual);
    auto avg_steps = engine.get_avg_inference_steps();

    SECTION("推理步数在合理范围内") {
        REQUIRE(avg_steps > 0);
        REQUIRE(avg_steps <= 50);
    }
}
