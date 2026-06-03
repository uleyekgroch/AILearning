/**
 * @file learner_factory.cpp
 * @brief Learner 工厂实现
 */

#include "ai_learning/core/learner_factory.hpp"
#include "ai_learning/core/learner.hpp"
#include "ai_learning/learning/predictive_coding_engine.hpp"
#include "ai_learning/learning/mlp_forward_engine.hpp"
#include "ai_learning/learning/light_predictive_engine.hpp"

#include <algorithm>

namespace ai_learning::core {

auto LearnerFactory::make_default_engine(const LearnerConfig& config)
    -> std::unique_ptr<learning::IPredictiveEngine> {
    return std::make_unique<learning::PredictiveCodingEngine>(
        learning::PredictiveCodingConfig{
            config.obs_dim,
            config.action_dim,
            config.hidden_dims.size() > 0 ? config.hidden_dims[0] : 64,
            config.hidden_dims.size() > 1 ? config.hidden_dims[1] : 32,
            config.learning_rate,
            config.inference_lr,
            config.max_inference_steps,
            config.convergence_threshold,
        });
}

auto LearnerFactory::create_default(const LearnerConfig& config)
    -> std::unique_ptr<Learner> {
    return create_with_engine(config, make_default_engine(config));
}

auto LearnerFactory::create_with_engine(
    const LearnerConfig& config,
    std::unique_ptr<learning::IPredictiveEngine> engine)
    -> std::unique_ptr<Learner> {
    return std::make_unique<Learner>(config, std::move(engine));
}

auto LearnerFactory::make_engine(const std::string& name,
                                  const LearnerConfig& config)
    -> std::unique_ptr<learning::IPredictiveEngine> {
    auto lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    auto h1 = config.hidden_dims.size() > 0 ? config.hidden_dims[0] : 64;
    auto h2 = config.hidden_dims.size() > 1 ? config.hidden_dims[1] : 32;

    if (lower == "mlp") {
        return std::make_unique<learning::MLPForwardEngine>(
            learning::MLPForwardConfig{
                config.obs_dim,
                config.action_dim,
                h1,
                h2,
                config.learning_rate,
                3.0,
            });
    }
    if (lower == "light") {
        return std::make_unique<learning::LightPredictiveEngine>(
            learning::LightPredictiveConfig{
                config.obs_dim,
                config.action_dim,
                h1,
                config.learning_rate,
                3.0,
            });
    }
    // 默认: "pc" 或其他任何值
    return make_default_engine(config);
}

}  // namespace ai_learning::core
