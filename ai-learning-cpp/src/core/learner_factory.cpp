/**
 * @file learner_factory.cpp
 * @brief Learner 工厂实现
 */

#include "ai_learning/core/learner_factory.hpp"
#include "ai_learning/core/learner.hpp"
#include "ai_learning/learning/predictive_coding_engine.hpp"

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

}  // namespace ai_learning::core
