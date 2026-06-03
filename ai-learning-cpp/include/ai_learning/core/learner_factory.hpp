/**
 * @file learner_factory.hpp
 * @brief Learner 工厂 — 将构造逻辑从 Learner 类中提取出来
 *
 * 设计目标：
 * - Learner 构造函数保持简洁（委托给工厂）
 * - 允许注入自定义预测引擎（如 Eigen 优化版、Transformer 版）
 * - 保留其他子系统的值语义（C++ 优势）
 */
#pragma once

#include "ai_learning/core/config.hpp"
#include "ai_learning/learning/ipredictive_engine.hpp"

#include <memory>

namespace ai_learning::core {

class Learner;

/// Learner 构建配置（扩展 LearnerConfig，添加引擎选项）
struct LearnerBuildConfig {
    LearnerConfig core_config;

    /// 自定义预测引擎（nullptr = 使用默认 Hebbian PC）
    std::unique_ptr<learning::IPredictiveEngine> custom_engine;
};

/// Learner 工厂
class LearnerFactory {
public:
    /// 使用默认配置构建 Learner
    static auto create_default(const LearnerConfig& config = LearnerConfig{})
        -> std::unique_ptr<Learner>;

    /// 使用自定义引擎构建 Learner
    static auto create_with_engine(
        const LearnerConfig& config,
        std::unique_ptr<learning::IPredictiveEngine> engine)
        -> std::unique_ptr<Learner>;

    /// 创建默认预测编码引擎
    static auto make_default_engine(const LearnerConfig& config)
        -> std::unique_ptr<learning::IPredictiveEngine>;
};

}  // namespace ai_learning::core
