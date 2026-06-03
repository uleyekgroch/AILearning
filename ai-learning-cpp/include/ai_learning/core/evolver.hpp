#pragma once
/**
 * @file evolver.hpp
 * @brief 自主进化模块 — 能力评估与自我改进
 *
 * 通过评估当前能力、识别薄弱项、生成针对性改进来驱动学习体进化。
 */

#include <map>
#include <string>
#include <vector>

namespace ai_learning::core {

// 前向声明
class Learner;

/// 能力测试结果
struct CapabilityResult {
    float score = 0.0F;          ///< 0~1 得分
    int tests_passed = 0;        ///< 通过的测试数
    int tests_total = 0;         ///< 总测试数
};

/// 改进记录
struct Improvement {
    std::string capability;      ///< 改进的能力名
    std::string action;          ///< 改进动作描述
    float expected_gain = 0.0F;  ///< 预期提升
};

/// 进化结果
struct EvolutionResult {
    int iterations = 0;
    std::vector<Improvement> improvements;
    float score_before = 0.0F;
    float score_after = 0.0F;
};

class Evolver {
public:
    explicit Evolver(Learner& learner);

    /// 评估所有能力
    [[nodiscard]] auto evaluate_capabilities() const
        -> std::map<std::string, CapabilityResult>;

    /// 获取低于阈值的能力名
    [[nodiscard]] auto get_weak_capabilities(float threshold = 0.5F) const
        -> std::vector<std::string>;

    /// 执行自主进化
    [[nodiscard]] auto evolve(int iterations = 1) -> EvolutionResult;

    /// 获取进化报告
    [[nodiscard]] auto get_evolution_report() const
        -> const std::vector<EvolutionResult>& {
        return evolution_history_;
    }

private:
    Learner& learner_;
    std::vector<EvolutionResult> evolution_history_;
    mutable std::map<std::string, CapabilityResult> cached_capabilities_;

    /// 测试语义理解
    [[nodiscard]] auto test_semantic() const -> CapabilityResult;
    /// 测试因果推理
    [[nodiscard]] auto test_causal() const -> CapabilityResult;
    /// 测试概念形成
    [[nodiscard]] auto test_concept() const -> CapabilityResult;
    /// 测试知识检索
    [[nodiscard]] auto test_retrieval() const -> CapabilityResult;

    /// 生成针对性改进
    [[nodiscard]] auto generate_improvement(
        const std::string& capability) -> Improvement;
};

}  // namespace ai_learning::core
