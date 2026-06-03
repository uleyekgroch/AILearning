/**
 * @file simulation.hpp
 * @brief 模拟推理 — 在脑中构建场景、追踪因果链
 *
 * 认知科学基础：
 *   1. 心理模拟理论（Barsalou 1999）：推理 = 脑中运行世界模型
 *   2. 因果推理（Pearl 2009）：观察→干预→反事实三个层级
 *   3. 类比推理（Gentner 1983）：匹配关系结构而非表面特征
 *
 * 对应 Python: SimulationReasoning
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {
class KnowledgeGraph;
}

namespace ai_learning::reasoning {

/// 模拟场景 — 在脑中构建的虚拟世界
struct SimulationScene {
    std::vector<std::string> concepts;
    std::vector<std::map<std::string, std::string>> features;
    std::vector<std::map<std::string, std::string>> relations;
    double confidence = 0.0;
};

/// 因果链
struct CausalChain {
    std::vector<std::string> steps;
    double confidence = 0.0;
    std::vector<std::string> evidence;
};

/// 推理结果
struct SimulationResult {
    SimulationScene scene;
    std::vector<CausalChain> causal_chains;
    std::vector<std::map<std::string, std::string>> counterfactuals;
    std::vector<std::map<std::string, std::string>> analogies;
    double confidence = 0.0;
    std::string reasoning_type;  // causal / counterfactual / analogical / factual
};

/// 模拟推理系统
class SimulationReasoning {
public:
    explicit SimulationReasoning(
        const domain::knowledge::KnowledgeGraph& kg);

    /// 推理主入口：场景构建 → 因果追踪 → 反事实 → 类比
    [[nodiscard]] auto reason(
        const std::string& question,
        const std::vector<std::string>& activated_concepts) const -> SimulationResult;

    /// 将推理结果表达为自然语言
    [[nodiscard]] auto express(
        const SimulationResult& result,
        const std::string& question) const -> std::string;

    /// 获取统计
    [[nodiscard]] auto get_stats() const -> std::map<std::string, double>;

private:
    // 场景构建
    [[nodiscard]] auto build_scene_(
        const std::vector<std::string>& concepts) const -> SimulationScene;

    // 因果链追踪
    [[nodiscard]] auto trace_causal_chains_(
        const SimulationScene& scene,
        const std::string& question) const -> std::vector<CausalChain>;

    // 反事实检测
    [[nodiscard]] static auto is_counterfactual_(
        const std::string& question) -> bool;

    // 反事实模拟
    [[nodiscard]] auto simulate_counterfactuals_(
        const SimulationScene& scene,
        const std::string& question) const
        -> std::vector<std::map<std::string, std::string>>;

    // 类比发现
    [[nodiscard]] auto find_analogies_(
        const SimulationScene& scene,
        const std::vector<std::string>& concepts) const
        -> std::vector<std::map<std::string, std::string>>;

    // 置信度评估
    [[nodiscard]] static auto assess_confidence_(
        const SimulationScene& scene,
        const std::vector<CausalChain>& chains,
        const std::vector<std::map<std::string, std::string>>& analogies)
        -> double;

    // 问题分类
    [[nodiscard]] static auto classify_question_(
        const std::string& question) -> std::string;

    const domain::knowledge::KnowledgeGraph& kg_;

    // 统计（mutable 允许 const 方法更新）
    mutable std::map<std::string, double> stats_;
};

}  // namespace ai_learning::reasoning
