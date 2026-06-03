/**
 * @file unified_engine.hpp
 * @brief 统一推理引擎 — 整合所有推理模块
 *
 * 推理管线：
 * 1. 直接查询（知识图谱）
 * 2. 因果推理（因果链）
 * 3. 归纳推理（模式发现）
 * 4. 类比推理（跨域映射）
 * 5. 反事实推理（如果...会怎样）
 * 6. 概率推理（统计推理）
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {
class KnowledgeGraph;
}

namespace ai_learning::memory {
class EpisodicMemory;
}

namespace ai_learning::learning {
class StatisticalLearner;
}

namespace ai_learning::reasoning {

/// 推理结果
struct ReasoningResult {
    std::string content;
    double confidence = 0.0;
    std::string method;          // direct / causal / inductive / analogical / counterfactual / probabilistic / multi_hop
    std::vector<std::string> evidence;
    std::vector<std::string> reasoning_chain;
};

/// 统一推理引擎
class UnifiedReasoningEngine {
public:
    explicit UnifiedReasoningEngine(
        domain::knowledge::KnowledgeGraph& kg);

    /// 对问题进行多模式推理，按置信度排序返回
    auto reason(const std::string& question) const
        -> std::vector<ReasoningResult>;

    /// 关键词提取（公共工具方法）
    static auto extract_keywords(const std::string& text)
        -> std::vector<std::string>;

    /// 设置外部依赖（可选）
    void set_statistical_learner(learning::StatisticalLearner* sl);
    void set_episodic_memory(memory::EpisodicMemory* em);

private:
    // 推理路径
    auto direct_query(const std::string& question) const
        -> std::vector<ReasoningResult>;
    auto causal_reasoning(const std::string& question) const
        -> std::vector<ReasoningResult>;
    auto inductive_reasoning(const std::string& question) const
        -> std::vector<ReasoningResult>;
    auto analogical_reasoning(const std::string& question) const
        -> std::vector<ReasoningResult>;
    auto counterfactual_reasoning(const std::string& question) const
        -> std::vector<ReasoningResult>;
    auto probabilistic_reasoning(const std::string& question) const
        -> std::vector<ReasoningResult>;
    auto multi_hop_reasoning(const std::string& start_entity,
                              const std::string& question) const
        -> std::vector<ReasoningResult>;

    // 辅助
    static auto find_common_patterns(
        const std::vector<std::string>& memories) -> std::vector<std::string>;

    // 依赖
    domain::knowledge::KnowledgeGraph& kg_;
    learning::StatisticalLearner* stat_learner_ = nullptr;
    memory::EpisodicMemory* episodic_ = nullptr;
};

}  // namespace ai_learning::reasoning
