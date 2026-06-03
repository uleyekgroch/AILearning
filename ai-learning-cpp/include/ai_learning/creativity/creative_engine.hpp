/**
 * @file creative_engine.hpp
 * @brief 创造力引擎 — 发散思维、远程联想与组合创新
 *
 * 理论基础：
 *   - Mednick (1962) Associative Theory of Creative Process: 远程联想
 *   - Koestler (1964) The Act of Creation: bisociation（双关联想）
 *   - Boden (2004) The Creative Mind: 组合/探索/转换创造力
 *   - Guilford (1967) Divergent Thinking: 流畅性/灵活性/独创性/精细化
 *   - Finke (1992) Creative Cognition: Geneplore模型（生成→探索）
 *
 * 四种创造力机制：
 *   1. Bisociation (双关联想) — 将两个无关概念结合
 *   2. Remote Association (远程联想) — 找到遥远概念间的联系
 *   3. Combinatorial (组合创新) — 已有元素的重新组合
 *   4. Transformational (转换创新) — 改变概念空间的规则
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace ai_learning::creativity {

/// 创意类型
enum class CreativityType { kBisociation, kRemoteAssociation, kCombinatorial, kTransformational };

/// 创意结果
struct CreativeIdea {
    std::string idea;                     ///< 创意描述
    CreativityType type;                  ///< 创意类型
    double novelty = 0.0;                ///< 新颖性 0~1
    double usefulness = 0.0;             ///< 有用性 0~1
    double surprise = 0.0;              ///< 意外性 0~1
    std::vector<std::string> sources;    ///< 灵感来源
    std::string explanation;             ///< 解释如何产生的
};

/// 概念节点
struct ConceptNode {
    std::string name;                     ///< 概念名称
    std::string domain;                   ///< 所属领域
    double activation = 0.0;             ///< 当前激活水平
    std::vector<std::string> associations; ///< 关联概念
    double abstractness = 0.5;           ///< 抽象程度 0~1
};

/// 发散思维结果
struct DivergentThinkingResult {
    std::string problem;                  ///< 问题
    std::vector<CreativeIdea> ideas;      ///< 生成的创意
    int fluency = 0;                      ///< 流畅性（创意数量）
    double flexibility = 0.0;            ///< 灵活性（不同类别数）
    double originality = 0.0;            ///< 独创性（统计稀有度）
    double elaboration = 0.0;            ///< 精细化程度
};

/// 创造力引擎配置
struct CreativityConfig {
    double activation_threshold = 0.3;    ///< 激活传播阈值
    double remote_association_range = 0.7; ///< 远程联想范围（越大越远）
    int max_ideas_per_query = 10;         ///< 每次查询最大创意数
    double novelty_weight = 0.5;          ///< 新颖性权重
    double usefulness_weight = 0.5;       ///< 有用性权重
    int seed = 42;
};

/// 创造力引擎 — 实现人类式创造性思维
class CreativeEngine {
public:
    explicit CreativeEngine(const CreativityConfig& config = CreativityConfig{});

    // ═══════════════════════════════════════════════════════════
    // 发散思维 (Divergent Thinking)
    // ═══════════════════════════════════════════════════════════

    /// 对问题进行发散思维
    auto diverge(const std::string& problem,
                 const std::vector<ConceptNode>& known_concepts)
        -> DivergentThinkingResult;

    /// 头脑风暴
    auto brainstorm(const std::string& topic, int count = 10)
        -> std::vector<std::string>;

    // ═══════════════════════════════════════════════════════════
    // 双关联想 (Bisociation)
    // ═══════════════════════════════════════════════════════════

    /// 将两个无关领域的矩阵结合起来
    auto bisociate(const std::string& domain_a,
                   const std::string& domain_b,
                   const std::vector<ConceptNode>& concepts_a,
                   const std::vector<ConceptNode>& concepts_b)
        -> std::vector<CreativeIdea>;

    // ═══════════════════════════════════════════════════════════
    // 远程联想 (Remote Association)
    // ═══════════════════════════════════════════════════════════

    /// 找到两个遥远概念之间的联系
    auto remote_associate(const std::string& concept_a,
                          const std::string& concept_b,
                          const std::vector<ConceptNode>& knowledge)
        -> std::optional<CreativeIdea>;

    /// 从概念网络中寻找远程关联链
    auto find_associative_chain(const std::string& start,
                                const std::string& end,
                                const std::vector<ConceptNode>& network,
                                int max_steps = 5)
        -> std::vector<std::string>;

    // ═══════════════════════════════════════════════════════════
    // 组合创新 (Combinatorial Creativity)
    // ═══════════════════════════════════════════════════════════

    /// 组合已有元素生成新创意
    auto combine(const std::vector<ConceptNode>& elements,
                 int combination_size = 3)
        -> std::vector<CreativeIdea>;

    /// 类比迁移创新
    auto analogize_innovate(const std::string& source_domain,
                            const std::string& target_domain,
                            const std::vector<ConceptNode>& source_concepts,
                            const std::vector<ConceptNode>& target_concepts)
        -> std::vector<CreativeIdea>;

    // ═══════════════════════════════════════════════════════════
    // 评估
    // ═══════════════════════════════════════════════════════════

    /// 评估创意的新颖性
    auto assess_novelty(const CreativeIdea& idea,
                        const std::vector<ConceptNode>& known) const -> double;

    /// 评估创意的有用性
    auto assess_usefulness(const CreativeIdea& idea,
                           const std::string& problem) const -> double;

    /// 评估创意的意外性
    auto assess_surprise(const CreativeIdea& idea) const -> double;

    // ═══════════════════════════════════════════════════════════
    // 查询
    // ═══════════════════════════════════════════════════════════

    [[nodiscard]] auto total_ideas_generated() const -> int {
        return total_ideas_;
    }

    [[nodiscard]] auto config() const -> const CreativityConfig& {
        return config_;
    }

private:
    CreativityConfig config_;
    mutable std::mt19937 rng_;
    int total_ideas_ = 0;

    /// 语义距离
    static auto semantic_distance_(const std::string& a, const std::string& b) -> double;

    /// 激活传播
    auto spread_activation_(const std::string& seed,
                            const std::vector<ConceptNode>& network,
                            int steps) -> std::vector<std::string>;

    /// 概念是否属于不同领域
    static auto are_different_domains_(const ConceptNode& a,
                                       const ConceptNode& b) -> bool;
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline CreativeEngine::CreativeEngine(const CreativityConfig& config)
    : config_(config), rng_(config.seed) {}

inline auto CreativeEngine::diverge(
    const std::string& problem,
    const std::vector<ConceptNode>& known_concepts)
    -> DivergentThinkingResult {
    DivergentThinkingResult result;
    result.problem = problem;

    // 1. 从问题关键词出发，激活相关概念
    auto activated = spread_activation_(problem, known_concepts, 3);

    // 2. 对每个激活概念生成变异
    std::set<std::string> categories;
    for (const auto& concept_name : activated) {
        // 找到概念节点
        auto it = std::find_if(known_concepts.begin(), known_concepts.end(),
            [&](const auto& c) { return c.name == concept_name; });
        if (it == known_concepts.end()) continue;

        // 生成创意变体
        CreativeIdea idea;
        idea.type = CreativityType::kCombinatorial;
        idea.idea = "将" + concept_name + "应用于" + problem;
        idea.sources = {concept_name};
        idea.novelty = assess_novelty(idea, known_concepts);
        idea.usefulness = assess_usefulness(idea, problem);
        idea.surprise = assess_surprise(idea);
        idea.explanation = "从" + concept_name + "的角度重新思考" + problem;

        result.ideas.push_back(idea);
        categories.insert(it->domain);
        total_ideas_++;
    }

    // 计算发散思维指标
    result.fluency = static_cast<int>(result.ideas.size());
    result.flexibility = static_cast<double>(categories.size())
        / std::max(1.0, static_cast<double>(known_concepts.size()) * 0.5);
    result.originality = result.ideas.empty() ? 0.0
        : std::accumulate(result.ideas.begin(), result.ideas.end(), 0.0,
            [](double sum, const auto& idea) { return sum + idea.novelty; })
          / result.ideas.size();

    return result;
}

inline auto CreativeEngine::brainstorm(
    const std::string& topic, int count)
    -> std::vector<std::string> {
    std::vector<std::string> ideas;
    std::vector<std::string> prefixes = {
        "如果反向思考", "从完全不同的角度", "简化到极致",
        "夸张地放大", "结合不相干的事物", "颠倒顺序",
        "去掉核心假设", "用比喻的方式"
    };

    for (int i = 0; i < count; ++i) {
        size_t prefix_idx = i % prefixes.size();
        ideas.push_back(prefixes[prefix_idx] + "来看" + topic);
    }
    return ideas;
}

inline auto CreativeEngine::bisociate(
    const std::string& domain_a,
    const std::string& domain_b,
    const std::vector<ConceptNode>& concepts_a,
    const std::vector<ConceptNode>& concepts_b)
    -> std::vector<CreativeIdea> {
    std::vector<CreativeIdea> ideas;

    for (const auto& ca : concepts_a) {
        for (const auto& cb : concepts_b) {
            if (!are_different_domains_(ca, cb)) continue;

            CreativeIdea idea;
            idea.type = CreativityType::kBisociation;
            idea.idea = "像" + cb.name + "一样" + ca.name;
            idea.sources = {ca.name, cb.name};
            idea.novelty = 0.8;  // 双关联想天然新颖
            idea.usefulness = 0.5;
            idea.surprise = 0.7;
            idea.explanation = "将" + domain_b + "领域的" + cb.name
                             + "概念引入" + domain_a + "领域";
            ideas.push_back(idea);
            total_ideas_++;

            if (static_cast<int>(ideas.size()) >= config_.max_ideas_per_query) break;
        }
        if (static_cast<int>(ideas.size()) >= config_.max_ideas_per_query) break;
    }
    return ideas;
}

inline auto CreativeEngine::remote_associate(
    const std::string& concept_a,
    const std::string& concept_b,
    const std::vector<ConceptNode>& knowledge)
    -> std::optional<CreativeIdea> {
    auto chain = find_associative_chain(concept_a, concept_b, knowledge, 5);
    if (chain.empty()) return std::nullopt;

    CreativeIdea idea;
    idea.type = CreativityType::kRemoteAssociation;
    idea.idea = concept_a + " → " + concept_b;
    idea.novelty = semantic_distance_(concept_a, concept_b);
    idea.usefulness = 0.4;
    idea.surprise = idea.novelty;
    idea.sources = chain;
    idea.explanation = "通过关联链：";
    for (size_t i = 0; i < chain.size(); ++i) {
        if (i > 0) idea.explanation += " → ";
        idea.explanation += chain[i];
    }
    total_ideas_++;
    return idea;
}

inline auto CreativeEngine::find_associative_chain(
    const std::string& start,
    const std::string& end,
    const std::vector<ConceptNode>& network,
    int max_steps)
    -> std::vector<std::string> {
    // 简单的BFS搜索关联链
    std::vector<std::string> chain = {start};
    std::set<std::string> visited = {start};

    for (int step = 0; step < max_steps; ++step) {
        std::string& current = chain.back();
        // 找最近的关联
        double best_dist = 1e9;
        std::string best_next;
        for (const auto& node : network) {
            if (visited.count(node.name)) continue;
            double d = semantic_distance_(current, node.name);
            double d_to_target = semantic_distance_(node.name, end);
            double score = d + d_to_target * 2.0;
            if (score < best_dist) {
                best_dist = score;
                best_next = node.name;
            }
        }
        if (best_next.empty()) break;
        chain.push_back(best_next);
        visited.insert(best_next);
        if (best_next == end) break;
    }
    return chain;
}

inline auto CreativeEngine::combine(
    const std::vector<ConceptNode>& elements,
    int combination_size)
    -> std::vector<CreativeIdea> {
    std::vector<CreativeIdea> ideas;
    if (elements.size() < static_cast<size_t>(combination_size)) return ideas;

    // 随机采样组合
    std::uniform_int_distribution<size_t> dist(0, elements.size() - 1);
    for (int i = 0; i < config_.max_ideas_per_query; ++i) {
        CreativeIdea idea;
        idea.type = CreativityType::kCombinatorial;
        std::vector<std::string> picked;
        for (int j = 0; j < combination_size; ++j) {
            size_t idx = dist(rng_);
            picked.push_back(elements[idx].name);
        }
        idea.idea = "结合" + picked[0];
        for (size_t j = 1; j < picked.size(); ++j) {
            idea.idea += "与" + picked[j];
        }
        idea.sources = picked;
        idea.novelty = 0.6;
        idea.usefulness = 0.5;
        idea.surprise = 0.5;
        ideas.push_back(idea);
        total_ideas_++;
    }
    return ideas;
}

inline auto CreativeEngine::analogize_innovate(
    const std::string& source_domain,
    const std::string& target_domain,
    const std::vector<ConceptNode>& source_concepts,
    const std::vector<ConceptNode>& target_concepts)
    -> std::vector<CreativeIdea> {
    return bisociate(target_domain, source_domain,
                     target_concepts, source_concepts);
}

inline auto CreativeEngine::assess_novelty(
    const CreativeIdea& idea,
    const std::vector<ConceptNode>& known) const -> double {
    // 新颖性 = 1 - 与已知概念的最大相似度
    double max_sim = 0.0;
    for (const auto& source : idea.sources) {
        for (const auto& node : known) {
            double sim = 1.0 - semantic_distance_(source, node.name);
            max_sim = std::max(max_sim, sim);
        }
    }
    return 1.0 - max_sim;
}

inline auto CreativeEngine::assess_usefulness(
    const CreativeIdea& idea,
    const std::string& problem) const -> double {
    // 有用性 = 创意与问题的关联度
    return 1.0 - semantic_distance_(idea.idea, problem) * 0.5;
}

inline auto CreativeEngine::assess_surprise(
    const CreativeIdea& idea) const -> double {
    return idea.novelty * 0.8 + 0.2;
}

inline auto CreativeEngine::semantic_distance_(
    const std::string& a, const std::string& b) -> double {
    // 简化的语义距离：基于字符串相似度
    if (a == b) return 0.0;
    size_t common = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (a[i] == b[i]) common++;
    }
    double sim = static_cast<double>(common)
        / std::max(a.size(), b.size());
    return 1.0 - sim;
}

inline auto CreativeEngine::spread_activation_(
    const std::string& seed,
    const std::vector<ConceptNode>& network,
    int steps)
    -> std::vector<std::string> {
    std::vector<std::string> activated;
    std::set<std::string> visited;
    std::vector<std::string> frontier = {seed};

    for (int s = 0; s < steps && !frontier.empty(); ++s) {
        std::vector<std::string> next;
        for (const auto& f : frontier) {
            for (const auto& node : network) {
                if (visited.count(node.name)) continue;
                if (semantic_distance_(f, node.name) < config_.remote_association_range) {
                    next.push_back(node.name);
                    visited.insert(node.name);
                    activated.push_back(node.name);
                }
            }
        }
        frontier = next;
    }
    return activated;
}

inline auto CreativeEngine::are_different_domains_(
    const ConceptNode& a, const ConceptNode& b) -> bool {
    return a.domain != b.domain;
}

}  // namespace ai_learning::creativity