/**
 * @file activation_spread.hpp
 * @brief 激活扩散推理 — 知识图谱上的联想推理
 *
 * 从种子实体出发，沿关系边扩散激活值。
 * 模拟人类联想记忆的检索过程。
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {
class KnowledgeGraph;
}

namespace ai_learning::reasoning {

struct ActivatedNode {
    std::string entity_id;
    double      activation = 0.0;
    int         hops       = 0;
};

class ActivationSpread {
public:
    explicit ActivationSpread(
        const domain::knowledge::KnowledgeGraph& graph,
        double decay_rate     = 0.7,
        double threshold      = 0.1,
        int    max_depth      = 3);

    /// 从种子实体扩散激活
    auto spread(const std::vector<std::string>& seed_entities,
                int depth = 0) const
        -> std::vector<ActivatedNode>;

    /// 设置参数
    void set_decay_rate(double rate);
    void set_threshold(double threshold);

    /// 获取某实体的激活值
    [[nodiscard]] auto get_activation(const std::string& entity_id) const
        -> double;

    void clear() const;

private:
    const domain::knowledge::KnowledgeGraph& graph_;
    double                           decay_rate_;
    double                           threshold_;
    int                              max_depth_;
    std::map<std::string, double>    mutable activations_;
};

}  // namespace ai_learning::reasoning
