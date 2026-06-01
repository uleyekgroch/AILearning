/**
 * @file world_model.cpp
 * @brief 世界模型实现 — 因果 DAG + Pearl 三层推理 + DreamerV3 想象规划
 *
 * 参考：
 *   - DreamerV3 (Hafner et al., Nature 2025)
 *   - Pearl 因果阶梯 (Pearl 2009)
 *   - Active Inference (Friston 2025)
 */

#include "ai_learning/reasoning/world_model.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

namespace ai_learning::reasoning {

// ── 构造 ──────────────────────────────────────────────────────────

WorldModel::WorldModel(int seed) : rng_(seed) {}

// ── 观察（Pearl 第一层：Seeing）───────────────────────────────────

void WorldModel::observe_sequence(const std::vector<std::string>& events,
                                    const std::string& outcome) {
    ++observations_processed_;

    // 记录所有事件
    for (const auto& e : events) {
        ensure_node_(e);
        event_counts_[e]++;
    }
    ensure_node_(outcome);
    event_counts_[outcome]++;

    // 更新共现计数
    for (size_t i = 0; i < events.size(); ++i) {
        cooccurrence_[events[i]][outcome]++;
        for (size_t j = i + 1; j < events.size(); ++j) {
            cooccurrence_[events[i]][events[j]]++;
            cooccurrence_[events[j]][events[i]]++;
        }
    }

    // 从共现中推断因果
    for (const auto& event : events) {
        auto inferred = infer_causal_direction_(event, outcome);
        if (inferred.has_value()) {
            // 检查是否已存在
            bool exists = false;
            for (auto& edge : edges_) {
                if (edge.from == inferred->from && edge.to == inferred->to) {
                    // 更新强度
                    update_strength_(edge, 1.0);
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                edges_.push_back(inferred.value());
            }
        }
    }
}

void WorldModel::observe_correlation(const std::string& var_a,
                                       const std::string& var_b,
                                       double correlation) {
    ensure_node_(var_a);
    ensure_node_(var_b);

    cooccurrence_[var_a][var_b] += static_cast<int>(correlation * 100);
    cooccurrence_[var_b][var_a] += static_cast<int>(correlation * 100);

    // 强相关才添加边
    if (std::abs(correlation) > 0.6) {
        CausalEdge edge;
        edge.from = var_a;
        edge.to = var_b;
        edge.type = CausalEdgeType::kCorrelates;
        edge.strength = std::abs(correlation);
        edges_.push_back(edge);
    }
}

void WorldModel::observe_intervention(
    const std::string& action,
    const std::map<std::string, double>& before,
    const std::map<std::string, double>& after) {

    ensure_node_(action);

    for (const auto& [var, val_after] : after) {
        ensure_node_(var);

        auto it = before.find(var);
        double val_before = (it != before.end()) ? it->second : 0.0;
        double delta = val_after - val_before;

        if (std::abs(delta) > 0.01) {
            // 检测到变化 → 可能是因果关系
            CausalEdge edge;
            edge.from = action;
            edge.to = var;
            edge.type = (delta > 0) ? CausalEdgeType::kCauses
                                     : CausalEdgeType::kPrevents;
            edge.strength = std::min(1.0, std::abs(delta));

            // 更新或添加
            bool found = false;
            for (auto& e : edges_) {
                if (e.from == edge.from && e.to == edge.to) {
                    update_strength_(e, edge.strength);
                    found = true;
                    break;
                }
            }
            if (!found) {
                edges_.push_back(edge);
            }
        }
    }
}

// ── 因果推理 ──────────────────────────────────────────────────────

auto WorldModel::predict(const std::vector<std::string>& causes) const
    -> std::vector<std::pair<std::string, double>> {
    // BFS 扩展因果链
    std::unordered_map<std::string, double> effects;

    // 初始化直接效果
    std::queue<std::pair<std::string, double>> queue;
    for (const auto& cause : causes) {
        queue.push({cause, 1.0});
    }

    std::unordered_set<std::string> visited(causes.begin(), causes.end());

    while (!queue.empty()) {
        auto [node, prob] = queue.front();
        queue.pop();

        for (const auto& edge : edges_) {
            if (edge.from == node && !visited.count(edge.to)) {
                double new_prob = prob * edge.strength;
                if (new_prob > 0.05) {  // 阈值过滤
                    auto& current = effects[edge.to];
                    current = std::max(current, new_prob);

                    if (new_prob > 0.1) {
                        visited.insert(edge.to);
                        queue.push({edge.to, new_prob});
                    }
                }
            }
        }
    }

    // 移除输入
    for (const auto& c : causes) {
        effects.erase(c);
    }

    // 排序
    std::vector<std::pair<std::string, double>> sorted(
        effects.begin(), effects.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return sorted;
}

auto WorldModel::intervene(const std::vector<Intervention>& interventions) const
    -> std::map<std::string, double> {
    // do-calculus 简化：切断被干预变量的所有入边，设置为固定值
    std::unordered_map<std::string, double> values;
    std::unordered_set<std::string> intervened;

    for (const auto& iv : interventions) {
        values[iv.variable] = iv.value;
        intervened.insert(iv.variable);
    }

    // 对非干预节点，通过因果传播计算值
    // 简化实现：基于因果边计算概率
    for (const auto& node : nodes_) {
        if (intervened.count(node)) continue;

        double total = 0.0;
        int causes = 0;
        for (const auto& edge : edges_) {
            if (edge.to == node && edge.type == CausalEdgeType::kCauses) {
                auto it = values.find(edge.from);
                if (it != values.end()) {
                    total += it->second * edge.strength;
                    ++causes;
                }
            }
        }
        if (causes > 0) {
            values[node] = total / causes;
        }
    }

    return std::map<std::string, double>(values.begin(), values.end());
}

auto WorldModel::counterfactual(const std::string& observed_outcome,
                                  const std::vector<Intervention>& /*was*/,
                                  const std::vector<Intervention>& what_if) const
    -> CounterfactualResult {
    CounterfactualResult result;
    result.original_scenario = observed_outcome;

    // 构建干预描述
    std::ostringstream desc;
    for (const auto& iv : what_if) {
        desc << iv.description << "; ";
    }
    result.intervention = desc.str();

    // 反事实推理：在修改的因果模型中推理
    // 简化实现：计算干预后的预期值变化

    // 1. 找到影响 observed_outcome 的所有路径
    auto relevant_edges = std::vector<CausalEdge>();
    for (const auto& edge : edges_) {
        if (edge.to == observed_outcome) {
            relevant_edges.push_back(edge);
        }
    }

    // 2. 应用反事实干预
    auto modified = intervene(what_if);

    // 3. 预测结果
    if (modified.contains(observed_outcome)) {
        result.predicted_outcome = observed_outcome + " 概率: " +
            std::to_string(modified.at(observed_outcome));
    } else {
        result.predicted_outcome = "无影响";
    }

    // 4. 构建推理链
    for (const auto& iv : what_if) {
        result.reasoning_chain.push_back(
            "假设: " + iv.description);

        for (const auto& edge : relevant_edges) {
            if (edge.from == iv.variable) {
                result.reasoning_chain.push_back(
                    iv.variable + " → " + observed_outcome +
                    " (强度: " + std::to_string(edge.strength) + ")");
            }
        }
    }

    // 5. 置信度
    result.confidence = 0.5;  // 基础置信度
    for (const auto& edge : relevant_edges) {
        result.confidence *= edge.strength;
    }
    result.confidence = std::min(result.confidence, 0.95);

    return result;
}

auto WorldModel::find_causal_path(const std::string& from,
                                    const std::string& to) const
    -> std::vector<std::vector<std::string>> {
    std::vector<std::vector<std::string>> all_paths;
    std::vector<std::string> path = {from};

    dfs_paths_(from, to, path, all_paths, 10);

    return all_paths;
}

// ── 想象规划（DreamerV3 风格）────────────────────────────────────

auto WorldModel::imagine_plan(const std::string& goal, int max_depth) const
    -> ImaginationPlan {
    ImaginationPlan plan;
    plan.goal = goal;

    // 反向推理：从目标出发，找哪些行动能导致目标
    std::set<std::string> visited;
    std::queue<std::pair<std::string, int>> queue;
    queue.push({goal, 0});
    visited.insert(goal);

    std::vector<std::string> actions;

    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop();

        if (depth >= max_depth) continue;

        for (const auto& edge : edges_) {
            if (edge.to == node && !visited.count(edge.from)) {
                visited.insert(edge.from);

                // 检查这个节点是否是"可执行"的（没有入边 or 是叶节点）
                bool is_action = true;
                for (const auto& e : edges_) {
                    if (e.to == edge.from) {
                        is_action = false;
                        break;
                    }
                }

                if (is_action) {
                    actions.push_back(edge.from);
                }

                queue.push({edge.from, depth + 1});
            }
        }
    }

    // 评估预期奖励：追踪每个 action 到 goal 的因果路径强度
    double total_reward = 0.0;
    double total_uncertainty = 0.0;

    for (const auto& action : actions) {
        // BFS 从 action 到 goal，追踪最短路径的累积强度
        std::unordered_map<std::string, double> reachable;
        std::queue<std::pair<std::string, double>> bfs;
        bfs.push({action, 1.0});
        std::set<std::string> bfs_visited;
        bfs_visited.insert(action);

        while (!bfs.empty()) {
            auto [node, prob] = bfs.front();
            bfs.pop();

            for (const auto& edge : edges_) {
                if (edge.from == node && !bfs_visited.count(edge.to)) {
                    double new_prob = prob * edge.strength;
                    reachable[edge.to] = std::max(reachable[edge.to], new_prob);
                    bfs_visited.insert(edge.to);
                    if (new_prob > 0.05) {
                        bfs.push({edge.to, new_prob});
                    }
                }
            }
        }

        if (reachable.count(goal)) {
            total_reward += reachable[goal];
            total_uncertainty += 1.0 - reachable[goal];
        }
    }

    if (!actions.empty()) {
        plan.expected_reward = total_reward / actions.size();
        plan.uncertainty = total_uncertainty / actions.size();
    }

    // 反转为正序（从行动到目标）
    std::reverse(actions.begin(), actions.end());
    plan.steps = actions;

    // 生成备选方案
    if (actions.size() > 2) {
        for (size_t skip = 1; skip < actions.size() - 1; ++skip) {
            std::ostringstream desc;
            desc << "跳过 " << actions[skip];
            plan.alternatives.push_back(desc.str());
        }
    }

    return plan;
}

auto WorldModel::evaluate_plan(const std::vector<std::string>& actions) const
    -> std::pair<double, double> {
    if (actions.empty()) return {0.0, 1.0};

    double total_reward = 0.0;
    double total_uncertainty = 0.0;
    int steps = 0;

    for (size_t i = 0; i < actions.size(); ++i) {
        for (size_t j = i + 1; j < actions.size(); ++j) {
            for (const auto& edge : edges_) {
                if (edge.from == actions[i] && edge.to == actions[j]) {
                    total_reward += edge.strength;
                    total_uncertainty += 1.0 - edge.strength;
                    ++steps;
                }
            }
        }
    }

    if (steps == 0) return {0.0, 1.0};

    return {total_reward / steps, total_uncertainty / steps};
}

// ── 模型管理 ──────────────────────────────────────────────────────

void WorldModel::add_causal_rule(const CausalEdge& edge) {
    ensure_node_(edge.from);
    ensure_node_(edge.to);

    // 检查是否已存在
    for (auto& e : edges_) {
        if (e.from == edge.from && e.to == edge.to) {
            e.strength = edge.strength;
            e.type = edge.type;
            return;
        }
    }

    edges_.push_back(edge);
}

bool WorldModel::remove_causal_rule(const std::string& from,
                                      const std::string& to) {
    auto it = std::remove_if(edges_.begin(), edges_.end(),
        [&](const CausalEdge& e) {
            return e.from == from && e.to == to;
        });

    if (it != edges_.end()) {
        edges_.erase(it, edges_.end());
        return true;
    }
    return false;
}

auto WorldModel::stats() const -> WorldModelStats {
    WorldModelStats s;
    s.node_count = static_cast<int>(nodes_.size());
    s.edge_count = static_cast<int>(edges_.size());
    s.observations_processed = observations_processed_;
    s.causal_rules_learned = static_cast<int>(
        std::count_if(edges_.begin(), edges_.end(),
            [](const CausalEdge& e) {
                return e.type == CausalEdgeType::kCauses;
            }));
    return s;
}

auto WorldModel::validate() const -> std::vector<std::string> {
    std::vector<std::string> issues;

    // 检查自环
    for (const auto& edge : edges_) {
        if (edge.from == edge.to) {
            issues.push_back("自环: " + edge.from);
        }
    }

    // 检查环（简化：只检查长度2的环）
    for (size_t i = 0; i < edges_.size(); ++i) {
        for (size_t j = i + 1; j < edges_.size(); ++j) {
            if (edges_[i].from == edges_[j].to &&
                edges_[i].to == edges_[j].from) {
                issues.push_back("环: " + edges_[i].from + " ↔ " + edges_[i].to);
            }
        }
    }

    return issues;
}

// ── Private ───────────────────────────────────────────────────────

void WorldModel::ensure_node_(const std::string& name) {
    if (std::find(nodes_.begin(), nodes_.end(), name) == nodes_.end()) {
        nodes_.push_back(name);
    }
}

auto WorldModel::infer_causal_direction_(const std::string& a,
                                           const std::string& b) const
    -> std::optional<CausalEdge> {
    auto it_a = cooccurrence_.find(a);
    if (it_a == cooccurrence_.end()) return std::nullopt;

    auto it_b = it_a->second.find(b);
    if (it_b == it_a->second.end()) return std::nullopt;

    // 计算条件概率 P(B|A)
    int count_a_b = it_b->second;
    int count_a = 0;
    auto it_count = event_counts_.find(a);
    if (it_count != event_counts_.end()) {
        count_a = it_count->second;
    }

    if (count_a == 0) return std::nullopt;

    double prob_b_given_a = static_cast<double>(count_a_b) /
                             static_cast<double>(count_a);

    if (prob_b_given_a > 0.5) {
        return CausalEdge{a, b, CausalEdgeType::kCauses, prob_b_given_a, {}};
    }

    return std::nullopt;
}

void WorldModel::update_strength_(CausalEdge& edge, double evidence) {
    // 贝叶斯更新：加权平均新证据
    double alpha = 0.1;  // 学习率
    edge.strength = edge.strength * (1.0 - alpha) + evidence * alpha;
    edge.strength = std::clamp(edge.strength, 0.0, 1.0);
}

auto WorldModel::dfs_paths_(const std::string& current,
                              const std::string& target,
                              std::vector<std::string>& path,
                              std::vector<std::vector<std::string>>& all_paths,
                              int depth) const -> void {
    if (depth <= 0) return;

    if (current == target) {
        all_paths.push_back(path);
        return;
    }

    for (const auto& edge : edges_) {
        if (edge.from == current) {
            // 避免环
            if (std::find(path.begin(), path.end(), edge.to) == path.end()) {
                path.push_back(edge.to);
                dfs_paths_(edge.to, target, path, all_paths, depth - 1);
                path.pop_back();
            }
        }
    }
}

}  // namespace ai_learning::reasoning
