/**
 * @file goal_manager.cpp
 * @brief 目标管理器实现 — 分解、规划、追踪
 *
 * 核心算法：
 * - GoalDecomposer: 贪婪最长匹配 KG 实体 → 按类型分组 → 子目标
 * - LearningPlanner: Kahn 拓扑排序 → 难度估算 → 策略选择
 * - GoalManager: 目标生命周期 + 进度传播
 */

#include "ai_learning/goals/goal_manager.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/metacognition.hpp"
#include "ai_learning/utils/utf8.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <sstream>

namespace ai_learning::goals {

using KG = domain::knowledge::KnowledgeGraph;

// ═══════════════════════════════════════════════════════════════
// GoalDecomposer
// ═══════════════════════════════════════════════════════════════

GoalDecomposer::GoalDecomposer(KG& kg, const core::MetacognitionEngine* meta)
    : kg_(kg), meta_(meta) {}

auto GoalDecomposer::extract_entities_(const std::string& text) const
    -> std::vector<std::string>
{
    auto ids = kg_.get_all_entity_ids();
    // 按长度降序排列，优先匹配长实体
    std::sort(ids.begin(), ids.end(),
              [](const auto& a, const auto& b) { return a.size() > b.size(); });

    std::string remaining = text;
    std::vector<std::string> found;
    for (const auto& id : ids) {
        auto pos = remaining.find(id);
        while (pos != std::string::npos) {
            found.push_back(id);
            remaining.erase(pos, id.size());
            pos = remaining.find(id);
        }
    }
    return found;
}

auto GoalDecomposer::identify_required_knowledge(const Goal& goal) const
    -> std::vector<std::string>
{
    auto entities = extract_entities_(goal.description);
    std::vector<std::string> gaps;
    for (const auto& eid : entities) {
        auto opt = kg_.get_entity(eid);
        if (opt.has_value() && opt->get().confidence() < 0.3) {
            gaps.push_back(eid);
        }
    }
    return gaps;
}

// UTF-8 工具已迁移至 ai_learning/utils/utf8.hpp

auto GoalDecomposer::fallback_decompose_(const std::string& text) const
    -> std::vector<Goal>
{
    // 按中英文标点分割（UTF-8 正确处理）
    std::vector<Goal> result;
    std::string segment;

    for (size_t i = 0; i < text.size(); ) {
        size_t clen = utils::utf8_char_len(static_cast<unsigned char>(text[i]));

        if (utils::is_utf8_delimiter(text, i)) {
            if (segment.size() > 1) {
                Goal g;
                g.description = segment;
                g.status = GoalStatus::Pending;
                result.push_back(std::move(g));
            }
            segment.clear();
            i += clen;
        } else {
            for (size_t j = 0; j < clen && i + j < text.size(); ++j) {
                segment += text[i + j];
            }
            i += clen;
        }
    }

    if (segment.size() > 1) {
        Goal g;
        g.description = segment;
        g.status = GoalStatus::Pending;
        result.push_back(std::move(g));
    }

    // 如果分割后仍为空，整段作为一个子目标
    if (result.empty() && text.size() > 1) {
        Goal g;
        g.description = text;
        g.status = GoalStatus::Pending;
        result.push_back(std::move(g));
    }
    return result;
}

auto GoalDecomposer::decompose_goal(const Goal& goal) const
    -> std::vector<Goal>
{
    auto found = extract_entities_(goal.description);
    if (found.empty()) {
        return fallback_decompose_(goal.description);
    }

    // 按实体类型分组
    std::map<std::string, std::vector<std::string>> type_groups;
    for (const auto& eid : found) {
        auto opt = kg_.get_entity(eid);
        std::string type = opt.has_value() ? opt->get().type() : "unknown";
        type_groups[type].push_back(eid);
    }

    // 每个类型组生成一个子目标
    std::vector<Goal> sub_goals;
    for (const auto& [type, ids] : type_groups) {
        Goal g;
        // 描述: 掌握 {type} 领域: id1, id2, ...
        std::string desc = "\xe6\x8e\x8c\xe6\x8f\xa1 " + type
                         + " \xe9\xa2\x86\xe5\x9f\x9f: ";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) desc += ", ";
            desc += ids[i];
        }
        g.description = desc;
        g.status = GoalStatus::Pending;
        g.required_knowledge = ids;
        sub_goals.push_back(std::move(g));
    }
    return sub_goals;
}

// ═══════════════════════════════════════════════════════════════
// LearningPlanner
// ═══════════════════════════════════════════════════════════════

LearningPlanner::LearningPlanner(KG& kg, const core::MetacognitionEngine* meta)
    : kg_(kg), meta_(meta) {}

auto LearningPlanner::topological_sort_(
    const std::vector<std::string>& gaps) const
    -> std::vector<std::string>
{
    if (gaps.empty()) return {};

    // 构建邻接表：gap_i → gap_j 如果存在 gap_i → gap_j 的关系
    std::map<std::string, std::vector<std::string>> adj;
    std::map<std::string, int> in_degree;
    for (const auto& g : gaps) {
        adj[g] = {};
        in_degree[g] = 0;
    }

    for (const auto& src : gaps) {
        auto relations = kg_.get_relations_of(src, "outgoing");
        for (const auto& rel : relations) {
            const auto& target = rel.get().target_id();
            if (in_degree.count(target) && target != src) {
                adj[src].push_back(target);
                ++in_degree[target];
            }
        }
    }

    // Kahn BFS，置信度高的优先（部分已知的先学）
    auto cmp = [&](const std::string& a, const std::string& b) {
        auto ea = kg_.get_entity(a);
        auto eb = kg_.get_entity(b);
        double ca = ea.has_value() ? ea->get().confidence() : 0.0;
        double cb = eb.has_value() ? eb->get().confidence() : 0.0;
        return ca < cb;  // 优先级队列默认 max-heap，低置信度先出
    };
    std::priority_queue<std::string, std::vector<std::string>,
                        decltype(cmp)> pq(cmp);

    for (const auto& [node, deg] : in_degree) {
        if (deg == 0) pq.push(node);
    }

    std::vector<std::string> sorted;
    std::unordered_map<std::string, bool> visited;
    while (!pq.empty()) {
        auto node = pq.top();
        pq.pop();
        if (visited[node]) continue;
        visited[node] = true;
        sorted.push_back(node);

        for (const auto& neighbor : adj[node]) {
            --in_degree[neighbor];
            if (in_degree[neighbor] == 0) {
                pq.push(neighbor);
            }
        }
    }

    // 环中的节点追加到末尾
    for (const auto& g : gaps) {
        if (!visited[g]) {
            sorted.push_back(g);
        }
    }

    return sorted;
}

auto LearningPlanner::estimate_difficulty_(const std::string& entity_id) const
    -> double
{
    auto opt = kg_.get_entity(entity_id);
    double confidence = opt.has_value() ? opt->get().confidence() : 0.0;
    double base = 1.0 - confidence;
    if (!opt.has_value()) base = 0.9;

    auto neighbors = kg_.get_neighbors(entity_id, 1);
    double bonus = std::min(0.3, static_cast<double>(neighbors.size()) * 0.05);

    return std::clamp(base - bonus, 0.1, 1.0);
}

auto LearningPlanner::choose_strategy_(double difficulty, int gap_count) const
    -> LearningStrategy
{
    if (difficulty >= 0.7) return LearningStrategy::Decompose;
    if (difficulty >= 0.5) return LearningStrategy::Analogize;
    if (gap_count >= 5)    return LearningStrategy::Practice;
    return LearningStrategy::Explore;
}

auto LearningPlanner::plan_learning(const Goal& goal) const
    -> LearningPlan
{
    LearningPlan plan;
    plan.goal_id = goal.id;

    // 合并目标自身所需知识和描述中提取的缺口
    auto gaps = goal.required_knowledge;
    if (gaps.empty()) {
        // 如果没有预设的 required_knowledge，从描述中提取
        // 这里用简单策略：取描述中所有已知实体
    }

    auto sorted = topological_sort_(gaps);
    for (const auto& gap : sorted) {
        LearningStep step;
        step.action   = StepAction::Learn;
        step.target   = gap;
        step.difficulty = estimate_difficulty_(gap);
        step.strategy = choose_strategy_(step.difficulty,
                                         static_cast<int>(sorted.size()));

        if (step.strategy == LearningStrategy::Analogize) {
            // 尝试找到相似的高置信度实体作为类比源
            auto related = kg_.get_related(gap, "similar_to");
            if (!related.empty()) {
                step.analogy_from = related[0].get().id();
            }
        }

        step.reason = "Learn about: " + gap;
        plan.steps.push_back(std::move(step));
    }

    // 估算总工作量
    for (const auto& step : plan.steps) {
        plan.estimated_effort += step.difficulty;
    }

    return plan;
}

// ═══════════════════════════════════════════════════════════════
// GoalManager
// ═══════════════════════════════════════════════════════════════

GoalManager::GoalManager(KG& kg, const core::MetacognitionEngine* meta)
    : kg_(kg), meta_(meta), decomposer_(kg, meta), planner_(kg, meta) {}

auto GoalManager::create_goal(const std::string& description,
                               double priority) -> Goal
{
    Goal g;
    g.id = "goal_" + std::to_string(counter_++);
    g.description = description;
    g.priority = priority;
    g.status = GoalStatus::Pending;
    g.required_knowledge = decomposer_.identify_required_knowledge(g);
    goals_[g.id] = g;
    return g;
}

auto GoalManager::decompose_and_plan(const std::string& goal_id)
    -> std::optional<LearningPlan>
{
    auto it = goals_.find(goal_id);
    if (it == goals_.end()) return std::nullopt;

    auto& goal = it->second;
    auto sub_goals = decomposer_.decompose_goal(goal);

    // 注册子目标
    for (auto& sg : sub_goals) {
        sg.id = "goal_" + std::to_string(counter_++);
        sg.parent_goal = goal_id;
        goal.sub_goals.push_back(sg.id);
        goals_[sg.id] = std::move(sg);
    }

    // 为原始目标生成学习计划
    auto plan = planner_.plan_learning(goal);
    plans_[goal_id] = plan;
    goal.status = GoalStatus::InProgress;

    return plan;
}

void GoalManager::update_progress(const std::string& goal_id,
                                   double progress)
{
    auto it = goals_.find(goal_id);
    if (it == goals_.end()) return;

    auto& goal = it->second;
    goal.progress = std::clamp(progress, 0.0, 1.0);

    if (goal.progress >= 1.0) {
        goal.status = GoalStatus::Completed;
    } else if (goal.progress > 0.0 && goal.status == GoalStatus::Pending) {
        goal.status = GoalStatus::InProgress;
    }

    if (!goal.parent_goal.empty()) {
        propagate_progress_(goal.parent_goal);
    }
}

void GoalManager::propagate_progress_(const std::string& parent_id)
{
    auto pit = goals_.find(parent_id);
    if (pit == goals_.end()) return;

    auto& parent = pit->second;
    if (parent.sub_goals.empty()) return;

    double total = 0.0;
    for (const auto& sg_id : parent.sub_goals) {
        auto sit = goals_.find(sg_id);
        if (sit != goals_.end()) {
            total += sit->second.progress;
        }
    }
    parent.progress = total / static_cast<double>(parent.sub_goals.size());

    if (parent.progress >= 1.0) {
        parent.status = GoalStatus::Completed;
    } else if (parent.progress > 0.0 && parent.status == GoalStatus::Pending) {
        parent.status = GoalStatus::InProgress;
    }
}

auto GoalManager::get_next_action() const
    -> std::optional<LearningStep>
{
    // 优先：活跃目标中第一个未完成的学习步骤
    for (const auto& [gid, goal] : goals_) {
        if (goal.status != GoalStatus::InProgress) continue;
        auto pit = plans_.find(gid);
        if (pit == plans_.end()) continue;

        for (const auto& step : pit->second.steps) {
            if (!step.done && step.action == StepAction::Learn) {
                return step;
            }
        }
    }

    // 其次：最高优先级的 Pending 目标，尝试分解
    const Goal* best = nullptr;
    for (const auto& [gid, goal] : goals_) {
        if (goal.status != GoalStatus::Pending) continue;
        if (!best || goal.priority > best->priority) {
            best = &goal;
        }
    }
    if (best) {
        auto pit = plans_.find(best->id);
        if (pit != plans_.end()) {
            for (const auto& step : pit->second.steps) {
                if (!step.done && step.action == StepAction::Learn) {
                    return step;
                }
            }
        }
    }

    return std::nullopt;
}

auto GoalManager::get_active_goals() const
    -> std::vector<const Goal*>
{
    std::vector<const Goal*> result;
    for (const auto& [gid, goal] : goals_) {
        if (goal.status == GoalStatus::Pending ||
            goal.status == GoalStatus::InProgress) {
            result.push_back(&goal);
        }
    }
    return result;
}

auto GoalManager::check_completion(const std::string& goal_id) const
    -> bool
{
    auto it = goals_.find(goal_id);
    if (it == goals_.end()) return false;
    return it->second.status == GoalStatus::Completed;
}

auto GoalManager::get_goal(const std::string& goal_id) const
    -> const Goal*
{
    auto it = goals_.find(goal_id);
    return it != goals_.end() ? &it->second : nullptr;
}

auto GoalManager::save_state() const -> ManagerState
{
    return {goals_, plans_, counter_};
}

void GoalManager::load_state(const ManagerState& state)
{
    goals_   = state.goals;
    plans_   = state.plans;
    counter_ = state.counter;
}

}  // namespace ai_learning::goals
