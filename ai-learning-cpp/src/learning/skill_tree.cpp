/**
 * @file skill_tree.cpp
 * @brief 技能树系统实现
 */

#include "ai_learning/learning/skill_tree.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>

namespace ai_learning::learning {

SkillTree::SkillTree(const SkillTreeConfig& config) : config_(config) {}

void SkillTree::add_skill(const SkillNode& skill) {
    skills_[skill.id] = skill;
}

auto SkillTree::get_skill(const std::string& id) const
    -> std::optional<SkillNode> {
    auto it = skills_.find(id);
    if (it != skills_.end()) return it->second;
    return std::nullopt;
}

auto SkillTree::skills_by_domain(const std::string& domain) const
    -> std::vector<SkillNode> {
    std::vector<SkillNode> result;
    for (const auto& [_, s] : skills_) {
        if (s.domain == domain) result.push_back(s);
    }
    return result;
}

void SkillTree::update_mastery(const std::string& skill_id, double delta) {
    auto it = skills_.find(skill_id);
    if (it != skills_.end()) {
        it->second.mastery = std::clamp(it->second.mastery + delta, 0.0, 1.0);
    }
}

void SkillTree::set_mastery(const std::string& skill_id, double mastery) {
    auto it = skills_.find(skill_id);
    if (it != skills_.end()) {
        it->second.mastery = std::clamp(mastery, 0.0, 1.0);
    }
}

auto SkillTree::assess_skill(const std::string& skill_id, double test_score)
    -> SkillAssessment {
    SkillAssessment assessment;
    assessment.skill_id = skill_id;

    auto it = skills_.find(skill_id);
    if (it == skills_.end()) {
        assessment.progress = 0.0;
        assessment.passed = false;
        return assessment;
    }

    assessment.mastery_before = it->second.mastery;

    // 用测试分数更新掌握度
    double new_mastery = std::clamp(test_score, 0.0, 1.0);
    // 取当前和测试的加权平均
    double blended = it->second.mastery * 0.4 + new_mastery * 0.6;
    it->second.mastery = std::clamp(blended, 0.0, 1.0);

    assessment.mastery_after = it->second.mastery;
    assessment.progress = assessment.mastery_after - assessment.mastery_before;
    assessment.passed = it->second.mastery >= config_.mastery_threshold;

    return assessment;
}

auto SkillTree::next_to_learn() const -> std::optional<SkillNode> {
    auto available = available_skills();
    if (available.empty()) return std::nullopt;

    // ZPD 筛选：掌握度在 [zpd_low, zpd_high] 之间
    std::vector<SkillNode> in_zpd;
    for (const auto& s : available) {
        if (s.mastery >= config_.zpd_low && s.mastery <= config_.zpd_high) {
            in_zpd.push_back(s);
        }
    }

    // 如果 ZPD 内没有，用 available 中最低掌握度的
    if (in_zpd.empty()) in_zpd = available;

    // 按难度排序（适中优先）
    std::sort(in_zpd.begin(), in_zpd.end(),
        [](const SkillNode& a, const SkillNode& b) {
            // 选择难度最接近 0.5 的
            double da = std::abs(a.difficulty - 0.5);
            double db = std::abs(b.difficulty - 0.5);
            return da < db;
        });

    return in_zpd[0];
}

auto SkillTree::learning_path(const std::string& target_skill) const
    -> std::optional<LearningPath> {
    if (!skills_.contains(target_skill)) return std::nullopt;

    // 收集所有需要学的技能（从未掌握的前置到目标）
    std::set<std::string> needed;
    std::queue<std::string> queue;
    queue.push(target_skill);

    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();

        if (needed.contains(current)) continue;
        needed.insert(current);

        auto it = skills_.find(current);
        if (it != skills_.end()) {
            for (const auto& pre : it->second.prerequisites) {
                if (skills_.contains(pre)) {
                    auto pre_skill = skills_.at(pre);
                    if (pre_skill.mastery < config_.mastery_threshold) {
                        queue.push(pre);
                    }
                }
            }
        }
    }

    // 拓扑排序
    auto sorted = topological_sort_();
    std::vector<SkillNode> path_steps;
    double total_diff = 0.0;

    for (const auto& id : sorted) {
        if (needed.contains(id)) {
            auto skill = skills_.at(id);
            if (skill.mastery < config_.mastery_threshold) {
                path_steps.push_back(skill);
                total_diff += skill.difficulty;
            }
        }
    }

    return LearningPath{
        path_steps,
        total_diff,
        static_cast<int>(path_steps.size()),
        target_skill
    };
}

auto SkillTree::available_skills() const -> std::vector<SkillNode> {
    std::vector<SkillNode> result;
    for (const auto& [_, s] : skills_) {
        if (s.mastery < config_.mastery_threshold && prerequisites_met(s.id)) {
            result.push_back(s);
        }
    }
    return result;
}

auto SkillTree::zpd_skills() const -> std::vector<SkillNode> {
    auto available = available_skills();
    std::vector<SkillNode> result;
    for (const auto& s : available) {
        if (s.mastery >= config_.zpd_low && s.mastery <= config_.zpd_high) {
            result.push_back(s);
        }
    }
    return result;
}

auto SkillTree::prerequisites_met(const std::string& skill_id) const -> bool {
    auto it = skills_.find(skill_id);
    if (it == skills_.end()) return false;

    for (const auto& pre_id : it->second.prerequisites) {
        auto pre_it = skills_.find(pre_id);
        if (pre_it == skills_.end()) return false;
        if (pre_it->second.mastery < config_.mastery_threshold) return false;
    }
    return true;
}

auto SkillTree::all_dependencies(const std::string& skill_id) const
    -> std::vector<std::string> {
    std::vector<std::string> result;
    std::set<std::string> visited;
    std::queue<std::string> queue;

    auto it = skills_.find(skill_id);
    if (it == skills_.end()) return result;

    for (const auto& pre : it->second.prerequisites) {
        queue.push(pre);
    }

    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();
        if (visited.contains(current)) continue;
        visited.insert(current);
        result.push_back(current);

        auto cit = skills_.find(current);
        if (cit != skills_.end()) {
            for (const auto& pre : cit->second.prerequisites) {
                queue.push(pre);
            }
        }
    }

    return result;
}

auto SkillTree::discover_skill(const std::string& suggested_id,
                                const std::string& domain,
                                const std::vector<std::string>& prereqs)
    -> SkillNode {
    SkillNode new_skill{
        suggested_id,
        suggested_id,
        domain,
        prereqs,
        0.0,
        0.5,
        {},
        {}
    };

    // 估算难度 = 平均前置难度 + 0.1
    if (!prereqs.empty()) {
        double avg_pre_diff = 0.0;
        int count = 0;
        for (const auto& pid : prereqs) {
            auto it = skills_.find(pid);
            if (it != skills_.end()) {
                avg_pre_diff += it->second.difficulty;
                ++count;
            }
        }
        if (count > 0) {
            new_skill.difficulty = std::min(1.0, avg_pre_diff / count + 0.1);
        }
    }

    skills_[suggested_id] = new_skill;
    return new_skill;
}

auto SkillTree::stats() const -> std::map<std::string, double> {
    int mastered = 0;
    for (const auto& [_, s] : skills_) {
        if (s.mastery >= config_.mastery_threshold) ++mastered;
    }
    return {
        {"total_skills", static_cast<double>(skills_.size())},
        {"mastered_skills", static_cast<double>(mastered)},
        {"available_skills", static_cast<double>(available_skills().size())},
    };
}

auto SkillTree::topological_sort_() const -> std::vector<std::string> {
    std::map<std::string, int> in_degree;
    for (const auto& [id, _] : skills_) in_degree[id] = 0;
    for (const auto& [id, s] : skills_) {
        for (const auto& pre : s.prerequisites) {
            if (skills_.contains(pre)) {
                in_degree[id]++;
            }
        }
    }

    std::queue<std::string> queue;
    for (const auto& [id, deg] : in_degree) {
        if (deg == 0) queue.push(id);
    }

    std::vector<std::string> sorted;
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();
        sorted.push_back(current);

        for (const auto& [id, s] : skills_) {
            for (const auto& pre : s.prerequisites) {
                if (pre == current) {
                    in_degree[id]--;
                    if (in_degree[id] == 0) queue.push(id);
                }
            }
        }
    }

    return sorted;
}

auto SkillTree::has_cycle_(const std::string& start,
                            std::vector<std::string>& visited) const
    -> bool {
    (void)start;
    (void)visited;
    return false;  // 简化：DAG 假设无环
}

}  // namespace ai_learning::learning
