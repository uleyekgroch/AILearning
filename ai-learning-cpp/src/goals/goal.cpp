/**
 * @file goal.cpp
 * @brief Goal 值对象序列化 + 枚举转换
 */

#include "ai_learning/goals/goal.hpp"

#include <algorithm>
#include <sstream>

namespace ai_learning::goals {

// ── 枚举 → 字符串 ──────────────────────────────────────────────

auto goal_status_to_string(GoalStatus s) -> std::string {
    switch (s) {
        case GoalStatus::Pending:    return "pending";
        case GoalStatus::InProgress: return "in_progress";
        case GoalStatus::Completed:  return "completed";
        case GoalStatus::Blocked:    return "blocked";
        case GoalStatus::Abandoned:  return "abandoned";
    }
    return "pending";
}

auto string_to_goal_status(const std::string& s) -> GoalStatus {
    if (s == "in_progress") return GoalStatus::InProgress;
    if (s == "completed")   return GoalStatus::Completed;
    if (s == "blocked")     return GoalStatus::Blocked;
    if (s == "abandoned")   return GoalStatus::Abandoned;
    return GoalStatus::Pending;
}

auto strategy_to_string(LearningStrategy s) -> std::string {
    switch (s) {
        case LearningStrategy::Decompose:  return "decompose";
        case LearningStrategy::Analogize:  return "analogize";
        case LearningStrategy::Practice:   return "practice";
        case LearningStrategy::Explore:    return "explore";
    }
    return "explore";
}

auto action_to_string(StepAction a) -> std::string {
    switch (a) {
        case StepAction::Learn:    return "learn";
        case StepAction::Execute:  return "execute";
    }
    return "learn";
}

// ── 辅助：逗号连接 ──────────────────────────────────────────────

static auto join_comma(const std::vector<std::string>& v) -> std::string {
    std::string r;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) r += ",";
        r += v[i];
    }
    return r;
}

static auto split_comma(const std::string& s) -> std::vector<std::string> {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // 去除首尾空格
        auto start = token.find_first_not_of(' ');
        auto end   = token.find_last_not_of(' ');
        if (start != std::string::npos && end != std::string::npos) {
            result.push_back(token.substr(start, end - start + 1));
        }
    }
    return result;
}

// ── Goal::to_map ────────────────────────────────────────────────

auto Goal::to_map() const -> std::map<std::string, std::string> {
    return {
        {"id",                 id},
        {"description",        description},
        {"status",             goal_status_to_string(status)},
        {"sub_goals",          join_comma(sub_goals)},
        {"parent_goal",        parent_goal},
        {"required_knowledge", join_comma(required_knowledge)},
        {"priority",           std::to_string(priority)},
        {"progress",           std::to_string(progress)},
        {"created_step",       std::to_string(created_step)},
    };
}

// ── Goal::from_map ──────────────────────────────────────────────

auto Goal::from_map(const std::map<std::string, std::string>& m) -> Goal {
    Goal g;
    g.id          = m.count("id")          ? m.at("id")          : "";
    g.description = m.count("description") ? m.at("description") : "";
    g.status      = m.count("status")      ? string_to_goal_status(m.at("status"))
                                           : GoalStatus::Pending;
    g.sub_goals          = m.count("sub_goals")          ? split_comma(m.at("sub_goals"))          : std::vector<std::string>{};
    g.parent_goal        = m.count("parent_goal")        ? m.at("parent_goal")                     : "";
    g.required_knowledge = m.count("required_knowledge") ? split_comma(m.at("required_knowledge")) : std::vector<std::string>{};
    g.priority     = m.count("priority")     ? std::stod(m.at("priority"))     : 0.5;
    g.progress     = m.count("progress")     ? std::stod(m.at("progress"))     : 0.0;
    g.created_step = m.count("created_step") ? std::stoi(m.at("created_step")) : 0;
    return g;
}

}  // namespace ai_learning::goals
