/**
 * @file autotelic_generator.cpp
 * @brief 自成目标生成器实现
 */

#include "ai_learning/learning/autotelic_generator.hpp"
#include <algorithm>
#include <random>

namespace ai_learning::learning {

void AutotelicGenerator::record_progress(const std::string& domain, double progress) {
    if (learning_progress_map_.find(domain) == learning_progress_map_.end()) {
        learning_progress_map_[domain] = progress;
    } else {
        // 指数平滑移动平均
        learning_progress_map_[domain] = 0.8 * learning_progress_map_[domain] + 0.2 * progress;
    }
}

auto AutotelicGenerator::generate_goal() -> AutotelicGoal {
    AutotelicGoal goal;
    goal.goal_type = "idle_dreaming"; // 默认

    if (learning_progress_map_.empty()) {
        goal.goal_type = "random_babbling";
        goal.expected_learning_progress = 0.1;
        return goal;
    }

    // 寻找 Learning Progress 最大 (最能激发好奇心) 的领域
    auto best_it = std::max_element(
        learning_progress_map_.begin(), learning_progress_map_.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    goal.goal_type = "explore_relation";
    goal.target_cpt_a = best_it->first;
    goal.expected_learning_progress = best_it->second;

    return goal;
}

} // namespace ai_learning::learning