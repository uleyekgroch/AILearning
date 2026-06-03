/**
 * @file learner_autonomous.cpp
 * @brief Learner 自主学习系统 — 目标生成/循环/问题求解/里程碑
 */

#include "ai_learning/core/learner.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::core {

using namespace learning;

/// 内置学习策略：用 Learner 已有的文本学习+因果推理能力
class LearnerBuiltInStrategy : public learning::ILearningStrategy {
public:
    explicit LearnerBuiltInStrategy(
        std::function<TextLearnResult(const std::string&)> learn_fn)
        : learn_fn_(std::move(learn_fn)) {}

    auto execute(const learning::LearningGoal& goal,
                 const learning::LearningPlan& /*plan*/)
        -> learning::LearningOutcome override {
        // 用文本学习作为实际学习手段
        auto result = learn_fn_(goal.topic);

        double progress = 0.3;
        if (!result.entities.empty()) progress += 0.2;
        if (!result.triples.empty()) progress += 0.2;
        progress = std::min(1.0, progress);

        return learning::LearningOutcome{
            goal.topic,
            progress,
            0.3,     // surprise
            progress > 0.4,
            false    // goal_completed
        };
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "learner_builtin";
    }

private:
    std::function<TextLearnResult(const std::string&)> learn_fn_;
};

auto Learner::generate_learning_goal()
    -> learning::LearningGoal {
    // 收集已知主题
    std::vector<std::string> topics;
    for (const auto& [id, _] : skill_tree_.all_skills()) {
        topics.push_back(id);
    }
    if (topics.empty()) {
        topics.push_back("general");
    }

    // 收集掌握度
    std::map<std::string, double> mastery;
    for (const auto& [id, skill] : skill_tree_.all_skills()) {
        mastery[id] = skill.mastery;
    }

    double curiosity = engine_->get_curiosity();
    return motivation_.generate_goal(topics, curiosity, mastery);
}

auto Learner::autonomous_learning_run(int iterations)
    -> learning::AutonomousLoopReport {

    // 收集已知主题
    std::vector<std::string> topics;
    for (const auto& [id, _] : skill_tree_.all_skills()) {
        topics.push_back(id);
    }
    if (topics.empty()) {
        topics.push_back("general");
    }

    learning::AutonomousLearningLoop loop(motivation_, skill_tree_);
    LearnerBuiltInStrategy strategy(
        [this](const std::string& text) {
            return learn_from_text(text);
        });

    learning::AutonomousLoopConfig config;
    config.max_iterations = iterations;

    return loop.run(config, topics, strategy);
}

auto Learner::solve_problem(const std::string& problem_description)
    -> learning::Solution {
    // 收集已知事实（从知识图谱获取实体信息）
    std::vector<std::string> facts;
    return problem_solver_.solve(problem_description, skill_tree_, facts);
}

auto Learner::check_milestones()
    -> std::vector<learning::MilestoneEvent> {
    return milestones_.check_milestones(skill_tree_);
}

auto Learner::learning_progress() const
    -> learning::ProgressSnapshot {
    return milestones_.progress(skill_tree_);
}

}  // namespace ai_learning::core
