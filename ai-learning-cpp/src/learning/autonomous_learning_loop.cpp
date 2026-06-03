/**
 * @file autonomous_learning_loop.cpp
 * @brief 自主学习循环实现
 */

#include "ai_learning/learning/autonomous_learning_loop.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

namespace ai_learning::learning {

AutonomousLearningLoop::AutonomousLearningLoop(
    IntrinsicMotivationEngine& motivation,
    SkillTree& skill_tree)
    : motivation_(motivation), skill_tree_(skill_tree) {}

auto AutonomousLearningLoop::run(
    const AutonomousLoopConfig& config,
    const std::vector<std::string>& known_topics,
    ILearningStrategy& strategy,
    ILearningResourceProvider* resource_provider)
    -> AutonomousLoopReport {

    report_ = AutonomousLoopReport{};
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < config.max_iterations; ++i) {
        auto step = one_iteration(known_topics, strategy, resource_provider);
        step.iteration = i;

        report_.history.push_back(step);
        report_.total_iterations++;
        report_.goals_attempted++;
        if (step.completed) report_.goals_completed++;
        report_.total_progress += step.progress;

        // 检查动机阈值
        if (motivation_.total_drive() < config.motivation_threshold) break;

        // 周期性巩固
        if (config.consolidation_interval > 0 &&
            (i + 1) % config.consolidation_interval == 0) {
            motivation_.decay();
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    report_.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    report_.avg_motivation = report_.total_iterations > 0
        ? report_.total_progress / report_.total_iterations : 0.0;

    return report_;
}

auto AutonomousLearningLoop::one_iteration(
    const std::vector<std::string>& known_topics,
    ILearningStrategy& strategy,
    ILearningResourceProvider* resource_provider)
    -> LearningStepResult {

    LearningStepResult result;

    // Phase 1: 动机产生 → 选择学什么
    result.goal = select_goal_(known_topics);
    result.motivation_before = motivation_.total_drive();

    // Phase 2: 课程规划 → 怎么学
    result.plan = plan_learning_(result.goal);

    // Phase 3 ★v2: 主动推理 → 预期自由能评估 + 策略选择
    auto policy = active_infer_(result.plan);
    result.selected_policy = policy.name;
    result.expected_free_energy = policy.expected_free_energy;

    // 可选：从资源提供者获取材料
    if (resource_provider) {
        auto resources = resource_provider->search(result.goal.topic);
        for (const auto& r : resources) {
            result.plan.resources.push_back(r);
        }
    }

    // Phase 4: 学习执行 → 动手学
    auto outcome = execute_learning_(result.plan, strategy);

    // Phase 5: 反思 → 学到了什么
    result.reflection = reflect_(outcome, result.goal);

    // Phase 6: 整合 → 更新技能/动机
    integrate_(outcome, result.goal);

    result.progress = outcome.progress;
    result.completed = outcome.goal_completed;
    result.motivation_after = motivation_.total_drive();

    if (outcome.mastery_improved) {
        result.learned.push_back(result.goal.topic);
    }

    return result;
}

auto AutonomousLearningLoop::select_goal_(
    const std::vector<std::string>& known_topics)
    -> LearningGoal {

    // 从技能树获取当前掌握度
    std::map<std::string, double> mastery_map;
    for (const auto& [id, skill] : skill_tree_.all_skills()) {
        mastery_map[id] = skill.mastery;
    }

    // 加入已知主题
    for (const auto& t : known_topics) {
        if (!mastery_map.contains(t)) mastery_map[t] = 0.5;
    }

    // 构建所有候选主题
    std::vector<std::string> all_topics = known_topics;
    for (const auto& [id, _] : skill_tree_.all_skills()) {
        if (std::find(all_topics.begin(), all_topics.end(), id) == all_topics.end()) {
            all_topics.push_back(id);
        }
    }

    return motivation_.generate_goal(all_topics, curiosity_signal_, mastery_map);
}

auto AutonomousLearningLoop::plan_learning_(const LearningGoal& goal)
    -> LearningPlan {

    LearningPlan plan;
    plan.goal = goal;

    // 尝试从技能树获取学习路径
    auto path = skill_tree_.learning_path(goal.topic);
    if (path.has_value()) {
        for (const auto& step : path->steps) {
            plan.steps.push_back("Learn: " + step.name);
        }
        plan.strategy = "skill_tree_guided";
    } else {
        plan.steps.push_back("Explore: " + goal.topic);
        plan.steps.push_back("Practice: " + goal.topic);
        plan.steps.push_back("Verify: " + goal.topic);
        plan.strategy = "exploration";
    }

    return plan;
}

auto AutonomousLearningLoop::execute_learning_(
    const LearningPlan& plan,
    ILearningStrategy& strategy)
    -> LearningOutcome {

    return strategy.execute(plan.goal, plan);
}

auto AutonomousLearningLoop::reflect_(
    const LearningOutcome& outcome,
    const LearningGoal& goal)
    -> std::string {

    std::string reflection = "Learned about " + goal.topic;

    if (outcome.progress > 0.5) {
        reflection += " with good progress";
    } else if (outcome.progress > 0.2) {
        reflection += " with some progress";
    } else {
        reflection += " with minimal progress";
    }

    if (outcome.surprise > 0.5) {
        reflection += ". High surprise - unexpected findings!";
    }

    if (outcome.goal_completed) {
        reflection += ". Goal completed!";
    }

    return reflection;
}

void AutonomousLearningLoop::integrate_(
    const LearningOutcome& outcome,
    const LearningGoal& goal) {

    // 更新动机引擎
    motivation_.update_on_learning(outcome);

    // 更新技能树
    if (skill_tree_.get_skill(goal.topic).has_value()) {
        skill_tree_.update_mastery(goal.topic, outcome.progress * 0.5);
    }

    // 更新好奇心信号
    curiosity_signal_ = outcome.surprise * 0.3 + curiosity_signal_ * 0.7;
}

// ★v2: 主动推理步骤实现
auto AutonomousLearningLoop::active_infer_(const LearningPlan& plan)
    -> reasoning::Policy {
    if (!active_inference_) {
        return reasoning::Policy{"passive", {}, 0.0, 0.0, 0.0};
    }

    auto belief = active_inference_->current_belief();
    auto policies = active_inference_->generate_policies(belief, 3);
    return active_inference_->select_policy(policies, belief, plan.goal.topic);
}

}  // namespace ai_learning::learning
