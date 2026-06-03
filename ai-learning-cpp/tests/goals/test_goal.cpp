/**
 * @file test_goal.cpp
 * @brief Goal 值对象单元测试 — 构造、序列化、枚举转换
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "ai_learning/goals/goal.hpp"

using Catch::Matchers::WithinAbs;
using namespace ai_learning::goals;

TEST_CASE("Goal default construction", "[goals]") {
    Goal g;
    CHECK(g.id.empty());
    CHECK(g.description.empty());
    CHECK(g.status == GoalStatus::Pending);
    CHECK(g.sub_goals.empty());
    CHECK(g.parent_goal.empty());
    CHECK(g.required_knowledge.empty());
    CHECK_THAT(g.priority, WithinAbs(0.5, 1e-9));
    CHECK_THAT(g.progress, WithinAbs(0.0, 1e-9));
    CHECK(g.created_step == 0);
}

TEST_CASE("Goal construction with values", "[goals]") {
    Goal g;
    g.id = "goal_0";
    g.description = "learn physics";
    g.status = GoalStatus::InProgress;
    g.sub_goals = {"goal_1", "goal_2"};
    g.parent_goal = "goal_parent";
    g.required_knowledge = {"gravity", "force"};
    g.priority = 0.8;
    g.progress = 0.3;
    g.created_step = 5;

    CHECK(g.id == "goal_0");
    CHECK(g.description == "learn physics");
    CHECK(g.status == GoalStatus::InProgress);
    CHECK(g.sub_goals.size() == 2);
    CHECK(g.parent_goal == "goal_parent");
    CHECK(g.required_knowledge.size() == 2);
    CHECK_THAT(g.priority, WithinAbs(0.8, 1e-9));
    CHECK_THAT(g.progress, WithinAbs(0.3, 1e-9));
}

TEST_CASE("Goal to_map/from_map round-trip", "[goals]") {
    Goal original;
    original.id = "goal_42";
    original.description = "master calculus";
    original.status = GoalStatus::InProgress;
    original.sub_goals = {"goal_43", "goal_44"};
    original.parent_goal = "goal_10";
    original.required_knowledge = {"derivatives", "integrals", "limits"};
    original.priority = 0.9;
    original.progress = 0.6;
    original.created_step = 100;

    auto m = original.to_map();
    CHECK(m.at("id") == "goal_42");
    CHECK(m.at("description") == "master calculus");
    CHECK(m.at("status") == "in_progress");
    CHECK(m.at("parent_goal") == "goal_10");

    Goal restored = Goal::from_map(m);
    CHECK(restored.id == original.id);
    CHECK(restored.description == original.description);
    CHECK(restored.status == original.status);
    CHECK(restored.sub_goals == original.sub_goals);
    CHECK(restored.parent_goal == original.parent_goal);
    CHECK(restored.required_knowledge == original.required_knowledge);
    CHECK_THAT(restored.priority, WithinAbs(original.priority, 1e-9));
    CHECK_THAT(restored.progress, WithinAbs(original.progress, 1e-9));
    CHECK(restored.created_step == original.created_step);
}

TEST_CASE("Goal from_map handles missing keys", "[goals]") {
    std::map<std::string, std::string> empty_map;
    auto g = Goal::from_map(empty_map);
    CHECK(g.id.empty());
    CHECK(g.status == GoalStatus::Pending);
    CHECK_THAT(g.priority, WithinAbs(0.5, 1e-9));
    CHECK_THAT(g.progress, WithinAbs(0.0, 1e-9));
}

TEST_CASE("GoalStatus enum conversions", "[goals]") {
    CHECK(goal_status_to_string(GoalStatus::Pending) == "pending");
    CHECK(goal_status_to_string(GoalStatus::InProgress) == "in_progress");
    CHECK(goal_status_to_string(GoalStatus::Completed) == "completed");
    CHECK(goal_status_to_string(GoalStatus::Blocked) == "blocked");
    CHECK(goal_status_to_string(GoalStatus::Abandoned) == "abandoned");

    CHECK(string_to_goal_status("pending") == GoalStatus::Pending);
    CHECK(string_to_goal_status("in_progress") == GoalStatus::InProgress);
    CHECK(string_to_goal_status("completed") == GoalStatus::Completed);
    CHECK(string_to_goal_status("blocked") == GoalStatus::Blocked);
    CHECK(string_to_goal_status("abandoned") == GoalStatus::Abandoned);
    CHECK(string_to_goal_status("unknown") == GoalStatus::Pending);
}

TEST_CASE("LearningStrategy enum conversion", "[goals]") {
    CHECK(strategy_to_string(LearningStrategy::Decompose) == "decompose");
    CHECK(strategy_to_string(LearningStrategy::Analogize) == "analogize");
    CHECK(strategy_to_string(LearningStrategy::Practice) == "practice");
    CHECK(strategy_to_string(LearningStrategy::Explore) == "explore");
}

TEST_CASE("StepAction enum conversion", "[goals]") {
    CHECK(action_to_string(StepAction::Learn) == "learn");
    CHECK(action_to_string(StepAction::Execute) == "execute");
}
