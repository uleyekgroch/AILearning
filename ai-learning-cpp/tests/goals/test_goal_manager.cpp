/**
 * @file test_goal_manager.cpp
 * @brief GoalManager / GoalDecomposer / LearningPlanner 单元测试
 *
 * 测试覆盖：
 * - GoalDecomposer: 实体提取、按类型分组分解、回退分解
 * - LearningPlanner: 拓扑排序、难度估算、策略选择
 * - GoalManager: 创建、分解规划、进度更新、传播、状态持久化
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "ai_learning/goals/goal_manager.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/metacognition.hpp"

using Catch::Matchers::WithinAbs;
using namespace ai_learning::goals;
using namespace ai_learning::domain::knowledge;
using namespace ai_learning::core;

// ── 辅助：构建含已知实体的 KnowledgeGraph ──────────────────────────

static auto make_test_kg() -> KnowledgeGraph {
    KnowledgeGraph kg;

    // 物理领域实体（高置信度）
    kg.add_entity(Entity("gravity", "physics", {{"attr", "force"}}, 0.8));
    kg.add_entity(Entity("force", "physics", {{"attr", "vector"}}, 0.9));
    kg.add_entity(Entity("mass", "physics", {{"attr", "scalar"}}, 0.7));

    // 化学领域实体（低置信度 — 知识缺口）
    kg.add_entity(Entity("oxygen", "chemistry", {}, 0.1));
    kg.add_entity(Entity("hydrogen", "chemistry", {}, 0.2));

    // 添加关系
    kg.add_relation(Relation("gravity", "mass", "acts_on"));
    kg.add_relation(Relation("oxygen", "hydrogen", "reacts_with"));

    return kg;
}

// ═══════════════════════════════════════════════════════════════
// GoalDecomposer Tests
// ═══════════════════════════════════════════════════════════════

TEST_CASE("GoalDecomposer extracts embedded entities", "[goals][decomposer]") {
    auto kg = make_test_kg();
    GoalDecomposer decomposer(kg);

    Goal g;
    g.description = "learn about gravity and force";

    auto subs = decomposer.decompose_goal(g);
    CHECK_FALSE(subs.empty());

    bool found_physics = false;
    for (const auto& sub : subs) {
        if (sub.required_knowledge.size() >= 2) {
            found_physics = true;
        }
    }
    CHECK(found_physics);
}

TEST_CASE("GoalDecomposer groups by entity type", "[goals][decomposer]") {
    auto kg = make_test_kg();
    GoalDecomposer decomposer(kg);

    Goal g;
    g.description = "gravity and oxygen";

    auto subs = decomposer.decompose_goal(g);
    // gravity=physics, oxygen=chemistry -> 2 sub-goals
    CHECK(subs.size() == 2);
}

TEST_CASE("GoalDecomposer fallback when no entities match", "[goals][decomposer]") {
    KnowledgeGraph kg;  // empty KG
    GoalDecomposer decomposer(kg);

    Goal g;
    g.description = "some random text without entities";

    auto subs = decomposer.decompose_goal(g);
    CHECK_FALSE(subs.empty());
    CHECK(subs[0].description.size() > 1);
}

TEST_CASE("GoalDecomposer identify_required_knowledge", "[goals][decomposer]") {
    auto kg = make_test_kg();
    GoalDecomposer decomposer(kg);

    Goal g;
    g.description = "oxygen and hydrogen";

    auto gaps = decomposer.identify_required_knowledge(g);
    // oxygen (0.1) and hydrogen (0.2) are both < 0.3
    CHECK(gaps.size() == 2);
}

// ═══════════════════════════════════════════════════════════════
// LearningPlanner Tests
// ═══════════════════════════════════════════════════════════════

TEST_CASE("LearningPlanner produces steps for gaps", "[goals][planner]") {
    auto kg = make_test_kg();
    LearningPlanner planner(kg);

    Goal g;
    g.id = "goal_test";
    g.required_knowledge = {"oxygen", "hydrogen"};

    auto plan = planner.plan_learning(g);
    CHECK(plan.goal_id == "goal_test");
    CHECK(plan.steps.size() == 2);
    CHECK(plan.estimated_effort > 0.0);

    for (const auto& step : plan.steps) {
        CHECK_FALSE(step.target.empty());
        CHECK_THAT(step.difficulty, WithinAbs(0.5, 0.5));  // within [0.1, 1.0]
    }
}

TEST_CASE("LearningPlanner handles empty required_knowledge", "[goals][planner]") {
    KnowledgeGraph kg;
    LearningPlanner planner(kg);

    Goal g;
    g.id = "goal_empty";

    auto plan = planner.plan_learning(g);
    CHECK(plan.steps.empty());
}

// ═══════════════════════════════════════════════════════════════
// GoalManager Tests
// ═══════════════════════════════════════════════════════════════

TEST_CASE("GoalManager create_goal", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g = gm.create_goal("learn gravity", 0.7);
    CHECK_FALSE(g.id.empty());
    CHECK(g.id.substr(0, 5) == "goal_");
    CHECK(g.description == "learn gravity");
    CHECK_THAT(g.priority, WithinAbs(0.7, 1e-9));
    CHECK(g.status == GoalStatus::Pending);
    CHECK_THAT(g.progress, WithinAbs(0.0, 1e-9));
}

TEST_CASE("GoalManager create_goal auto-identifies knowledge", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g = gm.create_goal("learn oxygen and hydrogen", 0.5);
    // oxygen (0.1) and hydrogen (0.2) are low-confidence
    CHECK(g.required_knowledge.size() == 2);
}

TEST_CASE("GoalManager decompose_and_plan", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g = gm.create_goal("gravity and oxygen", 0.5);
    auto plan = gm.decompose_and_plan(g.id);

    REQUIRE(plan.has_value());
    CHECK(plan->goal_id == g.id);

    auto* updated = gm.get_goal(g.id);
    REQUIRE(updated != nullptr);
    CHECK(updated->status == GoalStatus::InProgress);
    CHECK_FALSE(updated->sub_goals.empty());
}

TEST_CASE("GoalManager decompose_and_plan unknown goal", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto plan = gm.decompose_and_plan("nonexistent");
    CHECK_FALSE(plan.has_value());
}

TEST_CASE("GoalManager update_progress", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g = gm.create_goal("test", 0.5);

    gm.update_progress(g.id, 0.3);
    auto* updated = gm.get_goal(g.id);
    REQUIRE(updated != nullptr);
    CHECK_THAT(updated->progress, WithinAbs(0.3, 1e-9));
    CHECK(updated->status == GoalStatus::InProgress);

    gm.update_progress(g.id, 1.0);
    updated = gm.get_goal(g.id);
    REQUIRE(updated != nullptr);
    CHECK_THAT(updated->progress, WithinAbs(1.0, 1e-9));
    CHECK(updated->status == GoalStatus::Completed);
}

TEST_CASE("GoalManager update_progress clamps", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g = gm.create_goal("test", 0.5);

    gm.update_progress(g.id, -0.5);
    auto* updated = gm.get_goal(g.id);
    REQUIRE(updated != nullptr);
    CHECK_THAT(updated->progress, WithinAbs(0.0, 1e-9));

    gm.update_progress(g.id, 2.0);
    updated = gm.get_goal(g.id);
    REQUIRE(updated != nullptr);
    CHECK_THAT(updated->progress, WithinAbs(1.0, 1e-9));
}

TEST_CASE("GoalManager progress propagates to parent", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g = gm.create_goal("gravity and oxygen", 0.5);
    gm.decompose_and_plan(g.id);

    auto* parent = gm.get_goal(g.id);
    REQUIRE(parent != nullptr);
    REQUIRE_FALSE(parent->sub_goals.empty());

    // Complete first sub-goal
    gm.update_progress(parent->sub_goals[0], 1.0);

    parent = gm.get_goal(g.id);
    CHECK(parent->progress > 0.0);

    // Complete all sub-goals
    for (const auto& sg_id : parent->sub_goals) {
        gm.update_progress(sg_id, 1.0);
    }

    parent = gm.get_goal(g.id);
    CHECK_THAT(parent->progress, WithinAbs(1.0, 1e-9));
    CHECK(parent->status == GoalStatus::Completed);
}

TEST_CASE("GoalManager get_active_goals", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    gm.create_goal("goal a", 0.5);
    gm.create_goal("goal b", 0.7);

    auto active = gm.get_active_goals();
    CHECK(active.size() == 2);
}

TEST_CASE("GoalManager check_completion", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g = gm.create_goal("test", 0.5);
    CHECK_FALSE(gm.check_completion(g.id));

    gm.update_progress(g.id, 1.0);
    CHECK(gm.check_completion(g.id));
}

TEST_CASE("GoalManager get_goal returns nullptr for unknown", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    CHECK(gm.get_goal("nonexistent") == nullptr);
}

TEST_CASE("GoalManager save_state / load_state round-trip", "[goals][manager]") {
    auto kg = make_test_kg();
    GoalManager gm(kg);

    auto g1 = gm.create_goal("first goal", 0.8);
    gm.decompose_and_plan(g1.id);
    gm.update_progress(g1.id, 0.5);

    auto g2 = gm.create_goal("second goal", 0.3);

    auto state = gm.save_state();

    GoalManager gm2(kg);
    gm2.load_state(state);

    CHECK(gm2.check_completion(g1.id) == gm.check_completion(g1.id));
    auto* loaded = gm2.get_goal(g1.id);
    REQUIRE(loaded != nullptr);
    CHECK(loaded->description == "first goal");
    CHECK_THAT(loaded->progress, WithinAbs(0.5, 1e-9));

    auto* loaded2 = gm2.get_goal(g2.id);
    REQUIRE(loaded2 != nullptr);
    CHECK(loaded2->description == "second goal");
}
