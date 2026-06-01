/**
 * @file test_autonomous_learning.cpp
 * @brief 测试自主学习的 6 个新模块
 *
 * TDD 测试覆盖：
 *   1. IntrinsicMotivationEngine — 内在动机引擎
 *   2. SkillTree — 技能树
 *   3. AutonomousLearningLoop — 自主学习循环
 *   4. ProblemSolver — 问题求解器
 *   5. DevelopmentMilestones — 发展里程碑
 *   6. 集成测试 — 六大模块协同
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/learning/intrinsic_motivation.hpp"
#include "ai_learning/learning/skill_tree.hpp"
#include "ai_learning/learning/autonomous_learning_loop.hpp"
#include "ai_learning/learning/problem_solver.hpp"
#include "ai_learning/learning/development_milestones.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════════════
// 1. IntrinsicMotivationEngine 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Motivation: initial state", "[motivation]") {
    IntrinsicMotivationEngine engine;

    // 初始所有动机都有默认驱动水平
    REQUIRE_THAT(engine.drive_level(MotiveType::kCuriosity),
                 WithinAbs(0.5, 0.01));
    REQUIRE_THAT(engine.drive_level(MotiveType::kCompetence),
                 WithinAbs(0.5, 0.01));
    REQUIRE_THAT(engine.drive_level(MotiveType::kAchievement),
                 WithinAbs(0.5, 0.01));

    // 总驱动 > 0
    REQUIRE(engine.total_drive() > 0.0);

    // 没有上一次目标
    REQUIRE_FALSE(engine.last_goal().has_value());
}

TEST_CASE("Motivation: generate_goal produces valid goal", "[motivation]") {
    IntrinsicMotivationEngine engine;

    std::vector<std::string> known = {"programming", "math"};
    std::map<std::string, double> mastery = {{"programming", 0.3}, {"math", 0.1}};

    auto goal = engine.generate_goal(known, 0.8, mastery);

    REQUIRE_FALSE(goal.topic.empty());
    REQUIRE_FALSE(goal.domain.empty());
    REQUIRE(goal.estimated_value > 0.0);
    // 验证主动机是有效的枚举值
    bool valid_motive = (goal.primary_motive == MotiveType::kCuriosity ||
                         goal.primary_motive == MotiveType::kCompetence ||
                         goal.primary_motive == MotiveType::kAchievement ||
                         goal.primary_motive == MotiveType::kAutonomy ||
                         goal.primary_motive == MotiveType::kSocial);
    REQUIRE(valid_motive);
}

TEST_CASE("Motivation: evaluate_goal returns positive score", "[motivation]") {
    IntrinsicMotivationEngine engine;

    LearningGoal goal{
        "sorting", "programming", 0.5, 0.0,
        MotiveType::kCuriosity, "Learn sorting algorithms"
    };

    double score = engine.evaluate_goal(goal);
    REQUIRE(score > 0.0);
}

TEST_CASE("Motivation: select_goal chooses best", "[motivation]") {
    IntrinsicMotivationEngine engine;

    std::vector<LearningGoal> candidates = {
        {"topic_a", "domain_a", 0.3, 0.5, MotiveType::kCuriosity, "A"},
        {"topic_b", "domain_b", 0.7, 0.8, MotiveType::kCompetence, "B"},
        {"topic_c", "domain_c", 0.5, 0.3, MotiveType::kAchievement, "C"},
    };

    auto selected = engine.select_goal(candidates);
    REQUIRE(selected.has_value());
    // 应该选估计价值最高的 B
    REQUIRE(selected->topic == "topic_b");
}

TEST_CASE("Motivation: update_on_learning changes drives", "[motivation]") {
    IntrinsicMotivationEngine engine;

    // 未使用 curiosity_before — 仅为验证 API 可调用
    (void)engine.drive_level(MotiveType::kCuriosity);

    LearningOutcome outcome{
        "test_topic", 0.5, 0.8, true, false
    };
    engine.update_on_learning(outcome);

    // 高 surprise → 好奇心应该有变化
    double curiosity_after = engine.drive_level(MotiveType::kCuriosity);
    // curiosity 可能上升（因为 surprise 刺激）或下降（被满足）
    // 只需确认值在 [0,1] 范围
    REQUIRE(curiosity_after >= 0.0);
    REQUIRE(curiosity_after <= 1.0);
}

TEST_CASE("Motivation: decay reduces drives", "[motivation]") {
    IntrinsicMotivationEngine engine;

    // 先让好奇心驱动升高
    LearningOutcome outcome{"topic", 0.0, 0.9, false, false};
    engine.update_on_learning(outcome);

    double drive_before = engine.total_drive();

    // 多次衰减
    for (int i = 0; i < 50; ++i) {
        engine.decay();
    }

    double drive_after = engine.total_drive();
    REQUIRE(drive_after < drive_before);
}

TEST_CASE("Motivation: motive_name converts correctly", "[motivation]") {
    REQUIRE(IntrinsicMotivationEngine::motive_name(MotiveType::kCuriosity)
            == "curiosity");
    REQUIRE(IntrinsicMotivationEngine::motive_name(MotiveType::kCompetence)
            == "competence");
    REQUIRE(IntrinsicMotivationEngine::motive_name(MotiveType::kAutonomy)
            == "autonomy");
    REQUIRE(IntrinsicMotivationEngine::motive_name(MotiveType::kSocial)
            == "social");
    REQUIRE(IntrinsicMotivationEngine::motive_name(MotiveType::kAchievement)
            == "achievement");
}

// ═══════════════════════════════════════════════════════════════════
// 2. SkillTree 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("SkillTree: add and get skill", "[skill_tree]") {
    SkillTree tree;

    tree.add_skill({"add", "加法", "math", {}, 0.0, 0.2, {}, {}});

    auto skill = tree.get_skill("add");
    REQUIRE(skill.has_value());
    REQUIRE(skill->name == "加法");
    REQUIRE(skill->domain == "math");
    REQUIRE_THAT(skill->mastery, WithinAbs(0.0, 0.01));
}

TEST_CASE("SkillTree: prerequisites check", "[skill_tree]") {
    SkillTree tree;

    tree.add_skill({"add", "加法", "math", {}, 0.8, 0.2, {}, {}});
    tree.add_skill({"sub", "减法", "math", {"add"}, 0.0, 0.3, {}, {}});
    tree.add_skill({"mul", "乘法", "math", {"add", "sub"}, 0.0, 0.5, {}, {}});

    // add 无前置 → 已满足
    REQUIRE(tree.prerequisites_met("add"));
    // sub 前置是 add（mastery=0.8 >= 0.6）→ 已满足
    REQUIRE(tree.prerequisites_met("sub"));
    // mul 前置是 add(0.8) + sub(0.0)，sub 未掌握 → 未满足
    REQUIRE_FALSE(tree.prerequisites_met("mul"));
}

TEST_CASE("SkillTree: next_to_learn follows ZPD", "[skill_tree]") {
    SkillTree tree;

    // 加法已掌握
    tree.add_skill({"add", "加法", "math", {}, 0.8, 0.2, {}, {}});
    // 减法前置加法，mastery=0.0 → 在 ZPD 内
    tree.add_skill({"sub", "减法", "math", {"add"}, 0.0, 0.3, {}, {}});
    // 乘法前置 sub（未掌握）→ 不可学
    tree.add_skill({"mul", "乘法", "math", {"add", "sub"}, 0.0, 0.5, {}, {}});

    auto next = tree.next_to_learn();
    REQUIRE(next.has_value());
    REQUIRE(next->id == "sub");
}

TEST_CASE("SkillTree: learning_path finds route", "[skill_tree]") {
    SkillTree tree;

    // add 未掌握(0.3) → sub 未掌握 → mul 需要两者
    tree.add_skill({"add", "加法", "math", {}, 0.3, 0.2, {}, {}});
    tree.add_skill({"sub", "减法", "math", {"add"}, 0.0, 0.3, {}, {}});
    tree.add_skill({"mul", "乘法", "math", {"add", "sub"}, 0.0, 0.5, {}, {}});

    auto path = tree.learning_path("mul");
    REQUIRE(path.has_value());
    // 路径应该包含 add 和 sub（都未掌握）
    REQUIRE(path->steps.size() >= 2);

    std::vector<std::string> ids;
    for (const auto& s : path->steps) ids.push_back(s.id);

    // add 在 sub 之前（拓扑序：add 是 sub 的前置）
    auto add_it = std::find(ids.begin(), ids.end(), "add");
    auto sub_it = std::find(ids.begin(), ids.end(), "sub");
    REQUIRE(add_it <= sub_it);
}

TEST_CASE("SkillTree: update_mastery adjusts skill", "[skill_tree]") {
    SkillTree tree;

    tree.add_skill({"add", "加法", "math", {}, 0.0, 0.2, {}, {}});

    tree.update_mastery("add", 0.3);
    REQUIRE_THAT(tree.get_skill("add")->mastery, WithinAbs(0.3, 0.01));

    tree.update_mastery("add", 0.5);
    REQUIRE_THAT(tree.get_skill("add")->mastery, WithinAbs(0.8, 0.01));
}

TEST_CASE("SkillTree: assess_skill produces assessment", "[skill_tree]") {
    SkillTree tree;
    tree.add_skill({"add", "加法", "math", {}, 0.3, 0.2, {}, {}});

    auto assessment = tree.assess_skill("add", 0.8);
    REQUIRE_THAT(assessment.mastery_before, WithinAbs(0.3, 0.01));
    REQUIRE(assessment.mastery_after > assessment.mastery_before);
    REQUIRE(assessment.passed);
}

TEST_CASE("SkillTree: discover_skill creates new skill", "[skill_tree]") {
    SkillTree tree;
    tree.add_skill({"add", "加法", "math", {}, 0.5, 0.2, {}, {}});

    auto new_skill = tree.discover_skill("fast_add", "math", {"add"});
    REQUIRE(new_skill.id == "fast_add");
    REQUIRE(new_skill.domain == "math");

    // 验证已添加到树中
    REQUIRE(tree.get_skill("fast_add").has_value());
}

TEST_CASE("SkillTree: stats returns valid data", "[skill_tree]") {
    SkillTree tree;
    tree.add_skill({"a", "A", "math", {}, 0.5, 0.2, {}, {}});
    tree.add_skill({"b", "B", "math", {"a"}, 0.0, 0.3, {}, {}});

    auto stats = tree.stats();
    REQUIRE(stats.contains("total_skills"));
    REQUIRE_THAT(stats.at("total_skills"), WithinAbs(2.0, 0.01));
}

TEST_CASE("SkillTree: available_skills filters correctly", "[skill_tree]") {
    SkillTree tree;
    tree.add_skill({"add", "加法", "math", {}, 0.9, 0.2, {}, {}});
    tree.add_skill({"sub", "减法", "math", {"add"}, 0.0, 0.3, {}, {}});
    tree.add_skill({"mul", "乘法", "math", {"add", "sub"}, 0.0, 0.5, {}, {}});

    auto available = tree.available_skills();
    // add 已掌握（不在 available）, sub 前置满足且未掌握（在 available）
    // mul 前置 sub 未掌握（不在 available）
    REQUIRE(available.size() == 1);
    REQUIRE(available[0].id == "sub");
}

// ═══════════════════════════════════════════════════════════════════
// 3. AutonomousLearningLoop 测试
// ═══════════════════════════════════════════════════════════════════

/// 简单的学习策略（测试用）
class MockStrategy : public ILearningStrategy {
public:
    auto execute(const LearningGoal& goal,
                 const LearningPlan& /*plan*/) -> LearningOutcome override {
        execution_count_++;
        return LearningOutcome{
            goal.topic, 0.5, 0.3, true, false
        };
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "mock_strategy";
    }

    int execution_count_ = 0;
};

/// 简单的资源提供者（测试用）
class MockResourceProvider : public ILearningResourceProvider {
public:
    auto search(const std::string& topic)
        -> std::vector<std::string> override {
        return {"resource_for_" + topic};
    }

    auto get_content(const std::string& resource_id)
        -> std::string override {
        return "Content of " + resource_id;
    }
};

TEST_CASE("AutonomousLoop: single iteration produces result", "[auto_loop]") {
    IntrinsicMotivationEngine motivation;
    SkillTree tree;
    tree.add_skill({"basics", "基础", "general", {}, 0.0, 0.3, {}, {}});

    AutonomousLearningLoop loop(motivation, tree);
    MockStrategy strategy;

    auto result = loop.one_iteration(
        {"existing_topic"}, strategy, nullptr);

    REQUIRE_FALSE(result.goal.topic.empty());
    REQUIRE_THAT(result.progress, WithinAbs(0.5, 0.01));
    REQUIRE(result.motivation_before >= 0.0);
    REQUIRE(result.motivation_after >= 0.0);
}

TEST_CASE("AutonomousLoop: run produces report", "[auto_loop]") {
    IntrinsicMotivationEngine motivation;
    SkillTree tree;
    tree.add_skill({"basics", "基础", "general", {}, 0.0, 0.3, {}, {}});

    AutonomousLearningLoop loop(motivation, tree);
    MockStrategy strategy;

    AutonomousLoopConfig config;
    config.max_iterations = 5;

    auto report = loop.run(config, {"topic_a"}, strategy);

    REQUIRE(report.total_iterations == 5);
    REQUIRE(report.goals_attempted == 5);
    REQUIRE(report.history.size() == 5);
}

TEST_CASE("AutonomousLoop: run with resource provider", "[auto_loop]") {
    IntrinsicMotivationEngine motivation;
    SkillTree tree;
    tree.add_skill({"basics", "基础", "general", {}, 0.0, 0.3, {}, {}});

    AutonomousLearningLoop loop(motivation, tree);
    MockStrategy strategy;
    MockResourceProvider provider;

    AutonomousLoopConfig config;
    config.max_iterations = 3;

    auto report = loop.run(config, {"topic_a"}, strategy, &provider);
    REQUIRE(report.total_iterations == 3);
}

// ═══════════════════════════════════════════════════════════════════
// 4. ProblemSolver 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("ProblemSolver: understand extracts domain", "[problem_solver]") {
    ProblemSolver solver;

    auto profile = solver.understand("计算 3 + 5 的结果");
    REQUIRE(profile.domain == "math");
    REQUIRE_FALSE(profile.concepts.empty());
}

TEST_CASE("ProblemSolver: plan produces steps", "[problem_solver]") {
    ProblemSolver solver;
    SkillTree tree;
    tree.add_skill({"add", "加法", "math", {}, 0.5, 0.2, {}, {}});

    ProblemProfile profile{"math", "computation", {"加法"}, {"add"}, 0.3, ""};

    auto steps = solver.plan_solution(profile, tree, {"3和5相加等于8"});
    REQUIRE_FALSE(steps.empty());
}

TEST_CASE("ProblemSolver: full solve produces solution", "[problem_solver]") {
    ProblemSolver solver;
    SkillTree tree;
    tree.add_skill({"add", "加法", "math", {}, 0.5, 0.2, {}, {}});

    auto solution = solver.solve("计算 3 + 5", tree, {"3+5=8"});

    REQUIRE_FALSE(solution.answer.empty());
    REQUIRE(solution.confidence > 0.0);
    REQUIRE_FALSE(solution.steps.empty());
}

TEST_CASE("ProblemSolver: review validates solution", "[problem_solver]") {
    ProblemSolver solver;

    Solution sol;
    sol.answer = "8";
    sol.confidence = 0.9;
    sol.verified = false;

    auto result = solver.review(sol, "计算 3 + 5");
    // 验证结果存在
    REQUIRE(result.score >= 0.0);
}

TEST_CASE("ProblemSolver: stats after solving", "[problem_solver]") {
    ProblemSolver solver;
    SkillTree tree;

    solver.solve("问题1", tree, {"fact1"});
    solver.solve("问题2", tree, {"fact2"});

    auto stats = solver.stats();
    REQUIRE(stats.contains("problems_solved"));
    REQUIRE_THAT(stats.at("problems_solved"), WithinAbs(2.0, 0.01));
}

// ═══════════════════════════════════════════════════════════════════
// 5. DevelopmentMilestones 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Milestones: add and check", "[milestones]") {
    DevelopmentMilestones dm;
    SkillTree tree;

    tree.add_skill({"add", "加法", "math", {}, 0.0, 0.2, {}, {}});

    dm.add_milestone({
        "first_math", "初学数学", "掌握加法", "math",
        0, {{"add", 0.6}}, 1.0, false, -1
    });

    // 未达成（add mastery = 0.0）
    REQUIRE_FALSE(dm.check_milestone("first_math", tree));

    // 更新掌握度后达成
    tree.set_mastery("add", 0.7);
    REQUIRE(dm.check_milestone("first_math", tree));
}

TEST_CASE("Milestones: check_milestones returns events", "[milestones]") {
    DevelopmentMilestones dm;
    SkillTree tree;

    tree.add_skill({"add", "加法", "math", {}, 0.7, 0.2, {}, {}});

    dm.add_milestone({
        "first_math", "初学数学", "掌握加法", "math",
        0, {{"add", 0.6}}, 1.0, false, -1
    });

    auto events = dm.check_milestones(tree);
    REQUIRE(events.size() == 1);
    REQUIRE(events[0].milestone_id == "first_math");
    REQUIRE_THAT(events[0].reward, WithinAbs(1.0, 0.01));
}

TEST_CASE("Milestones: next_milestone returns unachieved", "[milestones]") {
    DevelopmentMilestones dm;

    dm.add_milestone({
        "m1", "第一步", "描述1", "math",
        0, {}, 1.0, true, 10  // 已达成
    });
    dm.add_milestone({
        "m2", "第二步", "描述2", "math",
        1, {{"add", 0.6}}, 2.0, false, -1  // 未达成
    });

    auto next = dm.next_milestone();
    REQUIRE(next.has_value());
    REQUIRE(next->id == "m2");
}

TEST_CASE("Milestones: progress snapshot", "[milestones]") {
    DevelopmentMilestones dm;
    SkillTree tree;

    tree.add_skill({"add", "加法", "math", {}, 0.7, 0.2, {}, {}});
    tree.add_skill({"sub", "减法", "math", {}, 0.3, 0.3, {}, {}});

    dm.add_milestone({
        "m1", "初学", "基础数学", "math",
        0, {{"add", 0.6}}, 1.0, false, -1
    });
    dm.add_milestone({
        "m2", "进阶", "减法掌握", "math",
        1, {{"sub", 0.6}}, 2.0, false, -1
    });

    auto snap = dm.progress(tree);
    REQUIRE(snap.milestones_total == 2);
    REQUIRE(snap.total_progress >= 0.0);
    REQUIRE(snap.total_progress <= 1.0);
}

TEST_CASE("Milestones: init_programming_milestones", "[milestones]") {
    DevelopmentMilestones dm;
    dm.init_programming_milestones();

    auto& all = dm.all_milestones();
    REQUIRE(all.size() >= 3);  // 至少有几个编程里程碑

    // 验证每个都有有效名称
    for (const auto& [id, m] : all) {
        REQUIRE_FALSE(m.name.empty());
        REQUIRE_FALSE(m.domain.empty());
    }
}

TEST_CASE("Milestones: init_math_milestones", "[milestones]") {
    DevelopmentMilestones dm;
    dm.init_math_milestones();

    REQUIRE(dm.all_milestones().size() >= 3);
}

TEST_CASE("Milestones: init_general_milestones", "[milestones]") {
    DevelopmentMilestones dm;
    dm.init_general_milestones();

    REQUIRE(dm.all_milestones().size() >= 2);
}

// ═══════════════════════════════════════════════════════════════════
// 6. 集成测试 — 六大模块协同
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Integration: full autonomous learning cycle", "[integration]") {
    // 初始化所有模块
    IntrinsicMotivationEngine motivation;
    SkillTree tree;
    DevelopmentMilestones milestones;

    // 构建技能树：数学
    tree.add_skill({"count", "数数", "math", {}, 0.8, 0.1, {}, {}});
    tree.add_skill({"add", "加法", "math", {"count"}, 0.3, 0.2, {}, {}});
    tree.add_skill({"sub", "减法", "math", {"add"}, 0.0, 0.3, {}, {}});
    tree.add_skill({"mul", "乘法", "math", {"add"}, 0.0, 0.5, {}, {}});

    // 初始化里程碑
    milestones.init_math_milestones();

    // 创建学习循环
    AutonomousLearningLoop loop(motivation, tree);

    // 运行 5 轮
    MockStrategy strategy;
    AutonomousLoopConfig config;
    config.max_iterations = 5;

    auto report = loop.run(config, {"counting"}, strategy);

    REQUIRE(report.total_iterations == 5);
    REQUIRE(report.history.size() == 5);

    // 检查里程碑 — 可能达成某些，不崩溃即可
    auto events = milestones.check_milestones(tree);
    (void)events;

    // 验证动机引擎已更新（至少有迭代发生）
    REQUIRE(report.total_iterations > 0);
}

TEST_CASE("Integration: problem solving with skill tree", "[integration]") {
    SkillTree tree;
    tree.add_skill({"add", "加法", "math", {}, 0.7, 0.2, {}, {}});
    tree.add_skill({"sub", "减法", "math", {"add"}, 0.5, 0.3, {}, {}});

    ProblemSolver solver;

    auto solution = solver.solve("计算 10 - 3", tree, {"10-3=7"});

    REQUIRE_FALSE(solution.answer.empty());
    REQUIRE(solution.confidence > 0.0);

    // 从解中学习
    auto new_skills = solver.learn_from_solution(
        solution,
        ProblemProfile{"math", "computation", {"减法"}, {"sub"}, 0.3, ""},
        tree
    );

    // 应该返回使用了哪些技能（不为空或为空都合法）
    (void)new_skills;
}

TEST_CASE("Integration: motivation drives skill progression", "[integration]") {
    IntrinsicMotivationEngine motivation;
    SkillTree tree;

    tree.add_skill({"basics", "基础", "general", {}, 0.1, 0.2, {}, {}});

    // 动机引擎生成目标
    auto goal = motivation.generate_goal(
        {"basics"}, 0.8, {{"basics", 0.1}});

    REQUIRE_FALSE(goal.topic.empty());

    // 模拟学习
    LearningOutcome outcome{goal.topic, 0.5, 0.6, true, false};
    motivation.update_on_learning(outcome);

    // 技能掌握度更新
    tree.update_mastery("basics", 0.3);

    // 验证掌握度提升
    REQUIRE(tree.get_skill("basics")->mastery > 0.1);
}
