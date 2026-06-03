/**
 * @file test_simulation_verification.cpp
 * @brief 测试模拟推理和知识验证
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/reasoning/simulation.hpp"
#include "ai_learning/learning/verification.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/domain/knowledge/entity.hpp"
#include "ai_learning/domain/knowledge/relation.hpp"
#include "ai_learning/core/learner.hpp"

using namespace ai_learning;
using namespace ai_learning::reasoning;
using namespace ai_learning::learning;
using namespace ai_learning::domain::knowledge;
using namespace ai_learning::core;

// ═══════════════════════════════════════════════════════════════
// 模拟推理测试
// ═══════════════════════════════════════════════════════════════

TEST_CASE("SimulationReasoning: 场景构建", "[simulation]") {
    KnowledgeGraph kg;

    // 添加因果知识
    kg.add_entity(Entity("下雨", "concept", {}, 0.8));
    kg.add_entity(Entity("地面湿", "concept", {}, 0.8));
    kg.add_relation(Relation("下雨", "地面湿", "导致", 0.9));

    SimulationReasoning sr(kg);

    auto result = sr.reason("下雨会导致什么", {"下雨", "地面湿"});

    REQUIRE_FALSE(result.scene.concepts.empty());
    REQUIRE(result.scene.concepts.size() >= 2);
    CHECK(result.confidence > 0.0);
}

TEST_CASE("SimulationReasoning: 因果链追踪", "[simulation]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("加热", "concept", {}, 0.9));
    kg.add_entity(Entity("水", "concept", {}, 0.9));
    kg.add_entity(Entity("水蒸气", "concept", {}, 0.9));
    kg.add_relation(Relation("加热", "水蒸气", "导致", 0.9));
    kg.add_relation(Relation("水", "水蒸气", "变成", 0.8));

    SimulationReasoning sr(kg);
    auto result = sr.reason("加热导致什么", {"加热", "水", "水蒸气"});

    CHECK(result.reasoning_type == "causal");
    // 应该找到至少一条因果链
    CHECK_FALSE(result.causal_chains.empty());
}

TEST_CASE("SimulationReasoning: 反事实推理", "[simulation]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("下雨", "concept", {}, 0.8));
    kg.add_entity(Entity("洪水", "concept", {}, 0.8));
    kg.add_relation(Relation("下雨", "洪水", "导致", 0.7));

    SimulationReasoning sr(kg);
    auto result = sr.reason("如果不下雨会怎样", {"下雨", "洪水"});

    CHECK(result.reasoning_type == "counterfactual");
    CHECK_FALSE(result.counterfactuals.empty());
}

TEST_CASE("SimulationReasoning: 表达为自然语言", "[simulation]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("闪电", "concept", {}, 0.8));
    kg.add_entity(Entity("雷声", "concept", {}, 0.8));
    kg.add_relation(Relation("闪电", "雷声", "导致", 0.9));

    SimulationReasoning sr(kg);
    auto result = sr.reason("闪电和雷声的关系", {"闪电", "雷声"});

    auto answer = sr.express(result, "闪电和雷声的关系");
    REQUIRE_FALSE(answer.empty());
    // 回答应包含关键概念
    bool has_concept = answer.find("闪电") != std::string::npos ||
                       answer.find("雷声") != std::string::npos ||
                       answer.find("相关") != std::string::npos;
    CHECK(has_concept);
}

TEST_CASE("SimulationReasoning: 统计", "[simulation]") {
    KnowledgeGraph kg;
    SimulationReasoning sr(kg);

    auto r1 = sr.reason("test", {"a", "b"});
    (void)r1;
    auto r2 = sr.reason("如果怎样", {"a"});
    (void)r2;

    auto stats = sr.get_stats();
    CHECK(stats.at("total_reasoning") == 2.0);
}

// ═══════════════════════════════════════════════════════════════
// 知识验证测试
// ═══════════════════════════════════════════════════════════════

TEST_CASE("KnowledgeVerifier: 验证通过", "[verification]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("人工智能", "concept", {}, 0.8));
    kg.add_entity(Entity("计算机科学", "concept", {}, 0.8));
    kg.add_relation(Relation("人工智能", "计算机科学", "是", 0.9));

    KnowledgeVerifier verifier;
    auto report = verifier.verify(
        {{"人工智能", "是", "计算机科学"}},
        kg
    );

    CHECK(report.passed);
    CHECK_THAT(report.score, Catch::Matchers::WithinAbs(1.0, 0.01));
    REQUIRE(report.tests.size() == 1);
    CHECK(report.tests[0].passed);
}

TEST_CASE("KnowledgeVerifier: 验证失败", "[verification]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("Python", "concept", {}, 0.8));
    // 不添加相关关系，所以验证应该失败

    KnowledgeVerifier verifier;
    auto report = verifier.verify(
        {{"Python", "是", "编程语言"}},
        kg
    );

    CHECK_FALSE(report.passed);
    CHECK_THAT(report.score, Catch::Matchers::WithinAbs(0.0, 0.01));
}

TEST_CASE("KnowledgeVerifier: 部分通过", "[verification]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("猫", "concept", {}, 0.8));
    kg.add_entity(Entity("动物", "concept", {}, 0.8));
    kg.add_relation(Relation("猫", "动物", "是", 0.9));

    KnowledgeVerifier verifier;
    auto report = verifier.verify(
        {{"猫", "是", "动物"}, {"猫", "属于", "哺乳动物"}},
        kg
    );

    CHECK_FALSE(report.passed);  // 第二个会失败
    CHECK_THAT(report.score, Catch::Matchers::WithinAbs(0.5, 0.01));
    REQUIRE(report.tests.size() == 2);
}

TEST_CASE("KnowledgeVerifier: 空三元组", "[verification]") {
    KnowledgeGraph kg;
    KnowledgeVerifier verifier;

    auto report = verifier.verify({}, kg);
    CHECK(report.passed);
    CHECK_THAT(report.score, Catch::Matchers::WithinAbs(0.0, 0.01));
    CHECK(report.tests.empty());
}

// ═══════════════════════════════════════════════════════════════
// 集成测试：Learner + 模拟推理 + 验证
// ═══════════════════════════════════════════════════════════════

TEST_CASE("Learner: 模拟推理集成", "[integration]") {
    LearnerConfig config;
    config.obs_dim    = 16;
    config.action_dim = 4;

    Learner learner(config);

    // 先学习因果知识
    learner.learn_from_text("下雨导致地面湿了");
    learner.learn_from_text("闪电引起雷声");

    // 查询因果问题
    auto answer = learner.think("闪电和雷声");
    // 应该能给出某种回答（不管是 KG 还是模拟推理）
    REQUIRE_FALSE(answer.empty());
    CHECK(answer != "抱歉，我暂时不知道答案");
}

TEST_CASE("Learner: 反事实推理集成", "[integration]") {
    LearnerConfig config;
    config.obs_dim    = 16;
    config.action_dim = 4;

    Learner learner(config);

    learner.learn_from_text("下雨导致地面湿了");

    // 模拟推理应该在 think() 中被调用
    // 但如果 TextLearner 的 KG 推理已经能回答就不需要模拟推理
    auto answer = learner.think("什么是下雨");
    CHECK_FALSE(answer.empty());
}
