/**
 * @file test_grounding_development_evolver.cpp
 * @brief 符号接地、语言发展、自主进化的测试
 */

#include <catch2/catch_test_macros.hpp>

#include "ai_learning/language/grounding.hpp"
#include "ai_learning/language/development.hpp"
#include "ai_learning/core/evolver.hpp"
#include "ai_learning/core/learner.hpp"

using namespace ai_learning;

// ============================================================================
// 符号接地测试
// ============================================================================

TEST_CASE("Grounding: 感知聚类创建", "[language][grounding]") {
    language::GroundingModule gm(4);

    auto id1 = gm.ground_from_perception({1.0F, 0.0F, 0.0F, 0.0F});
    auto id2 = gm.ground_from_perception({0.0F, 1.0F, 0.0F, 0.0F});

    CHECK(id1 != id2);  // 不同感知应创建不同聚类
    CHECK(gm.cluster_count() == 2);
}

TEST_CASE("Grounding: 相似感知合并到同一聚类", "[language][grounding]") {
    language::GroundingModule gm(4);

    auto id1 = gm.ground_from_perception({1.0F, 0.0F, 0.0F, 0.0F});
    auto id2 = gm.ground_from_perception({1.01F, 0.01F, 0.01F, 0.0F});

    CHECK(id1 == id2);  // 足够近的感知应合并
    CHECK(gm.cluster_count() == 1);
}

TEST_CASE("Grounding: 社会标注接地", "[language][grounding]") {
    language::GroundingModule gm(4);

    gm.ground_from_social("red", {1.0F, 0.0F, 0.0F, 0.0F}, "color");
    gm.ground_from_social("blue", {0.0F, 0.0F, 1.0F, 0.0F}, "color");

    CHECK(gm.symbol_count() == 2);

    auto red_meaning = gm.get_symbol_meaning("red");
    REQUIRE(red_meaning.has_value());
    CHECK_FALSE(red_meaning->referent_clusters.empty());
    CHECK(red_meaning->usage_count == 1);

    auto symbols = gm.get_grounded_symbols();
    CHECK(symbols.size() == 2);
}

TEST_CASE("Grounding: 重复社会标注增强置信度", "[language][grounding]") {
    language::GroundingModule gm(4);

    gm.ground_from_social("cat", {1.0F, 0.0F, 0.0F, 0.0F});
    float conf1 = gm.get_symbol_meaning("cat")->confidence;

    gm.ground_from_social("cat", {1.0F, 0.0F, 0.0F, 0.0F});
    float conf2 = gm.get_symbol_meaning("cat")->confidence;

    CHECK(conf2 > conf1);
    CHECK(gm.get_symbol_meaning("cat")->usage_count == 2);
}

TEST_CASE("Grounding: 相似概念检索", "[language][grounding]") {
    language::GroundingModule gm(4);

    (void)gm.ground_from_perception({1.0F, 0.0F, 0.0F, 0.0F});  // cluster 0
    (void)gm.ground_from_perception({0.0F, 0.0F, 0.0F, 1.0F});  // cluster 1

    auto similar = gm.find_similar_concepts({0.9F, 0.1F, 0.0F, 0.0F}, 2);
    REQUIRE(similar.size() >= 1);
    CHECK(similar[0].cluster_id == 0);  // 应匹配到 cluster 0
    CHECK(similar[0].similarity > 0.5F);
}

TEST_CASE("Grounding: 未接地符号返回空", "[language][grounding]") {
    language::GroundingModule gm(4);
    CHECK_FALSE(gm.get_symbol_meaning("unknown").has_value());
}

// ============================================================================
// 语言发展测试
// ============================================================================

TEST_CASE("Development: 初始阶段为感知运动", "[language][development]") {
    language::DevelopmentTracker dt;
    CHECK(dt.current_stage() == language::Stage::Sensorimotor);
    CHECK(dt.stage_name() == "sensorimotor");
}

TEST_CASE("Development: 预测准确率达标可晋升", "[language][development]") {
    language::DevelopmentTracker dt;

    language::Evaluation eval;
    eval.prediction_accuracy = 0.7F;  // > 0.6

    CHECK(dt.try_advance(eval));
    CHECK(dt.current_stage() == language::Stage::SingleWord);
}

TEST_CASE("Development: 词汇量达标可晋升", "[language][development]") {
    language::DevelopmentTracker dt;

    language::Evaluation eval;
    eval.vocabulary_size = 5;  // >= 5

    CHECK(dt.try_advance(eval));
}

TEST_CASE("Development: 条件不足不能晋升", "[language][development]") {
    language::DevelopmentTracker dt;

    language::Evaluation eval;
    eval.prediction_accuracy = 0.3F;
    eval.vocabulary_size = 2;

    CHECK_FALSE(dt.try_advance(eval));
    CHECK(dt.current_stage() == language::Stage::Sensorimotor);
}

TEST_CASE("Development: 逐级晋升到读写", "[language][development]") {
    language::DevelopmentTracker dt;

    // sensorimotor → single_word
    language::Evaluation e1;
    e1.prediction_accuracy = 0.7F;
    CHECK(dt.try_advance(e1));
    CHECK(dt.stage_name() == "single_word");

    // single_word → two_word
    language::Evaluation e2;
    e2.vocabulary_size = 12;
    CHECK(dt.try_advance(e2));
    CHECK(dt.stage_name() == "two_word");

    // two_word → complex
    language::Evaluation e3;
    e3.grammar_complexity = 0.6F;
    CHECK(dt.try_advance(e3));
    CHECK(dt.stage_name() == "complex");

    // complex → literacy
    language::Evaluation e4;
    e4.grammar_complexity = 0.7F;
    CHECK(dt.try_advance(e4));
    CHECK(dt.stage_name() == "literacy");

    // literacy 不能再晋升
    CHECK_FALSE(dt.try_advance(e4));
}

TEST_CASE("Development: 阶段转换记录历史", "[language][development]") {
    language::DevelopmentTracker dt;

    language::Evaluation e;
    e.prediction_accuracy = 0.7F;
    (void)dt.try_advance(e);

    REQUIRE(dt.history().size() == 1);
    CHECK(dt.history()[0].from_stage == language::Stage::Sensorimotor);
    CHECK(dt.history()[0].to_stage == language::Stage::SingleWord);
}

TEST_CASE("Development: 阶段名称转换", "[language][development]") {
    CHECK(language::stage_to_string(language::Stage::Sensorimotor) == "sensorimotor");
    CHECK(language::stage_to_string(language::Stage::Literacy) == "literacy");
    CHECK(language::stage_from_string("two_word") == language::Stage::TwoWord);
}

// ============================================================================
// 自主进化测试
// ============================================================================

TEST_CASE("Evolver: 能力评估返回所有能力", "[core][evolver]") {
    core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    core::Learner learner(config);
    core::Evolver evolver(learner);

    auto caps = evolver.evaluate_capabilities();
    CHECK(caps.count("semantic_understanding") == 1);
    CHECK(caps.count("causal_reasoning") == 1);
    CHECK(caps.count("concept_formation") == 1);
    CHECK(caps.count("knowledge_retrieval") == 1);
}

TEST_CASE("Evolver: 进化后分数应记录", "[core][evolver]") {
    core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    core::Learner learner(config);
    core::Evolver evolver(learner);

    auto result = evolver.evolve(1);
    CHECK(result.iterations == 1);
    CHECK(result.score_before >= 0.0F);
    // score_after 应该被设置
    CHECK(result.score_after >= 0.0F);
}

TEST_CASE("Evolver: 弱项识别", "[core][evolver]") {
    core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    core::Learner learner(config);
    core::Evolver evolver(learner);

    auto caps = evolver.evaluate_capabilities();
    // 初始化后应有能力评估结果
    CHECK_FALSE(caps.empty());
    // 各项分数应在合理范围
    for (const auto& [name, result] : caps) {
        CHECK(result.score >= 0.0F);
        CHECK(result.score <= 1.0F);
    }
}
