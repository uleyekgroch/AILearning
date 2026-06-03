/**
 * @file test_text_learner.cpp
 * @brief TextLearner 单元测试 — TDD
 */

#include <catch2/catch_test_macros.hpp>
#include "ai_learning/learning/text_learner.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

using namespace ai_learning::learning;
using namespace ai_learning::domain::knowledge;

TEST_CASE("TextLearner: 从简单文本学习") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    auto result = learner.learn_from_text(
        "人工智能是计算机科学的一个分支");

    SECTION("应提取实体") {
        REQUIRE_FALSE(result.entities.empty());
    }

    SECTION("应有学习统计") {
        REQUIRE(learner.stats().at("total_learned") == 1);
    }
}

TEST_CASE("TextLearner: 从因果文本学习") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    auto result = learner.learn_from_text(
        "因为下雨所以地面湿了");

    SECTION("应提取因果") {
        REQUIRE_FALSE(result.causal_links.empty());
    }
}

TEST_CASE("TextLearner: 从数值文本学习") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    auto result = learner.learn_from_text(
        "水在100度沸腾");

    SECTION("应提取数值") {
        REQUIRE_FALSE(result.numerical_facts.empty());
        REQUIRE(result.numerical_facts[0].value == 100.0);
    }
}

TEST_CASE("TextLearner: 学后思考") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    learner.learn_from_text("人工智能是计算机科学的一个分支");
    learner.learn_from_text("人工智能包括机器学习");

    auto answer = learner.think("什么是人工智能");

    SECTION("应有答案") {
        REQUIRE_FALSE(answer.empty());
        REQUIRE(answer.find("人工智能") != std::string::npos);
    }
}

TEST_CASE("TextLearner: STDP 连接") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    // 多次学习建立 STDP 连接
    for (int i = 0; i < 10; ++i) {
        learner.learn_from_text("人工智能是计算机科学的一个分支");
    }

    auto answer = learner.think("人工智能");
    REQUIRE_FALSE(answer.empty());
}

TEST_CASE("TextLearner: 海马记忆") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    learner.learn_from_text("牛顿发现了万有引力定律");
    learner.learn_from_text("爱因斯坦提出了相对论");

    auto answer = learner.think("牛顿");
    REQUIRE_FALSE(answer.empty());
}

TEST_CASE("TextLearner: 知识图谱增长") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    REQUIRE(kg.entity_count() == 0);

    learner.learn_from_text("人工智能是计算机科学的一个分支");

    SECTION("KG 应增长") {
        REQUIRE(kg.entity_count() > 0);
        REQUIRE(kg.relation_count() > 0);
    }
}

TEST_CASE("TextLearner: 验证反馈") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    auto result = learner.learn_from_text(
        "Python是一种编程语言");

    SECTION("验证结果应有分数") {
        REQUIRE(result.verification_score >= 0.0);
        REQUIRE(result.verification_score <= 1.0);
    }
}

TEST_CASE("TextLearner: 矛盾检测") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    learner.learn_from_text("地球是圆的");

    // 学习冲突知识
    auto result = learner.learn_from_text("地球是平的");

    SECTION("矛盾不会导致崩溃") {
        REQUIRE(result.entities.empty() == false);
    }
}
