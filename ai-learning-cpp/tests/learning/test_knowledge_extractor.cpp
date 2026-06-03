/**
 * @file test_knowledge_extractor.cpp
 * @brief KnowledgeExtractor 单元测试
 */

#include <catch2/catch_test_macros.hpp>
#include "ai_learning/learning/knowledge_extractor.hpp"

using namespace ai_learning::learning;

TEST_CASE("KnowledgeExtractor: 提取中文实体") {
    auto entities = KnowledgeExtractor::extract_entities(
        "人工智能是计算机科学的一个分支");

    SECTION("应提取关键实体") {
        REQUIRE_FALSE(entities.empty());
        // 应包含"人工智能"
        bool found_ai = false;
        for (const auto& e : entities) {
            if (e.find("人工智能") != std::string::npos) {
                found_ai = true;
                break;
            }
        }
        REQUIRE(found_ai);
    }

    SECTION("无虚词") {
        for (const auto& e : entities) {
            REQUIRE(e != "的");
            REQUIRE(e != "了");
            REQUIRE(e != "是");
        }
    }
}

TEST_CASE("KnowledgeExtractor: 提取关系三元组") {
    auto entities = std::vector<std::string>{"人工智能", "计算机科学"};
    auto triples = KnowledgeExtractor::extract_triples(
        "人工智能是计算机科学的一个分支", entities);

    SECTION("应提取'是'关系") {
        REQUIRE_FALSE(triples.empty());
        bool found_is_a = false;
        for (const auto& t : triples) {
            if (t.relation == "是") {
                found_is_a = true;
                REQUIRE(t.subject == "人工智能");
                REQUIRE(t.object.find("计算机") != std::string::npos);
            }
        }
        REQUIRE(found_is_a);
    }
}

TEST_CASE("KnowledgeExtractor: 提取因果关系") {
    auto links = KnowledgeExtractor::extract_causal_links(
        "因为下雨，所以地面湿了");

    REQUIRE_FALSE(links.empty());
    REQUIRE(links[0].cause.find("下雨") != std::string::npos);
    REQUIRE(links[0].effect.find("湿") != std::string::npos);
}

TEST_CASE("KnowledgeExtractor: 提取数值事实") {
    auto facts = KnowledgeExtractor::extract_numerical_facts(
        "水在100度沸腾");

    REQUIRE_FALSE(facts.empty());
    REQUIRE(facts[0].value == 100.0);
    REQUIRE(facts[0].unit == "摄氏度");
}

TEST_CASE("KnowledgeExtractor: 空文本") {
    auto entities = KnowledgeExtractor::extract_entities("");
    REQUIRE(entities.empty());

    auto triples = KnowledgeExtractor::extract_triples("", {});
    REQUIRE(triples.empty());
}

TEST_CASE("KnowledgeExtractor: 多个因果关系") {
    auto links = KnowledgeExtractor::extract_causal_links(
        "下雨导致地面湿了");

    REQUIRE_FALSE(links.empty());
}
