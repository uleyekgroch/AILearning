/**
 * @file test_mastery_assessor.cpp
 * @brief Bloom 掌握度评估器测试
 *
 * 验证：
 *   1. MasteryLevel 枚举转换
 *   2. mastery_level_from_score 阈值边界
 *   3. 零接触实体 → 低 recognition 分数
 *   4. 高接触/练习实体 → 高掌握度
 *   5. 维度评分正确性
 *   6. 领域报告聚合
 *   7. 推荐生成（弱维度触发推荐）
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/assessment/mastery_assessor.hpp"
#include "ai_learning/domain/knowledge/entity.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/domain/knowledge/relation.hpp"

using namespace ai_learning::assessment;
using namespace ai_learning::domain::knowledge;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════
// MasteryLevel 枚举
// ═══════════════════════════════════════════════════════════

TEST_CASE("MasteryLevel: 枚举转字符串", "[assessment]") {
    CHECK(mastery_level_to_string(MasteryLevel::Unknown)    == "Unknown");
    CHECK(mastery_level_to_string(MasteryLevel::Exposed)    == "Exposed");
    CHECK(mastery_level_to_string(MasteryLevel::Recognized) == "Recognized");
    CHECK(mastery_level_to_string(MasteryLevel::Understood) == "Understood");
    CHECK(mastery_level_to_string(MasteryLevel::Applied)    == "Applied");
    CHECK(mastery_level_to_string(MasteryLevel::Mastered)   == "Mastered");
}

TEST_CASE("MasteryLevel: 分数到等级的阈值边界", "[assessment]") {
    CHECK(mastery_level_from_score(0.0)  == MasteryLevel::Unknown);
    CHECK(mastery_level_from_score(0.09) == MasteryLevel::Unknown);
    CHECK(mastery_level_from_score(0.10) == MasteryLevel::Exposed);
    CHECK(mastery_level_from_score(0.29) == MasteryLevel::Exposed);
    CHECK(mastery_level_from_score(0.30) == MasteryLevel::Recognized);
    CHECK(mastery_level_from_score(0.49) == MasteryLevel::Recognized);
    CHECK(mastery_level_from_score(0.50) == MasteryLevel::Understood);
    CHECK(mastery_level_from_score(0.69) == MasteryLevel::Understood);
    CHECK(mastery_level_from_score(0.70) == MasteryLevel::Applied);
    CHECK(mastery_level_from_score(0.89) == MasteryLevel::Applied);
    CHECK(mastery_level_from_score(0.90) == MasteryLevel::Mastered);
    CHECK(mastery_level_from_score(1.0)  == MasteryLevel::Mastered);
}

// ═══════════════════════════════════════════════════════════
// 维度评分
// ═══════════════════════════════════════════════════════════

TEST_CASE("MasteryAssessor: 零接触实体 → 低 recognition", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    Entity e("e1", "concept",
             {{"exposure_count", "0"},
              {"practice_count", "0"},
              {"success_count",  "0"}},
             0.0);
    kg.add_entity(e);

    auto result = ma.assess(*kg.get_entity("e1"), kg);
    CHECK(result.dimensions.at("recognition") == 0.0);
    CHECK(result.overall_mastery < 0.2);
    CHECK(result.mastery_level == MasteryLevel::Unknown);
}

TEST_CASE("MasteryAssessor: 高接触/高练习 → 高掌握度", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    Entity e("e2", "concept",
             {{"exposure_count",  "50"},
              {"practice_count",  "20"},
              {"success_count",   "18"}},
             0.9);
    kg.add_entity(e);

    auto result = ma.assess(*kg.get_entity("e2"), kg);
    CHECK(result.dimensions.at("recognition") > 0.5);
    CHECK(result.dimensions.at("comprehension") > 0.5);
    CHECK(result.dimensions.at("application")   > 0.8);
    CHECK(result.dimensions.at("teaching")      > 0.5);
    CHECK(result.overall_mastery > 0.5);
}

TEST_CASE("MasteryAssessor: recognition 对数衰减", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    // exposure=20 → log(21)/log(21) = 1.0
    Entity e("r1", "test", {{"exposure_count", "20"}}, 0.5);
    kg.add_entity(e);
    auto r = ma.assess(*kg.get_entity("r1"), kg);
    CHECK_THAT(r.dimensions.at("recognition"), WithinAbs(1.0, 0.01));
}

TEST_CASE("MasteryAssessor: comprehension 公式", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    // practice=10, success=8 → rate=0.8, saturation=1.0 → 0.8
    Entity e("c1", "test",
             {{"practice_count", "10"}, {"success_count", "8"}},
             0.5);
    kg.add_entity(e);
    auto r = ma.assess(*kg.get_entity("c1"), kg);
    CHECK_THAT(r.dimensions.at("comprehension"), WithinAbs(0.8, 0.01));
}

TEST_CASE("MasteryAssessor: analysis 用关联关系数", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    Entity target("a1", "concept", {{"exposure_count", "5"}}, 0.5);
    Entity related1("r1", "concept", {}, 0.5);
    Entity related2("r2", "concept", {}, 0.5);
    kg.add_entity(target);
    kg.add_entity(related1);
    kg.add_entity(related2);
    kg.add_relation(Relation("a1", "r1", "related_to"));
    kg.add_relation(Relation("a1", "r2", "is_a"));

    auto r = ma.assess(*kg.get_entity("a1"), kg);
    // 2 relations / 10 = 0.2
    CHECK_THAT(r.dimensions.at("analysis"), WithinAbs(0.2, 0.01));
}

TEST_CASE("MasteryAssessor: teaching 等级查表", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    // confidence=0.95 → Mastered → teaching=1.0
    Entity e_mastered("t1", "test", {}, 0.95);
    kg.add_entity(e_mastered);
    auto r1 = ma.assess(*kg.get_entity("t1"), kg);
    CHECK_THAT(r1.dimensions.at("teaching"), WithinAbs(1.0, 0.01));

    // confidence=0.75 → Applied → teaching=0.6
    Entity e_applied("t2", "test", {}, 0.75);
    kg.add_entity(e_applied);
    auto r2 = ma.assess(*kg.get_entity("t2"), kg);
    CHECK_THAT(r2.dimensions.at("teaching"), WithinAbs(0.6, 0.01));

    // confidence=0.4 → Recognized → teaching=0.0
    Entity e_low("t3", "test", {}, 0.4);
    kg.add_entity(e_low);
    auto r3 = ma.assess(*kg.get_entity("t3"), kg);
    CHECK_THAT(r3.dimensions.at("teaching"), WithinAbs(0.0, 0.01));
}

// ═══════════════════════════════════════════════════════════
// 推荐
// ═══════════════════════════════════════════════════════════

TEST_CASE("MasteryAssessor: 弱维度触发推荐", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    // 低接触、低练习 → recognition + comprehension 弱
    Entity e("w1", "concept",
             {{"exposure_count", "0"},
              {"practice_count", "0"},
              {"success_count",  "0"}},
             0.0);
    kg.add_entity(e);

    auto r = ma.assess(*kg.get_entity("w1"), kg);
    REQUIRE_FALSE(r.recommendations.empty());

    // 应包含 recognition 和 comprehension 的推荐
    bool has_rec_rec = false;
    bool has_comp_rec = false;
    for (const auto& rec : r.recommendations) {
        if (rec.find("接触") != std::string::npos) has_rec_rec = true;
        if (rec.find("理解") != std::string::npos) has_comp_rec = true;
    }
    CHECK(has_rec_rec);
    CHECK(has_comp_rec);
}

TEST_CASE("MasteryAssessor: 全强维度无推荐", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    Entity strong("s1", "concept",
                  {{"exposure_count", "50"},
                   {"practice_count", "30"},
                   {"success_count",  "28"}},
                  0.95);
    kg.add_entity(strong);
    // 添加关系以提升 analysis + synthesis
    Entity rel("rel1", "concept", {}, 0.8);
    kg.add_entity(rel);
    for (int i = 0; i < 12; ++i) {
        Entity r("r" + std::to_string(i), "other_domain", {}, 0.5);
        kg.add_entity(r);
        kg.add_relation(Relation("s1", "r" + std::to_string(i),
                                  "related_to"));
    }

    auto result = ma.assess(*kg.get_entity("s1"), kg);
    // 高掌握度实体推荐可能很少或没有
    CHECK(result.overall_mastery > 0.5);
}

// ═══════════════════════════════════════════════════════════
// 领域报告
// ═══════════════════════════════════════════════════════════

TEST_CASE("MasteryAssessor: assess_domain 聚合报告", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    // 3 个同类型实体，掌握度各不同
    Entity e1("d1", "math",
              {{"exposure_count", "30"},
               {"practice_count", "15"},
               {"success_count",  "12"},
               {"name", "代数"}},
              0.8);
    Entity e2("d2", "math",
              {{"exposure_count", "5"},
               {"practice_count", "2"},
               {"success_count",  "1"},
               {"name", "几何"}},
              0.3);
    Entity e3("d3", "math",
              {{"exposure_count", "1"},
               {"practice_count", "0"},
               {"success_count",  "0"},
               {"name", "拓扑"}},
              0.1);
    kg.add_entity(e1);
    kg.add_entity(e2);
    kg.add_entity(e3);

    auto report = ma.assess_domain("math", kg);
    CHECK(report.domain == "math");
    CHECK(report.total_units == 3);
    CHECK(report.avg_mastery > 0.0);
    CHECK(report.level_distribution.size() >= 1);
    CHECK(report.coverage > 0.0);
    CHECK_FALSE(report.strongest_units.empty());
    CHECK_FALSE(report.weakest_units.empty());
    // 最强的应该是 "代数"
    CHECK(report.strongest_units[0] == "代数");
    // 最弱的应该是 "拓扑"
    CHECK(report.weakest_units.back() == "拓扑");
}

TEST_CASE("MasteryAssessor: assess_all_domains 多领域", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    Entity m1("m1", "math", {{"exposure_count", "10"}}, 0.5);
    Entity p1("p1", "physics", {{"exposure_count", "20"}}, 0.7);
    kg.add_entity(m1);
    kg.add_entity(p1);

    auto reports = ma.assess_all_domains(kg);
    CHECK(reports.count("math") == 1);
    CHECK(reports.count("physics") == 1);
    CHECK(reports.at("math").total_units == 1);
    CHECK(reports.at("physics").total_units == 1);
}

TEST_CASE("MasteryAssessor: assess_domain 空领域", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    auto report = ma.assess_domain("nonexistent", kg);
    CHECK(report.domain == "nonexistent");
    CHECK(report.total_units == 0);
    CHECK_THAT(report.avg_mastery, WithinAbs(0.0, 0.001));
}

// ═══════════════════════════════════════════════════════════
// 强弱项分类
// ═══════════════════════════════════════════════════════════

TEST_CASE("MasteryAssessor: strengths/weaknesses 分类", "[assessment]") {
    MasteryAssessor ma;
    KnowledgeGraph kg;

    // 只有 application 高（confidence=0.8），其余低
    Entity e("sw1", "concept",
             {{"exposure_count", "0"},
              {"practice_count", "0"},
              {"success_count",  "0"}},
             0.8);
    kg.add_entity(e);

    auto r = ma.assess(*kg.get_entity("sw1"), kg);
    // application = 0.8 >= 0.7 → strength
    bool has_app = false;
    for (const auto& s : r.strengths) {
        if (s == "application") has_app = true;
    }
    CHECK(has_app);

    // recognition = 0.0 < 0.3 → weakness
    bool has_rec = false;
    for (const auto& w : r.weaknesses) {
        if (w == "recognition") has_rec = true;
    }
    CHECK(has_rec);
}

// ═══════════════════════════════════════════════════════════
// ProficiencyTester
// ═══════════════════════════════════════════════════════════

TEST_CASE("ProficiencyTester: 空知识图谱 → 低分", "[assessment][proficiency]") {
    ProficiencyTester pt;
    KnowledgeGraph kg;

    auto report = pt.assess("word", kg);
    CHECK(report.score == 0.0);
    CHECK(report.level == "A1");
    CHECK(report.receptive_vocab == 0);
    CHECK(report.productive_vocab == 0);
    CHECK(report.level_distribution.empty());
}

TEST_CASE("ProficiencyTester: CEFR 等级判定阈值", "[assessment][proficiency]") {
    ProficiencyTester pt;
    KnowledgeGraph kg;

    // Score of 0.10 → A1 threshold
    Entity e1("w1", "word", {}, 0.10, "", {"A1"});
    kg.add_entity(e1);
    auto r1 = pt.assess("word", kg);
    CHECK(r1.level == "A1");
}

TEST_CASE("ProficiencyTester: CEFR 标签实体 → 正确 breadth 评分", "[assessment][proficiency]") {
    ProficiencyTester pt;
    KnowledgeGraph kg;

    // A1 word with high confidence
    Entity e1("w1", "word", {}, 0.9, "", {"A1"});
    // B2 word with medium confidence
    Entity e2("w2", "word", {}, 0.5, "", {"B2"});
    kg.add_entity(e1);
    kg.add_entity(e2);

    auto report = pt.assess("word", kg);
    CHECK(report.level_distribution.count("A1") == 1);
    CHECK(report.level_distribution.count("B2") == 1);
    CHECK(report.level_distribution.at("A1") == 1);
    CHECK(report.level_distribution.at("B2") == 1);
    CHECK(report.dimensions.at("vocabulary_breadth") > 0.0);
}

TEST_CASE("ProficiencyTester: 完整评估 → 有效报告", "[assessment][proficiency]") {
    ProficiencyTester pt;
    KnowledgeGraph kg;

    // Add diverse word entities with CEFR tags and properties
    Entity w1("w1", "word",
              {{"example_count", "3"}}, 0.8, "", {"A1", "word_family"});
    Entity w2("w2", "word",
              {{"example_count", "5"}}, 0.6, "", {"A2", "word_family"});
    Entity w3("w3", "word",
              {{"example_count", "2"}}, 0.7, "", {"B1"});
    Entity w4("w4", "word",
              {{"example_count", "0"}}, 0.4, "", {"B2"});
    Entity w5("w5", "word",
              {{"example_count", "4"}}, 0.9, "", {"C1", "word_family"});
    kg.add_entity(w1);
    kg.add_entity(w2);
    kg.add_entity(w3);
    kg.add_entity(w4);
    kg.add_entity(w5);

    // Add some relations for semantic network
    kg.add_relation(Relation("w1", "w2", "related_to"));
    kg.add_relation(Relation("w1", "w3", "related_to"));
    kg.add_relation(Relation("w2", "w3", "related_to"));

    auto report = pt.assess("word", kg);

    // All fields populated
    CHECK_FALSE(report.level.empty());
    CHECK(report.score > 0.0);
    CHECK(report.dimensions.count("vocabulary_breadth") == 1);
    CHECK(report.dimensions.count("vocabulary_depth") == 1);
    CHECK(report.dimensions.count("semantic_network") == 1);
    CHECK(report.dimensions.count("collocation") == 1);
    CHECK(report.dimensions.count("word_family") == 1);

    // Receptive vocab: confidence >= 0.3 → w1(0.8),w2(0.6),w3(0.7),w4(0.4),w5(0.9) = 5
    CHECK(report.receptive_vocab == 5);
    // Productive vocab: confidence >= 0.7 → w1,w3,w5 = 3
    CHECK(report.productive_vocab == 3);

    // Level distribution has entries for A1,A2,B1,B2,C1
    CHECK(report.level_distribution.size() == 5);
}

TEST_CASE("ProficiencyTester: 弱维度生成推荐", "[assessment][proficiency]") {
    ProficiencyTester pt;
    KnowledgeGraph kg;

    // Entities with no CEFR tags, no examples, no word_family, no relations
    Entity w1("w1", "word", {}, 0.1);
    Entity w2("w2", "word", {}, 0.05);
    kg.add_entity(w1);
    kg.add_entity(w2);

    auto report = pt.assess("word", kg);

    // Most dimensions should be very low → recommendations generated
    CHECK_FALSE(report.recommendations.empty());

    // Check specific recommendation strings appear
    bool has_breadth = false, has_depth = false, has_network = false;
    bool has_collocation = false, has_family = false;
    for (const auto& rec : report.recommendations) {
        if (rec.find("词汇量覆盖面") != std::string::npos) has_breadth = true;
        if (rec.find("理解深度") != std::string::npos) has_depth = true;
        if (rec.find("语义关联") != std::string::npos) has_network = true;
        if (rec.find("搭配") != std::string::npos) has_collocation = true;
        if (rec.find("词族") != std::string::npos) has_family = true;
    }
    CHECK(has_breadth);
    CHECK(has_depth);
    CHECK(has_network);
    CHECK(has_collocation);
    CHECK(has_family);
}
