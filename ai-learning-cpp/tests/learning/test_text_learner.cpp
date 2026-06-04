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

// ── 无监督分词 + 依存解析在线管线对接 ───────────────────────────

namespace {

/// 生成内容词多上下文的无空格语料（同 segment_eval），供解析器训练。
auto make_parse_corpus() -> std::vector<std::string> {
    std::vector<std::string> persons = {"学生", "老师", "医生", "工人", "农民"};
    std::vector<std::string> subjects = {"数学", "物理", "化学", "历史", "地理"};
    std::vector<std::string> animals = {"猫", "狗", "老虎", "兔子"};
    std::vector<std::string> mods = {"聪明", "勤奋", "优秀", "可爱",
                                     "重要", "有趣", "年轻"};
    std::vector<std::string> verbs = {"喜欢", "学习", "研究", "讨厌"};
    std::vector<std::string> pronouns = {"我", "他", "她", "你"};
    std::vector<std::string> degree = {"很", "非常", "比较", "特别"};

    std::vector<std::vector<std::string>> g;
    auto add = [&](std::vector<std::string> ws) { g.push_back(std::move(ws)); };
    for (const auto& p : persons) {
        add({p, "是", "人类"});
        add({"人类", "包括", p});
    }
    for (const auto& s : subjects) {
        add({s, "是", "学科"});
        add({s, "是", "知识"});
        add({"学科", "包括", s});
    }
    for (const auto& a : animals) {
        add({a, "是", "动物"});
        add({"动物", "包括", a});
    }
    for (const auto& m : mods) {
        for (const auto& p : persons) add({m, "的", p, "是", "人类"});
        for (const auto& s : subjects) add({m, "的", s, "是", "学科"});
        for (const auto& a : animals) add({m, "的", a, "是", "动物"});
    }
    for (const auto& p : persons)
        for (const auto& d : degree)
            for (const auto& m : mods) add({p, d, m});
    for (const auto& s : subjects)
        for (const auto& d : degree) add({s, d, "重要"});
    for (const auto& pr : pronouns)
        for (const auto& v : verbs)
            for (const auto& s : subjects) add({pr, v, s});
    for (const auto& p : persons)
        for (const auto& v : verbs)
            for (const auto& s : subjects) add({p, v, s});
    for (const auto& pr : pronouns)
        for (const auto& v : verbs)
            for (const auto& a : animals) add({pr, v, a});
    {
        auto base = g;
        for (int rep = 0; rep < 3; ++rep)
            for (auto& s : base) g.push_back(s);
    }

    std::vector<std::string> raw;
    raw.reserve(g.size());
    for (const auto& ws : g) {
        std::string j;
        for (const auto& w : ws) j += w;
        raw.push_back(j);
    }
    return raw;
}

}  // namespace

TEST_CASE("TextLearner: 依存解析默认关闭、语料不足训练失败") {
    KnowledgeGraph kg;
    TextLearner learner(kg);

    REQUIRE_FALSE(learner.dependency_parsing_enabled());
    learner.learn_from_text("猫是动物");
    REQUIRE_FALSE(learner.train_dependency_parser());  // 语料过少
    REQUIRE_FALSE(learner.dependency_parsing_enabled());
}

TEST_CASE("TextLearner: 训练后启用句法树抽取并产出完整短语主语") {
    KnowledgeGraph kg;
    TextLearner learner(kg);
    for (const auto& line : make_parse_corpus())
        learner.learn_from_text(line, "test");

    REQUIRE(learner.train_dependency_parser());
    REQUIRE(learner.dependency_parsing_enabled());

    auto res = learner.learn_from_text("聪明的学生是人类");
    bool found = false;
    for (const auto& t : res.triples)
        if (t.relation == "是" && t.object == "人类" &&
            t.subject.find("学生") != std::string::npos) {
            found = true;
        }
    REQUIRE(found);
}
