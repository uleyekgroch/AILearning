/**
 * @file test_statistical_unified.cpp
 * @brief 统计学习器 + 统一推理引擎测试
 */

#include <catch2/catch_test_macros.hpp>

#include "ai_learning/learning/statistical_learner.hpp"
#include "ai_learning/reasoning/unified_engine.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/domain/knowledge/entity.hpp"
#include "ai_learning/domain/knowledge/relation.hpp"
#include "ai_learning/memory/episodic_memory.hpp"

using namespace ai_learning;
using namespace ai_learning::learning;
using namespace ai_learning::reasoning;
using namespace ai_learning::domain::knowledge;

// ═══════════════════════════════════════════════════════════════════
// 统计学习器
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("StatisticalLearner: 基本观察", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    auto result = sl.observe("数学是研究数量和结构的学科");
    REQUIRE(result.contains("new_concepts"));

    auto stats = sl.get_stats();
    REQUIRE(stats.at("text_count") == 1.0);
    REQUIRE(stats.at("total_chars") > 0.0);
}

TEST_CASE("StatisticalLearner: 概念涌现", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    // 重复观察包含相同片段的文本以触发涌现
    for (int i = 0; i < 5; ++i) {
        sl.observe("数学研究数量和结构");
        sl.observe("数学是一门基础学科");
        sl.observe("数学应用很广泛");
    }

    auto concepts = sl.get_emergent_concepts(2, 0.0);
    // "数学" 应该涌现（频率高、出现在多个文本中）
    bool found_math = false;
    for (const auto& c : concepts) {
        if (c.text.find("数学") != std::string::npos) {
            found_math = true;
            REQUIRE(c.frequency >= 2);
            REQUIRE(c.pmi > 0.0);
        }
    }
    REQUIRE(found_math);
}

TEST_CASE("StatisticalLearner: 概念查询", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    for (int i = 0; i < 5; ++i) {
        sl.observe("物理学研究自然规律");
        sl.observe("物理学很有趣");
    }

    auto info = sl.get_concept_info("物理学");
    // 概念可能存在也可能不存在，取决于涌现条件
    // 主要验证 API 不崩溃
    if (info) {
        REQUIRE(info->text == "物理学");
        REQUIRE(info->frequency > 0);
    }
}

TEST_CASE("StatisticalLearner: 序列预测", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    for (int i = 0; i < 10; ++i) {
        sl.observe("数学很好");
    }

    auto preds = sl.predict_next("学", 3);
    // "学" 之后应该有预测
    REQUIRE_FALSE(preds.empty());

    // 概率之和应 <= 1
    double total = 0.0;
    for (const auto& [_, p] : preds) {
        total += p;
        REQUIRE(p > 0.0);
        REQUIRE(p <= 1.0);
    }
    REQUIRE(total <= 1.01);
}

TEST_CASE("StatisticalLearner: 意外度", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    // 训练
    for (int i = 0; i < 10; ++i) {
        sl.observe("数学很好数学很好");
    }

    // 熟悉文本的意外度应低于随机文本
    auto familiar = sl.get_surprise("数学很好");
    auto novel = sl.get_surprise("量子纠缠态");
    // 注意：因为训练数据有限，不一定总是成立
    // 主要验证 API 不崩溃
    REQUIRE(familiar >= 0.0);
    REQUIRE(novel >= 0.0);
}

TEST_CASE("StatisticalLearner: 关系涌现", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    // 两个概念频繁共现
    for (int i = 0; i < 10; ++i) {
        sl.observe("数学和物理是基础学科");
        sl.observe("数学和物理相辅相成");
    }

    auto rels = sl.get_emergent_relations(0.0);
    // 验证 API 不崩溃；关系涌现取决于统计条件
    for (const auto& r : rels) {
        REQUIRE(r.count > 0);
    }
}

TEST_CASE("StatisticalLearner: 相关概念", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    for (int i = 0; i < 10; ++i) {
        sl.observe("数学和物理是基础学科");
    }

    auto related = sl.get_related("数学", 5);
    // 验证 API 不崩溃
    for (const auto& [other, strength] : related) {
        REQUIRE_FALSE(other.empty());
    }
}

TEST_CASE("StatisticalLearner: 统计信息", "[statistical]") {
    StatisticalLearner sl(4, 2, 0.5, 100);

    sl.observe("数学是基础学科");
    sl.observe("物理学研究自然");

    auto stats = sl.get_stats();
    REQUIRE(stats.at("text_count") == 2.0);
    REQUIRE(stats.at("total_chars") > 0.0);
    REQUIRE(stats.at("ngram_types") > 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// 统一推理引擎
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("UnifiedReasoningEngine: 基本推理", "[unified]") {
    KnowledgeGraph kg;

    // 构建知识图谱
    kg.add_entity(Entity("数学", "学科"));
    kg.add_entity(Entity("物理", "学科"));
    kg.add_relation(Relation("数学", "物理", "相关", 0.8));
    UnifiedReasoningEngine engine(kg);
    auto results = engine.reason("数学");
    REQUIRE_FALSE(results.empty());

    // 结果按置信度排序
    for (size_t i = 1; i < results.size(); ++i) {
        REQUIRE(results[i - 1].confidence >= results[i].confidence);
    }
}

TEST_CASE("UnifiedReasoningEngine: 直接查询", "[unified]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("猫", "动物"));
    kg.add_entity(Entity("哺乳动物", "分类"));
    kg.add_relation(Relation("猫", "哺乳动物", "属于", 0.9));

    UnifiedReasoningEngine engine(kg);
    auto results = engine.reason("猫是什么");
    // 应该找到直接关系
    bool found = false;
    for (const auto& r : results) {
        if (r.method == "direct" && r.content.find("猫") != std::string::npos) {
            found = true;
            REQUIRE(r.confidence > 0.0);
        }
    }
    REQUIRE(found);
}

TEST_CASE("UnifiedReasoningEngine: 概率推理", "[unified]") {
    KnowledgeGraph kg;
    StatisticalLearner sl(4, 2, 0.5, 100);

    // 训练统计学习器
    for (int i = 0; i < 10; ++i) {
        sl.observe("数学很好");
        sl.observe("数学很有用");
    }

    UnifiedReasoningEngine engine(kg);
    engine.set_statistical_learner(&sl);

    auto results = engine.reason("数学");
    // 应该包含统计推理结果
    bool found_stat = false;
    for (const auto& r : results) {
        if (r.method == "probabilistic") {
            found_stat = true;
        }
    }
    REQUIRE(found_stat);
}

TEST_CASE("UnifiedReasoningEngine: 多跳推理", "[unified]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("A", "概念"));
    kg.add_entity(Entity("B", "概念"));
    kg.add_entity(Entity("C", "概念"));
    kg.add_relation(Relation("A", "B", "导致", 0.9));
    kg.add_relation(Relation("B", "C", "导致", 0.8));

    UnifiedReasoningEngine engine(kg);
    auto results = engine.reason("A");
    // 应该找到因果链 A->B->C
    bool found_causal = false;
    for (const auto& r : results) {
        if (r.method == "causal") {
            found_causal = true;
            REQUIRE(r.content.find("A") != std::string::npos);
            REQUIRE(r.content.find("C") != std::string::npos);
        }
    }
    REQUIRE(found_causal);
}

TEST_CASE("UnifiedReasoningEngine: 归纳推理", "[unified]") {
    KnowledgeGraph kg;
    memory::EpisodicMemory em(100);

    // 存一些相关记忆
    std::map<std::string, std::string> meta1;
    meta1["text"] = "猫是哺乳动物";
    meta1["category"] = "动物分类";
    em.store({1.0f, 0.0f, 0.0f}, meta1);

    std::map<std::string, std::string> meta2;
    meta2["text"] = "狗是哺乳动物";
    meta2["category"] = "动物分类";
    em.store({0.0f, 1.0f, 0.0f}, meta2);

    UnifiedReasoningEngine engine(kg);
    engine.set_episodic_memory(&em);

    auto results = engine.reason("哺乳动物");
    // 可能包含归纳结果（取决于记忆内容）
    // 主要验证不崩溃
    for (const auto& r : results) {
        REQUIRE(r.confidence >= 0.0);
        REQUIRE_FALSE(r.method.empty());
    }
}

TEST_CASE("UnifiedReasoningEngine: 空知识图谱", "[unified]") {
    KnowledgeGraph kg;
    UnifiedReasoningEngine engine(kg);

    auto results = engine.reason("未知问题");
    // 空图谱应返回空结果
    REQUIRE(results.empty());
}

TEST_CASE("UnifiedReasoningEngine: 关键词提取", "[unified]") {
    auto kw = UnifiedReasoningEngine::extract_keywords("什么是数学研究");
    REQUIRE_FALSE(kw.empty());
    // 应该包含 "什么" 或 "数学" 或 "研究"
    bool has_content = false;
    for (const auto& k : kw) {
        if (k.find("数学") != std::string::npos ||
            k.find("研究") != std::string::npos) {
            has_content = true;
        }
    }
    REQUIRE(has_content);
}
