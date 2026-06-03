/**
 * @file test_memory_reasoning_stdp.cpp
 * @brief 情景记忆、激活扩散推理、STDP 学习的测试
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/memory/episodic_memory.hpp"
#include "ai_learning/reasoning/activation_spread.hpp"
#include "ai_learning/learning/stdp_learning.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

using namespace ai_learning;
using namespace memory;
using namespace reasoning;
using namespace learning;
using namespace domain::knowledge;

// ════════════════════════════════════════════════════════════════
// Episodic Memory Tests
// ════════════════════════════════════════════════════════════════

TEST_CASE("EpisodicMemory: 存储和检索", "[memory]") {
    EpisodicMemory mem(100);

    // 存储 3 个向量
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> v3 = {0.9f, 0.1f, 0.0f};  // 接近 v1

    mem.store(v1, {{"label", "A"}});
    mem.store(v2, {{"label", "B"}});
    mem.store(v3, {{"label", "C"}});

    // 用 v1 检索，A 和 C 应排在前面
    auto results = mem.retrieve(v1, 3);
    REQUIRE(results.size() == 3);
    // v1 自身相似度 1.0，v3 相似度 ~0.99，v2 相似度 0.0
    REQUIRE(results[0].metadata.at("label") == "A");
    REQUIRE(results[1].metadata.at("label") == "C");
    REQUIRE(results[2].metadata.at("label") == "B");
}

TEST_CASE("EpisodicMemory: 容量限制", "[memory]") {
    EpisodicMemory mem(3);

    for (int i = 0; i < 5; ++i) {
        std::vector<float> v = {static_cast<float>(i)};
        mem.store(v, {{"idx", std::to_string(i)}});
    }

    // 只保留最新 3 条
    REQUIRE(mem.size() == 3);

    // 应该有 idx=2,3,4
    auto results = mem.retrieve({4.0f}, 5);
    REQUIRE(results.size() == 3);
}

TEST_CASE("EpisodicMemory: 巩固", "[memory]") {
    EpisodicMemory mem(100, 0.5, 0.4);

    std::vector<float> v = {1.0f, 0.0f};
    mem.store(v, {{"label", "item1"}});
    mem.store(v, {{"label", "item2"}});

    REQUIRE(mem.size() == 2);
    auto report = mem.consolidate();
    // 衰减 0.5 + boost 0.1 = 0.6，仍高于 0.4 阈值
    REQUIRE(mem.size() == 2);
    REQUIRE(report.at("remaining") == 2.0);

    // 多次巩固直到被移除（每次衰减 0.5，boost 0.1）
    // 等比数列：0.6 → 0.4 → 0.3（移除）或更精确：
    // 第2次：0.6*0.5=0.3, boost→0.4, 不移除
    // 第3次：0.4*0.5=0.2 < 0.4 → 但先 boost 上一次的 0.4→0.5? 不对
    // 实际顺序：先衰减全部，再移除弱的，再 boost
    // 第2次：0.6*0.5=0.3 < 0.4 → 移除!
    mem.consolidate();
    REQUIRE(mem.size() == 0);
}

// ════════════════════════════════════════════════════════════════
// Activation Spread Tests
// ════════════════════════════════════════════════════════════════

TEST_CASE("ActivationSpread: 简单扩散", "[reasoning]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("人工智能", "概念"));
    kg.add_entity(Entity("计算机科学", "学科"));
    kg.add_entity(Entity("机器学习", "领域"));

    kg.add_relation(Relation("人工智能", "计算机科学", "属于", 0.9));
    kg.add_relation(Relation("计算机科学", "机器学习", "包含", 0.8));

    ActivationSpread as(kg, 0.7, 0.1, 3);
    auto activated = as.spread({"人工智能"});

    // 应该激活计算机科学和机器学习
    REQUIRE(activated.size() >= 2);

    // 计算机科学应该排在前面（1跳）
    bool has_cs = false;
    bool has_ml = false;
    for (const auto& node : activated) {
        if (node.entity_id == "计算机科学") has_cs = true;
        if (node.entity_id == "机器学习") has_ml = true;
    }
    REQUIRE(has_cs);
    REQUIRE(has_ml);
}

TEST_CASE("ActivationSpread: 衰减效应", "[reasoning]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("A", "node"));
    kg.add_entity(Entity("B", "node"));
    kg.add_entity(Entity("C", "node"));

    kg.add_relation(Relation("A", "B", "to", 0.9));
    kg.add_relation(Relation("B", "C", "to", 0.9));

    ActivationSpread as(kg, 0.5, 0.01, 3);
    auto activated = as.spread({"A"});

    // B 的激活度应大于 C
    double act_b = as.get_activation("B");
    double act_c = as.get_activation("C");
    REQUIRE(act_b > act_c);
    REQUIRE_THAT(act_b, Catch::Matchers::WithinAbs(0.5, 0.01));
    REQUIRE_THAT(act_c, Catch::Matchers::WithinAbs(0.25, 0.01));
}

TEST_CASE("ActivationSpread: 阈值过滤", "[reasoning]") {
    KnowledgeGraph kg;

    kg.add_entity(Entity("A", "node"));
    kg.add_entity(Entity("B", "node"));

    kg.add_relation(Relation("A", "B", "to", 0.9));

    // 阈值很高，应该过滤掉弱激活
    ActivationSpread as(kg, 0.5, 0.9, 3);
    auto activated = as.spread({"A"});

    // B 的激活 = 0.5 < 0.9 阈值，应该被过滤
    REQUIRE(activated.empty());
}

// ════════════════════════════════════════════════════════════════
// STDP Learning Tests
// ════════════════════════════════════════════════════════════════

TEST_CASE("STDP: LTP 长时程增强", "[learning]") {
    STDP stdp(0.1, 10.0, 0.95);

    // pre 先于 post → 正向时序 → LTP
    stdp.strengthen("下雨", "地面湿", 1.0);
    double w = stdp.get_weight("下雨", "地面湿");
    REQUIRE(w > 0.0);

    // 多次强化应该累积
    stdp.strengthen("下雨", "地面湿", 1.0);
    double w2 = stdp.get_weight("下雨", "地面湿");
    REQUIRE(w2 > w);
}

TEST_CASE("STDP: LTD 长时程抑制", "[learning]") {
    STDP stdp(0.1, 10.0, 0.95);

    // 反向时序 → LTD
    stdp.strengthen("地面湿", "下雨", -1.0);
    double w = stdp.get_weight("地面湿", "下雨");
    REQUIRE(w < 0.0);
}

TEST_CASE("STDP: 强连接检索", "[learning]") {
    STDP stdp(0.1, 10.0, 0.95);

    stdp.strengthen("下雨", "地面湿", 1.0);
    stdp.strengthen("下雨", "打伞", 2.0);
    stdp.strengthen("下雨", "路滑", 1.0);

    auto strong = stdp.get_strong_connections("下雨", 0.05);
    REQUIRE(strong.size() >= 2);

    // 打伞 应该排在前面（时序差 2.0，但 exp(-2/10) 略小）
    // 实际上 timing_delta=1.0 的 delta 更大
    // 所以地面湿和路滑应该排在打伞前面
    REQUIRE(strong[0].second >= strong.back().second);
}

TEST_CASE("STDP: 全局衰减", "[learning]") {
    STDP stdp(0.1, 10.0, 0.5);

    stdp.strengthen("A", "B", 1.0);
    double w_before = stdp.get_weight("A", "B");
    REQUIRE(w_before > 0.0);

    stdp.decay_all();
    double w_after = stdp.get_weight("A", "B");
    REQUIRE(w_after < w_before);
    REQUIRE_THAT(w_after, Catch::Matchers::WithinAbs(w_before * 0.5, 0.001));

    // 极弱连接应被清理
    for (int i = 0; i < 20; ++i) {
        stdp.decay_all();
    }
    // 连接应该被移除
    REQUIRE(stdp.get_weight("A", "B") == 0.0);
    REQUIRE(stdp.connection_count() == 0);
}
