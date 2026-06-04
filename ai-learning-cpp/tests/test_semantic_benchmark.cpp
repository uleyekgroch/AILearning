/**
 * @file test_semantic_benchmark.cpp
 * @brief 可证伪的语义基准 — 检验"嵌入语义"是否真的编码了含义
 *
 * 动机：既有测试多为 `similarity > 0` 这类无法证伪的弱断言。
 * 本文件用**相对序**与**provider 覆盖**断言，使"语义是否真实"变得可证伪：
 *   B1 分布语义：同簇概念应比跨簇概念更相似（数学~物理 > 数学~香蕉）。
 *   B2 CreativeEngine 依赖倒置：注入真实语义距离后，创意评估随语义变化，
 *      且 provider 必须覆盖字符级 Jaccard（字面重合但语义远 → 距离仍大）。
 *   B3 端到端：Learner 学习语料后，creative_engine() 的评估由真实分布语义驱动。
 */

#include <catch2/catch_test_macros.hpp>

#include "ai_learning/core/learner.hpp"
#include "ai_learning/creativity/creative_engine.hpp"
#include "ai_learning/learning/distributional_semantics.hpp"

using namespace ai_learning;
using ai_learning::creativity::ConceptNode;
using ai_learning::creativity::CreativeEngine;
using ai_learning::creativity::CreativeIdea;
using ai_learning::creativity::CreativityType;

namespace {

/// 只设 name 的 ConceptNode（其余字段用默认值，避免漏初始化）。
auto node(const std::string& name) -> ConceptNode {
    ConceptNode c;
    c.name = name;
    return c;
}

/// 两个清晰可分的语义簇：科学簇 vs 食物簇。
auto make_two_cluster_corpus() -> std::vector<std::string> {
    return {
        "数学 物理 化学 都是 科学",
        "数学 研究 数量 与 结构",
        "物理 研究 物质 与 能量",
        "化学 研究 物质 的 变化",
        "数学 物理 需要 大量 计算",
        "物理 化学 都 属于 自然 科学",
        "科学 研究 依赖 数学 公式",
        "数学 是 物理 与 化学 的 基础",
        "香蕉 苹果 都是 水果",
        "香蕉 是 甜 的 水果",
        "苹果 是 常见 的 水果",
        "水果 包含 香蕉 与 苹果",
        "香蕉 苹果 都 可以 吃",
        "甜 的 水果 有 香蕉 和 苹果",
    };
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════
// B1 分布语义：同簇 > 跨簇（可证伪的相对序）
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("语义基准: 同簇概念比跨簇概念更相似", "[semantic][benchmark]") {
    learning::DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    learning::DistributionalSemantics ds(config);
    ds.learn_batch(make_two_cluster_corpus());

    REQUIRE(ds.has_cpt("数学"));
    REQUIRE(ds.has_cpt("物理"));
    REQUIRE(ds.has_cpt("香蕉"));

    double sim_in_cluster = ds.similarity("数学", "物理").similarity;
    double sim_cross_cluster = ds.similarity("数学", "香蕉").similarity;

    // 关键的可证伪断言：若嵌入只是噪声/字面匹配，这条会失败。
    CHECK(sim_in_cluster > sim_cross_cluster);
}

// ═══════════════════════════════════════════════════════════════════
// B2 CreativeEngine 依赖倒置：注入语义 → 评估随语义变化 + 覆盖 Jaccard
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("语义基准: CreativeEngine 创意新颖性由注入语义驱动",
          "[semantic][benchmark][creativity]") {
    CreativeEngine engine;
    REQUIRE_FALSE(engine.has_semantic_provider());

    // 确定性 provider：给定一张已知距离表，避免语料噪声，断言可严格证伪。
    engine.set_semantic_distance_provider(
        [](const std::string& a, const std::string& b)
            -> std::optional<double> {
            // 对称：(猫,狗)=0.1 近；(猫,量子)=0.9 远
            auto key = std::minmax(a, b);
            if (key.first == "狗" && key.second == "猫") return 0.1;
            if (key.first == "猫" && key.second == "量子") return 0.9;
            if (key.first == "量子" && key.second == "狗") return 0.9;
            return std::nullopt;
        });
    REQUIRE(engine.has_semantic_provider());

    CreativeIdea idea;
    idea.type = CreativityType::kRemoteAssociation;
    idea.sources = {"猫"};

    // 与"狗"(近)相比，"量子"(远)应给出更高的新颖性。
    double novelty_near = engine.assess_novelty(idea, {node("狗")});
    double novelty_far = engine.assess_novelty(idea, {node("量子")});
    CHECK(novelty_far > novelty_near);
}

TEST_CASE("语义基准: 注入的 provider 覆盖字符级 Jaccard",
          "[semantic][benchmark][creativity]") {
    // "苹果" 与 "苹果手机" 字面高度重合（Jaccard 距离小），
    // 但若 provider 判定二者语义远，则评估必须采信 provider。
    CreativeEngine engine;

    CreativeIdea idea;
    idea.idea = "苹果";
    // 无 provider：字符级 Jaccard，重合多 → 距离小 → usefulness 高。
    double useful_jaccard = engine.assess_usefulness(idea, "苹果手机");

    engine.set_semantic_distance_provider(
        [](const std::string&, const std::string&) -> std::optional<double> {
            return 0.9;  // 语义远
        });
    double useful_semantic = engine.assess_usefulness(idea, "苹果手机");

    // usefulness = 1 - 0.5 * distance：provider 距离更大 → usefulness 更低。
    CHECK(useful_semantic < useful_jaccard);
}

// ═══════════════════════════════════════════════════════════════════
// B3 端到端：Learner 学习后，CreativeEngine 由真实分布语义驱动
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("语义基准: Learner 接线后 CreativeEngine 使用真实语义",
          "[semantic][benchmark][integration]") {
    core::LearnerConfig config;
    config.embedding_learning_enabled = true;  // 开启嵌入管线以喂养 ds_
    core::Learner learner(config);

    for (const auto& line : make_two_cluster_corpus()) {
        learner.learn_from_text(line);  // 该路径才会喂养 ds_
    }

    // Learner 构造时已把 ds_ 接到 creative_engine_（依赖倒置接线点）。
    CHECK(learner.creative_engine().has_semantic_provider());
    // 学习流水线确实填充了分布语义空间（而非空壳）。
    CHECK_FALSE(learner.distributional_semantics().all_cpts().empty());
}
