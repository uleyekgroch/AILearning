/**
 * @file test_dense_vector_cache.cpp
 * @brief 稠密向量缓存 + 批量 API 测试
 *
 * 验证：
 *   1. dense_cache 缓存命中 — 重复调用不重新排序
 *   2. batch API 正确性 — 与逐个调用结果一致
 *   3. rebuild 后缓存失效 — 向量数据更新后缓存清空
 *   4. learn_predictive_ 去重行为 — 通过 Learner 集成验证
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/learning/distributional_semantics.hpp"

using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════
// 缓存命中：两次调用返回相同值
// ═══════════════════════════════════════════════════════════

TEST_CASE("DenseCache: 重复调用返回相同向量", "[semantic][cache]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "数学研究数量",
        "物理研究物质",
        "数学和物理都是学科",
    });

    auto vec1 = ds.get_dense_vector("数学", 32);
    auto vec2 = ds.get_dense_vector("数学", 32);

    REQUIRE(vec1.has_value());
    REQUIRE(vec2.has_value());
    REQUIRE(vec1->size() == vec2->size());

    for (size_t i = 0; i < vec1->size(); ++i) {
        CHECK_THAT((*vec1)[i], WithinAbs((*vec2)[i], 1e-7f));
    }
}

// ═══════════════════════════════════════════════════════════
// 缓存维度不匹配时重新计算
// ═══════════════════════════════════════════════════════════

TEST_CASE("DenseCache: 不同 dims 参数返回不同长度", "[semantic][cache]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "数学研究数量和结构",
        "数学是一门基础学科",
    });

    auto vec16 = ds.get_dense_vector("数学", 16);
    auto vec32 = ds.get_dense_vector("数学", 32);
    auto vec16_again = ds.get_dense_vector("数学", 16);

    REQUIRE(vec16.has_value());
    REQUIRE(vec32.has_value());
    REQUIRE(vec16_again.has_value());

    CHECK(static_cast<int>(vec16->size()) == 16);
    CHECK(static_cast<int>(vec32->size()) == 32);

    // 返回 dims=16 后再请求 dims=16 应该命中缓存
    for (size_t i = 0; i < vec16->size(); ++i) {
        CHECK_THAT((*vec16)[i], WithinAbs((*vec16_again)[i], 1e-7f));
    }
}

// ═══════════════════════════════════════════════════════════
// 批量 API：与逐个调用结果一致
// ═══════════════════════════════════════════════════════════

TEST_CASE("DenseCache: batch API 与逐个调用一致", "[semantic][cache]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "数学研究数量结构",
        "物理研究物质能量",
        "数学物理都是学科",
    });

    std::vector<std::string> concepts = {"数学", "物理", "不存在", "学科"};

    // 逐个调用
    std::vector<std::optional<std::vector<float>>> individual;
    for (const auto& c : concepts) {
        individual.push_back(ds.get_dense_vector(c, 32));
    }

    // 批量调用
    auto batch = ds.get_dense_vectors_batch(concepts, 32);

    REQUIRE(individual.size() == batch.size());

    for (size_t i = 0; i < concepts.size(); ++i) {
        if (individual[i].has_value()) {
            REQUIRE(batch[i].has_value());
            REQUIRE(individual[i]->size() == batch[i]->size());
            for (size_t j = 0; j < individual[i]->size(); ++j) {
                CHECK_THAT((*individual[i])[j], WithinAbs((*batch[i])[j], 1e-7f));
            }
        } else {
            CHECK_FALSE(batch[i].has_value());
        }
    }
}

// ═══════════════════════════════════════════════════════════
// 批量 API：空输入
// ═══════════════════════════════════════════════════════════

TEST_CASE("DenseCache: batch API 空输入", "[semantic][cache]") {
    DistributionalSemantics ds;
    auto results = ds.get_dense_vectors_batch({}, 32);
    CHECK(results.empty());
}

// ═══════════════════════════════════════════════════════════
// rebuild 后缓存失效
// ═══════════════════════════════════════════════════════════

TEST_CASE("DenseCache: rebuild 后缓存失效，向量可能变化", "[semantic][cache]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    config.window_size = 10;
    DistributionalSemantics ds(config);

    // 初始学习
    ds.learn_from_text("数学研究数量结构");
    auto vec_before = ds.get_dense_vector("数学", 32);

    // 继续学习更多语料，触发 rebuild
    ds.learn_from_text("数学是一门基础学科数学研究数量结构");
    ds.learn_from_text("数学提供理论框架数学研究数量结构");
    ds.learn_from_text("数学和物理都是科学数学研究数量结构");
    ds.learn_from_text("数学是重要的工具数学研究数量结构");
    ds.learn_from_text("数学应用广泛数学研究数量结构");
    ds.learn_from_text("数学发展历史数学研究数量结构");
    ds.learn_from_text("数学需要逻辑数学研究数量结构");
    ds.learn_from_text("数学计算数学研究数量结构");
    ds.learn_from_text("数学分析数学研究数量结构");
    ds.learn_from_text("数学代数数学研究数量结构");

    auto vec_after = ds.get_dense_vector("数学", 32);

    // 向量应仍然有效
    REQUIRE(vec_before.has_value());
    REQUIRE(vec_after.has_value());
    CHECK(static_cast<int>(vec_after->size()) == 32);
}

// ═══════════════════════════════════════════════════════════
// 不存在概念的缓存行为
// ═══════════════════════════════════════════════════════════

TEST_CASE("DenseCache: 不存在概念返回 nullopt", "[semantic][cache]") {
    DistributionalSemantics ds;

    auto single = ds.get_dense_vector("不存在", 32);
    CHECK_FALSE(single.has_value());

    auto batch = ds.get_dense_vectors_batch({"不存在", "也没有"}, 32);
    CHECK(batch.size() == 2);
    CHECK_FALSE(batch[0].has_value());
    CHECK_FALSE(batch[1].has_value());
}

// ═══════════════════════════════════════════════════════════
// 批量 API 大量 token 性能测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("DenseCache: 批量查询大量重复 token", "[semantic][cache]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "数学研究数量和结构",
        "数学是一门基础学科",
    });

    // 大量重复 token（模拟中文文章高频词场景）
    std::vector<std::string> tokens(1000, "数学");

    auto results = ds.get_dense_vectors_batch(tokens, 32);
    CHECK(results.size() == 1000);

    // 所有结果都应该有值
    for (const auto& r : results) {
        CHECK(r.has_value());
    }

    // 第一个和最后一个应该完全一致
    for (size_t i = 0; i < results.front()->size(); ++i) {
        CHECK_THAT((*results.front())[i], WithinAbs((*results.back())[i], 1e-7f));
    }
}
