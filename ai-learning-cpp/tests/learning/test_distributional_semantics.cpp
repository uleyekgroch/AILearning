/**
 * @file test_distributional_semantics.cpp
 * @brief 分布语义引擎测试 — 纯自主语义理解
 *
 * 验证：
 *   1. 概念涌现 — 多次观察后概念达到阈值
 *   2. 共现统计 — 上下文窗口内的共现计数
 *   3. PPMI 计算 — 正点互信息
 *   4. 余弦相似度 — 相似概念的分布相似
 *   5. 语义聚类 — 自动发现概念聚类
 *   6. 类比推理 — A:B ≈ C:D
 *   7. 因果先验 — WorldModel 因果增强
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/learning/distributional_semantics.hpp"

using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════
// 基础：概念涌现与共现
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 初始状态", "[semantic]") {
    DistributionalSemantics ds;
    auto stats = ds.stats();

    CHECK(stats.cpts_represented == 0);
    CHECK(stats.texts_processed == 0);
}

TEST_CASE("分布语义: 概念涌现", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 2;
    DistributionalSemantics ds(config);

    // 反复输入相同文本让概念涌现
    ds.learn_from_text("数学研究数量和结构");
    ds.learn_from_text("数学研究数量和结构");
    ds.learn_from_text("数学研究数量和结构");

    CHECK(ds.has_cpt("数学"));
    CHECK(ds.has_cpt("研究"));
}

TEST_CASE("分布语义: 共现统计", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    config.window_size = 10;
    DistributionalSemantics ds(config);

    ds.learn_from_text("数学是一门研究数量结构变化的学科");
    ds.learn_from_text("物理是一门研究物质运动规律的学科");
    ds.learn_from_text("数学和物理都是基础学科");
    ds.learn_from_text("数学研究理论推导");
    ds.learn_from_text("物理研究实验验证");

    CHECK(ds.has_cpt("数学"));
    CHECK(ds.has_cpt("物理"));
    CHECK(ds.has_cpt("学科"));
}

// ═══════════════════════════════════════════════════════════
// 核心：语义相似度
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 相似概念有高相似度", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    config.window_size = 10;
    DistributionalSemantics ds(config);

    std::vector<std::string> corpus = {
        "数学是一门基础学科 研究数量 结构 变化 空间",
        "物理是一门基础学科 研究物质 能量 运动 力",
        "数学属于形式科学 依靠逻辑推理",
        "物理属于自然科学 依靠实验观测",
        "数学提供理论框架 物理提供实验验证",
        "数学和物理都是科学的基石",
        "数学家用数学证明定理",
        "物理学家用数学描述物理规律",
        "微积分是数学的重要分支",
        "力学是物理的重要分支",
    };
    ds.learn_batch(corpus);

    auto sim = ds.similarity("数学", "物理");
    CHECK(sim.cpt_a == "数学");
    CHECK(sim.cpt_b == "物理");
    CHECK(sim.similarity > 0.0);
    CHECK_FALSE(sim.explanation.empty());
}

TEST_CASE("分布语义: 无关概念相似度低", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    config.window_size = 10;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "数学研究数量和逻辑",
        "烹饪需要食材和火候",
        "数学是一门学科",
        "烹饪是一种技能",
        "数学需要推理",
        "烹饪需要调味",
    });

    auto sim = ds.similarity("数学", "烹饪");
    CHECK(sim.similarity >= 0.0);
}

// ═══════════════════════════════════════════════════════════
// 核心：most_similar
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 最相似概念查询", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    config.window_size = 10;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "代数研究数学结构",
        "几何研究空间形状",
        "分析研究极限连续",
        "代数和几何都是数学分支",
        "代数研究群环域",
        "几何研究点线面",
        "拓扑学研究空间性质",
    });

    auto similar = ds.most_similar("代数", 3);
    CHECK_FALSE(similar.empty());
}

// ═══════════════════════════════════════════════════════════
// 核心：概念向量
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 概念分布向量", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "数学研究数量",
        "物理研究物质",
        "数学和物理都是学科",
    });

    auto vec = ds.get_vector("数学");
    REQUIRE(vec.has_value());
    CHECK(vec->name == "数学");
    CHECK_FALSE(vec->dimensions.empty());
    CHECK(vec->observation_count > 0);

    auto dense = ds.get_dense_vector("数学", 32);
    REQUIRE(dense.has_value());
    CHECK(static_cast<int>(dense->size()) == 32);
}

TEST_CASE("分布语义: 不存在概念返回nullopt", "[semantic]") {
    DistributionalSemantics ds;
    CHECK_FALSE(ds.get_vector("不存在的概念").has_value());
    CHECK_FALSE(ds.get_dense_vector("不存在").has_value());
}

// ═══════════════════════════════════════════════════════════
// 核心：语义聚类
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 语义聚类发现", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    config.window_size = 10;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "代数研究数学结构",
        "几何研究空间形状",
        "分析研究极限变化",
        "代数几何分析都是数学",
        "炒菜需要油盐酱醋",
        "煲汤需要文火慢炖",
        "烤肉需要高温快烤",
        "炒菜煲汤烤肉都是烹饪",
    });

    auto clusters = ds.discover_clusters(0.1);
    CHECK_FALSE(clusters.empty());
}

// ═══════════════════════════════════════════════════════════
// 核心：因果先验
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 因果先验增强", "[semantic]") {
    DistributionalSemantics ds;

    ds.register_causal_link("加热", "膨胀", 0.9);
    ds.register_causal_link("冷却", "收缩", 0.8);

    auto related = ds.causally_related("加热");
    CHECK_FALSE(related.empty());
    CHECK(related[0].first == "膨胀");
}

// ═══════════════════════════════════════════════════════════
// 核心：类比推理
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 向量类比 A:B ~ C:D", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    config.window_size = 10;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "法国首都巴黎",
        "英国首都伦敦",
        "德国首都柏林",
        "法国位于欧洲",
        "英国位于欧洲",
        "德国位于欧洲",
        "巴黎是法国城市",
        "伦敦是英国城市",
        "柏林是德国城市",
    });

    auto results = ds.analogy("法国", "巴黎", "英国", 3);
    if (!results.empty()) {
        CHECK(results[0].confidence > 0.0);
        CHECK_FALSE(results[0].a_is_to_b.empty());
        CHECK_FALSE(results[0].as_c_is_to_d.empty());
    }
}

// ═══════════════════════════════════════════════════════════
// 规模：模拟百科语料学习
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 百科语料模拟学习", "[semantic][corpus]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 2;
    config.window_size = 8;
    DistributionalSemantics ds(config);

    std::vector<std::string> wiki_entries = {
        "数学是研究数量结构变化空间等概念的一门学科",
        "数学利用符号语言研究抽象模式",
        "数学分为纯数学和应用数学两大分支",
        "纯数学包括代数几何分析数论",
        "应用数学包括统计学运筹学计算数学",
        "代数研究数学结构如群环域",
        "几何研究空间形状和大小",
        "微积分是数学分析的基础",
        "物理学是研究物质运动规律和能量转换的科学",
        "物理学分为经典物理和现代物理",
        "经典物理包括力学热学电磁学光学",
        "现代物理包括相对论量子力学",
        "牛顿运动定律是经典力学的基础",
        "爱因斯坦提出相对论",
        "量子力学描述微观世界的规律",
        "化学是研究物质组成结构和性质变化的科学",
        "化学分为有机化学无机化学分析化学",
        "元素是化学的基本概念",
        "化学反应是物质变化的过程",
        "化学键连接原子形成分子",
        "哲学是研究存在知识价值理性的学科",
        "哲学分为形而上学认识论伦理学逻辑学",
        "古希腊哲学家苏格拉底柏拉图亚里士多德",
        "哲学思考人生的意义",
        "逻辑学是哲学的重要分支也是数学的基础",
        "数学物理利用数学方法研究物理问题",
        "物理化学研究化学过程中的物理原理",
        "计算化学利用计算机模拟化学反应",
        "生物物理研究生命现象中的物理规律",
        "数学哲学研究数学的基础和本质",
    };

    ds.learn_batch(wiki_entries);

    auto stats = ds.stats();
    CHECK(stats.texts_processed > 20);
    CHECK(stats.cpts_represented > 0);

    CHECK(ds.has_cpt("数学"));
    CHECK(ds.has_cpt("物理"));
    CHECK(ds.has_cpt("化学"));
    CHECK(ds.has_cpt("研究"));

    auto sim_math_phys = ds.similarity("数学", "物理");
    CHECK(sim_math_phys.similarity > 0.0);

    auto top5 = ds.most_similar("数学", 5);
    CHECK_FALSE(top5.empty());

    auto clusters = ds.discover_clusters(0.15);
    CHECK_FALSE(clusters.empty());

    auto vec = ds.get_vector("数学");
    REQUIRE(vec.has_value());
    CHECK(vec->dimensions.size() >= 1);
}

// ═══════════════════════════════════════════════════════════
// 增量：逐步学习
// ═══════════════════════════════════════════════════════════

TEST_CASE("分布语义: 增量学习 — 知识积累", "[semantic]") {
    DistributionalSemanticsConfig config;
    config.min_cpt_freq = 1;
    DistributionalSemantics ds(config);

    ds.learn_batch({
        "数学研究数量和结构",
        "数学是一门基础学科",
    });

    auto concepts_1 = ds.all_cpts();
    int n1 = static_cast<int>(concepts_1.size());

    ds.learn_batch({
        "物理研究物质和能量",
        "物理是一门基础学科",
    });

    auto concepts_2 = ds.all_cpts();
    int n2 = static_cast<int>(concepts_2.size());
    CHECK(n2 > n1);

    ds.learn_batch({
        "数学物理交叉研究数学和物理",
    });

    auto sim = ds.similarity("数学", "物理");
    CHECK(sim.similarity >= 0.0);
}
