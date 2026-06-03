/**
 * @file test_world_model.cpp
 * @brief 世界模型单元测试 — 因果 DAG + Pearl 三层推理 + 想象规划
 *
 * 验证：
 *   1. 添加节点和因果边
 *   2. 观察事件序列 → 因果推断
 *   3. 预测（Pearl 第一层）
 *   4. 干预推理（Pearl 第二层）
 *   5. 反事实推理（Pearl 第三层）
 *   6. 因果路径查找
 *   7. 想象规划
 *   8. 模型管理：添加/移除/验证
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/reasoning/world_model.hpp"

using namespace ai_learning::reasoning;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════
// 基础：构造和空状态
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 初始空状态", "[world_model]") {
    WorldModel wm;

    auto stats = wm.stats();
    CHECK(stats.node_count == 0);
    CHECK(stats.edge_count == 0);
    CHECK(stats.observations_processed == 0);

    CHECK(wm.nodes().empty());
    CHECK(wm.causal_edges().empty());
}

// ═══════════════════════════════════════════════════════════
// 添加因果规则
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 添加因果规则", "[world_model]") {
    WorldModel wm;

    CausalEdge edge{"rain", "wet_ground", CausalEdgeType::kCauses, 0.9, ""};
    wm.add_causal_rule(edge);

    CHECK(wm.nodes().size() == 2);
    CHECK(wm.causal_edges().size() == 1);

    auto stats = wm.stats();
    CHECK(stats.node_count == 2);
    CHECK(stats.edge_count == 1);
    CHECK(stats.causal_rules_learned == 1);
}

TEST_CASE("世界模型: 重复添加同一规则更新强度", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.5, ""});
    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});

    // 不应重复创建边
    CHECK(wm.causal_edges().size() == 1);
    // 强度应更新
    CHECK_THAT(wm.causal_edges()[0].strength, WithinAbs(0.9, 0.01));
}

TEST_CASE("世界模型: 移除因果规则", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"X", "Y", CausalEdgeType::kCauses, 0.8, ""});
    CHECK(wm.causal_edges().size() == 1);

    bool removed = wm.remove_causal_rule("X", "Y");
    CHECK(removed);
    CHECK(wm.causal_edges().empty());

    // 再次移除应返回 false
    removed = wm.remove_causal_rule("X", "Y");
    CHECK_FALSE(removed);
}

// ═══════════════════════════════════════════════════════════
// 观察事件序列
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 观察事件序列", "[world_model]") {
    WorldModel wm;

    // 多次观察相同事件→结果模式
    for (int i = 0; i < 5; ++i) {
        wm.observe_sequence({"rain"}, "wet_ground");
    }

    auto stats = wm.stats();
    CHECK(stats.node_count == 2);
    CHECK(stats.observations_processed == 5);

    // 应推断出因果边 rain → wet_ground
    CHECK_FALSE(wm.causal_edges().empty());
}

TEST_CASE("世界模型: 观察相关性", "[world_model]") {
    WorldModel wm;

    wm.observe_correlation("temperature", "ice_melt", 0.85);

    CHECK(wm.nodes().size() == 2);
    CHECK_FALSE(wm.causal_edges().empty());

    // 相关性 > 0.6 应创建边
    auto& edges = wm.causal_edges();
    bool found = false;
    for (const auto& e : edges) {
        if (e.type == CausalEdgeType::kCorrelates) found = true;
    }
    CHECK(found);
}

TEST_CASE("世界模型: 弱相关不创建边", "[world_model]") {
    WorldModel wm;

    wm.observe_correlation("A", "B", 0.3);

    // 相关性 < 0.6 不应创建边
    CHECK(wm.causal_edges().empty());
}

TEST_CASE("世界模型: 观察干预效果", "[world_model]") {
    WorldModel wm;

    std::map<std::string, double> before = {{"speed", 10.0}, {"fuel", 50.0}};
    std::map<std::string, double> after  = {{"speed", 20.0}, {"fuel", 40.0}};

    wm.observe_intervention("press_accelerator", before, after);

    // observe_intervention creates nodes for: action + each variable in after
    CHECK(wm.nodes().size() == 3);  // press_accelerator + speed + fuel
    CHECK_FALSE(wm.causal_edges().empty());
}

// ═══════════════════════════════════════════════════════════
// 预测（Pearl 第一层）
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 预测因果关系", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"rain", "wet_ground", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"wet_ground", "slippery", CausalEdgeType::kCauses, 0.7, ""});

    auto effects = wm.predict({"rain"});

    // 应预测出 wet_ground 和 slippery
    bool has_wet = false, has_slippery = false;
    for (const auto& [name, prob] : effects) {
        if (name == "wet_ground") has_wet = true;
        if (name == "slippery") has_slippery = true;
    }
    CHECK(has_wet);
    CHECK(has_slippery);
}

TEST_CASE("世界模型: 预测概率递减", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.8, ""});
    wm.add_causal_rule({"B", "C", CausalEdgeType::kCauses, 0.7, ""});

    auto effects = wm.predict({"A"});

    // 找到 B 和 C 的概率
    double prob_b = 0.0, prob_c = 0.0;
    for (const auto& [name, prob] : effects) {
        if (name == "B") prob_b = prob;
        if (name == "C") prob_c = prob;
    }

    CHECK_THAT(prob_b, WithinAbs(0.8, 0.01));
    CHECK(prob_c < prob_b);  // 距离越远概率越低
}

TEST_CASE("世界模型: 预测未知原因返回空", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});

    auto effects = wm.predict({"unknown"});
    CHECK(effects.empty());
}

// ═══════════════════════════════════════════════════════════
// 干预推理（Pearl 第二层）
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 干预推理", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"exercise", "health", CausalEdgeType::kCauses, 0.8, ""});

    auto result = wm.intervene({{"exercise", 1.0, "开始锻炼"}});

    CHECK(result.count("exercise") > 0);
    CHECK_THAT(result.at("exercise"), WithinAbs(1.0, 0.01));
}

// ═══════════════════════════════════════════════════════════
// 反事实推理（Pearl 第三层）
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 反事实推理", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"rain", "flood", CausalEdgeType::kCauses, 0.9, ""});

    auto result = wm.counterfactual(
        "flood",
        {{"rain", 1.0, "下了大雨"}},
        {{"rain", 0.0, "如果不下雨"}}
    );

    CHECK(result.original_scenario == "flood");
    CHECK_FALSE(result.intervention.empty());
    CHECK(result.confidence >= 0.0);
    CHECK(result.confidence <= 1.0);
}

// ═══════════════════════════════════════════════════════════
// 因果路径查找
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 因果路径查找", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"B", "C", CausalEdgeType::kCauses, 0.8, ""});
    wm.add_causal_rule({"C", "D", CausalEdgeType::kCauses, 0.7, ""});

    auto paths = wm.find_causal_path("A", "D");

    CHECK_FALSE(paths.empty());
    // 至少有一条路径: A→B→C→D
    bool found_direct = false;
    for (const auto& path : paths) {
        if (path.size() == 4 &&
            path[0] == "A" && path[1] == "B" &&
            path[2] == "C" && path[3] == "D") {
            found_direct = true;
        }
    }
    CHECK(found_direct);
}

TEST_CASE("世界模型: 无路径返回空", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"C", "D", CausalEdgeType::kCauses, 0.9, ""});

    auto paths = wm.find_causal_path("A", "D");
    CHECK(paths.empty());
}

// ═══════════════════════════════════════════════════════════
// 想象规划（DreamerV3 风格）
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 想象规划", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"study", "knowledge", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"knowledge", "exam_pass", CausalEdgeType::kCauses, 0.8, ""});

    auto plan = wm.imagine_plan("exam_pass");

    CHECK(plan.goal == "exam_pass");
    CHECK_FALSE(plan.steps.empty());
}

TEST_CASE("世界模型: 评估行动计划", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"B", "C", CausalEdgeType::kCauses, 0.7, ""});

    auto [reward, uncertainty] = wm.evaluate_plan({"A", "B", "C"});

    CHECK(reward >= 0.0);
    CHECK(uncertainty >= 0.0);
}

TEST_CASE("世界模型: 空行动计划", "[world_model]") {
    WorldModel wm;

    auto [reward, uncertainty] = wm.evaluate_plan({});

    CHECK_THAT(reward, WithinAbs(0.0, 0.01));
    CHECK_THAT(uncertainty, WithinAbs(1.0, 0.01));
}

// ═══════════════════════════════════════════════════════════
// 模型验证
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 验证无问题", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"B", "C", CausalEdgeType::kCauses, 0.8, ""});

    auto issues = wm.validate();
    CHECK(issues.empty());
}

TEST_CASE("世界模型: 验证检测自环", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "A", CausalEdgeType::kCauses, 0.5, ""});

    auto issues = wm.validate();
    CHECK_FALSE(issues.empty());
}

TEST_CASE("世界模型: 验证检测环", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"B", "A", CausalEdgeType::kCauses, 0.9, ""});

    auto issues = wm.validate();
    CHECK_FALSE(issues.empty());
}

// ═══════════════════════════════════════════════════════════
// 统计
// ═══════════════════════════════════════════════════════════

TEST_CASE("世界模型: 统计正确反映状态", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 0.9, ""});
    wm.add_causal_rule({"B", "C", CausalEdgeType::kCorrelates, 0.5, ""});
    wm.add_causal_rule({"D", "E", CausalEdgeType::kPrevents, 0.6, ""});

    auto stats = wm.stats();
    CHECK(stats.node_count == 5);
    CHECK(stats.edge_count == 3);
    CHECK(stats.causal_rules_learned == 1);  // 只有 kCauses
}
