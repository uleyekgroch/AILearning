/**
 * @file test_human_capabilities.cpp
 * @brief 四大人类核心能力集成测试
 *
 * TDD 红灯阶段：先定义所有测试（预期行为），确认编译后逐步绿灯。
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/learning/execution_sandbox.hpp"
#include "ai_learning/learning/self_modifier.hpp"
#include "ai_learning/reasoning/world_model.hpp"
#include "ai_learning/learning/metacognition.hpp"
#include "ai_learning/core/learner.hpp"

#include <cmath>

using namespace ai_learning::core;
using namespace ai_learning::learning;
using namespace ai_learning::reasoning;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════════════
// 能力 1：动手做（Execution Sandbox）
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("ExecutionSandbox: 编译有效 C++ 代码", "[sandbox]") {
    SandboxConfig config;
    config.timeout_ms = 5000;
    LocalProcessSandbox sandbox(config);

    std::string valid_code = R"(
#include <iostream>
int main() {
    std::cout << "Hello from sandbox" << std::endl;
    return 0;
}
)";

    auto result = sandbox.execute(valid_code, "cpp");
    CHECK(result.success);
    CHECK(result.exit_code == 0);
    CHECK(result.stdout_output.find("Hello from sandbox") != std::string::npos);
}

TEST_CASE("ExecutionSandbox: 捕获编译错误", "[sandbox]") {
    SandboxConfig config;
    config.timeout_ms = 5000;
    LocalProcessSandbox sandbox(config);

    std::string invalid_code = R"(
int main() {
    undeclared_var = 42;  // 编译错误
    return 0;
}
)";

    auto result = sandbox.compile(invalid_code);
    CHECK_FALSE(result.success);
    CHECK_FALSE(result.stderr_output.empty());
    CHECK(result.error_type == "compile_error");
}

TEST_CASE("ExecutionSandbox: 捕获运行时错误", "[sandbox]") {
    SandboxConfig config;
    config.timeout_ms = 3000;
    LocalProcessSandbox sandbox(config);

    std::string runtime_error_code = R"(
#include <vector>
int main() {
    std::vector<int> v;
    int x = v.at(100);  // 抛出 out_of_range
    return x;
}
)";

    auto result = sandbox.execute(runtime_error_code, "cpp");
    CHECK_FALSE(result.success);
    CHECK(result.exit_code != 0);
}

TEST_CASE("ExecutionSandbox: 超时保护", "[sandbox]") {
    SandboxConfig config;
    config.timeout_ms = 500;  // 500ms 超时
    LocalProcessSandbox sandbox(config);

    std::string infinite_loop = R"(
int main() {
    while(true) {}  // 死循环
    return 0;
}
)";

    auto result = sandbox.execute(infinite_loop, "cpp");
    CHECK(result.error_type == "timeout");
}

TEST_CASE("FeedbackParser: 解析编译错误", "[sandbox]") {
    CompileError err{
        "test.cpp", 10, "error", "use of undeclared identifier 'foo'", "foo = 42"
    };

    auto analysis = FeedbackParser::analyze_compile_error(err);
    CHECK(analysis.contains("error_type"));
    CHECK_FALSE(analysis.at("error_type").empty());
}

TEST_CASE("FeedbackParser: 提取性能指标", "[sandbox]") {
    std::string output = "Tests passed: 42/50\nTime: 123ms\nScore: 0.84";
    auto metrics = FeedbackParser::extract_performance_metrics(output);

    CHECK(metrics.contains("tests_passed"));
    CHECK(metrics.contains("time_ms"));
}

// ═══════════════════════════════════════════════════════════════════
// 能力 2：自我修改（Self-Modification）
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("SelfModifier: 参数变异提案", "[self_modify]") {
    LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    Learner learner(config);

    SelfModifier modifier;
    auto proposals = modifier.propose_param_mutations(learner);

    CHECK_FALSE(proposals.empty());
    for (const auto& p : proposals) {
        CHECK_FALSE(p.param_name.empty());
        CHECK_FALSE(p.rationale.empty());
    }
}

TEST_CASE("SelfModifier: 应用和回滚参数修改", "[self_modify]") {
    LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    config.learning_rate = 0.01;
    Learner learner(config);

    ParameterMutation mutation{
        "learning_rate", 0.01, 0.05, "测试学习率提升", 0.1
    };

    auto applied = SelfModifier::apply_mutation(learner, mutation);
    CHECK(applied);

    auto rolled_back = SelfModifier::rollback_mutation(learner, mutation);
    CHECK(rolled_back);
}

TEST_CASE("SelfModifier: 进化循环（适应度驱动）", "[self_modify]") {
    LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    Learner learner(config);

    SelfModifier modifier;
    auto fitness = [](Learner& l) -> double {
        // 简单适应度：知识图谱大小
        auto stats = l.get_stats();
        return stats.at("entity_count") + stats.at("relation_count");
    };

    // 先学一些知识
    learner.learn_from_text("人工智能是计算机科学的分支");

    auto result = modifier.evolve_once(learner, fitness);
    CHECK(result.score_before >= 0.0);
    CHECK(result.improvement >= -1.0);  // 可能负向
}

TEST_CASE("StrategySelector: 推荐策略", "[self_modify]") {
    // 简单记忆任务 → 死记硬背或间隔重复
    auto s1 = StrategySelector::recommend("memorization", 0.3, 0.5);
    CHECK(s1 == StrategySelector::Strategy::kSpacedRepetition);

    // 探索任务 → 试错法
    auto s2 = StrategySelector::recommend("exploration", 0.1, 0.8);
    CHECK(s2 == StrategySelector::Strategy::kTrialAndError);

    // 理解任务 → 类比迁移
    auto s3 = StrategySelector::recommend("understanding", 0.5, 0.3);
    CHECK(s3 == StrategySelector::Strategy::kAnalogicalTransfer);
}

// ═══════════════════════════════════════════════════════════════════
// 能力 3：世界模型和因果推理（World Model）
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("WorldModel: 观察序列并学习因果", "[world_model]") {
    WorldModel wm;

    wm.observe_sequence({"下雨", "地面湿"}, "地面滑");
    wm.observe_sequence({"下雨", "地面湿"}, "地面滑");
    wm.observe_sequence({"洒水", "地面湿"}, "地面滑");

    auto stats = wm.stats();
    CHECK(stats.node_count >= 3);
    CHECK(stats.observations_processed >= 3);
}

TEST_CASE("WorldModel: 因果预测", "[world_model]") {
    WorldModel wm;

    // 添加因果规则
    wm.add_causal_rule({"open_file", "get_fd", CausalEdgeType::kCauses, 0.9, {}});;
    wm.add_causal_rule({"get_fd", "write_data", CausalEdgeType::kEnables, 0.95, {}});;
    wm.add_causal_rule({"write_data", "data_on_disk", CausalEdgeType::kCauses, 0.99, {}});;

    auto predictions = wm.predict({"open_file"});
    CHECK_FALSE(predictions.empty());

    // 应该能预测到 get_fd
    bool found_fd = false;
    for (const auto& [name, prob] : predictions) {
        if (name == "get_fd") {
            found_fd = true;
            CHECK_THAT(prob, WithinAbs(0.9, 0.1));
        }
    }
    CHECK(found_fd);
}

TEST_CASE("WorldModel: 干预推理（Pearl 第二层）", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"smoking", "lung_cancer", CausalEdgeType::kCauses, 0.8, {}});;
    wm.add_causal_rule({"exercise", "health", CausalEdgeType::kCauses, 0.6, {}});;
    wm.add_causal_rule({"genetics", "lung_cancer", CausalEdgeType::kCauses, 0.3, {}});;

    // 如果干预：停止吸烟
    auto result = wm.intervene({
        {"smoking", 0.0, "停止吸烟"}
    });

    CHECK(result.contains("lung_cancer"));
    CHECK(result.at("lung_cancer") < 0.8);  // 肺癌概率应降低
}

TEST_CASE("WorldModel: 反事实推理（Pearl 第三层）", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"rain", "wet_road", CausalEdgeType::kCauses, 0.95, {}});;
    wm.add_causal_rule({"wet_road", "accident", CausalEdgeType::kCauses, 0.4, {}});;

    // 反事实：如果当时没下雨，会怎样？
    auto cf = wm.counterfactual(
        "accident",
        {{"rain", 1.0, "下了雨"}},
        {{"rain", 0.0, "如果没下雨"}}
    );

    CHECK_FALSE(cf.predicted_outcome.empty());
    CHECK(cf.confidence >= 0.0);
    CHECK(cf.confidence <= 1.0);
}

TEST_CASE("WorldModel: 想象规划（DreamerV3 风格）", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"study", "knowledge", CausalEdgeType::kCauses, 0.8, {}});;
    wm.add_causal_rule({"practice", "skill", CausalEdgeType::kCauses, 0.7, {}});;
    wm.add_causal_rule({"knowledge", "exam_pass", CausalEdgeType::kCauses, 0.9, {}});;
    wm.add_causal_rule({"skill", "exam_pass", CausalEdgeType::kCauses, 0.6, {}});;

    auto plan = wm.imagine_plan("exam_pass", 3);
    CHECK_FALSE(plan.steps.empty());
    CHECK(plan.expected_reward > 0.0);
}

TEST_CASE("WorldModel: 因果路径查询", "[world_model]") {
    WorldModel wm;

    wm.add_causal_rule({"A", "B", CausalEdgeType::kCauses, 1.0, {}});;
    wm.add_causal_rule({"B", "C", CausalEdgeType::kCauses, 1.0, {}});;
    wm.add_causal_rule({"A", "D", CausalEdgeType::kCauses, 0.5, {}});;

    auto paths = wm.find_causal_path("A", "C");
    CHECK_FALSE(paths.empty());

    // 至少有一条路径 A→B→C
    bool found_abc = false;
    for (const auto& path : paths) {
        if (path.size() == 3 && path[0] == "A" && path[2] == "C") {
            found_abc = true;
        }
    }
    CHECK(found_abc);
}

// ═══════════════════════════════════════════════════════════════════
// 能力 4：元认知（Metacognition）
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Metacognition: 知识置信度评估", "[metacognition]") {
    MetacognitionEngine meta(0.5);

    // 记录学习结果
    meta.record_outcome("AI基础", true);
    meta.record_outcome("AI基础", true);
    meta.record_outcome("AI基础", false);

    auto confidence = meta.assess_confidence("AI基础");
    CHECK(confidence.topic == "AI基础");
    CHECK(confidence.times_used == 3);
    CHECK(confidence.times_correct == 2);
    CHECK_THAT(confidence.confidence, WithinAbs(2.0/3.0, 0.1));
}

TEST_CASE("Metacognition: 知识盲区检测", "[metacognition]") {
    LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    Learner learner(config);

    MetacognitionEngine meta;
    auto gaps = meta.detect_gaps(learner);

    // 新建的学习体应该有很多盲区
    CHECK_FALSE(gaps.empty());
    for (const auto& gap : gaps) {
        CHECK_FALSE(gap.topic.empty());
        CHECK(gap.urgency >= 0.0);
        CHECK(gap.urgency <= 1.0);
    }
}

TEST_CASE("Metacognition: 我是否知道？", "[metacognition]") {
    MetacognitionEngine meta(0.5);

    // 没有记录 → 不知道
    CHECK_FALSE(meta.knows_about("量子物理"));

    // 多次正确 → 知道
    for (int i = 0; i < 10; ++i) {
        meta.record_outcome("编程基础", true);
    }
    CHECK(meta.knows_about("编程基础"));

    // 混合记录，低于阈值 → 不确定
    meta.record_outcome("模糊概念", true);
    meta.record_outcome("模糊概念", false);
    meta.record_outcome("模糊概念", false);
    meta.record_outcome("模糊概念", false);
    CHECK_FALSE(meta.knows_about("模糊概念"));
}

TEST_CASE("Metacognition: 学习策略评估", "[metacognition]") {
    MetacognitionEngine meta;

    // 性能持续下降 → 需要换策略
    auto assessment = meta.evaluate_strategy(0.3, -0.1);
    CHECK_THAT(assessment.effectiveness, WithinAbs(0.3, 0.2));
    CHECK_FALSE(assessment.recommended_strategy.empty());

    // 性能良好 → 保持当前策略
    auto assessment2 = meta.evaluate_strategy(0.9, 0.05);
    CHECK(assessment2.effectiveness > 0.5);
}

TEST_CASE("Metacognition: 主动信息寻求", "[metacognition]") {
    MetacognitionEngine meta(0.5);

    // 对低置信度主题应寻求信息
    auto need = meta.should_seek_info("完全未知的领域");
    CHECK(need.has_value());
    CHECK_FALSE(need->query.empty());
    CHECK(need->priority > 0.0);

    // 对高置信度主题不需要
    for (int i = 0; i < 20; ++i) {
        meta.record_outcome("已掌握知识", true);
    }
    auto need2 = meta.should_seek_info("已掌握知识");
    CHECK_FALSE(need2.has_value());
}

TEST_CASE("Metacognition: 生成完整报告", "[metacognition]") {
    LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;
    Learner learner(config);

    learner.learn_from_text("人工智能是计算机科学的分支");

    MetacognitionEngine meta;
    auto report = meta.generate_report(learner);

    CHECK(report.overall_confidence >= 0.0);
    CHECK(report.overall_confidence <= 1.0);
    CHECK(report.learning_efficiency >= 0.0);
    CHECK_FALSE(report.gaps.empty());
    CHECK_FALSE(report.dimension_scores.empty());
}

TEST_CASE("Metacognition: 生成学习计划", "[metacognition]") {
    MetacognitionEngine meta;

    std::vector<KnowledgeGap> gaps = {
        {"CUDA编程", 0.9, "需要理解GPU计算", {"C++", "并行计算"}, "学习CUDA教程"},
        {"分布式系统", 0.7, "需要理解一致性", {"网络", "算法"}, "阅读论文"},
    };

    auto plan = meta.generate_learning_plan(gaps);
    CHECK(plan.size() == gaps.size());

    // 按紧迫性排序
    CHECK(plan[0].priority >= plan[1].priority);
}
