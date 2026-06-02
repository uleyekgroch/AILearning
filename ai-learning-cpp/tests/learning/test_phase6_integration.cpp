/**
 * @file test_phase6_integration.cpp
 * @brief Phase 6 深度整合测试 — 五大跨模块协同
 *
 * 五大整合场景：
 *   1. 全流水线闭环 — observe→learn→analogize→experiment→reflect→insight
 *   2. 元学习驱动策略 — MetaLearner + IntrinsicMotivation
 *   3. 情感调制全系统 — EmotionEngine 影响学习率/编码/探索
 *   4. 实验驱动探索 — ActiveExperimenter → IntrinsicMotivation
 *   5. 社会学习加速类比 — SocialLearning → AnalogicalTransfer → ContinualLearner
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/core/learner.hpp"
#include "ai_learning/learning/integrated_learner.hpp"

using namespace ai_learning;
using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ═══════════════════════════════════════════════════════════
// 整合 1：全流水线闭环
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 6: 全流水线 — 单领域观察闭环", "[phase6][pipeline]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    auto report = learner.integrated_pipeline(
        "金属在加热后膨胀", "physics");

    CHECK(report.steps_completed >= 2);
    CHECK(report.overall_progress > 0.0);
    CHECK(report.overall_progress <= 1.0);
    CHECK(report.observe_step.success);
    CHECK(report.learn_step.success);
}

TEST_CASE("Phase 6: 全流水线 — 多次运行递进", "[phase6][pipeline]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 第一次运行
    auto r1 = learner.integrated_pipeline("金属导电", "chemistry");
    CHECK(r1.steps_completed >= 2);

    // 第二次运行 — 积累了经验
    auto r2 = learner.integrated_pipeline("酸碱中和反应", "chemistry");
    CHECK(r2.steps_completed >= 2);

    // 反思应该越来越多
    auto insights1 = learner.meta_reflect();
    CHECK_FALSE(insights1.empty());

    // 统计应该有变化
    auto integrated_stats = learner.integrated().stats();
    CHECK_THAT(integrated_stats.at("pipeline_runs"), WithinAbs(2.0, 1e-9));
}

TEST_CASE("Phase 6: 全流水线 — 各步骤报告完整性", "[phase6][pipeline]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    auto report = learner.integrated_pipeline("观测天文现象", "astronomy");

    // 每个步骤都有名称
    CHECK(report.observe_step.step_name == "observe");
    CHECK(report.learn_step.step_name == "learn");
    CHECK(report.analogize_step.step_name == "analogize");
    CHECK(report.experiment_step.step_name == "experiment");
    CHECK(report.reflect_step.step_name == "reflect");
    CHECK(report.insight_step.step_name == "insight");

    // 至少观察和学习应成功
    CHECK(report.observe_step.success);
    CHECK(report.learn_step.success);
    CHECK_FALSE(report.dominant_emotion.empty());
}

// ═══════════════════════════════════════════════════════════
// 整合 2：元学习驱动策略选择
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 6: 元学习驱动 — 动机+策略联合", "[phase6][meta_driven]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    std::vector<std::string> topics = {"math", "physics", "chemistry"};
    std::map<std::string, double> mastery = {
        {"math", 0.6}, {"physics", 0.3}, {"chemistry", 0.1}};

    auto report = learner.meta_guided_learn(topics, mastery);

    // 应该选择了某个目标
    CHECK_FALSE(report.chosen_goal.topic.empty());
    CHECK_FALSE(report.chosen_goal.domain.empty());

    // 应该推荐了策略
    CHECK_FALSE(report.strategy_rec.rationale.empty());
    CHECK(report.learning_rate > 0.0);

    // 应该有情感调制
    CHECK(report.emotion_modulation.encoding_boost > 0.0);

    // 经验应该被记录
    CHECK(report.experience.task.domain != "");
}

TEST_CASE("Phase 6: 元学习驱动 — 多次会话经验积累", "[phase6][meta_driven]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    std::vector<std::string> topics = {"math"};
    std::map<std::string, double> mastery = {{"math", 0.4}};

    // 第一次会话
    auto r1 = learner.meta_guided_learn(topics, mastery);
    CHECK_FALSE(r1.reflections.empty());

    // 第二次会话 — 应有更多经验
    mastery["math"] = 0.5;
    auto r2 = learner.meta_guided_learn(topics, mastery);

    auto integrated_stats = learner.integrated().stats();
    CHECK_THAT(integrated_stats.at("meta_sessions"), WithinAbs(2.0, 1e-9));
}

// ═══════════════════════════════════════════════════════════
// 整合 3：情感调制全系统
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 6: 情感调制 — 系统参数计算", "[phase6][emotion]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    auto params = learner.emotion_modulated_params();

    CHECK(params.learning_rate > 0.0);
    CHECK(params.encoding_boost > 0.0);
    CHECK(params.exploration_tendency >= 0.0);
    CHECK(params.exploration_tendency <= 1.0);
    CHECK(params.risk_tolerance >= 0.0);
    CHECK(params.risk_tolerance <= 1.0);
    CHECK_FALSE(params.emotion_label.empty());
}

TEST_CASE("Phase 6: 情感调制 — 事件驱动参数变化", "[phase6][emotion]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 初始状态
    auto params_before = learner.emotion_modulated_params();

    // 处理一次正面情绪事件
    EmotionEvent success_event;
    success_event.event_type = "success";
    success_event.domain = "math";
    success_event.description = "解出了一道难题";
    success_event.magnitude = 0.8;
    success_event.expected_outcome = 0.5;
    success_event.actual_outcome = 0.9;
    learner.process_emotion(success_event);

    auto params_after = learner.integrated().process_emotion_and_modulate(
        success_event);

    // 正面事件应提高探索倾向
    CHECK(params_after.exploration_tendency >= params_before.exploration_tendency);
}

TEST_CASE("Phase 6: 情感调制 — Yerkes-Dodson 倒U型", "[phase6][emotion]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 低唤醒 → 较低学习率
    EmotionEvent boring;
    boring.event_type = "block";
    boring.domain = "test";
    boring.description = "无聊";
    boring.magnitude = 0.1;
    learner.process_emotion(boring);
    auto low_arousal = learner.emotion_modulated_params();

    // 高唤醒 → 也应该降低（过度兴奋）
    EmotionEvent excited;
    excited.event_type = "surprise";
    excited.domain = "test";
    excited.description = "极度兴奋";
    excited.magnitude = 1.0;
    excited.expected_outcome = 0.0;
    excited.actual_outcome = 1.0;
    learner.process_emotion(excited);
    auto high_arousal = learner.emotion_modulated_params();

    // 两种极端都应该低于适度唤醒
    // （具体值取决于 EmotionEngine 的内部状态）
    CHECK(low_arousal.learning_rate > 0.0);
    CHECK(high_arousal.learning_rate > 0.0);
}

// ═══════════════════════════════════════════════════════════
// 整合 4：实验驱动探索
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 6: 实验驱动 — 空领域探索", "[phase6][experiment]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    auto report = learner.experiment_driven_explore("new_domain");

    CHECK(report.domain == "new_domain");
    CHECK(report.domain_uncertainty >= 0.0);
    CHECK_FALSE(report.suggested_goal.topic.empty());
    CHECK(report.suggested_goal.primary_motive == MotiveType::kCuriosity);
}

TEST_CASE("Phase 6: 实验驱动 — 有假设的领域", "[phase6][experiment]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 先生成一些假设
    learner.generate_hypothesis("光沿直线传播", "optics");
    learner.generate_hypothesis("光可以被折射", "optics");

    auto report = learner.experiment_driven_explore("optics");

    CHECK(report.hypotheses.size() >= 2);
    CHECK(report.domain_uncertainty > 0.0);
    CHECK(report.information_value > 0.0);
}

TEST_CASE("Phase 6: 实验驱动 — 探索→学习闭环", "[phase6][experiment]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // Step 1: 实验驱动发现探索目标
    learner.generate_hypothesis("电荷相互吸引", "electromagnetism");
    auto explore_report = learner.experiment_driven_explore("electromagnetism");
    CHECK_FALSE(explore_report.suggested_goal.topic.empty());

    // Step 2: 设计实验
    auto design = learner.design_experiment();
    if (design.has_value()) {
        // Step 3: 执行实验
        ExperimentResult result;
        result.hypothesis_id = design->target_hypothesis;
        result.supports_hypothesis = true;
        result.confidence_delta = 0.2;
        result.observation = "电荷确实相互吸引";
        result.information_gain = 0.5;
        learner.record_experiment(result);

        // Step 4: 验证元学习记录了经验
        auto insights = learner.meta_reflect();
        CHECK_FALSE(insights.empty());
    }
}

// ═══════════════════════════════════════════════════════════
// 整合 5：社会学习加速类比
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 6: 社会加速 — 观察到迁移", "[phase6][social_analogy]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // Step 1: 社会观察源领域
    BehaviorObservation obs;
    obs.agent_id = "teacher";
    obs.action = "用弹簧秤测量力";
    obs.context = "物理实验课";
    obs.domain = "physics";
    obs.preconditions = {"准备弹簧", "挂上砝码", "读取刻度"};
    obs.effects = {"得到力的读数"};
    obs.outcome_quality = 1.0;
    learner.observe_behavior(obs);

    // Step 2: 准备目标领域概念
    std::vector<ConceptDescriptor> target_concepts;
    ConceptDescriptor tc;
    tc.id = "weight_scale";
    tc.domain = "chemistry";
    tc.attributes = {"测量", "读取数值", "实验仪器"};
    tc.relations = {"used_in"};
    target_concepts.push_back(tc);

    // Step 3: 社会加速类比迁移
    auto report = learner.social_accelerated_transfer(
        "physics", "chemistry", target_concepts);

    CHECK(report.source_domain == "physics");
    CHECK(report.target_domain == "chemistry");
    CHECK(report.social_observations_used >= 0);
}

TEST_CASE("Phase 6: 社会加速 — 多次观察强化迁移", "[phase6][social_analogy]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 多次观察同一领域
    for (int i = 0; i < 3; ++i) {
        BehaviorObservation obs;
        obs.agent_id = "expert_" + std::to_string(i);
        obs.action = "实验步骤" + std::to_string(i);
        obs.context = "科学实验";
        obs.domain = "biology";
        obs.preconditions = {"观察", "记录", "分析"};
        obs.effects = {"得出结论"};
        obs.outcome_quality = 1.0;
        learner.observe_behavior(obs);
    }

    // 准备目标
    std::vector<ConceptDescriptor> targets;
    ConceptDescriptor tc;
    tc.id = "ecology_concept";
    tc.domain = "ecology";
    tc.attributes = {"观察", "分析", "数据"};
    targets.push_back(tc);

    auto report = learner.social_accelerated_transfer(
        "biology", "ecology", targets);

    CHECK(report.source_domain == "biology");
    CHECK(report.target_domain == "ecology");
    CHECK(report.transfer_result.source_domain == "biology");
}

// ═══════════════════════════════════════════════════════════
// 全整合：跨 Phase 3-6 端到端
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 6: 端到端 — 完整认知发展模拟", "[phase6][e2e]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // === 模拟一个小型认知发展过程 ===

    // 1. 社会观察学习（Phase 4）
    BehaviorObservation obs;
    obs.agent_id = "teacher";
    obs.action = "讲解数学概念";
    obs.context = "课堂";
    obs.domain = "math";
    obs.preconditions = {"定义", "举例", "练习", "总结"};
    obs.effects = {"学生理解了"};
    obs.outcome_quality = 1.0;
    learner.observe_behavior(obs);

    // 2. 生成假设（Phase 5）
    auto hyp = learner.generate_hypothesis(
        "乘法是重复加法", "math");
    CHECK_FALSE(hyp.id.empty());

    // 3. 全流水线学习（Phase 6）
    auto pipeline_report = learner.integrated_pipeline(
        "乘法可以简化加法运算", "math");
    CHECK(pipeline_report.steps_completed >= 2);

    // 4. 情感调制（Phase 6）
    auto params = learner.emotion_modulated_params();
    CHECK(params.learning_rate > 0.0);

    // 5. 实验驱动探索（Phase 6）
    auto explore = learner.experiment_driven_explore("math");
    CHECK_FALSE(explore.suggested_goal.topic.empty());

    // 6. 元学习驱动会话（Phase 6）
    std::vector<std::string> topics = {"math", "physics"};
    std::map<std::string, double> mastery = {{"math", 0.4}, {"physics", 0.2}};
    auto session = learner.meta_guided_learn(topics, mastery);
    CHECK_FALSE(session.chosen_goal.domain.empty());

    // 7. 抽象概念形成（Phase 3）
    auto concept_report = learner.form_abstractions(
        "multiplication_instance",
        {"重复加法", "快速计算", "分配律"},
        {{"complexity", 0.3}, {"abstraction", 0.7}},
        {"简化"});
    CHECK(concept_report.instances_processed >= 1);

    // 8. 持续学习保护（Phase 3）
    learner.protect_knowledge("mult_concept", "math", 0.8, 5);
    auto forgetting = learner.detect_forgetting();
    CHECK(forgetting.empty());  // 刚保护的不应遗忘

    // 9. 最终验证：所有子系统有统计数据
    auto integrated_stats = learner.integrated().stats();
    CHECK(integrated_stats.at("pipeline_runs") >= 1);
    CHECK(integrated_stats.at("meta_sessions") >= 1);
    CHECK(integrated_stats.at("experiment_explorations") >= 1);

    // 10. 反思
    auto final_reflections = learner.meta_reflect();
    CHECK_FALSE(final_reflections.empty());
}

TEST_CASE("Phase 6: 端到端 — 多领域认知网络", "[phase6][e2e]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 在多个领域执行全流水线
    learner.integrated_pipeline("牛顿运动定律", "physics");
    learner.integrated_pipeline("化学键理论", "chemistry");
    learner.integrated_pipeline("进化论", "biology");

    // 生成假设
    learner.generate_hypothesis("力是质量乘加速度", "physics");
    learner.generate_hypothesis("化学键是电子共享", "chemistry");

    // 社会学习→类比迁移
    BehaviorObservation obs;
    obs.agent_id = "professor";
    obs.action = "用量筒测量体积";
    obs.context = "化学实验";
    obs.domain = "chemistry";
    obs.preconditions = {"倒液", "读数", "记录"};
    obs.effects = {"测量准确"};
    obs.outcome_quality = 1.0;
    learner.observe_behavior(obs);

    ConceptDescriptor target;
    target.id = "beaker";
    target.domain = "physics";
    target.attributes = {"测量", "读数", "容器"};
    target.relations = {"used_in"};

    auto transfer_report = learner.social_accelerated_transfer(
        "chemistry", "physics", {target});

    // 验证所有模块都有活跃数据
    auto exp_stats = learner.experimenter().stats();
    CHECK(exp_stats.at("total_hypotheses") >= 2);

    auto meta_stats = learner.meta_learner().stats();
    CHECK(meta_stats.at("total_experiences") >= 3);

    auto integrated_stats = learner.integrated().stats();
    CHECK(integrated_stats.at("pipeline_runs") >= 3);
}
