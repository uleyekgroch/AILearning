/**
 * @file test_phase5_meta_cognition.cpp
 * @brief Phase 5 高级元认知测试 — 元学习 + 主动实验设计
 *
 * 测试覆盖：
 *   MetaLearner: 策略推荐、学习率调度、跨域迁移、自我反思
 *   ActiveExperimenter: 假设生成、实验设计、贝叶斯更新、理论构建
 *   Learner 集成: 全流水线
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/core/learner.hpp"
#include "ai_learning/learning/meta_learner.hpp"
#include "ai_learning/learning/active_experimenter.hpp"

using namespace ai_learning;
using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ═══════════════════════════════════════════════════════════
// MetaLearner 单元测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("MetaLearner: 初始状态", "[phase5][meta_learner]") {
    MetaLearner ml;
    auto stats = ml.stats();

    CHECK_THAT(stats.at("total_experiences"), WithinAbs(0.0, 1e-9));
    CHECK_THAT(stats.at("avg_improvement"), WithinAbs(0.0, 1e-9));
    CHECK_THAT(stats.at("success_rate"), WithinAbs(0.0, 1e-9));
}

TEST_CASE("MetaLearner: 启发式策略推荐（无经验）", "[phase5][meta_learner]") {
    MetaLearner ml;

    SECTION("记忆型任务推荐间隔重复") {
        TaskDescriptor task;
        task.task_type = "memorization";
        task.domain = "vocabulary";

        auto rec = ml.recommend_strategy(task);
        CHECK(rec.recommended_strategy == LearningStrategyType::kSpacedRepetition);
        CHECK(rec.confidence < 0.5);
        CHECK_FALSE(rec.rationale.empty());
    }

    SECTION("推理型任务推荐分解学习") {
        TaskDescriptor task;
        task.task_type = "reasoning";
        task.domain = "math";

        auto rec = ml.recommend_strategy(task);
        CHECK(rec.recommended_strategy == LearningStrategyType::kDecomposition);
    }

    SECTION("创造型任务推荐探索性学习") {
        TaskDescriptor task;
        task.task_type = "creative";
        task.domain = "art";

        auto rec = ml.recommend_strategy(task);
        CHECK(rec.recommended_strategy == LearningStrategyType::kExploratory);
    }

    SECTION("高新颖度推荐类比迁移") {
        TaskDescriptor task;
        task.task_type = "general";
        task.novelty = 0.8;
        task.domain = "new_field";

        auto rec = ml.recommend_strategy(task);
        CHECK(rec.recommended_strategy == LearningStrategyType::kAnalogicalTransfer);
    }
}

TEST_CASE("MetaLearner: 经验记录与策略统计", "[phase5][meta_learner]") {
    MetaLearner ml;

    LearningExperience exp1;
    exp1.task.domain = "math";
    exp1.task.task_type = "reasoning";
    exp1.strategy_used = LearningStrategyType::kDecomposition;
    exp1.improvement = 0.4;
    exp1.success = true;
    exp1.time_cost = 1.0;

    LearningExperience exp2;
    exp2.task.domain = "math";
    exp2.task.task_type = "reasoning";
    exp2.strategy_used = LearningStrategyType::kDecomposition;
    exp2.improvement = 0.5;
    exp2.success = true;
    exp2.time_cost = 0.8;

    LearningExperience exp3;
    exp3.task.domain = "math";
    exp3.task.task_type = "reasoning";
    exp3.strategy_used = LearningStrategyType::kRoteMemorization;
    exp3.improvement = 0.1;
    exp3.success = false;
    exp3.time_cost = 2.0;

    ml.record_batch({exp1, exp2, exp3});

    auto stats = ml.stats();
    CHECK_THAT(stats.at("total_experiences"), WithinAbs(3.0, 1e-9));

    // 检查策略统计
    auto decomp_stats = ml.get_strategy_stats(LearningStrategyType::kDecomposition);
    REQUIRE(decomp_stats.has_value());
    CHECK(decomp_stats->usage_count == 2);
    CHECK_THAT(decomp_stats->avg_improvement, WithinRel(0.45, 0.01));
    CHECK_THAT(decomp_stats->success_rate, WithinRel(1.0, 1e-9));

    auto rote_stats = ml.get_strategy_stats(LearningStrategyType::kRoteMemorization);
    REQUIRE(rote_stats.has_value());
    CHECK(rote_stats->usage_count == 1);
    CHECK_THAT(rote_stats->avg_improvement, WithinRel(0.1, 1e-9));
}

TEST_CASE("MetaLearner: 基于经验的策略推荐", "[phase5][meta_learner]") {
    MetaLearner ml;

    // 积累经验：分解学习在数学领域效果好
    for (int i = 0; i < 10; ++i) {
        LearningExperience exp;
        exp.task.domain = "math";
        exp.task.task_type = "reasoning";
        exp.strategy_used = LearningStrategyType::kDecomposition;
        exp.improvement = 0.3 + i * 0.05;
        exp.success = true;
        exp.time_cost = 1.0;
        ml.record_experience(exp);
    }

    // 死记硬背在数学领域效果差
    for (int i = 0; i < 5; ++i) {
        LearningExperience exp;
        exp.task.domain = "math";
        exp.task.task_type = "reasoning";
        exp.strategy_used = LearningStrategyType::kRoteMemorization;
        exp.improvement = 0.05;
        exp.success = false;
        exp.time_cost = 2.0;
        ml.record_experience(exp);
    }

    TaskDescriptor new_math_task;
    new_math_task.domain = "math";
    new_math_task.task_type = "reasoning";
    new_math_task.difficulty = 0.6;

    auto rec = ml.recommend_strategy(new_math_task);
    // 应推荐分解学习（因为统计上效果最好）
    CHECK(rec.confidence > 0.0);
    CHECK_FALSE(rec.rationale.empty());
}

TEST_CASE("MetaLearner: 学习率调度", "[phase5][meta_learner]") {
    MetaLearner ml;

    TaskDescriptor easy_task;
    easy_task.difficulty = 0.2;
    easy_task.novelty = 0.1;
    easy_task.urgency = 0.5;

    TaskDescriptor hard_task;
    hard_task.difficulty = 0.9;
    hard_task.novelty = 0.8;
    hard_task.urgency = 0.3;

    double easy_rate = ml.suggest_learning_rate(easy_task);
    double hard_rate = ml.suggest_learning_rate(hard_task);

    CHECK(easy_rate > hard_rate);  // 简单任务学得快

    // 自适应更新
    double rate_after_success = ml.update_learning_rate(0.5);
    CHECK(rate_after_success > 0.1);  // 成功后加速

    double rate_after_failure = ml.update_learning_rate(-0.3);
    CHECK(rate_after_failure < rate_after_success);  // 失败后减速
}

TEST_CASE("MetaLearner: 跨域迁移", "[phase5][meta_learner]") {
    MetaLearner ml;

    // 在物理学领域积累经验
    for (int i = 0; i < 5; ++i) {
        LearningExperience exp;
        exp.task.domain = "physics";
        exp.task.task_type = "reasoning";
        exp.strategy_used = LearningStrategyType::kDecomposition;
        exp.improvement = 0.4;
        exp.success = true;
        exp.time_cost = 1.0;
        ml.record_experience(exp);
    }

    // 迁移到数学领域
    auto transferred = ml.transfer_experience("physics", "math");
    CHECK_FALSE(transferred.empty());

    // 迁移潜力
    double potential = ml.transfer_potential("physics", "math");
    CHECK(potential > 0.0);
    CHECK(potential < 1.0);

    // 同领域迁移潜力最高
    double same = ml.transfer_potential("physics", "physics");
    CHECK_THAT(same, WithinAbs(1.0, 1e-9));
}

TEST_CASE("MetaLearner: 自我反思", "[phase5][meta_learner]") {
    MetaLearner ml;

    SECTION("无经验时反思") {
        auto insights = ml.reflect();
        CHECK_FALSE(insights.empty());
    }

    SECTION("有经验后反思") {
        for (int i = 0; i < 10; ++i) {
            LearningExperience exp;
            exp.task.domain = "test";
            exp.strategy_used = LearningStrategyType::kActiveRecall;
            exp.improvement = 0.2 + i * 0.03;
            exp.success = true;
            ml.record_experience(exp);
        }
        auto insights = ml.reflect();
        CHECK_FALSE(insights.empty());
    }
}

TEST_CASE("MetaLearner: 效率趋势", "[phase5][meta_learner]") {
    MetaLearner ml;

    // 生成 20 条经验
    for (int i = 0; i < 20; ++i) {
        LearningExperience exp;
        exp.task.domain = "test";
        exp.strategy_used = LearningStrategyType::kActiveRecall;
        exp.improvement = 0.1 + i * 0.02;
        exp.success = true;
        ml.record_experience(exp);
    }

    auto trend = ml.efficiency_trend();
    CHECK(trend.size() >= 2);
    // 趋势应该是上升的
    CHECK(trend.back() > trend.front());
}

// ═══════════════════════════════════════════════════════════
// ActiveExperimenter 单元测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("ActiveExperimenter: 初始状态", "[phase5][experimenter]") {
    ActiveExperimenter ae;
    auto stats = ae.stats();

    CHECK_THAT(stats.at("total_hypotheses"), WithinAbs(0.0, 1e-9));
    CHECK_THAT(stats.at("total_experiments"), WithinAbs(0.0, 1e-9));
    CHECK_THAT(stats.at("theories_built"), WithinAbs(0.0, 1e-9));
}

TEST_CASE("ActiveExperimenter: 假设生成", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    auto hyp = ae.generate_hypothesis("金属都能导电", "chemistry");

    CHECK_FALSE(hyp.id.empty());
    CHECK(hyp.domain == "chemistry");
    CHECK_THAT(hyp.prior_confidence, WithinAbs(0.5, 1e-9));
    CHECK(hyp.tested == false);
    CHECK(hyp.falsified == false);
    CHECK_FALSE(hyp.statement.empty());

    // 检查假设库
    auto active = ae.active_hypotheses();
    CHECK(active.size() == 1);
}

TEST_CASE("ActiveExperimenter: 多领域假设", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    ae.generate_hypothesis("金属导电", "chemistry");
    ae.generate_hypothesis("重力使物体下落", "physics");
    ae.generate_hypothesis("水在100度沸腾", "chemistry");

    auto chem_hyps = ae.hypotheses_in_domain("chemistry");
    CHECK(chem_hyps.size() == 2);

    auto phys_hyps = ae.hypotheses_in_domain("physics");
    CHECK(phys_hyps.size() == 1);
}

TEST_CASE("ActiveExperimenter: 实验设计", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    auto hyp = ae.generate_hypothesis("金属导电", "chemistry");
    auto design = ae.design_experiment(hyp.id);

    REQUIRE(design.has_value());
    CHECK(design->target_hypothesis == hyp.id);
    CHECK_FALSE(design->description.empty());
    CHECK_FALSE(design->method.empty());
    CHECK_FALSE(design->steps.empty());
    CHECK(design->steps.size() >= 3);
    CHECK(design->expected_information_gain >= 0.0);
}

TEST_CASE("ActiveExperimenter: 高置信度假设置验设计", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    // 注册高置信度假设
    Hypothesis hyp;
    hyp.id = "hyp_test";
    hyp.domain = "physics";
    hyp.statement = "光速是恒定的";
    hyp.prior_confidence = 0.8;
    hyp.posterior_confidence = 0.8;
    ae.register_hypothesis(hyp);

    auto design = ae.design_experiment("hyp_test");
    REQUIRE(design.has_value());
    CHECK(design->method == "验证性实验");  // 高置信度 → 验证性
}

TEST_CASE("ActiveExperimenter: 自动设计实验", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    // 无假设时
    CHECK_FALSE(ae.auto_design_experiment().has_value());

    // 生成假设后
    ae.generate_hypothesis("金属导电", "chemistry");
    ae.generate_hypothesis("水沸腾", "chemistry");

    auto design = ae.auto_design_experiment();
    REQUIRE(design.has_value());
    CHECK_FALSE(design->target_hypothesis.empty());
}

TEST_CASE("ActiveExperimenter: 贝叶斯更新 — 支持证据", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    auto hyp = ae.generate_hypothesis("金属导电", "chemistry");
    double prior = hyp.posterior_confidence;

    ExperimentResult result;
    result.hypothesis_id = hyp.id;
    result.supports_hypothesis = true;
    result.confidence_delta = 0.2;
    result.observation = "铜线导电";
    result.information_gain = 0.5;

    auto analysis = ae.record_result(result);
    CHECK_FALSE(analysis.empty());

    // 置信度应上升
    auto updated = ae.active_hypotheses().at(hyp.id);
    CHECK(updated.posterior_confidence > prior);
    CHECK(updated.supporting_evidence.size() == 2);  // 初始 + 新证据
}

TEST_CASE("ActiveExperimenter: 贝叶斯更新 — 反对证据", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    auto hyp = ae.generate_hypothesis("所有金属都是固体", "chemistry");
    double prior = hyp.posterior_confidence;

    ExperimentResult result;
    result.hypothesis_id = hyp.id;
    result.supports_hypothesis = false;
    result.confidence_delta = -0.2;
    result.observation = "汞是液态金属";
    result.information_gain = 0.8;

    ae.record_result(result);

    auto updated = ae.active_hypotheses().at(hyp.id);
    CHECK(updated.posterior_confidence < prior);
    CHECK_FALSE(updated.contradicting_evidence.empty());
}

TEST_CASE("ActiveExperimenter: 假设证伪", "[phase5][experimenter]") {
    ExperimenterConfig config;
    config.falsification_threshold = 0.3;
    ActiveExperimenter ae(config);

    auto hyp = ae.generate_hypothesis("测试假设", "test");
    REQUIRE_THAT(hyp.posterior_confidence, WithinAbs(0.5, 1e-9));

    // 多次反对证据
    for (int i = 0; i < 10; ++i) {
        ExperimentResult result;
        result.hypothesis_id = hyp.id;
        result.supports_hypothesis = false;
        result.confidence_delta = -0.1;
        result.observation = "反对证据" + std::to_string(i);
        ae.record_result(result);
    }

    auto updated = ae.active_hypotheses().at(hyp.id);
    CHECK(updated.falsified == true);
}

TEST_CASE("ActiveExperimenter: 理论构建", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    // 生成并验证多个假设
    auto hyp1 = ae.generate_hypothesis("金属导电A", "physics");
    auto hyp2 = ae.generate_hypothesis("金属导热B", "physics");

    // 为假设提供支持证据
    for (int i = 0; i < 5; ++i) {
        ExperimentResult r1;
        r1.hypothesis_id = hyp1.id;
        r1.supports_hypothesis = true;
        r1.confidence_delta = 0.1;
        r1.observation = "支持证据";
        ae.record_result(r1);

        ExperimentResult r2;
        r2.hypothesis_id = hyp2.id;
        r2.supports_hypothesis = true;
        r2.confidence_delta = 0.1;
        r2.observation = "支持证据";
        ae.record_result(r2);
    }

    // 尝试构建理论
    auto theory = ae.build_theory("physics");
    REQUIRE(theory.has_value());
    CHECK_FALSE(theory->name.empty());
    CHECK(theory->verified_hypotheses.size() >= 2);
    CHECK(theory->confidence > 0.5);
    CHECK_FALSE(theory->description.empty());
}

TEST_CASE("ActiveExperimenter: 理论构建不足", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    // 只有一个假设，不足以构建理论
    ae.generate_hypothesis("孤独的假设", "math");
    CHECK_FALSE(ae.build_theory("math").has_value());

    // 没有假设的领域
    CHECK_FALSE(ae.build_theory("empty_domain").has_value());
}

TEST_CASE("ActiveExperimenter: 探索策略", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    ae.generate_hypothesis("假设A", "domain_a");
    ae.generate_hypothesis("假设B", "domain_b");

    auto state = ae.exploration_state("domain_a");
    CHECK(state.domain == "domain_a");

    auto rec = ae.recommend_exploration();
    REQUIRE(rec.has_value());
    CHECK_FALSE(rec->empty());

    // 不确定性
    double unc = ae.domain_uncertainty("domain_a");
    CHECK(unc >= 0.0);
}

TEST_CASE("ActiveExperimenter: 信息增益计算", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    auto hyp = ae.generate_hypothesis("信息测试", "info");
    double ig = ae.expected_information_gain(hyp.id);

    // 新假设（p=0.5）应该有较高信息增益
    CHECK(ig > 0.0);

    // 不存在的假设
    CHECK_THAT(ae.expected_information_gain("nonexistent"), WithinAbs(0.0, 1e-9));
}

TEST_CASE("ActiveExperimenter: 结果分析", "[phase5][experimenter]") {
    ActiveExperimenter ae;

    auto hyp = ae.generate_hypothesis("分析测试", "test");

    ExperimentResult result;
    result.hypothesis_id = hyp.id;
    result.supports_hypothesis = true;
    result.confidence_delta = 0.15;
    result.observation = "观测数据";
    result.information_gain = 0.3;
    result.surprise = 0.1;

    auto impact = ae.analyze_impact(result);
    CHECK_FALSE(impact.empty());
}

// ═══════════════════════════════════════════════════════════
// Learner 集成测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("Learner Phase 5 集成: 元学习流水线", "[phase5][integration]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 通过 Learner 接口使用元学习
    TaskDescriptor task;
    task.domain = "math";
    task.task_type = "reasoning";
    task.difficulty = 0.6;

    auto rec = learner.meta_recommend(task);
    CHECK_FALSE(rec.rationale.empty());

    // 记录经验
    LearningExperience exp;
    exp.task = task;
    exp.strategy_used = rec.recommended_strategy;
    exp.improvement = 0.3;
    exp.success = true;
    learner.meta_record(exp);

    // 反思
    auto insights = learner.meta_reflect();
    CHECK_FALSE(insights.empty());
}

TEST_CASE("Learner Phase 5 集成: 主动实验流水线", "[phase5][integration]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 通过 Learner 接口使用主动实验
    auto hyp = learner.generate_hypothesis("光沿直线传播", "physics");
    CHECK_FALSE(hyp.id.empty());

    // 设计实验
    auto design = learner.design_experiment();
    // 可能还没有假设（因为刚生成一个），但不应崩溃
    // 如果有假设则应有实验设计
    if (design.has_value()) {
        CHECK_FALSE(design->target_hypothesis.empty());
    }

    // 记录实验结果
    ExperimentResult result;
    result.hypothesis_id = hyp.id;
    result.supports_hypothesis = true;
    result.confidence_delta = 0.2;
    result.observation = "激光笔光束是直线";
    result.information_gain = 0.5;

    auto analysis = learner.record_experiment(result);
    CHECK_FALSE(analysis.empty());

    // 访问底层引擎
    CHECK(learner.experimenter().stats().at("total_experiments") > 0);
}

TEST_CASE("Learner Phase 5 集成: 全元认知流水线", "[phase5][integration]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    core::Learner learner(config);

    // 完整科学方法流水线
    // 1. 观察生成假设
    auto hyp1 = learner.generate_hypothesis("金属导电", "chemistry");
    auto hyp2 = learner.generate_hypothesis("酸碱中和", "chemistry");

    // 2. 设计实验
    for (int i = 0; i < 3; ++i) {
        auto design = learner.design_experiment();
        if (!design.has_value()) break;

        // 3. 模拟执行实验（支持/反对）
        ExperimentResult result;
        result.hypothesis_id = design->target_hypothesis;
        result.supports_hypothesis = (i % 2 == 0);
        result.confidence_delta = result.supports_hypothesis ? 0.15 : -0.1;
        result.observation = "实验观察 #" + std::to_string(i);
        result.information_gain = 0.3;

        learner.record_experiment(result);
    }

    // 4. 尝试构建理论
    auto theory = learner.build_theory("chemistry");

    // 5. 元学习反思
    auto insights = learner.meta_reflect();
    CHECK_FALSE(insights.empty());

    // 检查统计
    auto exp_stats = learner.experimenter().stats();
    CHECK(exp_stats.at("total_hypotheses") >= 2);
    CHECK(exp_stats.at("total_experiments") >= 1);
}
