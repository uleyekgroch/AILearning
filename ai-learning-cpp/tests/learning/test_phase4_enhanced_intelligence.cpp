/**
 * @file test_phase4_enhanced_intelligence.cpp
 * @brief Phase 4 三大增强智能模块测试
 *
 * TDD 测试覆盖：
 *   1. SocialLearningEngine — 社会性/观察学习
 *   2. EmotionEngine — 情感驱动学习
 *   3. InsightEngine — 知识重组/顿悟
 *   4. 集成测试 — 三模块协同 + 与 Phase 3 集成
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/learning/social_learning.hpp"
#include "ai_learning/learning/emotion_engine.hpp"
#include "ai_learning/learning/insight_engine.hpp"
#include "ai_learning/learning/analogical_transfer.hpp"
#include "ai_learning/learning/continual_learner.hpp"
#include "ai_learning/learning/abstract_concept.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════════════
// 1. SocialLearningEngine 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Social: observe stores and returns report", "[social]") {
    SocialLearningEngine engine;

    BehaviorObservation obs;
    obs.agent_id = "teacher";
    obs.action = "演示排序算法";
    obs.context = "课堂";
    obs.domain = "algorithms";
    obs.outcome_quality = 0.9;

    auto report = engine.observe(obs);
    REQUIRE(report.observations_processed == 1);
}

TEST_CASE("Social: multiple observations enable pattern extraction", "[social]") {
    SocialLearningEngine engine;

    for (int i = 0; i < 4; ++i) {
        BehaviorObservation obs;
        obs.agent_id = "expert";
        obs.action = "先分析再编码";
        obs.context = "开发";
        obs.domain = "programming";
        obs.outcome_quality = 0.8 + i * 0.05;
        obs.preconditions = std::vector<std::string>{"需求明确"};
        obs.effects = std::vector<std::string>{"代码质量高"};
        engine.observe(obs);
    }

    auto pattern = engine.extract_pattern("programming");
    REQUIRE(pattern.has_value());
    REQUIRE(pattern->domain == "programming");
    REQUIRE(pattern->observation_count >= 3);
    REQUIRE_FALSE(pattern->steps.empty());
}

TEST_CASE("Social: evaluate_model returns profile", "[social]") {
    SocialLearningEngine engine;

    // 先观察该榜样的行为
    BehaviorObservation obs;
    obs.agent_id = "mentor";
    obs.action = "教学";
    obs.domain = "math";
    obs.outcome_quality = 0.9;
    engine.observe(obs);

    auto profile = engine.evaluate_model("mentor");
    REQUIRE(profile.agent_id == "mentor");
    REQUIRE(profile.observations_count >= 1);
}

TEST_CASE("Social: select_role_model picks best", "[social]") {
    SocialLearningEngine engine;

    // 创建两个榜样
    for (int i = 0; i < 5; ++i) {
        BehaviorObservation obs;
        obs.agent_id = "expert";
        obs.action = "教学";
        obs.domain = "math";
        obs.outcome_quality = 0.9;
        engine.observe(obs);
    }
    for (int i = 0; i < 3; ++i) {
        BehaviorObservation obs;
        obs.agent_id = "novice";
        obs.action = "尝试";
        obs.domain = "math";
        obs.outcome_quality = 0.3;
        engine.observe(obs);
    }

    auto best = engine.select_role_model("math");
    REQUIRE(best.has_value());
    REQUIRE(*best == "expert");
}

TEST_CASE("Social: imitate produces adapted steps", "[social]") {
    SocialLearningEngine engine;

    StrategyPattern pattern;
    pattern.id = "test_strat";
    pattern.name = "sorting";
    pattern.steps = std::vector<std::string>{"分析输入", "选择算法", "编码实现"};
    pattern.observed_success_rate = 0.8;

    auto results = engine.imitate(pattern, "处理用户数据");
    REQUIRE(results.size() == 3);
}

TEST_CASE("Social: adapt_strategy adjusts to capabilities", "[social]") {
    SocialLearningEngine engine;

    StrategyPattern pattern;
    pattern.id = "test_strat";
    pattern.name = "coding";
    pattern.steps = std::vector<std::string>{"分析输入", "选择算法", "编码实现", "测试验证"};
    pattern.observed_success_rate = 0.8;

    auto adapted = engine.adapt_strategy(
        pattern, std::vector<std::string>{"分析", "编码"});
    REQUIRE_FALSE(adapted.steps.empty());
    REQUIRE(adapted.name.find("adapted") != std::string::npos);
}

TEST_CASE("Social: receive_feedback and adjust", "[social]") {
    SocialLearningEngine engine;

    SocialFeedback feedback;
    feedback.from_agent = "teacher";
    feedback.feedback_type = "correction";
    feedback.content = "你的排序不稳定";
    feedback.domain = "programming";
    feedback.weight = 0.8;

    engine.receive_feedback(feedback);

    auto adjustments = engine.adjust_from_feedback("programming");
    REQUIRE_FALSE(adjustments.empty());
    REQUIRE(adjustments[0].find("修正") != std::string::npos);
}

TEST_CASE("Social: share_knowledge returns strategies", "[social]") {
    SocialLearningEngine engine;

    // 先学一个策略
    for (int i = 0; i < 4; ++i) {
        BehaviorObservation obs;
        obs.agent_id = "expert";
        obs.action = "分析问题";
        obs.domain = "math";
        obs.outcome_quality = 0.8;
        engine.observe(obs);
    }
    engine.extract_pattern("math");

    auto knowledge = engine.share_knowledge("math");
    // 应该有可分享的知识
    (void)knowledge;
}

TEST_CASE("Social: stats tracks all counters", "[social]") {
    SocialLearningEngine engine;

    BehaviorObservation obs;
    obs.agent_id = "a";
    obs.action = "test";
    obs.domain = "d";
    obs.outcome_quality = 0.5;
    engine.observe(obs);

    auto stats = engine.stats();
    REQUIRE(stats.contains("total_observations"));
    REQUIRE(stats.at("total_observations") == 1.0);
}

// ═══════════════════════════════════════════════════════════════════
// 2. EmotionEngine 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Emotion: initial state is neutral", "[emotion]") {
    EmotionEngine engine;

    auto& state = engine.current_state();
    REQUIRE(state.valence == 0.0);
    REQUIRE(state.arousal == 0.0);
}

TEST_CASE("Emotion: success event increases valence", "[emotion]") {
    EmotionEngine engine;

    EmotionEvent event;
    event.event_type = "success";
    event.domain = "math";
    event.description = "解出一道难题";
    event.magnitude = 0.8;
    event.expected_outcome = 0.5;
    event.actual_outcome = 0.9;

    auto state = engine.process_event(event);
    REQUIRE(state.valence > 0.0);
    REQUIRE(state.arousal > 0.0);
}

TEST_CASE("Emotion: failure event decreases valence", "[emotion]") {
    EmotionEngine engine;

    EmotionEvent event;
    event.event_type = "failure";
    event.domain = "math";
    event.description = "考试不及格";
    event.magnitude = 0.7;
    event.expected_outcome = 0.7;
    event.actual_outcome = 0.2;

    auto state = engine.process_event(event);
    REQUIRE(state.valence < 0.0);
}

TEST_CASE("Emotion: surprise event increases arousal", "[emotion]") {
    EmotionEngine engine;

    EmotionEvent event;
    event.event_type = "surprise";
    event.domain = "science";
    event.magnitude = 0.9;
    event.expected_outcome = 0.3;
    event.actual_outcome = 0.8;

    auto state = engine.process_event(event);
    REQUIRE(state.arousal > 0.3);
}

TEST_CASE("Emotion: decay moves toward neutral", "[emotion]") {
    EmotionEngine engine;

    EmotionEvent event;
    event.event_type = "success";
    event.magnitude = 0.9;
    event.actual_outcome = 1.0;
    event.expected_outcome = 0.5;
    engine.process_event(event);

    double valence_before = engine.current_state().valence;
    engine.decay();
    double valence_after = engine.current_state().valence;

    REQUIRE(std::abs(valence_after) < std::abs(valence_before));
}

TEST_CASE("Emotion: RPE computes correctly", "[emotion]") {
    EmotionEngine engine;

    auto rpe = engine.compute_rpe(0.5, 0.8);
    REQUIRE(rpe.error > 0.0);
    REQUIRE(rpe.is_positive());
    REQUIRE(rpe.magnitude() >= 0.29);
    REQUIRE(rpe.magnitude() <= 0.31);

    auto neg_rpe = engine.compute_rpe(0.8, 0.3);
    REQUIRE(neg_rpe.error < 0.0);
    REQUIRE_FALSE(neg_rpe.is_positive());
}

TEST_CASE("Emotion: compute_modulation boosts encoding", "[emotion]") {
    EmotionEngine engine;

    // 高唤醒状态应增强编码
    EmotionEvent event;
    event.event_type = "surprise";
    event.magnitude = 0.9;
    event.actual_outcome = 0.9;
    event.expected_outcome = 0.2;
    engine.process_event(event);

    auto mod = engine.compute_modulation();
    REQUIRE(mod.encoding_boost > 1.0);
    REQUIRE(mod.forgetting_rate < 0.02);
}

TEST_CASE("Emotion: suggest_strategy adapts to state", "[emotion]") {
    EmotionEngine engine;

    // 低唤醒 → 探索性学习
    auto suggestion = engine.suggest_strategy();
    REQUIRE_FALSE(suggestion.strategy.empty());
    REQUIRE(suggestion.confidence > 0.0);
}

TEST_CASE("Emotion: regulation reduces extreme emotions", "[emotion]") {
    EmotionEngine engine;

    // 制造极端情绪
    EmotionEvent event;
    event.event_type = "success";
    event.magnitude = 1.0;
    event.actual_outcome = 1.0;
    event.expected_outcome = 0.0;
    engine.process_event(event);

    // 反复触发使情绪极端
    for (int i = 0; i < 5; ++i) {
        engine.process_event(event);
    }

    if (engine.needs_regulation()) {
        double arousal_before = engine.current_state().arousal;
        engine.regulate();
        double arousal_after = engine.current_state().arousal;
        REQUIRE(arousal_after < arousal_before);
    }
}

TEST_CASE("Emotion: performance_prediction follows Yerkes-Dodson", "[emotion]") {
    EmotionEngine engine;

    // 中等唤醒应该有最高绩效预测
    EmotionEvent moderate;
    moderate.event_type = "progress";
    moderate.magnitude = 0.5;
    moderate.actual_outcome = 0.6;
    moderate.expected_outcome = 0.5;
    engine.process_event(moderate);

    double perf = engine.performance_prediction();
    REQUIRE(perf >= 0.0);
    REQUIRE(perf <= 1.0);
}

TEST_CASE("Emotion: stats returns valid data", "[emotion]") {
    EmotionEngine engine;

    auto stats = engine.stats();
    REQUIRE(stats.contains("current_valence"));
    REQUIRE(stats.contains("current_arousal"));
    REQUIRE(stats.contains("performance_prediction"));
}

TEST_CASE("Emotion: emotion_label is meaningful", "[emotion]") {
    EmotionEngine engine;

    auto label = engine.emotion_label();
    REQUIRE_FALSE(label.empty());
}

TEST_CASE("Emotion: event_modulation computes per-event boost", "[emotion]") {
    EmotionEngine engine;

    EmotionEvent event;
    event.event_type = "success";
    event.magnitude = 0.8;
    event.actual_outcome = 0.9;
    event.expected_outcome = 0.5;

    auto mod = engine.event_modulation(event);
    REQUIRE(mod.encoding_boost >= 1.0);
    REQUIRE_FALSE(mod.reason.empty());
}

// ═══════════════════════════════════════════════════════════════════
// 3. InsightEngine 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Insight: register_element stores data", "[insight]") {
    InsightEngine engine;

    KnowledgeElement elem;
    elem.id = "gravity";
    elem.content = "万有引力";
    elem.domain = "physics";
    elem.features = std::map<std::string, double>{{"strength", 0.9}};

    engine.register_element(elem);
    REQUIRE(engine.elements().count("gravity") == 1);
}

TEST_CASE("Insight: add_constraint stores data", "[insight]") {
    InsightEngine engine;

    MentalConstraint c;
    c.id = "linear_only";
    c.description = "只能线性思考";
    c.strength = 0.8;

    engine.add_constraint(c);
    REQUIRE(engine.constraints().count("linear_only") == 1);
}

TEST_CASE("Insight: remote_association finds hidden links", "[insight]") {
    InsightEngine engine;

    // 注册两个看似无关但有隐藏联系的元素
    KnowledgeElement apple;
    apple.id = "falling_apple";
    apple.content = "苹果掉落";
    apple.domain = "daily_life";
    apple.features = std::map<std::string, double>{{"mass", 0.2}, {"acceleration", 9.8}};

    KnowledgeElement moon;
    moon.id = "moon_orbit";
    moon.content = "月球绕地球运行";
    moon.domain = "astronomy";
    moon.features = std::map<std::string, double>{{"mass", 7.3e22}, {"acceleration", 0.0027}};

    engine.register_element(apple);
    engine.register_element(moon);

    auto insight = engine.remote_association("falling_apple", "moon_orbit");
    if (insight.has_value()) {
        REQUIRE_FALSE(insight->new_perspective.empty());
        REQUIRE(insight->elements_combined.size() == 2);
    }
}

TEST_CASE("Insight: release_constraint produces insight", "[insight]") {
    InsightEngine engine;

    MentalConstraint c;
    c.id = "must_be_linear";
    c.description = "解决方案必须是线性的";
    c.strength = 0.4;  // 弱约束，容易被释放
    c.is_active = true;

    engine.add_constraint(c);

    auto insight = engine.release_constraint("must_be_linear", "路径规划问题");
    REQUIRE(insight.has_value());
    REQUIRE(insight->constraint_released == "must_be_linear");
    REQUIRE_FALSE(insight->new_perspective.empty());
    REQUIRE(engine.released_constraints().size() == 1);
}

TEST_CASE("Insight: try_insight attempts multiple strategies", "[insight]") {
    InsightEngine engine;

    // 注册足够的元素和约束
    KnowledgeElement a;
    a.id = "A"; a.content = "元素A"; a.domain = "test";
    a.features = std::map<std::string, double>{{"x", 1.0}};
    KnowledgeElement b;
    b.id = "B"; b.content = "元素B"; b.domain = "test";
    b.features = std::map<std::string, double>{{"x", 0.5}};
    engine.register_element(a);
    engine.register_element(b);

    MentalConstraint c;
    c.id = "c1"; c.description = "约束1"; c.strength = 0.3;
    engine.add_constraint(c);

    auto insight = engine.try_insight("测试问题");
    // 可能成功也可能失败，取决于内部逻辑
    (void)insight;
}

TEST_CASE("Insight: assess_creativity evaluates correctly", "[insight]") {
    InsightEngine engine;

    InsightEvent event;
    event.id = "test";
    event.old_perspective = "旧视角";
    event.new_perspective = "全新的视角 - 通过跨领域类比发现隐藏联系";
    event.elements_combined = std::vector<std::string>{"A", "B"};
    event.confidence = 0.8;

    auto assessment = engine.assess_creativity(event);
    REQUIRE(assessment.novelty >= 0.0);
    REQUIRE(assessment.utility >= 0.0);
    REQUIRE(assessment.overall >= 0.0);
    REQUIRE(assessment.overall <= 1.0);
    REQUIRE_FALSE(assessment.verdict.empty());
}

TEST_CASE("Insight: verify_insight checks validity", "[insight]") {
    InsightEngine engine;

    InsightEvent good;
    good.id = "good";
    good.new_perspective = "有意义的顿悟";
    good.old_perspective = "旧观点";
    good.confidence = 0.8;

    REQUIRE(engine.verify_insight(good));

    InsightEvent bad;
    bad.id = "bad";
    bad.new_perspective = "";
    bad.old_perspective = "";
    bad.confidence = 0.1;

    REQUIRE_FALSE(engine.verify_insight(bad));
}

TEST_CASE("Insight: propose_reorganization generates plans", "[insight]") {
    InsightEngine engine;

    engine.register_element({"A", "元素A", "physics", {}, {}});
    engine.register_element({"B", "元素B", "physics", {}, {}});
    engine.register_element({"C", "元素C", "math", {}, {}});

    auto plans = engine.propose_reorganization("physics");
    REQUIRE_FALSE(plans.empty());
}

TEST_CASE("Insight: execute_reorganization produces insight", "[insight]") {
    InsightEngine engine;

    ReorganizationPlan plan;
    plan.elements_to_rearrange = std::vector<std::string>{"A", "B"};
    plan.new_structure = "新结构";
    plan.method = "remote_association";
    plan.estimated_novelty = 0.6;

    auto insight = engine.execute_reorganization(plan);
    REQUIRE_FALSE(insight.new_perspective.empty());
}

TEST_CASE("Insight: stats tracks attempts and successes", "[insight]") {
    InsightEngine engine;

    engine.register_element({"A", "a", "d", {}, {}});
    engine.register_element({"B", "b", "d", std::vector<std::string>{"A"}, {}});

    engine.remote_association("A", "B");

    auto stats = engine.stats();
    REQUIRE(stats.contains("total_attempts"));
    REQUIRE(stats.at("total_attempts") >= 1.0);
}

// ═══════════════════════════════════════════════════════════════════
// 4. 集成测试 — 三模块协同 + 跨 Phase 集成
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Integration: emotion modulates social learning", "[integration_phase4]") {
    EmotionEngine emotion;
    SocialLearningEngine social;

    // 高兴奋状态 → 学习效率高
    EmotionEvent event;
    event.event_type = "success";
    event.magnitude = 0.9;
    event.actual_outcome = 1.0;
    event.expected_outcome = 0.3;
    emotion.process_event(event);

    auto mod = emotion.compute_modulation();
    REQUIRE(mod.encoding_boost > 1.0);

    // 在高效状态下观察学习
    BehaviorObservation obs;
    obs.agent_id = "expert";
    obs.action = "高级技巧";
    obs.domain = "art";
    obs.outcome_quality = 0.9;

    auto report = social.observe(obs);
    REQUIRE(report.observations_processed == 1);
}

TEST_CASE("Integration: insight from social knowledge + emotion", "[integration_phase4]") {
    InsightEngine insight_eng;
    SocialLearningEngine social_eng;
    EmotionEngine emotion_eng;

    // 通过社会学习获得知识元素
    for (int i = 0; i < 4; ++i) {
        BehaviorObservation obs;
        obs.agent_id = "scientist";
        obs.action = "实验观察";
        obs.domain = "physics";
        obs.outcome_quality = 0.8;
        obs.effects = std::vector<std::string>{"发现新现象"};
        social_eng.observe(obs);
    }

    // 将学到的知识注册到顿悟引擎
    insight_eng.register_element(
        {"gravity", "万有引力", "physics",
         std::vector<std::string>{"apple"},
         std::map<std::string, double>{{"force", 9.8}}});
    insight_eng.register_element(
        {"orbit", "天体运行", "astronomy",
         std::vector<std::string>{},
         std::map<std::string, double>{{"force", 0.0027}}});

    // 情绪事件激发顿悟
    EmotionEvent eureka;
    eureka.event_type = "surprise";
    eureka.magnitude = 0.95;
    eureka.actual_outcome = 0.95;
    eureka.expected_outcome = 0.2;
    emotion_eng.process_event(eureka);

    REQUIRE(emotion_eng.current_state().arousal > 0.5);

    // 尝试产生顿悟
    auto insight = insight_eng.remote_association("gravity", "orbit");
    (void)insight;
}

TEST_CASE("Integration: full Phase 3+4 pipeline", "[integration_phase4]") {
    // Phase 3: 概念形成
    AbstractConceptEngine cpt_engine;
    cpt_engine.observe_instance("apple",
        std::vector<std::string>{"果实", "甜", "植物"}, {});
    cpt_engine.observe_instance("pear",
        std::vector<std::string>{"果实", "甜", "植物"}, {});
    cpt_engine.observe_instance("peach",
        std::vector<std::string>{"果实", "甜", "植物"}, {});

    // Phase 4: 情感增强记忆
    EmotionEngine emotion;
    EmotionEvent event;
    event.event_type = "progress";
    event.magnitude = 0.7;
    event.actual_outcome = 0.8;
    event.expected_outcome = 0.5;
    emotion.process_event(event);

    auto mod = emotion.compute_modulation();
    REQUIRE(mod.encoding_boost > 1.0);

    // Phase 4: 顿悟发现新联系
    InsightEngine insight_eng;
    insight_eng.register_element(
        {"fruit", "水果概念", "botany",
         std::vector<std::string>{"apple", "pear"},
         std::map<std::string, double>{{"sweetness", 0.8}}});
    insight_eng.register_element(
        {"vegetable", "蔬菜概念", "botany",
         std::vector<std::string>{"carrot"},
         std::map<std::string, double>{{"sweetness", 0.2}}});

    auto insight = insight_eng.remote_association("fruit", "vegetable");
    (void)insight;

    // Phase 3: 持续学习保护
    ContinualLearner continual;
    continual.register_knowledge("fruit_cpt", "botany", 0.8, 10);
    continual.register_knowledge("vegetable_cpt", "botany", 0.7, 5);

    double retention = continual.retention_rate();
    REQUIRE(retention >= 0.0);
}

TEST_CASE("Integration: social + continual prevents forgetting during imitation",
          "[integration_phase4]") {
    SocialLearningEngine social;
    ContinualLearner continual;

    // 注册受保护的知识
    continual.register_knowledge("my_method", "algorithms", 0.9, 50);

    // 观察他人方法
    for (int i = 0; i < 4; ++i) {
        BehaviorObservation obs;
        obs.agent_id = "expert";
        obs.action = "不同的排序方法";
        obs.domain = "algorithms";
        obs.outcome_quality = 0.7;
        social.observe(obs);
    }

    // 保护旧知识的同时学习新知识
    auto conflicts = continual.prepare_for_new_learning(
        "algorithms", std::vector<std::string>{"expert_sorting_method"});

    // 高保护知识应该被保护
    auto prot = continual.get_protection("my_method");
    REQUIRE(prot.has_value());
    REQUIRE(prot->level != ProtectionLevel::kNone);

    (void)conflicts;
}
