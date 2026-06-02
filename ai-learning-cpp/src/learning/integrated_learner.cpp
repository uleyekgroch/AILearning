/**
 * @file integrated_learner.cpp
 * @brief Phase 6 深度整合实现
 */

#include "ai_learning/learning/integrated_learner.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace ai_learning::learning {

IntegratedLearner::IntegratedLearner(
    MetaLearner& meta,
    ActiveExperimenter& experimenter,
    EmotionEngine& emotion,
    SocialLearningEngine& social,
    AnalogicalTransferEngine& analogy,
    InsightEngine& insight,
    IntrinsicMotivationEngine& motivation,
    ContinualLearner& continual,
    AbstractConceptEngine& concept_eng)
    : meta_(meta),
      experimenter_(experimenter),
      emotion_(emotion),
      social_(social),
      analogy_(analogy),
      insight_(insight),
      motivation_(motivation),
      continual_(continual),
      concept_eng_(concept_eng) {}

// ── 整合 1：全流水线闭环 ──────────────────────────────────────

auto IntegratedLearner::run_full_pipeline(
    const std::string& observation_text,
    const std::string& domain)
    -> IntegratedPipelineReport
{
    pipeline_runs_++;
    IntegratedPipelineReport report;

    // Step 1: 观察（ActiveExperimenter 生成假设）
    report.observe_step.step_name = "observe";
    auto hyp = experimenter_.generate_hypothesis(observation_text, domain);
    if (!hyp.id.empty()) {
        report.observe_step.success = true;
        report.observe_step.summary = "生成假设: " + hyp.statement;
    }

    // Step 2: 学习（情感调制）
    report.learn_step.step_name = "learn";
    auto modulation = emotion_.compute_modulation();
    EmotionEvent learn_event;
    learn_event.event_type = "progress";
    learn_event.domain = domain;
    learn_event.description = "学习观察: " + observation_text;
    learn_event.magnitude = 0.5;
    auto emo_state = emotion_.process_event(learn_event);
    report.learn_step.success = true;
    report.learn_step.confidence_delta = modulation.encoding_boost - 1.0;
    report.learn_step.summary = "情感调制: encoding_boost=" +
        std::to_string(modulation.encoding_boost);

    // Step 3: 类比（尝试从已有知识迁移）
    report.analogize_step.step_name = "analogize";
    auto domain_hyps = experimenter_.hypotheses_in_domain(domain);
    if (domain_hyps.size() >= 2) {
        // 有足够假设，尝试类比
        report.analogize_step.success = true;
        report.analogize_step.summary = "领域内有 " +
            std::to_string(domain_hyps.size()) + " 个假设可供类比";
    } else {
        report.analogize_step.summary = "假设不足，跳过类比";
    }

    // Step 4: 实验（设计并执行）
    report.experiment_step.step_name = "experiment";
    auto design = experimenter_.auto_design_experiment();
    if (design.has_value()) {
        // 模拟实验结果（支持假设）
        ExperimentResult result;
        result.hypothesis_id = design->target_hypothesis;
        result.supports_hypothesis = true;
        result.confidence_delta = 0.15;
        result.observation = "验证实验结果";
        result.information_gain = design->expected_information_gain;
        result.surprise = 0.1;
        auto analysis = experimenter_.record_result(result);
        report.experiment_step.success = true;
        report.experiment_step.summary = analysis;
        report.experiment_step.confidence_delta = result.confidence_delta;
    } else {
        report.experiment_step.summary = "无需实验";
    }

    // Step 5: 反思（元学习自我评估）
    report.reflect_step.step_name = "reflect";
    auto reflections = meta_.reflect();
    if (!reflections.empty()) {
        report.reflect_step.success = true;
        report.reflect_step.summary = reflections[0];
        report.insights = reflections;
    }

    // Step 6: 顿悟（知识重组）
    report.insight_step.step_name = "insight";
    auto insight_event = insight_.try_insight(domain);
    if (insight_event.has_value()) {
        report.insight_step.success = true;
        report.insight_step.confidence_delta = insight_event->surprise_level;
        report.insight_step.summary = "顿悟: " + insight_event->new_perspective;

        // 顿悟触发兴奋情绪
        EmotionEvent excitement;
        excitement.event_type = "surprise";
        excitement.domain = domain;
        excitement.description = "产生顿悟";
        excitement.magnitude = insight_event->surprise_level * 0.8;
        emotion_.process_event(excitement);
    } else {
        report.insight_step.summary = "未产生顿悟";
    }

    // 尝试构建理论
    auto theory = experimenter_.build_theory(domain);
    if (theory.has_value()) {
        report.theories.push_back(theory->description);
    }

    // 综合指标
    int completed = 0;
    if (report.observe_step.success) completed++;
    if (report.learn_step.success) completed++;
    if (report.analogize_step.success) completed++;
    if (report.experiment_step.success) completed++;
    if (report.reflect_step.success) completed++;
    if (report.insight_step.success) completed++;
    report.steps_completed = completed;
    report.overall_progress = static_cast<double>(completed) / 6.0;

    auto final_state = emotion_.current_state();
    report.emotion_valence = final_state.valence;
    report.dominant_emotion = emotion_.emotion_label();

    // 更新内在动机
    LearningOutcome outcome;
    outcome.topic = domain;
    outcome.progress = report.overall_progress;
    outcome.surprise = report.insight_step.confidence_delta;
    outcome.mastery_improved = report.overall_progress > 0.3;
    outcome.goal_completed = report.overall_progress > 0.8;
    motivation_.update_on_learning(outcome);

    // 记录元学习经验
    LearningExperience meta_exp;
    meta_exp.task.domain = domain;
    meta_exp.task.task_type = "scientific";
    meta_exp.task.difficulty = 0.5;
    meta_exp.improvement = report.overall_progress;
    meta_exp.success = report.overall_progress > 0.3;
    meta_.record_experience(meta_exp);

    return report;
}

// ── 整合 2：元学习驱动策略选择 ────────────────────────────────

auto IntegratedLearner::meta_guided_session(
    const std::vector<std::string>& known_topics,
    const std::map<std::string, double>& mastery_map)
    -> MetaGuidedSessionReport
{
    meta_sessions_++;
    MetaGuidedSessionReport report;

    // Step 1: 动机系统产生学习目标
    report.chosen_goal = motivation_.generate_goal(
        known_topics, 0.5, mastery_map);

    // Step 2: 元学习推荐策略
    TaskDescriptor task;
    task.domain = report.chosen_goal.domain;
    task.task_type = report.chosen_goal.primary_motive ==
        MotiveType::kCuriosity ? "exploration" : "practice";
    task.difficulty = report.chosen_goal.difficulty;
    task.novelty = 1.0 - mastery_map.count(report.chosen_goal.domain)
        ? 0.3 : 0.8;

    report.strategy_rec = meta_.recommend_strategy(task);
    report.learning_rate = meta_.suggest_learning_rate(task);

    // Step 3: 情感调制
    report.emotion_modulation = emotion_.compute_modulation();

    // Step 4: 构造经验并记录
    report.experience.task = task;
    report.experience.strategy_used = report.strategy_rec.recommended_strategy;
    report.experience.improvement = report.strategy_rec.confidence * 0.5;
    report.experience.success = report.strategy_rec.confidence > 0.3;
    report.experience.time_cost = 1.0 / report.learning_rate;

    meta_.record_experience(report.experience);

    // Step 5: 反思
    report.reflections = meta_.reflect();

    // 更新学习率
    meta_.update_learning_rate(report.experience.improvement);

    return report;
}

// ── 整合 3：情感调制全系统 ────────────────────────────────────

auto IntegratedLearner::compute_system_params() const
    -> EmotionModulatedParams
{
    EmotionModulatedParams params;

    auto state = emotion_.current_state();
    auto modulation = emotion_.compute_modulation();

    // 唤醒度调制学习率：适度唤醒最佳（Yerkes-Dodson）
    double optimal_arousal = 0.5;
    double arousal_factor = 1.0 - std::abs(state.arousal - optimal_arousal);
    params.learning_rate = 0.1 * arousal_factor * modulation.encoding_boost;

    // 效价影响记忆编码
    params.encoding_boost = modulation.encoding_boost;

    // 正面效价增加探索，负面效价趋于保守
    params.exploration_tendency = 0.5 + state.valence * 0.3;

    // 唤醒度提高风险容忍（兴奋时更敢冒险）
    params.risk_tolerance = 0.3 + state.arousal * 0.4;

    params.emotion_label = emotion_.emotion_label();

    return params;
}

auto IntegratedLearner::process_emotion_and_modulate(
    const EmotionEvent& event)
    -> EmotionModulatedParams
{
    emotion_modulations_++;
    emotion_.process_event(event);
    return compute_system_params();
}

// ── 整合 4：实验驱动探索 ──────────────────────────────────────

auto IntegratedLearner::experiment_driven_exploration(
    const std::string& domain)
    -> ExperimentDrivenExplorationReport
{
    experiment_explorations_++;
    ExperimentDrivenExplorationReport report;
    report.domain = domain;

    // Step 1: 获取该领域已有假设
    report.hypotheses = experimenter_.hypotheses_in_domain(domain);

    // Step 2: 计算领域不确定性和信息价值
    report.domain_uncertainty = experimenter_.domain_uncertainty(domain);
    double total_info = 0.0;
    for (const auto& hyp : report.hypotheses) {
        total_info += experimenter_.expected_information_gain(hyp.id);
    }
    report.information_value = total_info;

    // Step 3: 设计最优实验
    report.design = experimenter_.auto_design_experiment();

    // Step 4: 将高不确定性领域转化为内在动机目标
    report.suggested_goal.topic = domain;
    report.suggested_goal.domain = domain;
    report.suggested_goal.difficulty = report.domain_uncertainty;
    report.suggested_goal.estimated_value = total_info;
    report.suggested_goal.primary_motive = MotiveType::kCuriosity;
    report.suggested_goal.description = "探索 " + domain +
        " (不确定性=" + std::to_string(report.domain_uncertainty) + ")";

    return report;
}

// ── 整合 5：社会学习加速类比 ──────────────────────────────────

auto IntegratedLearner::social_analogical_transfer(
    const std::string& source_domain,
    const std::string& target_domain,
    const std::vector<ConceptDescriptor>& target_concepts)
    -> SocialAnalogicalReport
{
    social_transfers_++;
    SocialAnalogicalReport report;
    report.source_domain = source_domain;
    report.target_domain = target_domain;

    // Step 1: 从社会学习引擎提取源领域的策略模式
    auto pattern = social_.extract_pattern(source_domain);
    int observations_used = 0;

    // Step 2: 构建源领域概念描述符
    std::vector<ConceptDescriptor> source_concepts;
    if (pattern.has_value()) {
        observations_used++;
        ConceptDescriptor src;
        src.id = pattern->id;
        src.domain = source_domain;
        src.attributes = pattern->steps;
        src.relations = {"learned_via_observation"};
        source_concepts.push_back(src);
    }

    // Step 3: 添加从社会学习积累的领域知识
    auto domain_pattern = social_.extract_pattern(source_domain);
    if (domain_pattern.has_value() &&
        std::find_if(source_concepts.begin(), source_concepts.end(),
            [&](const auto& c) { return c.id == domain_pattern->id; })
        == source_concepts.end()) {
        observations_used++;
        ConceptDescriptor src;
        src.id = domain_pattern->id + "_v2";
        src.domain = source_domain;
        src.attributes = domain_pattern->steps;
        src.relations = {"social_source"};
        source_concepts.push_back(src);
    }

    report.social_observations_used = observations_used;

    // Step 4: 执行类比迁移
    if (!source_concepts.empty() && !target_concepts.empty()) {
        report.transfer_result = analogy_.transfer(
            source_concepts, target_concepts,
            {"社会观察知识"});
        report.transfer_quality = report.transfer_result.transfer_quality;
    } else {
        report.transfer_result.source_domain = source_domain;
        report.transfer_result.target_domain = target_domain;
        report.transfer_quality = 0.0;
    }

    // Step 5: 保护迁移的知识（防遗忘）
    for (const auto& tk : report.transfer_result.transferred_knowledge) {
        continual_.register_knowledge(tk, target_domain, 0.6, 1);
    }

    return report;
}

// ── 统计 ──────────────────────────────────────────────────────

auto IntegratedLearner::stats() const -> std::map<std::string, double>
{
    return {
        {"pipeline_runs", static_cast<double>(pipeline_runs_)},
        {"meta_sessions", static_cast<double>(meta_sessions_)},
        {"emotion_modulations", static_cast<double>(emotion_modulations_)},
        {"experiment_explorations", static_cast<double>(experiment_explorations_)},
        {"social_transfers", static_cast<double>(social_transfers_)},
    };
}

}  // namespace ai_learning::learning
