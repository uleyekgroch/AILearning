/**
 * @file learner_higher_cognition.cpp
 * @brief Learner 高级认知能力 — Phase 3~6 + Bloom 评估
 *
 * 从 learner.cpp 分离，遵守单文件 ≤800 行规范。
 */

#include "ai_learning/core/learner.hpp"

namespace ai_learning::core {

using namespace learning;

// ── Phase 3：高级认知能力 ────────────────────────────────────────

auto Learner::analogical_transfer(
    const std::vector<learning::ConceptDescriptor>& source_concepts,
    const std::vector<learning::ConceptDescriptor>& target_concepts,
    const std::vector<std::string>& source_facts)
    -> learning::TransferResult
{
    auto result = analogy_engine_.transfer(
        source_concepts, target_concepts, source_facts);

    // 将迁移成功的知识注册到持续学习保护
    for (const auto& knowledge : result.transferred_knowledge) {
        continual_.register_knowledge(
            knowledge, result.target_domain, result.transfer_quality, 1);
    }

    return result;
}

void Learner::protect_knowledge(const std::string& knowledge_id,
                                 const std::string& domain,
                                 double confidence,
                                 int usage_count) {
    continual_.register_knowledge(knowledge_id, domain, confidence, usage_count);
}

auto Learner::detect_forgetting() const
    -> std::vector<learning::ForgettingAlert> {
    return continual_.detect_forgetting();
}

auto Learner::form_abstractions(
    const std::string& instance_id,
    const std::vector<std::string>& attributes,
    const std::map<std::string, double>& features,
    const std::vector<std::string>& relations)
    -> learning::ConceptFormationReport
{
    return concept_engine_.observe_instance(
        instance_id, attributes, features, relations);
}

// ── Phase 4：增强智能 ────────────────────────────────────────

auto Learner::observe_behavior(
    const learning::BehaviorObservation& observation)
    -> learning::SocialLearningReport
{
    // 情感调制：高唤醒状态增强社会学习效果
    auto report = social_engine_.observe(observation);

    // 将学到的策略注册到持续学习保护
    for (const auto& strategy : report.learned_strategies) {
        continual_.register_knowledge(
            strategy.id, strategy.domain,
            strategy.observed_success_rate, strategy.observation_count);
    }

    return report;
}

auto Learner::process_emotion(const learning::EmotionEvent& event)
    -> learning::EmotionState
{
    return emotion_engine_.process_event(event);
}

auto Learner::try_insight(const std::string& problem_context)
    -> std::optional<learning::InsightEvent>
{
    auto insight = insight_engine_.try_insight(problem_context);

    // 如果产生顿悟，触发情感事件（兴奋）
    if (insight.has_value()) {
        learning::EmotionEvent event;
        event.event_type = "surprise";
        event.domain = "insight";
        event.description = insight->new_perspective;
        event.magnitude = insight->surprise_level;
        event.actual_outcome = insight->confidence;
        event.expected_outcome = 0.3;
        emotion_engine_.process_event(event);

        // 保护顿悟知识
        continual_.register_knowledge(
            insight->id, "insight",
            insight->confidence, 1);
    }

    return insight;
}

// ── Phase 5：高级元认知 ────────────────────────────────────

auto Learner::meta_recommend(
    const learning::TaskDescriptor& task) const
    -> learning::MetaLearningRecommendation
{
    return meta_learner_.recommend_strategy(task);
}

void Learner::meta_record(
    const learning::LearningExperience& experience)
{
    meta_learner_.record_experience(experience);
}

auto Learner::meta_reflect() const -> std::vector<std::string>
{
    return meta_learner_.reflect();
}

auto Learner::generate_hypothesis(
    const std::string& observation,
    const std::string& domain)
    -> learning::Hypothesis
{
    return experimenter_.generate_hypothesis(observation, domain);
}

auto Learner::design_experiment()
    -> std::optional<learning::ExperimentDesign>
{
    return experimenter_.auto_design_experiment();
}

auto Learner::record_experiment(
    const learning::ExperimentResult& result)
    -> std::string
{
    return experimenter_.record_result(result);
}

auto Learner::build_theory(const std::string& domain)
    -> std::optional<learning::Theory>
{
    return experimenter_.build_theory(domain);
}

// ── Phase 6：深度整合 ────────────────────────────────────

auto Learner::integrated_pipeline(const std::string& observation,
                                   const std::string& domain)
    -> learning::IntegratedPipelineReport
{
    return integrated_.run_full_pipeline(observation, domain);
}

auto Learner::meta_guided_learn(
    const std::vector<std::string>& known_topics,
    const std::map<std::string, double>& mastery_map)
    -> learning::MetaGuidedSessionReport
{
    return integrated_.meta_guided_session(known_topics, mastery_map);
}

auto Learner::emotion_modulated_params() const
    -> learning::EmotionModulatedParams
{
    return integrated_.compute_system_params();
}

auto Learner::experiment_driven_explore(const std::string& domain)
    -> learning::ExperimentDrivenExplorationReport
{
    return integrated_.experiment_driven_exploration(domain);
}

auto Learner::social_accelerated_transfer(
    const std::string& source_domain,
    const std::string& target_domain,
    const std::vector<learning::ConceptDescriptor>& target_concepts)
    -> learning::SocialAnalogicalReport
{
    return integrated_.social_analogical_transfer(
        source_domain, target_domain, target_concepts);
}

// ── Bloom 掌握度评估 ────────────────────────────────────────────

auto Learner::assess(const std::string& domain) const
    -> assessment::AssessmentResult
{
    // If domain specified, find first entity of that type
    if (!domain.empty()) {
        auto entities = kg_.query(domain);
        if (!entities.empty()) {
            return mastery_assessor_.assess(entities.front(), kg_);
        }
    }
    // Otherwise, find the first entity in the KG
    auto ids = kg_.get_all_entity_ids();
    if (ids.empty()) {
        return assessment::AssessmentResult{};
    }
    auto opt = kg_.get_entity(ids.front());
    if (opt.has_value()) {
        return mastery_assessor_.assess(opt->get(), kg_);
    }
    return assessment::AssessmentResult{};
}

auto Learner::assess_domain(const std::string& domain) const
    -> assessment::DomainReport
{
    return mastery_assessor_.assess_domain(domain, kg_);
}

auto Learner::assess_all() const
    -> std::map<std::string, assessment::DomainReport>
{
    return mastery_assessor_.assess_all_domains(kg_);
}

auto Learner::assess_proficiency(const std::string& entity_type) const
    -> assessment::ProficiencyReport
{
    return proficiency_tester_.assess(entity_type, kg_);
}

}  // namespace ai_learning::core
