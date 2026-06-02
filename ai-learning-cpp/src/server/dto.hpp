/**
 * @file dto.hpp
 * @brief 数据传输对象 — JSON 序列化/反序列化工具
 *
 * 集中定义所有 API 请求/响应的 JSON 映射。
 * 使用 nlohmann/json 的 to_json/from_json 自定义序列化。
 */
#pragma once

#include "ai_learning/core/types.hpp"
#include "ai_learning/learning/knowledge_extractor.hpp"
#include "ai_learning/learning/intrinsic_motivation.hpp"
#include "ai_learning/learning/autonomous_learning_loop.hpp"
#include "ai_learning/learning/development_milestones.hpp"
#include "ai_learning/reasoning/unified_engine.hpp"
#include "ai_learning/domain/memory/i_memory.hpp"
#include "ai_learning/learning/analogical_transfer.hpp"
#include "ai_learning/learning/continual_learner.hpp"
#include "ai_learning/learning/abstract_concept.hpp"
#include "ai_learning/learning/social_learning.hpp"
#include "ai_learning/learning/emotion_engine.hpp"
#include "ai_learning/learning/insight_engine.hpp"
#include "ai_learning/learning/meta_learner.hpp"
#include "ai_learning/learning/active_experimenter.hpp"
#include "ai_learning/learning/integrated_learner.hpp"

#include <nlohmann/json.hpp>

namespace ai_learning::server::dto {

using json = nlohmann::json;

// ── 知识提取类型 → JSON ──────────────────────────────────────────

inline auto to_json_triple(const learning::Triple& t) -> json {
    return json{
        {"subject", t.subject},
        {"relation", t.relation},
        {"object", t.object},
        {"confidence", t.confidence}
    };
}

inline auto to_json_causal_link(const learning::CausalLink& cl) -> json {
    return json{
        {"cause", cl.cause},
        {"effect", cl.effect}
    };
}

inline auto to_json_numerical_fact(const learning::NumericalFact& nf) -> json {
    return json{
        {"attribute", nf.attribute},
        {"value", nf.value},
        {"unit", nf.unit}
    };
}

inline auto to_json_text_learn_result(const learning::TextLearnResult& r) -> json {
    json triples_arr = json::array();
    for (const auto& t : r.triples) {
        triples_arr.push_back(to_json_triple(t));
    }
    json causal_arr = json::array();
    for (const auto& cl : r.causal_links) {
        causal_arr.push_back(to_json_causal_link(cl));
    }
    json num_arr = json::array();
    for (const auto& nf : r.numerical_facts) {
        num_arr.push_back(to_json_numerical_fact(nf));
    }
    return json{
        {"entities", r.entities},
        {"triples", triples_arr},
        {"causal_links", causal_arr},
        {"numerical_facts", num_arr},
        {"verification_passed", r.verification_passed},
        {"verification_score", r.verification_score}
    };
}

// ── 推理结果 → JSON ──────────────────────────────────────────────

inline auto to_json_reasoning_result(const reasoning::ReasoningResult& r) -> json {
    return json{
        {"content", r.content},
        {"confidence", r.confidence},
        {"method", r.method},
        {"evidence", r.evidence},
        {"reasoning_chain", r.reasoning_chain}
    };
}

// ── 记忆条目 → JSON ──────────────────────────────────────────────

inline auto to_json_memory_item(const domain::MemoryItem& m) -> json {
    return json{
        {"memory_id", m.memory_id},
        {"metadata", m.metadata},
        {"strength", m.strength},
        {"importance", m.importance},
        {"created_at", m.created_at},
        {"last_accessed", m.last_accessed}
    };
}

// ── 自主学习循环报告 → JSON ──────────────────────────────────────

inline auto to_json_learning_goal(const learning::LearningGoal& g) -> json {
    return json{
        {"topic", g.topic},
        {"domain", g.domain},
        {"difficulty", g.difficulty},
        {"estimated_value", g.estimated_value},
        {"primary_motive", learning::IntrinsicMotivationEngine::motive_name(g.primary_motive)},
        {"description", g.description}
    };
}

inline auto to_json_step_result(const learning::LearningStepResult& s) -> json {
    return json{
        {"iteration", s.iteration},
        {"goal", to_json_learning_goal(s.goal)},
        {"progress", s.progress},
        {"motivation_before", s.motivation_before},
        {"motivation_after", s.motivation_after},
        {"learned", s.learned},
        {"reflection", s.reflection},
        {"completed", s.completed}
    };
}

inline auto to_json_loop_report(const learning::AutonomousLoopReport& r) -> json {
    json history_arr = json::array();
    for (const auto& h : r.history) {
        history_arr.push_back(to_json_step_result(h));
    }
    return json{
        {"total_iterations", r.total_iterations},
        {"goals_attempted", r.goals_attempted},
        {"goals_completed", r.goals_completed},
        {"total_progress", r.total_progress},
        {"avg_motivation", r.avg_motivation},
        {"elapsed_ms", r.elapsed_ms},
        {"history", history_arr}
    };
}

// ── 进度快照 → JSON ──────────────────────────────────────────────

inline auto to_json_progress_snapshot(const learning::ProgressSnapshot& p) -> json {
    return json{
        {"iteration", p.iteration},
        {"total_progress", p.total_progress},
        {"milestones_achieved", p.milestones_achieved},
        {"milestones_total", p.milestones_total},
        {"current_level", p.current_level},
        {"domain_progress", p.domain_progress}
    };
}

// ── JSON 解析工具 ────────────────────────────────────────────────

/// 从 JSON 数组解析 vector<float>
inline auto parse_float_vector(const json& arr) -> std::vector<float> {
    std::vector<float> result;
    if (!arr.is_array()) return result;
    result.reserve(arr.size());
    for (const auto& v : arr) {
        result.push_back(v.get<float>());
    }
    return result;
}

/// 从 JSON 对象解析 map<string, vector<float>>
inline auto parse_raw_input(const json& obj)
    -> std::map<std::string, std::vector<float>> {
    std::map<std::string, std::vector<float>> result;
    if (!obj.is_object()) return result;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        result[it.key()] = parse_float_vector(it.value());
    }
    return result;
}

/// 从 JSON 解析 MemoryItem 的 metadata
inline auto parse_metadata(const json& obj)
    -> std::map<std::string, std::string> {
    std::map<std::string, std::string> result;
    if (!obj.is_object()) return result;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (it.value().is_string()) {
            result[it.key()] = it.value().get<std::string>();
        }
    }
    return result;
}

/// 构造统一错误响应
inline auto make_error_response(const std::string& message) -> json {
    return json{{"error", message}};
}

// ── Phase 3：类比迁移 → JSON ──────────────────────────────────────────

inline auto to_json_concept_descriptor(const learning::ConceptDescriptor& cd) -> json {
    return json{
        {"id", cd.id},
        {"domain", cd.domain},
        {"attributes", cd.attributes},
        {"relations", cd.relations},
        {"features", cd.features}
    };
}

inline auto parse_concept_descriptor(const json& j) -> learning::ConceptDescriptor {
    learning::ConceptDescriptor cd;
    cd.id = j.value("id", "");
    cd.domain = j.value("domain", "");
    if (j.contains("attributes") && j["attributes"].is_array()) {
        cd.attributes = j["attributes"].get<std::vector<std::string>>();
    }
    if (j.contains("relations") && j["relations"].is_array()) {
        cd.relations = j["relations"].get<std::vector<std::string>>();
    }
    if (j.contains("features") && j["features"].is_object()) {
        cd.features = j["features"].get<std::map<std::string, double>>();
    }
    return cd;
}

inline auto to_json_structure_mapping(const learning::StructureMapping& sm) -> json {
    return json{
        {"source_concept", sm.source_concept},
        {"target_concept", sm.target_concept},
        {"attribute_map", sm.attribute_map},
        {"relation_map", sm.relation_map},
        {"alignment_score", sm.alignment_score},
        {"surface_similarity", sm.surface_similarity},
        {"relational_depth", sm.relational_depth}
    };
}

inline auto to_json_transfer_result(const learning::TransferResult& r) -> json {
    json mappings_arr = json::array();
    for (const auto& m : r.mappings) {
        mappings_arr.push_back(to_json_structure_mapping(m));
    }
    return json{
        {"source_domain", r.source_domain},
        {"target_domain", r.target_domain},
        {"mappings", mappings_arr},
        {"transferred_knowledge", r.transferred_knowledge},
        {"failed_transfers", r.failed_transfers},
        {"transfer_quality", r.transfer_quality},
        {"reasoning", r.reasoning}
    };
}

// ── Phase 3：持续学习 → JSON ──────────────────────────────────────────

inline auto to_json_forgetting_alert(const learning::ForgettingAlert& fa) -> json {
    return json{
        {"knowledge_id", fa.knowledge_id},
        {"original_confidence", fa.original_confidence},
        {"current_confidence", fa.current_confidence},
        {"degradation", fa.degradation},
        {"recommendation", fa.recommendation}
    };
}

// ── Phase 3：抽象概念 → JSON ──────────────────────────────────────────

inline auto to_json_concept_formation_report(const learning::ConceptFormationReport& r) -> json {
    json concepts_arr = json::array();
    for (const auto& c : r.new_concepts) {
        concepts_arr.push_back(json{
            {"id", c.id},
            {"name", c.name},
            {"core_attributes", c.core_attributes},
            {"variable_attributes", c.variable_attributes},
            {"prototype_features", c.prototype_features},
            {"source_instances", c.source_instances},
            {"strength", c.strength},
            {"parent_concept", c.parent_concept},
            {"child_concepts", c.child_concepts}
        });
    }
    json rules_arr = json::array();
    for (const auto& rule : r.new_rules) {
        rules_arr.push_back(json{
            {"id", rule.id},
            {"concrete_pattern", rule.concrete_pattern},
            {"abstract_pattern", rule.abstract_pattern},
            {"variable_part", rule.variable_part},
            {"confidence", rule.confidence},
            {"support_count", rule.support_count},
            {"counter_examples", rule.counter_examples}
        });
    }
    json diffs_arr = json::array();
    for (const auto& d : r.differentiations) {
        diffs_arr.push_back(json{
            {"original_concept", d.original_concept},
            {"new_subconcepts", d.new_subconcepts},
            {"reason", d.reason},
            {"differentiating_attribute", d.differentiating_attribute}
        });
    }
    return json{
        {"new_concepts", concepts_arr},
        {"new_rules", rules_arr},
        {"differentiations", diffs_arr},
        {"instances_processed", r.instances_processed},
        {"coherence_score", r.coherence_score}
    };
}

// ── Phase 4：社会学习 → JSON ──────────────────────────────────────────

inline auto parse_behavior_observation(const json& j) -> learning::BehaviorObservation {
    learning::BehaviorObservation obs;
    obs.agent_id = j.value("agent_id", "");
    obs.action = j.value("action", "");
    obs.context = j.value("context", "");
    obs.domain = j.value("domain", "");
    obs.outcome_quality = j.value("outcome_quality", 0.0);
    if (j.contains("preconditions") && j["preconditions"].is_array()) {
        obs.preconditions = j["preconditions"].get<std::vector<std::string>>();
    }
    if (j.contains("effects") && j["effects"].is_array()) {
        obs.effects = j["effects"].get<std::vector<std::string>>();
    }
    return obs;
}

inline auto to_json_social_learning_report(const learning::SocialLearningReport& r) -> json {
    json strategies_arr = json::array();
    for (const auto& s : r.learned_strategies) {
        strategies_arr.push_back(json{
            {"id", s.id},
            {"name", s.name},
            {"domain", s.domain},
            {"steps", s.steps},
            {"preconditions", s.preconditions},
            {"expected_outcomes", s.expected_outcomes},
            {"observed_success_rate", s.observed_success_rate},
            {"observation_count", s.observation_count},
            {"source_agent", s.source_agent}
        });
    }
    return json{
        {"learned_strategies", strategies_arr},
        {"knowledge_gained", r.knowledge_gained},
        {"failed_imitations", r.failed_imitations},
        {"imitation_success_rate", r.imitation_success_rate},
        {"observations_processed", r.observations_processed}
    };
}

// ── Phase 4：情感 → JSON ──────────────────────────────────────────────

inline auto parse_emotion_event(const json& j) -> learning::EmotionEvent {
    learning::EmotionEvent evt;
    evt.event_type = j.value("event_type", "");
    evt.domain = j.value("domain", "");
    evt.description = j.value("description", "");
    evt.magnitude = j.value("magnitude", 0.5);
    evt.expected_outcome = j.value("expected_outcome", 0.5);
    evt.actual_outcome = j.value("actual_outcome", 0.5);
    return evt;
}

inline auto to_json_emotion_state(const learning::EmotionState& s) -> json {
    return json{
        {"valence", s.valence},
        {"arousal", s.arousal},
        {"dominance", s.dominance},
        {"label", s.label},
        {"intensity", s.intensity()}
    };
}

// ── Phase 4：顿悟 → JSON ──────────────────────────────────────────────

inline auto to_json_insight_event(const learning::InsightEvent& e) -> json {
    return json{
        {"id", e.id},
        {"trigger", e.trigger},
        {"elements_combined", e.elements_combined},
        {"old_perspective", e.old_perspective},
        {"new_perspective", e.new_perspective},
        {"constraint_released", e.constraint_released},
        {"surprise_level", e.surprise_level},
        {"confidence", e.confidence},
        {"verified", e.verified},
        {"creativity_score", e.creativity_score},
        {"utility_score", e.utility_score}
    };
}

// ── Phase 5：元学习 → JSON ────────────────────────────────────────────

inline auto parse_task_descriptor(const json& j) -> learning::TaskDescriptor {
    learning::TaskDescriptor td;
    td.domain = j.value("domain", "");
    td.task_type = j.value("task_type", "");
    td.difficulty = j.value("difficulty", 0.5);
    td.novelty = j.value("novelty", 0.5);
    td.urgency = j.value("urgency", 0.5);
    td.prior_knowledge_count = j.value("prior_knowledge_count", 0);
    return td;
}

inline auto strategy_type_to_string(learning::LearningStrategyType type) -> std::string {
    switch (type) {
        case learning::LearningStrategyType::kRoteMemorization: return "rote_memorization";
        case learning::LearningStrategyType::kSpacedRepetition: return "spaced_repetition";
        case learning::LearningStrategyType::kActiveRecall: return "active_recall";
        case learning::LearningStrategyType::kTrialAndError: return "trial_and_error";
        case learning::LearningStrategyType::kAnalogicalTransfer: return "analogical_transfer";
        case learning::LearningStrategyType::kDecomposition: return "decomposition";
        case learning::LearningStrategyType::kExplanationBased: return "explanation_based";
        case learning::LearningStrategyType::kExploratory: return "exploratory";
        case learning::LearningStrategyType::kStructuredPractice: return "structured_practice";
        default: return "unknown";
    }
}

inline auto to_json_meta_recommendation(const learning::MetaLearningRecommendation& r) -> json {
    json alternatives_arr = json::array();
    for (const auto& alt : r.alternatives) {
        alternatives_arr.push_back(strategy_type_to_string(alt));
    }
    return json{
        {"recommended_strategy", strategy_type_to_string(r.recommended_strategy)},
        {"confidence", r.confidence},
        {"suggested_learning_rate", r.suggested_learning_rate},
        {"rationale", r.rationale},
        {"alternatives", alternatives_arr}
    };
}

// ── Phase 5：实验 → JSON ──────────────────────────────────────────────

inline auto parse_experiment_result(const json& j) -> learning::ExperimentResult {
    learning::ExperimentResult r;
    r.experiment_id = j.value("experiment_id", "");
    r.hypothesis_id = j.value("hypothesis_id", "");
    r.supports_hypothesis = j.value("supports_hypothesis", false);
    r.confidence_delta = j.value("confidence_delta", 0.0);
    r.observation = j.value("observation", "");
    r.analysis = j.value("analysis", "");
    r.information_gain = j.value("information_gain", 0.0);
    r.surprise = j.value("surprise", 0.0);
    return r;
}

inline auto to_json_experiment_design(const learning::ExperimentDesign& d) -> json {
    return json{
        {"id", d.id},
        {"target_hypothesis", d.target_hypothesis},
        {"description", d.description},
        {"method", d.method},
        {"steps", d.steps},
        {"expected_positive", d.expected_positive},
        {"expected_negative", d.expected_negative},
        {"expected_information_gain", d.expected_information_gain},
        {"cost", d.cost}
    };
}

// ── Phase 6：整合 → JSON ──────────────────────────────────────────────

inline auto to_json_pipeline_step(const learning::PipelineStepReport& s) -> json {
    return json{
        {"step_name", s.step_name},
        {"success", s.success},
        {"confidence_delta", s.confidence_delta},
        {"summary", s.summary}
    };
}

inline auto to_json_integrated_pipeline(const learning::IntegratedPipelineReport& r) -> json {
    return json{
        {"observe_step", to_json_pipeline_step(r.observe_step)},
        {"learn_step", to_json_pipeline_step(r.learn_step)},
        {"analogize_step", to_json_pipeline_step(r.analogize_step)},
        {"experiment_step", to_json_pipeline_step(r.experiment_step)},
        {"reflect_step", to_json_pipeline_step(r.reflect_step)},
        {"insight_step", to_json_pipeline_step(r.insight_step)},
        {"overall_progress", r.overall_progress},
        {"emotion_valence", r.emotion_valence},
        {"dominant_emotion", r.dominant_emotion},
        {"insights", r.insights},
        {"theories", r.theories},
        {"steps_completed", r.steps_completed}
    };
}

inline auto to_json_meta_guided_session(const learning::MetaGuidedSessionReport& r) -> json {
    return json{
        {"chosen_goal", to_json_learning_goal(r.chosen_goal)},
        {"strategy_recommendation", to_json_meta_recommendation(r.strategy_rec)},
        {"learning_rate", r.learning_rate},
        {"emotion_modulation", json{
            {"encoding_boost", r.emotion_modulation.encoding_boost},
            {"consolidation_boost", r.emotion_modulation.consolidation_boost},
            {"retrieval_boost", r.emotion_modulation.retrieval_boost},
            {"forgetting_rate", r.emotion_modulation.forgetting_rate},
            {"reason", r.emotion_modulation.reason}
        }},
        {"reflections", r.reflections}
    };
}

inline auto to_json_emotion_modulated_params(const learning::EmotionModulatedParams& p) -> json {
    return json{
        {"learning_rate", p.learning_rate},
        {"encoding_boost", p.encoding_boost},
        {"exploration_tendency", p.exploration_tendency},
        {"risk_tolerance", p.risk_tolerance},
        {"emotion_label", p.emotion_label}
    };
}

}  // namespace ai_learning::server::dto
