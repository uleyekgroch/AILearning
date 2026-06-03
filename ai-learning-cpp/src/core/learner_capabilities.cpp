/**
 * @file learner_capabilities.cpp
 * @brief Learner 四大人类核心能力 — 动手做/自我修改/因果推理/元认知
 */

#include "ai_learning/core/learner.hpp"

namespace ai_learning::core {

using namespace learning;

// -- 能力 1：动手做 -----------------------------------------------

auto Learner::learn_by_doing(const std::string& code,
                               const std::string& language)
    -> learning::ExecutionFeedback {
    (void)language;  // sandbox_.run_tests 统一用 cpp
    auto feedback = sandbox_.run_tests(code);

    // 从执行反馈中学习
    if (!feedback.compiled) {
        // 编译错误 -> 学习因果规则
        for (const auto& err : feedback.errors) {
            auto analysis = learning::FeedbackParser::analyze_compile_error(err);
            if (analysis.contains("category")) {
                world_model_.add_causal_rule({
                    analysis["category"], "compile_error",
                    reasoning::CausalEdgeType::kCauses, 0.9, {}
                });
            }
        }
    } else if (feedback.ran) {
        // 成功执行 -> 记录到元认知
        metacognition_.record_outcome("code_execution", true);
    } else {
        // 运行时错误 -> 学习
        metacognition_.record_outcome("code_execution", false);
    }

    return feedback;
}

// -- 能力 2：自我修改 -----------------------------------------------

auto Learner::self_evolve()
    -> MutationResult {
    auto fitness = [this](Learner& l) -> double {
        auto stats = l.get_stats();
        return stats.at("entity_count") * 2.0 +
               stats.at("relation_count") * 3.0 +
               stats.at("learning_progress") * 10.0;
    };

    return self_modifier_.evolve_once(*this, fitness);
}

// -- 能力 3：因果推理 -----------------------------------------------

void Learner::learn_causal(const std::vector<std::string>& events,
                             const std::string& outcome) {
    world_model_.observe_sequence(events, outcome);
}

auto Learner::reason_causal(const std::string& question) const
    -> reasoning::CounterfactualResult {
    // 简化的因果推理：将问题解析为反事实
    return world_model_.counterfactual(question, {}, {});
}

auto Learner::plan_with_world_model(const std::string& goal) const
    -> reasoning::ImaginationPlan {
    return world_model_.imagine_plan(goal, 5);
}

// -- 能力 4：元认知 -----------------------------------------------

auto Learner::metacognitive_report()
    -> MetacognitiveReport {
    return metacognition_.generate_report(*this);
}

auto Learner::knows_about(const std::string& topic) const
    -> bool {
    return metacognition_.knows_about(topic);
}

auto Learner::what_should_i_learn() const
    -> std::vector<KnowledgeGap> {
    return metacognition_.detect_gaps(*this);
}

}  // namespace ai_learning::core
