/**
 * @file development.cpp
 * @brief 语言发展阶段实现
 */

#include "ai_learning/language/development.hpp"

namespace ai_learning::language {

auto stage_to_string(Stage s) -> std::string {
    switch (s) {
        case Stage::Sensorimotor: return "sensorimotor";
        case Stage::SingleWord:   return "single_word";
        case Stage::TwoWord:      return "two_word";
        case Stage::Complex:      return "complex";
        case Stage::Literacy:     return "literacy";
    }
    return "unknown";
}

auto stage_from_string(const std::string& s) -> Stage {
    if (s == "sensorimotor")  return Stage::Sensorimotor;
    if (s == "single_word")   return Stage::SingleWord;
    if (s == "two_word")      return Stage::TwoWord;
    if (s == "complex")       return Stage::Complex;
    if (s == "literacy")      return Stage::Literacy;
    return Stage::Sensorimotor;
}

DevelopmentTracker::DevelopmentTracker(Stage initial) : stage_(initial) {}

auto DevelopmentTracker::try_advance(const Evaluation& evaluation) -> bool {
    int idx = static_cast<int>(stage_);
    if (idx >= stage_count() - 1) {
        return false;  // 已在最高阶段
    }

    if (!check_promotion(evaluation)) {
        return false;
    }

    Stage old_stage = stage_;
    stage_ = static_cast<Stage>(idx + 1);
    history_.push_back({old_stage, stage_, evaluation});
    return true;
}

auto DevelopmentTracker::current_abilities() const -> std::vector<std::string> {
    switch (stage_) {
        case Stage::Sensorimotor:
            return {"object_tracking", "basic_reflex"};
        case Stage::SingleWord:
            return {"word_production", "word_comprehension"};
        case Stage::TwoWord:
            return {"composition", "basic_syntax"};
        case Stage::Complex:
            return {"grammar", "narrative", "theory_of_mind"};
        case Stage::Literacy:
            return {"reading", "writing", "abstract_reasoning"};
    }
    return {};
}

auto DevelopmentTracker::check_promotion(const Evaluation& e) const -> bool {
    // 每个阶段允许多条晋升路径（OR 逻辑）
    switch (stage_) {
        case Stage::Sensorimotor:
            // 预测准确 OR 积累了足够词汇
            return e.prediction_accuracy > 0.6F || e.vocabulary_size >= 5;

        case Stage::SingleWord:
            // 词汇量继续增长 OR 组合表达涌现
            return e.vocabulary_size >= 10 || e.composition_rate > 0.3F;

        case Stage::TwoWord:
            // 组合能力成熟 OR 语法开始涌现
            return e.composition_rate > 0.3F || e.grammar_complexity > 0.5F;

        case Stage::Complex:
            // 语法系统成熟
            return e.grammar_complexity > 0.5F;

        case Stage::Literacy:
            return false;  // 最高阶段
    }
    return false;
}

}  // namespace ai_learning::language
