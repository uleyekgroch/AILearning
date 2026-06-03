/**
 * @file knowledge_unit.cpp
 * @brief 通用知识单元实现
 */

#include "ai_learning/domain/knowledge/knowledge_unit.hpp"

#include <algorithm>

namespace ai_learning::domain::knowledge {

auto mastery_level_to_string(MasteryLevel level) -> std::string {
    switch (level) {
        case MasteryLevel::kUnknown:    return "unknown";
        case MasteryLevel::kExposed:    return "exposed";
        case MasteryLevel::kRecognized: return "recognized";
        case MasteryLevel::kUnderstood: return "understood";
        case MasteryLevel::kApplied:    return "applied";
        case MasteryLevel::kMastered:   return "mastered";
    }
    return "unknown";
}

auto mastery_level_from_string(const std::string& s) -> MasteryLevel {
    if (s == "exposed")    return MasteryLevel::kExposed;
    if (s == "recognized") return MasteryLevel::kRecognized;
    if (s == "understood") return MasteryLevel::kUnderstood;
    if (s == "applied")    return MasteryLevel::kApplied;
    if (s == "mastered")   return MasteryLevel::kMastered;
    return MasteryLevel::kUnknown;
}

KnowledgeUnit::KnowledgeUnit(std::string id, std::string name,
                             std::string domain,
                             std::string definition,
                             double difficulty)
    : id_(std::move(id)),
      name_(std::move(name)),
      domain_(std::move(domain)),
      definition_(std::move(definition)),
      difficulty_(std::max(0.0, std::min(1.0, difficulty))) {}

void KnowledgeUnit::update_mastery(bool success, double quality) {
    ++practice_count_;
    if (success) {
        ++success_count_;
        double delta = 0.1 * quality;
        mastery_ = std::min(1.0, mastery_ + delta);
    } else {
        mastery_ = std::max(0.0, mastery_ - 0.05);
    }
    update_mastery_level_();
}

void KnowledgeUnit::add_prerequisite(const std::string& prereq_id) {
    // 避免重复
    auto it = std::find(prerequisites_.begin(), prerequisites_.end(),
                        prereq_id);
    if (it == prerequisites_.end()) {
        prerequisites_.push_back(prereq_id);
    }
}

auto KnowledgeUnit::success_rate() const -> double {
    if (practice_count_ == 0) return 0.0;
    return static_cast<double>(success_count_) / practice_count_;
}

auto KnowledgeUnit::prerequisites_met(
    const std::map<std::string, const KnowledgeUnit*>& all_units,
    double threshold) const -> bool {
    if (prerequisites_.empty()) return true;
    for (const auto& prereq_id : prerequisites_) {
        auto it = all_units.find(prereq_id);
        if (it == all_units.end()) continue;
        if (it->second->mastery() < threshold) return false;
    }
    return true;
}

auto KnowledgeUnit::to_map() const -> std::map<std::string, std::string> {
    return {
        {"id",             id_},
        {"name",           name_},
        {"domain",         domain_},
        {"definition",     definition_},
        {"difficulty",     std::to_string(difficulty_)},
        {"mastery",        std::to_string(mastery_)},
        {"mastery_level",  mastery_level_to_string(mastery_level_)},
        {"exposure_count", std::to_string(exposure_count_)},
        {"practice_count", std::to_string(practice_count_)},
        {"success_count",  std::to_string(success_count_)},
    };
}

void KnowledgeUnit::update_mastery_level_() {
    if (mastery_ < 0.1)       mastery_level_ = MasteryLevel::kUnknown;
    else if (mastery_ < 0.3)  mastery_level_ = MasteryLevel::kExposed;
    else if (mastery_ < 0.5)  mastery_level_ = MasteryLevel::kRecognized;
    else if (mastery_ < 0.7)  mastery_level_ = MasteryLevel::kUnderstood;
    else if (mastery_ < 0.9)  mastery_level_ = MasteryLevel::kApplied;
    else                      mastery_level_ = MasteryLevel::kMastered;
}

}  // namespace ai_learning::domain::knowledge
