/**
 * @file intrinsic_motivation.cpp
 * @brief 内在动机引擎实现
 */

#include "ai_learning/learning/intrinsic_motivation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ai_learning::learning {

IntrinsicMotivationEngine::IntrinsicMotivationEngine() {
    motives_ = {
        {MotiveType::kCuriosity,   {MotiveType::kCuriosity,   0.5, 0.0, 0.01, "curiosity"}},
        {MotiveType::kCompetence,  {MotiveType::kCompetence,  0.5, 0.0, 0.01, "competence"}},
        {MotiveType::kAutonomy,    {MotiveType::kAutonomy,    0.5, 0.0, 0.01, "autonomy"}},
        {MotiveType::kSocial,      {MotiveType::kSocial,      0.3, 0.0, 0.005, "social"}},
        {MotiveType::kAchievement, {MotiveType::kAchievement, 0.5, 0.0, 0.01, "achievement"}},
    };
}

auto IntrinsicMotivationEngine::generate_goal(
    const std::vector<std::string>& known_topics,
    double curiosity_signal,
    const std::map<std::string, double>& mastery_map)
    -> LearningGoal {

    if (known_topics.empty()) {
        return {"general", "general", 0.5, 0.5, MotiveType::kCuriosity, "Explore general knowledge"};
    }

    // 对每个已知主题计算综合动机评分
    std::vector<std::pair<LearningGoal, double>> scored;
    for (const auto& topic : known_topics) {
        double mastery = mastery_map.count(topic) ? mastery_map.at(topic) : 0.0;

        LearningGoal goal{
            topic, topic, 1.0 - mastery, 0.0,
            MotiveType::kCuriosity,
            "Learn more about " + topic
        };

        double c_score = curiosity_score_(topic, curiosity_signal);
        double comp_score = competence_score_(topic, mastery_map);
        double ach_score = achievement_score_(goal.difficulty);
        double auto_score = autonomy_score_(true);

        goal.estimated_value = c_score * 0.4 + comp_score * 0.3 + ach_score * 0.2 + auto_score * 0.1;

        // 确定主导动机
        double max_score = c_score;
        goal.primary_motive = MotiveType::kCuriosity;
        if (comp_score > max_score) { max_score = comp_score; goal.primary_motive = MotiveType::kCompetence; }
        if (ach_score > max_score) { max_score = ach_score; goal.primary_motive = MotiveType::kAchievement; }

        scored.emplace_back(goal, goal.estimated_value);
    }

    auto result = softmax_select_(scored);
    last_goal_ = result;
    return result;
}

auto IntrinsicMotivationEngine::evaluate_goal(const LearningGoal& goal) const
    -> double {
    double base = (goal.estimated_value > 0.0) ? goal.estimated_value : 0.3;
    double drive = drive_level(goal.primary_motive);
    return base * (0.5 + 0.5 * drive);
}

auto IntrinsicMotivationEngine::select_goal(
    const std::vector<LearningGoal>& candidates)
    -> std::optional<LearningGoal> {
    if (candidates.empty()) return std::nullopt;

    const LearningGoal* best = &candidates[0];
    double best_score = evaluate_goal(candidates[0]);

    for (size_t i = 1; i < candidates.size(); ++i) {
        double score = evaluate_goal(candidates[i]);
        if (score > best_score) {
            best_score = score;
            best = &candidates[i];
        }
    }

    last_goal_ = *best;
    return *best;
}

void IntrinsicMotivationEngine::update_on_learning(const LearningOutcome& outcome) {
    // 高 surprise → 刺激好奇心
    if (outcome.surprise > 0.5) {
        motives_[MotiveType::kCuriosity].drive_level =
            std::min(1.0, motives_[MotiveType::kCuriosity].drive_level + outcome.surprise * 0.2);
    } else {
        // 低 surprise → 好奇心被部分满足，略微下降
        motives_[MotiveType::kCuriosity].satisfaction += 0.1;
    }

    // 掌握度提升 → 刺激掌握欲
    if (outcome.mastery_improved) {
        motives_[MotiveType::kCompetence].drive_level =
            std::min(1.0, motives_[MotiveType::kCompetence].drive_level + 0.15);
        motives_[MotiveType::kCompetence].satisfaction += 0.2;
    }

    // 目标完成 → 成就感
    if (outcome.goal_completed) {
        motives_[MotiveType::kAchievement].drive_level =
            std::min(1.0, motives_[MotiveType::kAchievement].drive_level + 0.3);
        motives_[MotiveType::kAchievement].satisfaction += 0.5;
    }

    // 有进步 → 自主性增强
    if (outcome.progress > 0.3) {
        motives_[MotiveType::kAutonomy].satisfaction += outcome.progress * 0.1;
    }
}

void IntrinsicMotivationEngine::decay() {
    for (auto& [type, state] : motives_) {
        state.drive_level = std::max(0.1, state.drive_level - state.decay_rate);
        state.satisfaction = std::max(0.0, state.satisfaction - state.decay_rate * 0.5);
    }
}

auto IntrinsicMotivationEngine::drive_level(MotiveType type) const -> double {
    auto it = motives_.find(type);
    return it != motives_.end() ? it->second.drive_level : 0.0;
}

auto IntrinsicMotivationEngine::total_drive() const -> double {
    double total = 0.0;
    for (const auto& [_, state] : motives_) {
        total += state.drive_level;
    }
    return total;
}

auto IntrinsicMotivationEngine::last_goal() const -> std::optional<LearningGoal> {
    return last_goal_;
}

auto IntrinsicMotivationEngine::motive_name(MotiveType type) -> std::string {
    switch (type) {
        case MotiveType::kCuriosity:   return "curiosity";
        case MotiveType::kCompetence:  return "competence";
        case MotiveType::kAutonomy:    return "autonomy";
        case MotiveType::kSocial:      return "social";
        case MotiveType::kAchievement: return "achievement";
    }
    return "unknown";
}

auto IntrinsicMotivationEngine::curiosity_score_(
    const std::string& /*topic*/, double curiosity_signal) const -> double {
    return motives_.at(MotiveType::kCuriosity).drive_level * curiosity_signal;
}

auto IntrinsicMotivationEngine::competence_score_(
    const std::string& topic,
    const std::map<std::string, double>& mastery) const -> double {
    double m = mastery.count(topic) ? mastery.at(topic) : 0.0;
    // 低掌握度 → 高掌握欲
    return motives_.at(MotiveType::kCompetence).drive_level * (1.0 - m);
}

auto IntrinsicMotivationEngine::autonomy_score_(bool self_generated) const -> double {
    return self_generated ? motives_.at(MotiveType::kAutonomy).drive_level : 0.3;
}

auto IntrinsicMotivationEngine::achievement_score_(double difficulty) const -> double {
    // 中等难度最有吸引力（Flow Theory）
    double optimal = 1.0 - std::abs(difficulty - 0.5) * 2.0;
    return motives_.at(MotiveType::kAchievement).drive_level * optimal;
}

auto IntrinsicMotivationEngine::softmax_select_(
    const std::vector<std::pair<LearningGoal, double>>& scored)
    -> LearningGoal {
    if (scored.empty()) {
        return {"general", "general", 0.5, 0.5, MotiveType::kCuriosity, "default"};
    }

    double max_val = scored[0].second;
    for (const auto& [_, v] : scored) max_val = std::max(max_val, v);

    // 选择最高分
    const LearningGoal* best = &scored[0].first;
    double best_val = scored[0].second;
    for (const auto& [g, v] : scored) {
        if (v > best_val) { best_val = v; best = &g; }
    }
    return *best;
}

}  // namespace ai_learning::learning
