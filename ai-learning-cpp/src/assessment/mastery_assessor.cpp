/**
 * @file mastery_assessor.cpp
 * @brief Bloom 掌握度评估器实现
 *
 * 评分算法（从 Python 移植）：
 *   recognition   = log(1 + exposure) / log(21)          对数衰减
 *   comprehension = success_rate * min(1, practice/10)   正确率 * 练习饱和
 *   application   = entity.confidence()                   直接取置信度
 *   analysis      = min(1, relations/10)                  关联关系数
 *   synthesis     = min(1, total/15) + cross*0.05         跨领域加分
 *   teaching      = 按等级查表 Mastered=1.0 ...
 */

#include "ai_learning/assessment/mastery_assessor.hpp"

#include "ai_learning/domain/knowledge/entity.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/domain/knowledge/relation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

namespace ai_learning::assessment {

// ── 辅助 ────────────────────────────────────────────────────

namespace {

/// 从 Entity 属性中解析整数，缺失或非法返回 0
auto get_int_prop(const domain::knowledge::Entity& e,
                  const std::string& key) -> int {
    if (!e.has_property(key)) return 0;
    try { return std::stoi(e.get_property(key)); }
    catch (...) { return 0; }
}

/// 将 double 限制在 [lo, hi]
auto clamp(double v, double lo, double hi) -> double {
    return std::max(lo, std::min(hi, v));
}

}  // anonymous namespace

// ── MasteryLevel 枚举辅助 ───────────────────────────────────

auto mastery_level_to_string(MasteryLevel l) -> std::string {
    switch (l) {
        case MasteryLevel::Unknown:    return "Unknown";
        case MasteryLevel::Exposed:    return "Exposed";
        case MasteryLevel::Recognized: return "Recognized";
        case MasteryLevel::Understood: return "Understood";
        case MasteryLevel::Applied:    return "Applied";
        case MasteryLevel::Mastered:   return "Mastered";
    }
    return "Unknown";
}

auto mastery_level_from_score(double score) -> MasteryLevel {
    if (score >= 0.9) return MasteryLevel::Mastered;
    if (score >= 0.7) return MasteryLevel::Applied;
    if (score >= 0.5) return MasteryLevel::Understood;
    if (score >= 0.3) return MasteryLevel::Recognized;
    if (score >= 0.1) return MasteryLevel::Exposed;
    return MasteryLevel::Unknown;
}

// ── MasteryAssessor ─────────────────────────────────────────

MasteryAssessor::MasteryAssessor()
    : weights_{
        {"recognition",   0.15},
        {"comprehension", 0.25},
        {"application",   0.25},
        {"analysis",      0.15},
        {"synthesis",     0.10},
        {"teaching",      0.10}
    } {}

// ── 6 维度评分 ──────────────────────────────────────────────

auto MasteryAssessor::score_recognition(
        const domain::knowledge::Entity& entity) const -> double {
    // log(1 + exposure_count) / log(1 + 20), 上限 1.0
    int exposure = get_int_prop(entity, "exposure_count");
    return clamp(std::log(1.0 + exposure) / std::log(21.0), 0.0, 1.0);
}

auto MasteryAssessor::score_comprehension(
        const domain::knowledge::Entity& entity) const -> double {
    int practice = get_int_prop(entity, "practice_count");
    int success  = get_int_prop(entity, "success_count");
    double success_rate = static_cast<double>(success) /
                          std::max(1, practice);
    double saturation = std::min(1.0, static_cast<double>(practice) / 10.0);
    return clamp(success_rate * saturation, 0.0, 1.0);
}

auto MasteryAssessor::score_application(
        const domain::knowledge::Entity& entity) const -> double {
    return clamp(entity.confidence(), 0.0, 1.0);
}

auto MasteryAssessor::score_analysis(
        const domain::knowledge::Entity& entity,
        const domain::knowledge::KnowledgeGraph& kg) const -> double {
    auto related = kg.get_related(entity.id(), "related_to");
    auto is_a    = kg.get_related(entity.id(), "is_a");
    auto part_of = kg.get_related(entity.id(), "part_of");
    int total = static_cast<int>(related.size() + is_a.size() +
                                 part_of.size());
    return clamp(static_cast<double>(total) / 10.0, 0.0, 1.0);
}

auto MasteryAssessor::score_synthesis(
        const domain::knowledge::Entity& entity,
        const domain::knowledge::KnowledgeGraph& kg) const -> double {
    // 所有关系总数
    auto all_relations = kg.get_relations_of(entity.id());
    int total = static_cast<int>(all_relations.size());

    // 跨领域关系：关系对端实体类型不同于当前实体
    int cross_domain = 0;
    for (const auto& rel_ref : all_relations) {
        const auto& rel = rel_ref.get();
        std::string other_id = (rel.source_id() == entity.id())
                               ? rel.target_id()
                               : rel.source_id();
        auto other = kg.get_entity(other_id);
        if (other.has_value() &&
            other->get().type() != entity.type()) {
            ++cross_domain;
        }
    }

    double base = clamp(static_cast<double>(total) / 15.0, 0.0, 1.0);
    double bonus = std::min(0.2, cross_domain * 0.05);
    return clamp(base + bonus, 0.0, 1.0);
}

auto MasteryAssessor::score_teaching(
        const domain::knowledge::Entity& entity) const -> double {
    // 基于 mastery_level 查表
    MasteryLevel level = mastery_level_from_score(entity.confidence());
    switch (level) {
        case MasteryLevel::Mastered:   return 1.0;
        case MasteryLevel::Applied:    return 0.6;
        case MasteryLevel::Understood: return 0.3;
        default:                       return 0.0;
    }
}

// ── 推荐生成 ────────────────────────────────────────────────

auto MasteryAssessor::generate_recommendations(
        const std::map<std::string, double>& dimensions) const
    -> std::vector<std::string> {
    static const std::map<std::string, std::string> kRecommendations = {
        {"recognition",   "增加接触频率"},
        {"comprehension", "加强理解练习"},
        {"application",   "增加应用实践"},
        {"analysis",      "建立更多关联"},
        {"synthesis",     "尝试跨领域整合"},
        {"teaching",      "尝试教授他人以巩固知识"},
    };
    std::vector<std::string> recs;
    for (const auto& [dim, score] : dimensions) {
        if (score < 0.3) {
            auto it = kRecommendations.find(dim);
            if (it != kRecommendations.end()) {
                recs.push_back(it->second);
            }
        }
    }
    return recs;
}

// ── assess: 单实体评估 ──────────────────────────────────────

auto MasteryAssessor::assess(
        const domain::knowledge::Entity& entity,
        const domain::knowledge::KnowledgeGraph& kg) const
    -> AssessmentResult {

    // 1. 6 维度评分
    std::map<std::string, double> dims;
    dims["recognition"]   = score_recognition(entity);
    dims["comprehension"] = score_comprehension(entity);
    dims["application"]   = score_application(entity);
    dims["analysis"]      = score_analysis(entity, kg);
    dims["synthesis"]     = score_synthesis(entity, kg);
    dims["teaching"]      = score_teaching(entity);

    // 2. 加权平均
    double overall = 0.0;
    for (const auto& [dim, score] : dims) {
        auto wit = weights_.find(dim);
        double w = (wit != weights_.end()) ? wit->second : 0.0;
        overall += w * score;
    }
    overall = clamp(overall, 0.0, 1.0);

    // 3. 确定等级
    MasteryLevel level = mastery_level_from_score(overall);

    // 4. 强项 / 弱项
    std::vector<std::string> strengths;
    std::vector<std::string> weaknesses;
    for (const auto& [dim, score] : dims) {
        if (score >= 0.7) strengths.push_back(dim);
        if (score < 0.3)  weaknesses.push_back(dim);
    }

    // 5. 推荐
    auto recs = generate_recommendations(dims);

    // 6. 组装结果
    AssessmentResult result;
    result.entity_id      = entity.id();
    result.entity_name    = entity.get_property("name").empty()
                            ? entity.id()
                            : entity.get_property("name");
    result.domain         = entity.type();
    result.overall_mastery = overall;
    result.mastery_level  = level;
    result.dimensions     = std::move(dims);
    result.strengths      = std::move(strengths);
    result.weaknesses     = std::move(weaknesses);
    result.recommendations = std::move(recs);
    return result;
}

// ── assess_domain: 领域级评估 ────────────────────────────────

auto MasteryAssessor::assess_domain(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const
    -> DomainReport {

    DomainReport report;
    report.domain = entity_type;

    // 查询该类型所有实体
    auto entities = kg.query(entity_type);
    report.total_units = static_cast<int>(entities.size());
    if (entities.empty()) return report;

    // 逐个评估
    std::vector<AssessmentResult> results;
    results.reserve(entities.size());
    double mastery_sum = 0.0;
    double relation_sum = 0.0;
    int covered = 0;

    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        auto r = assess(entity, kg);
        mastery_sum += r.overall_mastery;
        if (r.overall_mastery >= 0.3) ++covered;

        // 统计关系数
        auto rels = kg.get_relations_of(entity.id());
        relation_sum += static_cast<double>(rels.size());

        // 等级分布
        std::string level_str = mastery_level_to_string(r.mastery_level);
        report.level_distribution[level_str]++;

        results.push_back(std::move(r));
    }

    report.avg_mastery = mastery_sum / static_cast<double>(report.total_units);
    report.coverage = static_cast<double>(covered) /
                      static_cast<double>(report.total_units);
    report.avg_relations = relation_sum /
                           static_cast<double>(report.total_units);

    // 找 top 5 / bottom 5
    std::sort(results.begin(), results.end(),
              [](const AssessmentResult& a, const AssessmentResult& b) {
                  return a.overall_mastery > b.overall_mastery;
              });

    int top_n = std::min(5, static_cast<int>(results.size()));
    for (int i = 0; i < top_n; ++i) {
        report.strongest_units.push_back(results[i].entity_name);
    }
    int bot_start = static_cast<int>(results.size()) - top_n;
    for (int i = bot_start; i < static_cast<int>(results.size()); ++i) {
        report.weakest_units.push_back(results[i].entity_name);
    }

    return report;
}

// ── assess_all_domains ──────────────────────────────────────

auto MasteryAssessor::assess_all_domains(
        const domain::knowledge::KnowledgeGraph& kg) const
    -> std::map<std::string, DomainReport> {

    auto type_dist = kg.type_distribution();
    std::map<std::string, DomainReport> reports;
    for (const auto& [type, count] : type_dist) {
        if (count > 0) {
            reports[type] = assess_domain(type, kg);
        }
    }
    return reports;
}

// ── ProficiencyTester ────────────────────────────────────────

ProficiencyTester::ProficiencyTester()
    : cefr_thresholds_{
        {"A1", 0.10}, {"A2", 0.20}, {"B1", 0.35},
        {"B2", 0.50}, {"C1", 0.70}, {"C2", 0.85}, {"Professional", 0.95}
    },
    dimension_weights_{
        {"vocabulary_breadth", 0.30},
        {"vocabulary_depth",   0.25},
        {"semantic_network",   0.20},
        {"collocation",        0.15},
        {"word_family",        0.10}
    },
    cefr_level_weights_{
        {"A1", 0.05}, {"A2", 0.10}, {"B1", 0.15},
        {"B2", 0.20}, {"C1", 0.25}, {"C2", 0.25}
    } {}

auto ProficiencyTester::score_vocabulary_breadth(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double {
    auto entities = kg.query(entity_type);
    if (entities.empty()) return 0.0;

    // Group entities by CEFR tag
    std::map<std::string, std::vector<double>> level_mastery;
    double total_weight = 0.0;
    double weighted_sum = 0.0;

    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        for (const auto& tag : entity.tags()) {
            auto wit = cefr_level_weights_.find(tag);
            if (wit != cefr_level_weights_.end()) {
                level_mastery[tag].push_back(entity.confidence());
                break;  // Use first matching CEFR tag
            }
        }
    }

    // Weighted average across levels
    for (const auto& [level, confidences] : level_mastery) {
        if (confidences.empty()) continue;
        double avg = 0.0;
        for (double c : confidences) avg += c;
        avg /= static_cast<double>(confidences.size());

        auto wit = cefr_level_weights_.find(level);
        double w = (wit != cefr_level_weights_.end()) ? wit->second : 0.0;
        weighted_sum += w * avg;
        total_weight += w;
    }

    if (total_weight == 0.0) return 0.0;
    return clamp(weighted_sum / total_weight, 0.0, 1.0);
}

auto ProficiencyTester::score_vocabulary_depth(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double {
    auto entities = kg.query(entity_type);
    if (entities.empty()) return 0.0;

    double confidence_sum = 0.0;
    int mastered_count = 0;
    int total = static_cast<int>(entities.size());

    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        confidence_sum += entity.confidence();
        if (entity.confidence() >= 0.7) ++mastered_count;
    }

    double avg_mastery = confidence_sum / static_cast<double>(total);
    double mastery_rate = static_cast<double>(mastered_count) /
                          static_cast<double>(total);
    return clamp(0.6 * avg_mastery + 0.4 * mastery_rate, 0.0, 1.0);
}

auto ProficiencyTester::score_semantic_network(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double {
    auto entities = kg.query(entity_type);
    if (entities.empty()) return 0.0;

    double relation_sum = 0.0;
    int isolated = 0;
    int total = static_cast<int>(entities.size());

    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        auto rels = kg.get_relations_of(entity.id());
        double rel_count = static_cast<double>(rels.size());
        relation_sum += rel_count;
        if (rels.empty()) ++isolated;
    }

    double density = std::min(1.0, (relation_sum / static_cast<double>(total)) / 10.0);
    double isolation_rate = static_cast<double>(isolated) / static_cast<double>(total);
    return clamp(density * (1.0 - 0.5 * isolation_rate), 0.0, 1.0);
}

auto ProficiencyTester::score_collocation(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double {
    auto entities = kg.query(entity_type);
    if (entities.empty()) return 0.0;

    int with_examples = 0;
    double example_sum = 0.0;
    int total = static_cast<int>(entities.size());

    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        if (entity.has_property("example_count")) {
            try {
                int count = std::stoi(entity.get_property("example_count"));
                if (count > 0) {
                    ++with_examples;
                    example_sum += static_cast<double>(count);
                }
            } catch (...) {
                // Invalid property, skip
            }
        }
    }

    double fraction_with = static_cast<double>(with_examples) / static_cast<double>(total);
    double avg_examples = (with_examples > 0)
                          ? example_sum / static_cast<double>(with_examples)
                          : 0.0;
    return clamp(0.5 * fraction_with + 0.5 * std::min(1.0, avg_examples / 3.0),
                 0.0, 1.0);
}

auto ProficiencyTester::score_word_family(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double {
    auto entities = kg.query(entity_type);
    if (entities.empty()) return 0.0;

    int with_family = 0;
    int total = static_cast<int>(entities.size());

    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        if (entity.has_tag("word_family")) {
            ++with_family;
        }
    }

    return static_cast<double>(with_family) / static_cast<double>(total);
}

auto ProficiencyTester::determine_level(double score) const -> std::string {
    std::string level = "A1";  // Default minimum
    for (const auto& [name, threshold] : cefr_thresholds_) {
        if (score >= threshold) {
            level = name;
        }
    }
    return level;
}

auto ProficiencyTester::assess(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> ProficiencyReport {

    ProficiencyReport report;

    // 1. Query all entities of the given type
    auto entities = kg.query(entity_type);

    // 2. Score each dimension
    report.dimensions["vocabulary_breadth"] = score_vocabulary_breadth(entity_type, kg);
    report.dimensions["vocabulary_depth"]   = score_vocabulary_depth(entity_type, kg);
    report.dimensions["semantic_network"]   = score_semantic_network(entity_type, kg);
    report.dimensions["collocation"]        = score_collocation(entity_type, kg);
    report.dimensions["word_family"]        = score_word_family(entity_type, kg);

    // 3. Compute weighted overall score
    double overall = 0.0;
    for (const auto& [dim, score] : report.dimensions) {
        auto wit = dimension_weights_.find(dim);
        double w = (wit != dimension_weights_.end()) ? wit->second : 0.0;
        overall += w * score;
    }
    report.score = clamp(overall, 0.0, 1.0);

    // 4. Determine CEFR level
    report.level = determine_level(report.score);

    // 5. Count receptive/productive vocabulary
    report.semantic_depth         = report.dimensions.at("semantic_network");
    report.collocation_knowledge = report.dimensions.at("collocation");
    report.word_family_coverage   = report.dimensions.at("word_family");

    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        if (entity.confidence() >= 0.3) ++report.receptive_vocab;   // Recognized+
        if (entity.confidence() >= 0.7) ++report.productive_vocab;  // Applied+
    }

    // 6. Build level_distribution map
    for (const auto& entity_ref : entities) {
        const auto& entity = entity_ref.get();
        for (const auto& tag : entity.tags()) {
            if (cefr_level_weights_.count(tag) > 0) {
                report.level_distribution[tag]++;
                break;  // Count each entity once for its first CEFR tag
            }
        }
    }

    // 7. Generate recommendations for weak dimensions
    static const std::map<std::string, std::string> kDimRecommendations = {
        {"vocabulary_breadth", "扩大词汇量覆盖面"},
        {"vocabulary_depth",   "加深词汇理解深度"},
        {"semantic_network",   "建立词汇间语义关联"},
        {"collocation",        "学习词汇搭配用法"},
        {"word_family",        "扩展词族知识"},
    };
    for (const auto& [dim, score] : report.dimensions) {
        if (score < 0.3) {
            auto it = kDimRecommendations.find(dim);
            if (it != kDimRecommendations.end()) {
                report.recommendations.push_back(it->second);
            }
        }
    }

    return report;
}

}  // namespace ai_learning::assessment
