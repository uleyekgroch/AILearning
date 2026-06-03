/**
 * @file mastery_assessor.hpp
 * @brief Bloom 掌握度评估器 — 基于布鲁姆分类法的知识掌握评估
 *
 * 6 维度评估（对应 Bloom 6 层）：
 *   1. recognition     — 记忆（接触频率）
 *   2. comprehension   — 理解（练习正确率）
 *   3. application     — 应用（直接使用 confidence）
 *   4. analysis        — 分析（关联关系数量）
 *   5. synthesis       — 综合（跨领域关联）
 *   6. teaching        — 教授（掌握度等级）
 *
 * Entity 属性映射：
 *   exposure_count  → entity.get_property("exposure_count")
 *   practice_count  → entity.get_property("practice_count")
 *   success_count   → entity.get_property("success_count")
 *   mastery         → entity.confidence()
 *   domain          → entity.type()
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {
    class KnowledgeGraph;
    class Entity;
}

namespace ai_learning::assessment {

/// Bloom 掌握度等级
enum class MasteryLevel {
    Unknown    = 0,
    Exposed    = 1,
    Recognized = 2,
    Understood = 3,
    Applied    = 4,
    Mastered   = 5
};

[[nodiscard]] auto mastery_level_to_string(MasteryLevel l) -> std::string;
[[nodiscard]] auto mastery_level_from_score(double score) -> MasteryLevel;

/// 单个实体的评估结果
struct AssessmentResult {
    std::string                         entity_id;
    std::string                         entity_name;
    std::string                         domain;
    double                              overall_mastery  = 0.0;
    MasteryLevel                        mastery_level    = MasteryLevel::Unknown;
    std::map<std::string, double>       dimensions;      // dimension_name -> score
    std::vector<std::string>            strengths;       // dimensions >= 0.7
    std::vector<std::string>            weaknesses;      // dimensions < 0.3
    std::vector<std::string>            recommendations;
};

/// 领域级聚合报告
struct DomainReport {
    std::string                         domain;
    int                                 total_units      = 0;
    double                              avg_mastery       = 0.0;
    std::map<std::string, int>          level_distribution;
    double                              coverage          = 0.0;
    double                              avg_relations     = 0.0;
    std::vector<std::string>            strongest_units;  // top 5
    std::vector<std::string>            weakest_units;    // bottom 5
};

/// Bloom 掌握度评估器
class MasteryAssessor {
public:
    MasteryAssessor();

    /// 评估单个实体（Bloom 6 维度）
    [[nodiscard]] auto assess(
        const domain::knowledge::Entity& entity,
        const domain::knowledge::KnowledgeGraph& kg) const -> AssessmentResult;

    /// 评估指定类型（领域）下所有实体
    [[nodiscard]] auto assess_domain(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> DomainReport;

    /// 评估知识图谱中所有领域
    [[nodiscard]] auto assess_all_domains(
        const domain::knowledge::KnowledgeGraph& kg) const
        -> std::map<std::string, DomainReport>;

private:
    // Bloom 6 维度权重
    std::map<std::string, double> weights_;

    [[nodiscard]] auto score_recognition(
        const domain::knowledge::Entity& entity) const -> double;
    [[nodiscard]] auto score_comprehension(
        const domain::knowledge::Entity& entity) const -> double;
    [[nodiscard]] auto score_application(
        const domain::knowledge::Entity& entity) const -> double;
    [[nodiscard]] auto score_analysis(
        const domain::knowledge::Entity& entity,
        const domain::knowledge::KnowledgeGraph& kg) const -> double;
    [[nodiscard]] auto score_synthesis(
        const domain::knowledge::Entity& entity,
        const domain::knowledge::KnowledgeGraph& kg) const -> double;
    [[nodiscard]] auto score_teaching(
        const domain::knowledge::Entity& entity) const -> double;

    [[nodiscard]] auto generate_recommendations(
        const std::map<std::string, double>& dimensions) const
        -> std::vector<std::string>;
};

/// CEFR language proficiency report
struct ProficiencyReport {
    std::string                         level;              // "A1" to "C2" or "Professional"
    double                              score            = 0.0;
    int                                 receptive_vocab  = 0;   // entities at Recognized+
    int                                 productive_vocab = 0;   // entities at Applied+
    double                              semantic_depth   = 0.0;
    double                              collocation_knowledge = 0.0;
    double                              word_family_coverage   = 0.0;
    std::map<std::string, double>       dimensions;
    std::vector<std::string>            recommendations;
    std::map<std::string, int>          level_distribution;    // CEFR tag -> count
};

/// CEFR language proficiency tester
class ProficiencyTester {
public:
    ProficiencyTester();

    /// Assess proficiency for entities of a given type (default "word")
    [[nodiscard]] auto assess(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> ProficiencyReport;

private:
    std::map<std::string, double> cefr_thresholds_;  // "A1" -> 0.10, ..., "Professional" -> 0.95
    std::map<std::string, double> dimension_weights_; // vocabulary_breadth, vocabulary_depth, etc.
    std::map<std::string, double> cefr_level_weights_; // A1=0.05, A2=0.10, ..., C2=0.25

    [[nodiscard]] auto score_vocabulary_breadth(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double;
    [[nodiscard]] auto score_vocabulary_depth(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double;
    [[nodiscard]] auto score_semantic_network(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double;
    [[nodiscard]] auto score_collocation(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double;
    [[nodiscard]] auto score_word_family(
        const std::string& entity_type,
        const domain::knowledge::KnowledgeGraph& kg) const -> double;
    [[nodiscard]] auto determine_level(double score) const -> std::string;
};

}  // namespace ai_learning::assessment
