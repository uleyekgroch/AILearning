/**
 * @file verification.cpp
 * @brief 知识验证器实现
 */

#include "ai_learning/learning/verification.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

namespace ai_learning::learning {

using domain::knowledge::KnowledgeGraph;

auto KnowledgeVerifier::verify(
    const std::vector<std::tuple<std::string, std::string, std::string>>& triples,
    const KnowledgeGraph& kg) const -> VerificationReport {

    VerificationReport report;

    for (const auto& [subject, relation, obj] : triples) {
        auto question = generate_question(subject, relation);
        auto answer   = query_answer(subject, kg);

        bool passed = answer.find(obj) != std::string::npos;

        report.tests.push_back({
            .question = std::move(question),
            .expected = obj,
            .actual   = answer.substr(0, 50),
            .passed   = passed,
        });

        if (!passed) report.passed = false;
    }

    if (!report.tests.empty()) {
        int passed_count = 0;
        for (const auto& t : report.tests)
            if (t.passed) ++passed_count;
        report.score = static_cast<double>(passed_count)
                     / static_cast<double>(report.tests.size());
    }

    return report;
}

auto KnowledgeVerifier::generate_question(
    const std::string& subject,
    const std::string& relation) -> std::string {

    if (relation == "是")          return "什么是" + subject;
    if (relation == "属于")        return subject + "属于什么";
    if (relation == "位于")        return subject + "位于哪里";
    if (relation == "导致")        return subject + "会导致什么";
    return subject + relation + "什么";
}

auto KnowledgeVerifier::query_answer(
    const std::string& subject,
    const KnowledgeGraph& kg) -> std::string {

    auto rels = kg.get_relations_of(subject);
    std::string answer;
    int count = 0;
    for (const auto& r : rels) {
        if (count > 0) answer += " ";
        answer += r.get().target_id();
        if (++count >= 2) break;  // 最多 2 个
    }
    return answer;
}

}  // namespace ai_learning::learning
