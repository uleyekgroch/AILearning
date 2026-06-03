/**
 * @file verification.hpp
 * @brief 知识验证器 — 反馈闭环保证学习质量
 *
 * 验证流程：
 *   1. 从三元组生成验证问题
 *   2. 查询知识图谱检查答案
 *   3. 计算验证分数
 *
 * 对应 Python: Learner._verify_learned_knowledge()
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {
class KnowledgeGraph;  // 前向声明
}

namespace ai_learning::learning {

/// 单项验证结果
struct VerificationTest {
    std::string question;
    std::string expected;
    std::string actual;
    bool passed = false;
};

/// 验证报告
struct VerificationReport {
    bool passed = true;
    double score = 0.0;
    std::vector<VerificationTest> tests;
};

/// 知识验证器
class KnowledgeVerifier {
public:
    /// 生成验证问题并检查知识图谱
    [[nodiscard]] auto verify(
        const std::vector<std::tuple<std::string, std::string, std::string>>& triples,
        const domain::knowledge::KnowledgeGraph& kg) const -> VerificationReport;

private:
    /// 从三元组生成验证问题
    [[nodiscard]] static auto generate_question(
        const std::string& subject,
        const std::string& relation) -> std::string;

    /// 查询知识图谱获取答案
    [[nodiscard]] static auto query_answer(
        const std::string& subject,
        const domain::knowledge::KnowledgeGraph& kg) -> std::string;
};

}  // namespace ai_learning::learning
