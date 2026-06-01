/**
 * @file problem_solver.hpp
 * @brief 问题求解器 — 用知识解决专业问题
 *
 * 参考：
 *   - Polyá's "How to Solve It" (1945): 四步求解法
 *   - Newell & Simon's GPS (1972): 通用问题求解器
 *   - EPAM/SOAR 认知架构：识别-行动循环
 *
 * 求解流程（Polyá 四步法）：
 *   1. Understand — 理解问题（识别领域、类型、约束）
 *   2. Plan — 制定方案（查找相关技能、生成候选解）
 *   3. Execute — 执行方案（逐步推理 + 沙箱验证）
 *   4. Review — 回顾反思（验证结果、学习新知识）
 */
#pragma once

#include "ai_learning/learning/skill_tree.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 问题分析结果
struct ProblemProfile {
    std::string domain;                 ///< 识别的领域 "math" / "programming" / ...
    std::string problem_type;           ///< 问题类型 "computation" / "debugging" / ...
    std::vector<std::string> concepts;  ///< 涉及的概念
    std::vector<std::string> required_skills;  ///< 所需技能 ID
    double estimated_difficulty = 0.5;  ///< 难度估计
    std::string reformulation;          ///< 重新表述的问题
};

/// 求解步骤
struct SolutionStep {
    std::string action;                 ///< 动作描述
    std::string reasoning;              ///< 推理过程
    std::string code;                   ///< 相关代码（可选）
    std::map<std::string, std::string> metadata;  ///< 额外信息
};

/// 求解方案
struct Solution {
    std::vector<SolutionStep> steps;    ///< 求解步骤
    std::string answer;                 ///< 最终答案
    double confidence = 0.0;            ///< 置信度
    std::string reasoning_trace;        ///< 完整推理过程
    bool verified = false;              ///< 是否已验证
    std::vector<std::string> skills_used;  ///< 使用了的技能
};

/// 验证结果
struct VerificationResult {
    bool correct = false;               ///< 是否正确
    double score = 0.0;                 ///< 评分 0~1
    std::string feedback;               ///< 反馈信息
    std::vector<std::string> errors;    ///< 错误列表
};

/// 问题求解器配置
struct ProblemSolverConfig {
    int max_reasoning_steps = 20;       ///< 最大推理步数
    double confidence_threshold = 0.5;  ///< 置信度阈值
    bool verify_solutions = true;       ///< 是否验证方案
};

/// 问题求解器
class ProblemSolver {
public:
    explicit ProblemSolver(
        const ProblemSolverConfig& config = ProblemSolverConfig{});

    // ── 核心接口 ──────────────────────────────────────────

    /// 求解问题（Polyá 四步法）
    /// @param problem_description 自然语言问题描述
    /// @param skill_tree 技能树（查找相关技能）
    /// @param known_facts 已知事实（从知识图谱获取）
    auto solve(const std::string& problem_description,
               const SkillTree& skill_tree,
               const std::vector<std::string>& known_facts)
        -> Solution;

    // ── 分步接口 ──────────────────────────────────────────

    /// Step 1: 理解问题
    auto understand(const std::string& description) -> ProblemProfile;

    /// Step 2: 制定方案
    auto plan_solution(const ProblemProfile& profile,
                       const SkillTree& skill_tree,
                       const std::vector<std::string>& known_facts)
        -> std::vector<SolutionStep>;

    /// Step 3: 执行方案（生成答案）
    auto execute_plan(const std::vector<SolutionStep>& steps,
                      const ProblemProfile& profile)
        -> Solution;

    /// Step 4: 回顾验证
    auto review(const Solution& solution,
                const std::string& original_problem)
        -> VerificationResult;

    // ── 学习反馈 ──────────────────────────────────────────

    /// 从问题求解中学习新技能
    auto learn_from_solution(const Solution& solution,
                             const ProblemProfile& profile,
                             SkillTree& skill_tree)
        -> std::vector<std::string>;

    // ── 配置 ──────────────────────────────────────────────

    [[nodiscard]] auto config() const -> const ProblemSolverConfig& {
        return config_;
    }

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

private:
    ProblemSolverConfig config_;

    // 统计
    int problems_solved_ = 0;
    int problems_failed_ = 0;
    double total_confidence_ = 0.0;

    /// 识别问题领域
    auto identify_domain_(const std::string& desc) const -> std::string;

    /// 提取涉及概念
    auto extract_concepts_(const std::string& desc) const
        -> std::vector<std::string>;

    /// 查找相关技能
    auto find_relevant_skills_(const std::string& domain,
                                const SkillTree& skill_tree) const
        -> std::vector<std::string>;

    /// 从已知事实推理
    auto reason_from_facts_(const std::vector<std::string>& facts,
                             const std::string& question) const
        -> std::string;
};

}  // namespace ai_learning::learning
