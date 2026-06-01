/**
 * @file problem_solver_milestones.cpp
 * @brief 问题求解器 + 发展里程碑系统 实现
 */

#include "ai_learning/learning/problem_solver.hpp"
#include "ai_learning/learning/development_milestones.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace ai_learning::learning {

// ═══════════════════════════════════════════════════════════════════
// ProblemSolver
// ═══════════════════════════════════════════════════════════════════

ProblemSolver::ProblemSolver(const ProblemSolverConfig& config)
    : config_(config) {}

auto ProblemSolver::solve(const std::string& problem_description,
                           const SkillTree& skill_tree,
                           const std::vector<std::string>& known_facts)
    -> Solution {

    // Step 1: 理解
    auto profile = understand(problem_description);

    // Step 2: 规划
    auto steps = plan_solution(profile, skill_tree, known_facts);

    // Step 3: 执行
    auto solution = execute_plan(steps, profile);

    // Step 4: 验证
    if (config_.verify_solutions) {
        auto verification = review(solution, problem_description);
        solution.verified = verification.correct;
        solution.confidence = verification.score;
    }

    // 记录使用了的技能
    solution.skills_used = find_relevant_skills_(profile.domain, skill_tree);

    // 更新统计
    if (solution.confidence >= config_.confidence_threshold) {
        ++problems_solved_;
    } else {
        ++problems_failed_;
    }
    total_confidence_ += solution.confidence;

    return solution;
}

auto ProblemSolver::understand(const std::string& description)
    -> ProblemProfile {

    ProblemProfile profile;
    profile.domain = identify_domain_(description);
    profile.concepts = extract_concepts_(description);
    profile.reformulation = description;

    // 查找所需技能
    for (const auto& c : profile.concepts) {
        profile.required_skills.push_back(c);
    }

    // 难度估计
    profile.estimated_difficulty = std::min(1.0, description.size() / 100.0);

    if (profile.problem_type.empty()) {
        profile.problem_type = "general";
    }

    return profile;
}

auto ProblemSolver::plan_solution(
    const ProblemProfile& profile,
    const SkillTree& skill_tree,
    const std::vector<std::string>& known_facts)
    -> std::vector<SolutionStep> {

    std::vector<SolutionStep> steps;

    // Step 1: 分析已知条件
    steps.push_back({
        "analyze",
        "分析问题: " + profile.reformulation + " (领域: " + profile.domain + ")",
        "",
        {{"domain", profile.domain}}
    });

    // Step 2: 检索相关知识
    auto relevant_skills = find_relevant_skills_(profile.domain, skill_tree);
    std::string skills_str;
    for (const auto& s : relevant_skills) {
        if (!skills_str.empty()) skills_str += ", ";
        skills_str += s;
    }
    steps.push_back({
        "retrieve",
        "相关技能: " + (skills_str.empty() ? "无" : skills_str),
        "",
        {{"skills_count", std::to_string(relevant_skills.size())}}
    });

    // Step 3: 推理
    auto reasoning = reason_from_facts_(known_facts, profile.reformulation);
    steps.push_back({
        "reason",
        reasoning,
        "",
        {{"facts_used", std::to_string(known_facts.size())}}
    });

    return steps;
}

auto ProblemSolver::execute_plan(
    const std::vector<SolutionStep>& steps,
    const ProblemProfile& /*profile*/)
    -> Solution {

    Solution solution;

    std::string reasoning_trace;
    for (const auto& step : steps) {
        solution.steps.push_back(step);
        reasoning_trace += step.action + ": " + step.reasoning + "\n";
    }

    // 从推理步骤中提取答案
    if (!steps.empty()) {
        solution.answer = steps.back().reasoning;
    }

    solution.reasoning_trace = reasoning_trace;
    solution.confidence = steps.size() > 2 ? 0.7 : 0.3;

    return solution;
}

auto ProblemSolver::review(const Solution& solution,
                            const std::string& /*original_problem*/)
    -> VerificationResult {

    VerificationResult result;

    if (solution.answer.empty()) {
        result.correct = false;
        result.score = 0.0;
        result.feedback = "No answer generated";
        return result;
    }

    // 基本验证：答案非空且有推理过程
    result.score = 0.5;
    result.correct = true;

    if (!solution.reasoning_trace.empty()) {
        result.score += 0.2;
    }
    if (solution.steps.size() >= 3) {
        result.score += 0.1;
    }

    result.score = std::min(1.0, result.score);
    result.feedback = "Solution verified with " +
                      std::to_string(static_cast<int>(result.score * 100)) +
                      "% confidence";

    return result;
}

auto ProblemSolver::learn_from_solution(
    const Solution& solution,
    const ProblemProfile& profile,
    SkillTree& skill_tree)
    -> std::vector<std::string> {

    std::vector<std::string> learned_skills;

    for (const auto& skill_id : solution.skills_used) {
        if (skill_tree.get_skill(skill_id).has_value()) {
            skill_tree.update_mastery(skill_id, 0.1);
            learned_skills.push_back(skill_id);
        }
    }

    // 如果用了未知技能，发现新技能
    if (solution.skills_used.empty() && !profile.domain.empty()) {
        auto new_skill = skill_tree.discover_skill(
            profile.domain + "_skill", profile.domain, {});
        learned_skills.push_back(new_skill.id);
    }

    return learned_skills;
}

auto ProblemSolver::stats() const -> std::map<std::string, double> {
    double avg_conf = (problems_solved_ + problems_failed_) > 0
        ? total_confidence_ / (problems_solved_ + problems_failed_) : 0.0;
    return {
        {"problems_solved", static_cast<double>(problems_solved_)},
        {"problems_failed", static_cast<double>(problems_failed_)},
        {"avg_confidence", avg_conf},
    };
}

auto ProblemSolver::identify_domain_(const std::string& desc) const
    -> std::string {
    // 关键词匹配
    if (desc.find("计算") != std::string::npos ||
        desc.find("加") != std::string::npos ||
        desc.find("减") != std::string::npos ||
        desc.find("乘") != std::string::npos ||
        desc.find("除") != std::string::npos ||
        desc.find("数") != std::string::npos ||
        desc.find("数学") != std::string::npos ||
        desc.find("方程") != std::string::npos) {
        return "math";
    }

    if (desc.find("代码") != std::string::npos ||
        desc.find("编程") != std::string::npos ||
        desc.find("函数") != std::string::npos ||
        desc.find("循环") != std::string::npos ||
        desc.find("程序") != std::string::npos ||
        desc.find("算法") != std::string::npos) {
        return "programming";
    }

    return "general";
}

auto ProblemSolver::extract_concepts_(const std::string& desc) const
    -> std::vector<std::string> {
    // 简化的概念提取：按分隔符和中文分隔词分词
    std::vector<std::string> concepts;
    std::string current;

    // 中文分隔词
    const std::vector<std::string> delimiters = {
        "，", "、", "的", "和", "与", "。", "；", "："
    };

    for (size_t i = 0; i < desc.size(); ) {
        bool is_delim = false;

        // 检查 ASCII 分隔符
        char c = desc[i];
        if (c == ' ' || c == ',') {
            is_delim = true;
            ++i;
        } else {
            // 检查中文分隔词（UTF-8 编码，每个中文字 3 字节）
            for (const auto& d : delimiters) {
                if (desc.substr(i, d.size()) == d) {
                    is_delim = true;
                    i += d.size();
                    break;
                }
            }
        }

        if (is_delim) {
            if (current.size() >= 2) {
                concepts.push_back(current);
            }
            current.clear();
        } else {
            current += desc[i];
            ++i;
        }
    }
    if (current.size() >= 2) {
        concepts.push_back(current);
    }

    // 限制数量
    if (concepts.size() > 10) {
        concepts.resize(10);
    }

    return concepts;
}

auto ProblemSolver::find_relevant_skills_(
    const std::string& domain,
    const SkillTree& skill_tree) const
    -> std::vector<std::string> {

    auto domain_skills = skill_tree.skills_by_domain(domain);
    std::vector<std::string> ids;
    for (const auto& s : domain_skills) {
        ids.push_back(s.id);
    }
    return ids;
}

auto ProblemSolver::reason_from_facts_(
    const std::vector<std::string>& facts,
    const std::string& question) const
    -> std::string {

    if (facts.empty()) {
        return "No known facts available for: " + question;
    }

    // 找到与问题最相关的事实
    std::string best_fact;
    int best_overlap = 0;
    for (const auto& f : facts) {
        int overlap = 0;
        for (size_t i = 0; i + 1 < question.size(); ++i) {
            if (f.find(question.substr(i, 2)) != std::string::npos) {
                ++overlap;
            }
        }
        if (overlap > best_overlap) {
            best_overlap = overlap;
            best_fact = f;
        }
    }

    if (best_overlap > 0) {
        return "Based on known fact: " + best_fact;
    }

    return "Using general reasoning with " +
           std::to_string(facts.size()) + " facts for: " + question;
}

// ═══════════════════════════════════════════════════════════════════
// DevelopmentMilestones
// ═══════════════════════════════════════════════════════════════════

DevelopmentMilestones::DevelopmentMilestones() = default;

void DevelopmentMilestones::add_milestone(const Milestone& milestone) {
    milestones_[milestone.id] = milestone;
}

auto DevelopmentMilestones::get_milestone(const std::string& id) const
    -> std::optional<Milestone> {
    auto it = milestones_.find(id);
    if (it != milestones_.end()) return it->second;
    return std::nullopt;
}

auto DevelopmentMilestones::check_milestones(const SkillTree& skill_tree)
    -> std::vector<MilestoneEvent> {

    std::vector<MilestoneEvent> new_events;

    for (auto& [id, m] : milestones_) {
        if (m.achieved) continue;

        if (check_milestone(id, skill_tree)) {
            m.achieved = true;
            m.achieved_at = current_iteration_++;

            MilestoneEvent event;
            event.milestone_id = id;
            event.milestone_name = m.name;
            event.reward = m.reward;
            event.iteration = m.achieved_at;

            for (const auto& [skill_id, _] : m.conditions) {
                event.skills_mastered.push_back(skill_id);
            }

            new_events.push_back(event);
            events_.push_back(event);
        }
    }

    return new_events;
}

auto DevelopmentMilestones::check_milestone(
    const std::string& id, const SkillTree& skill_tree) const
    -> bool {

    auto it = milestones_.find(id);
    if (it == milestones_.end()) return false;
    if (it->second.achieved) return true;

    for (const auto& [skill_id, min_mastery] : it->second.conditions) {
        auto skill = skill_tree.get_skill(skill_id);
        if (!skill.has_value()) return false;
        if (skill->mastery < min_mastery) return false;
    }

    return true;
}

auto DevelopmentMilestones::next_milestone() const
    -> std::optional<Milestone> {

    std::optional<Milestone> result;
    int min_level = 999;

    for (const auto& [_, m] : milestones_) {
        if (!m.achieved && m.level < min_level) {
            min_level = m.level;
            result = m;
        }
    }

    return result;
}

auto DevelopmentMilestones::nearest_milestone(const SkillTree& skill_tree) const
    -> std::optional<Milestone> {

    std::optional<Milestone> result;
    double best_progress = -1.0;

    for (const auto& [_, m] : milestones_) {
        if (m.achieved) continue;
        double prog = milestone_progress_(m, skill_tree);
        if (prog > best_progress) {
            best_progress = prog;
            result = m;
        }
    }

    return result;
}

auto DevelopmentMilestones::progress(const SkillTree& /*skill_tree*/) const
    -> ProgressSnapshot {

    ProgressSnapshot snap;
    snap.milestones_total = static_cast<int>(milestones_.size());

    for (const auto& [_, m] : milestones_) {
        if (m.achieved) {
            ++snap.milestones_achieved;
        }
    }

    snap.total_progress = snap.milestones_total > 0
        ? static_cast<double>(snap.milestones_achieved) / snap.milestones_total
        : 0.0;

    // 各领域进度
    std::map<std::string, std::pair<int, int>> domain_counts;  // achieved, total
    for (const auto& [_, m] : milestones_) {
        auto& [ach, tot] = domain_counts[m.domain];
        ++tot;
        if (m.achieved) ++ach;
    }
    for (const auto& [domain, counts] : domain_counts) {
        auto [ach, tot] = counts;
        snap.domain_progress[domain] = tot > 0 ? static_cast<double>(ach) / tot : 0.0;
    }

    // 等级描述
    if (snap.total_progress < 0.2) snap.current_level = "Beginner";
    else if (snap.total_progress < 0.4) snap.current_level = "Elementary";
    else if (snap.total_progress < 0.6) snap.current_level = "Intermediate";
    else if (snap.total_progress < 0.8) snap.current_level = "Advanced";
    else snap.current_level = "Expert";

    return snap;
}

auto DevelopmentMilestones::stats() const -> std::map<std::string, double> {
    int achieved = 0;
    for (const auto& [_, m] : milestones_) {
        if (m.achieved) ++achieved;
    }
    return {
        {"total_milestones", static_cast<double>(milestones_.size())},
        {"achieved", static_cast<double>(achieved)},
        {"events", static_cast<double>(events_.size())},
    };
}

void DevelopmentMilestones::init_programming_milestones() {
    add_milestone({"prog_hello", "Hello World",
                   "能够编写第一个程序", "programming",
                   0, {}, 1.0, false, -1});
    add_milestone({"prog_vars", "变量使用",
                   "理解和使用变量", "programming",
                   1, {}, 1.5, false, -1});
    add_milestone({"prog_loops", "循环控制",
                   "掌握循环结构", "programming",
                   1, {}, 2.0, false, -1});
    add_milestone({"prog_functions", "函数定义",
                   "能够定义和调用函数", "programming",
                   2, {}, 2.5, false, -1});
    add_milestone({"prog_algorithms", "基本算法",
                   "实现基本排序和搜索算法", "programming",
                   3, {}, 5.0, false, -1});
}

void DevelopmentMilestones::init_math_milestones() {
    add_milestone({"math_count", "数数入门",
                   "掌握基本的计数能力", "math",
                   0, {}, 1.0, false, -1});
    add_milestone({"math_add", "加法掌握",
                   "掌握加法运算", "math",
                   0, {}, 1.5, false, -1});
    add_milestone({"math_sub", "减法掌握",
                   "掌握减法运算", "math",
                   1, {}, 1.5, false, -1});
    add_milestone({"math_mul", "乘法掌握",
                   "掌握乘法运算", "math",
                   2, {}, 2.0, false, -1});
}

void DevelopmentMilestones::init_general_milestones() {
    add_milestone({"gen_first_concept", "第一个概念",
                   "理解第一个概念", "general",
                   0, {}, 1.0, false, -1});
    add_milestone({"gen_first_relation", "第一个关系",
                   "建立实体间的第一个关系", "general",
                   0, {}, 1.5, false, -1});
    add_milestone({"gen_first_inference", "第一次推理",
                   "完成第一次成功推理", "general",
                   1, {}, 2.0, false, -1});
}

auto DevelopmentMilestones::milestone_progress_(
    const Milestone& m, const SkillTree& skill_tree) const
    -> double {

    if (m.conditions.empty()) return 0.5;  // 无条件 → 50% 默认

    double total = 0.0;
    for (const auto& [skill_id, min_mastery] : m.conditions) {
        auto skill = skill_tree.get_skill(skill_id);
        if (!skill.has_value()) continue;
        total += std::min(1.0, skill->mastery / min_mastery);
    }

    return total / static_cast<double>(m.conditions.size());
}

}  // namespace ai_learning::learning
