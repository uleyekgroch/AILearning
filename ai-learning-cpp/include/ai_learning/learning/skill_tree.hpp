/**
 * @file skill_tree.hpp
 * @brief 技能树系统 — 知识的结构化组织和学习路径规划
 *
 * 参考：
 *   - Vygotsky's Zone of Proximal Development (ZPD, 1978)
 *   - Curriculum Learning (Bengio et al., 2009)
 *   - Mastery Learning (Bloom, 1968)
 *
 * 核心能力：
 *   1. 技能依赖图（DAG）— 前置条件关系
 *   2. 掌握度评估 — 每个技能的掌握程度 0~1
 *   3. 自适应课程 — ZPD 排序：不太简单也不太难
 *   4. 动态发现 — 通过探索发现新技能
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 技能节点
struct SkillNode {
    std::string id;                   ///< 唯一标识 "cpp_basics"
    std::string name;                 ///< 显示名称 "C++ 基础"
    std::string domain;               ///< 所属领域 "programming"

    /// 前置技能 ID 列表（必须先掌握）
    std::vector<std::string> prerequisites;

    /// 当前掌握度 0~1
    double mastery = 0.0;

    /// 难度估计 0~1
    double difficulty = 0.5;

    /// 验证任务（怎么测试是否掌握）
    std::vector<std::string> test_tasks;

    /// 学习资源描述
    std::vector<std::string> resources;
};

/// 学习路径（有序技能序列）
struct LearningPath {
    std::vector<SkillNode> steps;     ///< 有序步骤
    double total_difficulty = 0.0;    ///< 总难度
    int estimated_effort = 0;         ///< 预估学习量
    std::string target_skill;         ///< 目标技能 ID
};

/// 技能评估结果
struct SkillAssessment {
    std::string skill_id;
    double mastery_before = 0.0;
    double mastery_after = 0.0;
    double progress = 0.0;            ///< mastery_after - mastery_before
    bool passed = false;              ///< 是否达到掌握阈值
};

/// 技能树配置
struct SkillTreeConfig {
    double mastery_threshold = 0.6;   ///< 掌握阈值
    double zpd_low = 0.2;            ///< ZPD 下界
    double zpd_high = 0.7;           ///< ZPD 上界
};

/// 技能树 — DAG 结构，支持拓扑排序和 ZPD 课程
class SkillTree {
public:
    explicit SkillTree(const SkillTreeConfig& config = SkillTreeConfig{});

    // ── 技能管理 ──────────────────────────────────────────

    /// 添加技能
    void add_skill(const SkillNode& skill);

    /// 获取技能
    [[nodiscard]] auto get_skill(const std::string& id) const
        -> std::optional<SkillNode>;

    /// 获取所有技能
    [[nodiscard]] auto all_skills() const
        -> const std::map<std::string, SkillNode>& {
        return skills_;
    }

    /// 获取某领域的所有技能
    [[nodiscard]] auto skills_by_domain(const std::string& domain) const
        -> std::vector<SkillNode>;

    // ── 掌握度管理 ────────────────────────────────────────

    /// 更新技能掌握度
    void update_mastery(const std::string& skill_id, double delta);

    /// 设置技能掌握度
    void set_mastery(const std::string& skill_id, double mastery);

    /// 评估技能掌握度（通过测试任务验证）
    auto assess_skill(const std::string& skill_id,
                      double test_score) -> SkillAssessment;

    // ── 课程规划 ──────────────────────────────────────────

    /// 获取下一个应该学的技能（ZPD）
    [[nodiscard]] auto next_to_learn() const -> std::optional<SkillNode>;

    /// 获取学习路径（从当前到目标技能）
    [[nodiscard]] auto learning_path(const std::string& target_skill) const
        -> std::optional<LearningPath>;

    /// 获取所有可学技能（前置条件已满足 + 未掌握）
    [[nodiscard]] auto available_skills() const -> std::vector<SkillNode>;

    /// 获取 ZPD 内的技能（难度适中）
    [[nodiscard]] auto zpd_skills() const -> std::vector<SkillNode>;

    // ── 依赖查询 ──────────────────────────────────────────

    /// 检查前置条件是否全部满足
    [[nodiscard]] auto prerequisites_met(const std::string& skill_id) const
        -> bool;

    /// 获取所有依赖（递归）
    [[nodiscard]] auto all_dependencies(const std::string& skill_id) const
        -> std::vector<std::string>;

    // ── 动态发现 ──────────────────────────────────────────

    /// 动态发现新技能（从已有技能推导）
    auto discover_skill(const std::string& suggested_id,
                        const std::string& domain,
                        const std::vector<std::string>& prereqs)
        -> SkillNode;

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const SkillTreeConfig& {
        return config_;
    }

private:
    SkillTreeConfig config_;
    std::map<std::string, SkillNode> skills_;

    /// 拓扑排序（BFS）
    [[nodiscard]] auto topological_sort_() const -> std::vector<std::string>;

    /// 检查循环依赖
    [[nodiscard]] auto has_cycle_(const std::string& start,
                                   std::vector<std::string>& visited) const
        -> bool;
};

}  // namespace ai_learning::learning
