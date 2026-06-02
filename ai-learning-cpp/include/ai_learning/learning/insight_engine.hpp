/**
 * @file insight_engine.hpp
 * @brief 知识重组/顿悟 — 重新组织知识产生创造性突破
 *
 * 参考：
 *   - Ohlsson 顿悟理论 (2011)：约束释放 + 重新编码 + 扩展
 *   - Gestalt 重组 (Köhler, 1925)：看问题的新方式
 *   - 组合创造性 (Mednick, 1962)：远程联想测试
 *   - 结构映射与类比 (Gentner, 1983)：跨域重解释
 *   - 扩散激活理论 (Mednick, 1962)：创造性联想
 *
 * 核心能力：
 *   1. 知识重组 — 重新排列知识元素，发现隐藏结构
 *   2. 远程联想 — 连接看似无关的概念
 *   3. 约束释放 — 打破思维定势
 *   4. 重组验证 — 检验顿悟是否合理
 *   5. 创造性评估 — 评估新想法的原创性和实用性
 *
 * 人类顿悟机制：
 *   问题：9 点连线 ──→ 约束："线不能超出点阵"
 *   顿悟：释放约束 → 突然看到解法
 *
 *   "苹果掉落" + "月亮绕地球" → 万有引力（Newton 的顿悟）
 *   "洗澡水溢出" + "王冠真假" → 浮力原理（Archimedes 的顿悟）
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>
#include <set>

namespace ai_learning::learning {

/// 知识元素 — 重组的基本单元
struct KnowledgeElement {
    std::string id;                         ///< 元素 ID
    std::string content;                    ///< 内容描述
    std::string domain;                     ///< 所属领域
    std::vector<std::string> connections;   ///< 与其他元素的显式连接
    std::map<std::string, double> features; ///< 数值特征
};

/// 思维约束 — 隐含假设
struct MentalConstraint {
    std::string id;                         ///< 约束 ID
    std::string description;                ///< 约束描述（"线不能超出边界"）
    std::string source;                     ///< 来源（经验/教科书/假设）
    double strength = 1.0;                  ///< 约束强度 0~1
    bool is_active = true;                  ///< 是否激活
};

/// 顿悟事件 — 一次重组产生的新认知
struct InsightEvent {
    std::string id;                         ///< 事件 ID
    std::string trigger;                    ///< 触发因素
    std::vector<std::string> elements_combined; ///< 组合的元素
    std::string old_perspective;            ///< 旧视角
    std::string new_perspective;            ///< 新视角
    std::string constraint_released;        ///< 释放的约束（如果有）
    double surprise_level = 0.0;            ///< 惊讶度 0~1
    double confidence = 0.0;                ///< 置信度 0~1
    bool verified = false;                  ///< 是否已验证
    double creativity_score = 0.0;          ///< 创造性评分
    double utility_score = 0.0;             ///< 实用性评分
};

/// 重组方案 — 一次知识重组的完整方案
struct ReorganizationPlan {
    std::vector<std::string> elements_to_rearrange; ///< 需要重组的元素
    std::string new_structure;              ///< 新结构描述
    std::string method;                     ///< 重组方法（"constraint_release"/"remote_association"/"perspective_shift"）
    double estimated_novelty = 0.0;         ///< 估计新颖度
};

/// 创造性评估
struct CreativityAssessment {
    double novelty = 0.0;                   ///< 新颖度 0~1
    double utility = 0.0;                   ///< 实用性 0~1
    double surprise = 0.0;                  ///< 惊讶度 0~1
    double elegance = 0.0;                  ///< 优雅度 0~1
    double overall = 0.0;                   ///< 综合评分
    std::string verdict;                    ///< "breakthrough" / "incremental" / "trivial"
};

/// 顿悟引擎配置
struct InsightConfig {
    double novelty_threshold = 0.5;         ///< 新颖度阈值
    double confidence_threshold = 0.3;      ///< 验证置信度阈值
    int max_reorganization_attempts = 10;   ///< 最大重组尝试次数
    int max_constraints = 20;              ///< 最大约束数
    double constraint_release_threshold = 0.5; ///< 约束释放阈值
};

/// 知识重组/顿悟引擎
class InsightEngine {
public:
    explicit InsightEngine(
        const InsightConfig& config = InsightConfig{});

    // ── 知识管理 ──────────────────────────────────────

    /// 注册知识元素
    void register_element(const KnowledgeElement& element);

    /// 注册思维约束
    void add_constraint(const MentalConstraint& constraint);

    /// 获取所有知识元素
    [[nodiscard]] auto elements() const
        -> const std::map<std::string, KnowledgeElement>& {
        return elements_;
    }

    /// 获取所有约束
    [[nodiscard]] auto constraints() const
        -> const std::map<std::string, MentalConstraint>& {
        return constraints_;
    }

    // ── 重组触发 ──────────────────────────────────────

    /// 尝试产生顿悟（主动触发）
    auto try_insight(const std::string& problem_context)
        -> std::optional<InsightEvent>;

    /// 从两个看似无关的元素中发现隐藏联系（Mednick 远程联想）
    auto remote_association(const std::string& element_a,
                             const std::string& element_b)
        -> std::optional<InsightEvent>;

    /// 释放约束并重新求解（Ohlsson 约束释放）
    auto release_constraint(const std::string& constraint_id,
                             const std::string& problem)
        -> std::optional<InsightEvent>;

    /// 视角转换（Gestalt 重组）
    auto perspective_shift(const std::string& problem,
                            const std::string& new_viewpoint)
        -> std::optional<InsightEvent>;

    // ── 重组方案生成 ──────────────────────────────────────

    /// 生成重组方案
    auto propose_reorganization(const std::string& domain)
        -> std::vector<ReorganizationPlan>;

    /// 执行重组方案
    auto execute_reorganization(const ReorganizationPlan& plan)
        -> InsightEvent;

    // ── 验证与评估 ──────────────────────────────────────

    /// 验证顿悟是否自洽
    auto verify_insight(const InsightEvent& insight) -> bool;

    /// 评估创造性
    auto assess_creativity(const InsightEvent& insight) const
        -> CreativityAssessment;

    // ── 查询 ──────────────────────────────────────────

    /// 获取所有顿悟事件
    [[nodiscard]] auto insights() const
        -> const std::vector<InsightEvent>& {
        return insights_;
    }

    /// 获取被释放的约束
    [[nodiscard]] auto released_constraints() const
        -> const std::vector<std::string>& {
        return released_constraints_;
    }

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const InsightConfig& {
        return config_;
    }

private:
    InsightConfig config_;

    /// 知识元素库
    std::map<std::string, KnowledgeElement> elements_;

    /// 思维约束库
    std::map<std::string, MentalConstraint> constraints_;

    /// 顿悟事件历史
    std::vector<InsightEvent> insights_;

    /// 已释放的约束
    std::vector<std::string> released_constraints_;

    // 统计
    int total_attempts_ = 0;
    int successful_insights_ = 0;
    int constraints_released_ = 0;

    // ── 内部方法 ──────────────────────────────────────

    /// 计算两个元素之间的远程联想分数
    auto association_score_(const std::string& id_a,
                             const std::string& id_b) const -> double;

    /// 计算新颖度
    auto novelty_score_(const InsightEvent& insight) const -> double;

    /// 计算惊讶度
    auto surprise_score_(const InsightEvent& insight) const -> double;

    /// 下一个 ID
    int next_insight_id_ = 0;
    auto next_id_() -> std::string;
};

}  // namespace ai_learning::learning
