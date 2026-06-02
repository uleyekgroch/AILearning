/**
 * @file active_experimenter.hpp
 * @brief 主动实验设计 — 科学方法论的自主实现
 *
 * 参考：
 *   - Bayesian Experimental Design (Chaloner & Verdinelli, 1995)
 *   - Information-Theoretic Exploration (Still & Precup, 2012)
 *   - Active Learning (Settles, 2010): 不确定性采样
 *   - Self-Directed Learning (Bergstein, 2024): 主动探索
 *   - 科学方法论：观察→假设→预测→实验→分析→理论
 *
 * 核心能力：
 *   1. 假设生成 — 从观察中生成可验证的假设
 *   2. 实验设计 — 设计最大化信息增益的实验
 *   3. 结果分析 — 分析实验结果，接受/拒绝假设
 *   4. 理论构建 — 从验证的假设中构建理论
 *   5. 探索策略 — 主动探索策略（不确定性驱动）
 *
 * 人类科学方法：
 *   观察"金属都会导电" → 假设"所有金属都导电" →
 *   设计实验"测试汞是否导电" → 发现液态汞导电 → 支持假设
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>
#include <set>

namespace ai_learning::learning {

/// 假设
struct Hypothesis {
    std::string id;                         ///< 假设 ID
    std::string statement;                  ///< 假设陈述（"所有金属都导电"）
    std::string domain;                     ///< 所属领域
    double prior_confidence = 0.5;          ///< 先验置信度
    double posterior_confidence = 0.5;      ///< 后验置信度
    std::vector<std::string> supporting_evidence; ///< 支持证据
    std::vector<std::string> contradicting_evidence; ///< 反对证据
    bool tested = false;                    ///< 是否已测试
    bool falsified = false;                 ///< 是否已证伪
    double information_value = 0.0;         ///< 信息价值（测试它能获得多少信息）
};

/// 实验设计
struct ExperimentDesign {
    std::string id;                         ///< 实验 ID
    std::string target_hypothesis;          ///< 目标假设 ID
    std::string description;                ///< 实验描述
    std::string method;                     ///< 实验方法
    std::vector<std::string> steps;         ///< 实验步骤
    std::string expected_positive;          ///< 预期正面结果
    std::string expected_negative;          ///< 预期负面结果
    double expected_information_gain = 0.0; ///< 预期信息增益
    double cost = 1.0;                      ///< 实验成本
};

/// 实验结果
struct ExperimentResult {
    std::string experiment_id;
    std::string hypothesis_id;
    bool supports_hypothesis = false;       ///< 结果是否支持假设
    double confidence_delta = 0.0;          ///< 置信度变化
    std::string observation;                ///< 观察描述
    std::string analysis;                   ///< 分析描述
    double information_gain = 0.0;          ///< 实际信息增益
    double surprise = 0.0;                  ///< 惊讶度
};

/// 理论（由多个已验证假设组成）
struct Theory {
    std::string id;                         ///< 理论 ID
    std::string name;                       ///< 理论名称
    std::string domain;                     ///< 所属领域
    std::vector<std::string> verified_hypotheses; ///< 已验证的假设
    std::string description;                ///< 理论描述
    double confidence = 0.0;                ///< 综合置信度
    int supporting_experiments = 0;         ///< 支持实验数
    int total_experiments = 0;              ///< 总实验数
};

/// 探索状态
struct ExplorationState {
    std::string domain;                     ///< 探索领域
    double uncertainty = 0.5;               ///< 不确定性
    double information_density = 0.0;       ///< 信息密度
    int experiments_done = 0;               ///< 已完成实验
    double total_information_gained = 0.0;  ///< 累计信息增益
};

/// 主动实验设计配置
struct ExperimenterConfig {
    double hypothesis_threshold = 0.7;     ///< 接受假设的置信度阈值
    double falsification_threshold = 0.2;  ///< 证伪假设的阈值
    int max_active_hypotheses = 20;        ///< 最大活跃假设数
    double exploration_bonus = 0.1;         ///< 探索奖励
    double cost_sensitivity = 0.5;          ///< 成本敏感度
};

/// 主动实验设计引擎
class ActiveExperimenter {
public:
    explicit ActiveExperimenter(
        const ExperimenterConfig& config = ExperimenterConfig{});

    // ── 假设管理 ──────────────────────────────────────

    /// 从观察中生成假设
    auto generate_hypothesis(const std::string& observation,
                              const std::string& domain)
        -> Hypothesis;

    /// 手动注册假设
    void register_hypothesis(const Hypothesis& hypothesis);

    /// 获取所有活跃假设
    [[nodiscard]] auto active_hypotheses() const
        -> const std::map<std::string, Hypothesis>& {
        return hypotheses_;
    }

    /// 获取某领域的假设
    auto hypotheses_in_domain(const std::string& domain) const
        -> std::vector<Hypothesis>;

    // ── 实验设计 ──────────────────────────────────────

    /// 为假设设计实验（最大化信息增益）
    auto design_experiment(const std::string& hypothesis_id)
        -> std::optional<ExperimentDesign>;

    /// 自动选择最有价值的假设并设计实验
    auto auto_design_experiment()
        -> std::optional<ExperimentDesign>;

    /// 计算实验的预期信息增益
    auto expected_information_gain(const std::string& hypothesis_id) const
        -> double;

    // ── 实验执行 ──────────────────────────────────────

    /// 记录实验结果
    auto record_result(const ExperimentResult& result)
        -> std::string;

    /// 分析实验结果对假设的影响
    auto analyze_impact(const ExperimentResult& result) const
        -> std::string;

    // ── 理论构建 ──────────────────────────────────────

    /// 从已验证假设构建理论
    auto build_theory(const std::string& domain)
        -> std::optional<Theory>;

    /// 获取所有理论
    [[nodiscard]] auto theories() const
        -> const std::map<std::string, Theory>& {
        return theories_;
    }

    // ── 探索策略 ──────────────────────────────────────

    /// 获取探索状态
    [[nodiscard]] auto exploration_state(const std::string& domain) const
        -> ExplorationState;

    /// 推荐下一个探索目标
    auto recommend_exploration() const
        -> std::optional<std::string>;

    /// 计算某领域的不确定性
    auto domain_uncertainty(const std::string& domain) const -> double;

    // ── 查询 ──────────────────────────────────────────

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const ExperimenterConfig& {
        return config_;
    }

private:
    ExperimenterConfig config_;

    /// 假设库
    std::map<std::string, Hypothesis> hypotheses_;

    /// 理论库
    std::map<std::string, Theory> theories_;

    /// 实验结果历史
    std::vector<ExperimentResult> results_;

    /// 领域探索状态
    std::map<std::string, ExplorationState> exploration_states_;

    // ID 计数器
    int hypothesis_id_counter_ = 0;
    int experiment_id_counter_ = 0;
    int theory_id_counter_ = 0;

    // ── 内部方法 ──────────────────────────────────────

    /// 贝叶斯更新假设置信度
    auto bayesian_update_(Hypothesis& hyp, bool evidence_supports) const
        -> double;

    /// 计算假设的信息价值
    auto information_value_(const Hypothesis& hyp) const -> double;

    /// 生成假设 ID
    auto next_hyp_id_() -> std::string;
    auto next_exp_id_() -> std::string;
    auto next_theory_id_() -> std::string;
};

}  // namespace ai_learning::learning
