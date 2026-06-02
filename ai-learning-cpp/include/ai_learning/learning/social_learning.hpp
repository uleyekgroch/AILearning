/**
 * @file social_learning.hpp
 * @brief 社会性/观察学习 — 从观察他人行为中学习
 *
 * 参考：
 *   - Bandura 社会学习理论 (1977)：观察→注意→保持→再现→动机
 *   - 逆强化学习 (Ng & Russell, 2000)：从专家行为推断奖励函数
 *   - 模仿学习 (Argall et al., 2009)：从示范中学习策略
 *   - 文化传递 (Boyd & Richerson, 1985)：社会信息传播
 *
 * 核心能力：
 *   1. 观察建模 — 从他人行为中提取策略模式
 *   2. 榜样评估 — 评估榜样的可信度和专业度
 *   3. 策略模仿 — 将观察到的策略适配到自身
 *   4. 知识传播 — 在群体中分享和接收知识
 *   5. 社会反馈 — 从他人反馈中调整行为
 *
 * 人类社会学习流程：
 *   观察老师骑自行车 → 注意关键动作 → 心理模拟 → 尝试 → 反馈调整
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>
#include <deque>
#include <set>

namespace ai_learning::learning {

/// 行为观察 — 记录一次观察到的行为
struct BehaviorObservation {
    std::string agent_id;              ///< 被观察者 ID
    std::string action;                ///< 行为描述
    std::string context;               ///< 行为发生的情境
    std::string domain;                ///< 所属领域
    std::vector<std::string> preconditions; ///< 行为前置条件
    std::vector<std::string> effects;       ///< 行为效果
    double outcome_quality = 0.0;      ///< 结果质量 0~1（好/坏）
};

/// 榜样评估 — 对一个榜样 agent 的评价
struct RoleModelProfile {
    std::string agent_id;              ///< 榜样 ID
    double expertise = 0.0;            ///< 专业度 0~1
    double trustworthiness = 0.0;      ///< 可信度 0~1
    double similarity = 0.0;           ///< 与自身的相似度 0~1
    std::vector<std::string> demonstrated_skills; ///< 展示过的技能
    std::map<std::string, double> domain_expertise; ///< 各领域专业度
    int observations_count = 0;        ///< 被观察次数
};

/// 提取的策略模式
struct StrategyPattern {
    std::string id;                    ///< 策略 ID
    std::string name;                  ///< 策略名称
    std::string domain;                ///< 适用领域
    std::vector<std::string> steps;    ///< 策略步骤
    std::vector<std::string> preconditions; ///< 前置条件
    std::vector<std::string> expected_outcomes; ///< 预期结果
    double observed_success_rate = 0.0; ///< 观察到的成功率
    int observation_count = 0;         ///< 观察次数
    std::string source_agent;          ///< 策略来源 agent
};

/// 社会反馈
struct SocialFeedback {
    std::string from_agent;            ///< 反馈来源
    std::string feedback_type;         ///< "praise" / "correction" / "suggestion"
    std::string content;               ///< 反馈内容
    std::string domain;                ///< 相关领域
    double weight = 1.0;               ///< 反馈权重（基于反馈者可信度）
};

/// 社会学习报告
struct SocialLearningReport {
    std::vector<StrategyPattern> learned_strategies; ///< 学到的策略
    std::vector<std::string> knowledge_gained;       ///< 获得的知识
    std::vector<std::string> failed_imitations;      ///< 失败的模仿
    double imitation_success_rate = 0.0;             ///< 模仿成功率
    int observations_processed = 0;                  ///< 处理的观察数
};

/// 社会学习配置
struct SocialLearningConfig {
    double min_observation_for_pattern = 3; ///< 形成策略模式的最少观察
    double expertise_threshold = 0.5;       ///< 榜样专业度阈值
    double similarity_weight = 0.3;         ///< 相似度权重
    double expertise_weight = 0.7;          ///< 专业度权重
    int max_role_models = 10;               ///< 最大榜样数
};

/// 社会学习引擎
class SocialLearningEngine {
public:
    explicit SocialLearningEngine(
        const SocialLearningConfig& config = SocialLearningConfig{});

    // ── 观察建模 ──────────────────────────────────────

    /// 观察一次行为（Bandura 注意阶段）
    auto observe(const BehaviorObservation& observation)
        -> SocialLearningReport;

    /// 批量观察
    auto observe_batch(const std::vector<BehaviorObservation>& observations)
        -> SocialLearningReport;

    /// 从观察中提取策略模式（Bandura 保持阶段）
    auto extract_pattern(const std::string& domain)
        -> std::optional<StrategyPattern>;

    // ── 榜样管理 ──────────────────────────────────────

    /// 评估一个榜样
    auto evaluate_model(const std::string& agent_id) const
        -> RoleModelProfile;

    /// 更新榜样评估（根据新的观察结果）
    void update_model_assessment(const std::string& agent_id,
                                  double outcome_quality);

    /// 选择最佳榜样（某领域）
    auto select_role_model(const std::string& domain) const
        -> std::optional<std::string>;

    // ── 策略模仿 ──────────────────────────────────────

    /// 尝试模仿一个策略（Bandura 再现阶段）
    auto imitate(const StrategyPattern& pattern,
                 const std::string& target_context)
        -> std::vector<std::string>;

    /// 将策略适配到自身情境
    auto adapt_strategy(const StrategyPattern& pattern,
                         const std::vector<std::string>& my_capabilities)
        -> StrategyPattern;

    // ── 社会反馈 ──────────────────────────────────────

    /// 接收反馈
    void receive_feedback(const SocialFeedback& feedback);

    /// 基于反馈调整策略
    auto adjust_from_feedback(const std::string& domain)
        -> std::vector<std::string>;

    // ── 知识传播 ──────────────────────────────────────

    /// 分享知识给"他人"（返回可分享的知识摘要）
    auto share_knowledge(const std::string& domain) const
        -> std::vector<std::string>;

    // ── 查询 ──────────────────────────────────────────

    /// 获取已学策略
    [[nodiscard]] auto learned_strategies() const
        -> const std::map<std::string, StrategyPattern>& {
        return strategies_;
    }

    /// 获取榜样列表
    [[nodiscard]] auto role_models() const
        -> const std::map<std::string, RoleModelProfile>& {
        return role_models_;
    }

    /// 获取反馈历史
    [[nodiscard]] auto feedback_history() const
        -> const std::deque<SocialFeedback>& {
        return feedback_history_;
    }

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const SocialLearningConfig& {
        return config_;
    }

private:
    SocialLearningConfig config_;

    /// 观察记录：domain → 行为列表
    std::map<std::string, std::vector<BehaviorObservation>> observations_;

    /// 榜样档案
    std::map<std::string, RoleModelProfile> role_models_;

    /// 已提取的策略模式
    std::map<std::string, StrategyPattern> strategies_;

    /// 反馈历史
    std::deque<SocialFeedback> feedback_history_;

    /// 统计
    int total_observations_ = 0;
    int successful_imitations_ = 0;
    int failed_imitations_ = 0;

    // ── 内部方法 ──────────────────────────────────────

    /// 从观察中归纳共同步骤
    auto extract_common_steps_(
        const std::vector<BehaviorObservation>& obs) const
        -> std::vector<std::string>;

    /// 计算榜样可信度分数
    auto model_credibility_(const std::string& agent_id) const -> double;

    /// 下一个策略 ID
    int next_strategy_id_ = 0;
    auto next_id_() -> std::string;
};

}  // namespace ai_learning::learning
