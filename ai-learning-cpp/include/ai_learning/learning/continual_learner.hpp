/**
 * @file continual_learner.hpp
 * @brief 持续终身学习 — 学新不忘旧（防灾难性遗忘）
 *
 * 参考：
 *   - Kirkpatrick et al. EWC (Elastic Weight Consolidation, PNAS 2017)
 *   - Progressive Neural Networks (Rusu et al., 2016)
 *   - PackNet (Mallya & Lazebnik, 2018)
 *   - Experience Replay (Robins, 1995)
 *   - Complementary Learning Systems (McClelland et al., 1995)
 *
 * 核心机制：
 *   1. Fisher 信息矩阵 — 评估每个参数对旧知识的重要性
 *   2. 弹性约束 — 学习新知识时，对重要参数施加惩罚
 *   3. 经验回放 — 间歇性复习旧知识
 *   4. 知识保护区 — 标记已掌握的核心知识，阻止覆写
 *   5. 灾难性遗忘检测 — 监控旧知识的退化程度
 *
 * 设计思路：
 *   不依赖具体神经网络参数（本项目用 flat array 权重），
 *   而是在"知识"层面进行保护：每个知识点有重要性分数，
 *   新知识学习会检查是否与受保护知识冲突。
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <deque>
#include <optional>
#include <functional>

namespace ai_learning::learning {

/// 知识保护等级
enum class ProtectionLevel {
    kNone = 0,       ///< 无保护（可自由覆写）
    kLow = 1,        ///< 低保护（允许小幅度修改）
    kMedium = 2,     ///< 中等保护（需要验证才能修改）
    kHigh = 3,       ///< 高保护（核心知识，仅允许扩展不允许修改）
};

/// 知识保护条目
struct KnowledgeProtection {
    std::string knowledge_id;         ///< 知识标识
    double importance = 0.0;          ///< Fisher 重要性 0~1
    ProtectionLevel level = ProtectionLevel::kNone;
    int times_reinforced = 0;         ///< 被巩固的次数
    double original_confidence = 0.0; ///< 原始置信度
    double current_confidence = 0.0;  ///< 当前置信度
};

/// 遗忘警报
struct ForgettingAlert {
    std::string knowledge_id;         ///< 正在被遗忘的知识
    double original_confidence = 0.0; ///< 原始置信度
    double current_confidence = 0.0;  ///< 当前置信度
    double degradation = 0.0;         ///< 退化量
    ProtectionLevel protection;       ///< 保护等级
    std::string recommendation;       ///< 建议（回放/巩固/提醒）
};

/// 经验回放缓冲区条目
struct ReplayEntry {
    std::vector<float> observation;   ///< 原始观测
    std::vector<float> prediction;    ///< 原始预测
    double importance = 0.0;          ///< 采样权重
    int replay_count = 0;             ///< 已回放次数
    std::string source_domain;        ///< 来源领域
};

/// 新旧知识冲突
struct KnowledgeConflict {
    std::string old_knowledge;        ///< 旧知识
    std::string new_knowledge;        ///< 新知识
    double conflict_score = 0.0;      ///< 冲突程度 0~1
    std::string resolution;           ///< 解决策略（"merge"/"override"/"keep_old"/"branch"）
    std::string reason;               ///< 理由
};

/// 持续学习统计
struct ContinualLearningStats {
    int total_knowledge_protected = 0;
    int total_replays = 0;
    int conflicts_detected = 0;
    int conflicts_resolved = 0;
    int forgetting_alerts = 0;
    double avg_knowledge_retention = 0.0;
};

/// 持续学习配置
struct ContinualConfig {
    double ewc_lambda = 1.0;          ///< EWC 正则化强度
    double importance_threshold = 0.5; ///< 重要性阈值（超过此值受保护）
    int replay_buffer_size = 100;     ///< 经验回放缓冲区大小
    int replay_interval = 10;         ///< 每 N 次学习回放一次
    double forgetting_threshold = 0.2; ///< 遗忘检测阈值
    bool auto_replay = true;          ///< 是否自动回放
};

/// 持续终身学习引擎
class ContinualLearner {
public:
    explicit ContinualLearner(
        const ContinualConfig& config = ContinualConfig{});

    // ── 知识保护（EWC 机制）─────────────────────────────

    /// 注册知识并评估其重要性
    /// @param knowledge_id 知识标识
    /// @param domain 所属领域
    /// @param confidence 当前置信度
    /// @param usage_count 已使用次数
    auto register_knowledge(const std::string& knowledge_id,
                            const std::string& domain,
                            double confidence,
                            int usage_count)
        -> KnowledgeProtection;

    /// 评估学习新知识时对旧知识的冲击
    auto assess_impact(const std::string& new_knowledge,
                       const std::string& domain)
        -> std::vector<KnowledgeConflict>;

    /// 解决新旧知识冲突
    auto resolve_conflict(const KnowledgeConflict& conflict)
        -> KnowledgeConflict;

    /// 更新知识的 Fisher 重要性
    void update_importance(const std::string& knowledge_id,
                           double new_evidence);

    // ── 经验回放 ──────────────────────────────────────

    /// 添加经验到回放缓冲区
    void add_replay_entry(const ReplayEntry& entry);

    /// 执行一轮经验回放
    /// @return 回放的知识数量
    auto replay(int count = 5) -> int;

    /// 采样需要回放的经验（优先重要和久未回放的）
    auto sample_replay(int count) const -> std::vector<ReplayEntry>;

    // ── 遗忘检测 ──────────────────────────────────────

    /// 检测遗忘（与上次记录比较）
    auto detect_forgetting() const -> std::vector<ForgettingAlert>;

    /// 计算知识保留率
    [[nodiscard]] auto retention_rate() const -> double;

    /// 计算特定领域的保留率
    [[nodiscard]] auto retention_rate(const std::string& domain) const -> double;

    // ── 学习协调 ──────────────────────────────────────

    /// 在学习新知识前调用：决定是否允许学习，以及如何保护旧知识
    auto prepare_for_new_learning(const std::string& new_domain,
                                  const std::vector<std::string>& new_knowledge)
        -> std::vector<KnowledgeConflict>;

    /// 学习完成后调用：更新保护状态
    void post_learning_update(const std::string& learned_domain,
                              double performance);

    // ── 查询 ──────────────────────────────────────────

    /// 获取知识的保护状态
    [[nodiscard]] auto get_protection(
        const std::string& knowledge_id) const
        -> std::optional<KnowledgeProtection>;

    /// 获取所有受保护的知识
    [[nodiscard]] auto protected_knowledge() const
        -> std::vector<KnowledgeProtection>;

    /// 获取统计
    [[nodiscard]] auto stats() const -> ContinualLearningStats;

    /// 获取配置
    [[nodiscard]] auto config() const -> const ContinualConfig& {
        return config_;
    }

    /// 获取回放缓冲区大小
    [[nodiscard]] auto replay_buffer_size() const -> int {
        return static_cast<int>(replay_buffer_.size());
    }

private:
    ContinualConfig config_;

    /// 知识保护注册表
    std::map<std::string, KnowledgeProtection> protection_registry_;

    /// 领域 → 该领域下的知识 ID
    std::map<std::string, std::vector<std::string>> domain_index_;

    /// 经验回放缓冲区
    std::deque<ReplayEntry> replay_buffer_;

    /// 学习历史（用于遗忘检测）
    std::map<std::string, std::deque<double>> confidence_history_;

    /// 统计
    int total_replays_ = 0;
    int conflicts_detected_ = 0;
    int conflicts_resolved_ = 0;
    int forgetting_alerts_ = 0;

    // ── 内部方法 ──────────────────────────────────────

    /// 计算两个知识点之间的冲突分数
    auto conflict_score_(const std::string& old_k,
                         const std::string& new_k) const -> double;

    /// 根据重要性确定保护等级
    auto determine_protection_level_(double importance) const
        -> ProtectionLevel;

    /// 优先级采样（结合重要性和时间）
    auto priority_scores_() const -> std::vector<double>;

    /// 检查回放缓冲区是否需要清理
    void trim_replay_buffer_();
};

}  // namespace ai_learning::learning
