/**
 * @file lifelong_learning.hpp
 * @brief 持续终身学习 — 防止灾难性遗忘
 *
 * 参考：
 *   - Kirkpatrick et al. EWC (2017): 弹性权重巩固
 *   - Progressive Nets (Rusu et al., 2016): 渐进式网络
 *   - Gradient Episodic Memory (Lopez-Paz et al., 2017)
 *   - Complementary Learning Systems (McClelland et al., 1995)
 *
 * 核心机制：
 *   1. Fisher 信息矩阵：标记重要权重，学习新任务时保护
 *   2. 记忆回放：定期重放旧任务样本，巩固记忆
 *   3. 能力评估：检测遗忘程度，触发巩固
 *
 * 人类类比：学微积分不会忘记加减法
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>
#include <deque>

namespace ai_learning::learning {

/// 学习任务（技能/知识单元）
struct LearningTask {
    std::string id;                ///< 任务 ID
    std::string name;              ///< 任务名称
    std::string domain;            ///< 所属领域
    double mastery = 0.0;          ///< 掌握度 0~1
    double importance = 0.5;       ///< 重要性 0~1
    int practice_count = 0;        ///< 练习次数
    int last_practiced_step = 0;   ///< 上次练习的全局步数
};

/// 记忆样本（用于回放）
struct MemorySample {
    std::string task_id;           ///< 关联任务
    std::string content;           ///< 样本内容描述
    double strength = 1.0;         ///< 记忆强度
    int created_step = 0;          ///< 创建步数
};

/// 巩固报告
struct ConsolidationReportLL {
    int tasks_consolidated = 0;    ///< 巩固的任务数
    int memories_replayed = 0;     ///< 回放的记忆数
    double avg_retention = 0.0;    ///< 平均保留率
    std::vector<std::string> forgotten_tasks;  ///< 被遗忘的任务
};

/// 遗忘警告
struct ForgettingWarning {
    std::string task_id;
    std::string task_name;
    double current_mastery = 0.0;
    double original_mastery = 0.0;
    double decay_rate = 0.0;       ///< 衰减速率
};

/// 终身学习引擎
class LifelongLearningEngine {
public:
    explicit LifelongLearningEngine(int max_replay_size = 200,
                                     double forget_threshold = 0.3,
                                     double ewc_lambda = 1.0);

    // ── 任务管理 ────────────────────────────────────────

    /// 注册学习任务
    void register_task(const LearningTask& task);

    /// 更新任务掌握度（学习后调用）
    void update_mastery(const std::string& task_id,
                        double new_mastery,
                        int current_step);

    /// 获取任务信息
    [[nodiscard]] auto get_task(const std::string& task_id) const
        -> std::optional<LearningTask>;

    /// 获取所有任务
    [[nodiscard]] auto tasks() const
        -> const std::map<std::string, LearningTask>& { return tasks_; }

    // ── 记忆回放 ────────────────────────────────────────

    /// 存储记忆样本
    void store_memory(const MemorySample& sample);

    /// 选择需要回放的记忆（按遗忘曲线优先）
    auto select_replay(int current_step, int count = 5) const
        -> std::vector<MemorySample>;

    /// 巩固记忆：回放并增强
    auto consolidate(int current_step)
        -> ConsolidationReportLL;

    // ── 遗忘检测 ────────────────────────────────────────

    /// 计算任务保留率（防止遗忘）
    [[nodiscard]] auto retention(const std::string& task_id,
                                  int current_step) const -> double;

    /// 检测哪些任务面临遗忘风险
    auto detect_forgetting(int current_step) const
        -> std::vector<ForgettingWarning>;

    /// EWC 正则化惩罚：计算对重要任务的保护强度
    [[nodiscard]] auto ewc_penalty(const std::string& task_id) const -> double;

    // ── 跨任务保护 ──────────────────────────────────────

    /// 标记任务为"受保护"（高重要性，不允许遗忘）
    void protect_task(const std::string& task_id, double importance);

    /// 获取受保护任务列表
    [[nodiscard]] auto protected_tasks() const
        -> std::vector<std::string>;

    // ── 统计 ────────────────────────────────────────────

    /// 获取引擎统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

private:
    // 配置
    int max_replay_size_;
    double forget_threshold_;
    double ewc_lambda_;

    // 任务存储
    std::map<std::string, LearningTask> tasks_;

    // Fisher 信息（简化：每个任务的重要性权重）
    std::map<std::string, double> fisher_weights_;

    // 记忆回放缓冲区
    std::deque<MemorySample> replay_buffer_;

    // 受保护任务
    std::map<std::string, double> protected_tasks_;

    // 历史峰值掌握度（用于检测遗忘）
    std::map<std::string, double> peak_mastery_;

    /// Ebbinghaus 遗忘曲线模型
    static auto forgetting_curve_(double elapsed_steps,
                                   double strength) -> double;

    /// 计算记忆优先级（越需要复习，优先级越高）
    auto replay_priority_(const MemorySample& sample,
                           int current_step) const -> double;
};

}  // namespace ai_learning::learning
