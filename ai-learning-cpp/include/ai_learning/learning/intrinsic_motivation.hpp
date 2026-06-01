/**
 * @file intrinsic_motivation.hpp
 * @brief 内在动机引擎 — 让系统"想要"学习
 *
 * 参考：
 *   - H-GRAIL (Romero et al., IEEE TCDS 2025): 分层内在动机
 *   - D2A (ICLR 2025): 欲望驱动的自主行为
 *   - Self-Determination Theory (Deci & Ryan, 1985): 三大基本需求
 *
 * 五种内在动机：
 *   1. Curiosity（好奇心）：预测误差 → 探索未知
 *   2. Competence（掌握欲）：能力提升 → 追求精通
 *   3. Autonomy（自主性）：自己选择 → 内在满足
 *   4. Social（社交性）：帮助/合作 → 归属感（预留）
 *   5. Achievement（成就感）：完成目标 → 自我效能
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 内在动机类型
enum class MotiveType {
    kCuriosity,    ///< 好奇心：预测误差 → 探索未知
    kCompetence,   ///< 掌握欲：能力提升 → 追求精通
    kAutonomy,     ///< 自主性：自己选择 → 内在满足
    kSocial,       ///< 社交性：帮助/合作 → 归属感
    kAchievement,  ///< 成就感：完成目标 → 自我效能
};

/// 学习目标
struct LearningGoal {
    std::string topic;               ///< 学习主题
    std::string domain;              ///< 所属领域
    double difficulty = 0.5;         ///< 难度估计 0~1
    double estimated_value = 0.0;    ///< 预期价值（动机评分）
    MotiveType primary_motive;       ///< 主导动机
    std::string description;         ///< 目标描述
};

/// 动机状态
struct MotiveState {
    MotiveType type;
    double drive_level = 0.5;        ///< 驱动水平 0~1（越高越想行动）
    double satisfaction = 0.0;       ///< 满足度 0~1
    double decay_rate = 0.01;        ///< 自然衰减率
    std::string name;                ///< 人类可读名称
};

/// 学习结果（用于更新动机）
struct LearningOutcome {
    std::string topic;
    double progress = 0.0;           ///< 进步程度 0~1
    double surprise = 0.0;           ///< 意外程度 0~1
    bool mastery_improved = false;   ///< 掌握度是否提升
    bool goal_completed = false;     ///< 目标是否完成
};

/// 内在动机引擎
class IntrinsicMotivationEngine {
public:
    IntrinsicMotivationEngine();

    // ── 目标生成 ──────────────────────────────────────────

    /// 生成当前最有动机去学的目标
    /// @param known_topics 已知主题列表
    /// @param curiosity_signal 当前好奇心信号（来自预测引擎）0~1
    /// @param mastery_map 各领域掌握度映射
    auto generate_goal(const std::vector<std::string>& known_topics,
                       double curiosity_signal,
                       const std::map<std::string, double>& mastery_map)
        -> LearningGoal;

    /// 评估某个目标值不值得追求
    auto evaluate_goal(const LearningGoal& goal) const -> double;

    /// 从多个候选目标中选择最佳
    auto select_goal(const std::vector<LearningGoal>& candidates)
        -> std::optional<LearningGoal>;

    // ── 动机更新 ──────────────────────────────────────────

    /// 学习后更新动机状态
    void update_on_learning(const LearningOutcome& outcome);

    /// 时间衰减（每轮学习循环调用一次）
    void decay();

    // ── 状态查询 ──────────────────────────────────────────

    /// 获取所有动机状态
    [[nodiscard]] auto motive_states() const
        -> const std::map<MotiveType, MotiveState>& {
        return motives_;
    }

    /// 获取指定动机的驱动水平
    [[nodiscard]] auto drive_level(MotiveType type) const -> double;

    /// 获取总体动机强度
    [[nodiscard]] auto total_drive() const -> double;

    /// 获取最近生成的目标
    [[nodiscard]] auto last_goal() const -> std::optional<LearningGoal>;

    /// 动机类型转字符串
    static auto motive_name(MotiveType type) -> std::string;

private:
    std::map<MotiveType, MotiveState> motives_;
    std::optional<LearningGoal> last_goal_;

    /// 计算好奇心评分
    auto curiosity_score_(const std::string& topic,
                          double curiosity_signal) const -> double;

    /// 计算掌握欲评分
    auto competence_score_(const std::string& topic,
                           const std::map<std::string, double>& mastery) const
        -> double;

    /// 计算自主性评分
    auto autonomy_score_(bool self_generated) const -> double;

    /// 计算成就评分
    auto achievement_score_(double difficulty) const -> double;

    /// Softmax 选择（静态辅助方法）
    static auto softmax_select_(
        const std::vector<std::pair<LearningGoal, double>>& scored)
        -> LearningGoal;
};

}  // namespace ai_learning::learning
