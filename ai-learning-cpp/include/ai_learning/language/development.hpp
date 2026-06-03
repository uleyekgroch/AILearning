#pragma once
/**
 * @file development.hpp
 * @brief 语言发展阶段 — Piaget 式渐进发展
 *
 * 阶段序列：sensorimotor → single_word → two_word → complex → literacy
 * 每阶段有晋升条件和可解锁能力。
 */

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::language {

/// 发展阶段枚举
enum class Stage {
    Sensorimotor,  ///< 感知运动
    SingleWord,    ///< 单字
    TwoWord,       ///< 双字
    Complex,       ///< 复杂
    Literacy       ///< 读写
};

/// 阶段名称转换
[[nodiscard]] auto stage_to_string(Stage s) -> std::string;
[[nodiscard]] auto stage_from_string(const std::string& s) -> Stage;

/// 能力评估指标
struct Evaluation {
    float prediction_accuracy = 0.0F;
    int vocabulary_size = 0;
    float composition_rate = 0.0F;
    float grammar_complexity = 0.0F;
};

/// 进化历史记录
struct StageHistoryEntry {
    Stage from_stage;
    Stage to_stage;
    Evaluation evaluation;
};

class DevelopmentTracker {
public:
    explicit DevelopmentTracker(Stage initial = Stage::Sensorimotor);

    /// 尝试晋升到下一阶段，返回是否成功
    [[nodiscard]] auto try_advance(const Evaluation& evaluation) -> bool;

    /// 获取当前阶段
    [[nodiscard]] auto current_stage() const -> Stage { return stage_; }

    /// 获取当前阶段索引 (0-based)
    [[nodiscard]] auto stage_index() const -> int {
        return static_cast<int>(stage_);
    }

    /// 获取当前阶段可解锁的能力
    [[nodiscard]] auto current_abilities() const -> std::vector<std::string>;

    /// 获取阶段历史
    [[nodiscard]] auto history() const -> const std::vector<StageHistoryEntry>& {
        return history_;
    }

    /// 获取阶段总数
    [[nodiscard]] static auto stage_count() -> int { return 5; }

    /// 获取阶段名称
    [[nodiscard]] auto stage_name() const -> std::string {
        return stage_to_string(stage_);
    }

private:
    Stage stage_;
    std::vector<StageHistoryEntry> history_;

    /// 检查晋升条件
    [[nodiscard]] auto check_promotion(const Evaluation& evaluation) const
        -> bool;
};

}  // namespace ai_learning::language
