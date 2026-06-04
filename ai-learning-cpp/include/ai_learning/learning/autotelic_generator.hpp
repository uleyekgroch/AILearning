/**
 * @file autotelic_generator.hpp
 * @brief 自成目标生成器 — AI 的内在好奇心与无聊驱动
 *
 * 理论基础：
 *   - Schmidhuber (1991): Curious Model-Building Control Systems
 *   - Colas et al. (2022): Autotelic Agents with Intrinsically Motivated Learning
 *
 * 核心机制：
 *   当 AI 处于没有外部任务的“无聊”状态时，通过评估各个知识领域的 Learning Progress (LP)，
 *   主动生成一个正好处于最近发展区 (ZPD) 的虚构目标。这驱使 AI 在后台自己玩、自己探索。
 */
#pragma once

#include <string>
#include <vector>
#include <map>

namespace ai_learning::learning {

struct AutotelicGoal {
    std::string goal_type;    ///< 例如 "explore_relation", "simulate_future", "verify_rule"
    std::string target_cpt_a; ///< 目标概念 A
    std::string target_cpt_b; ///< 目标概念 B
    double expected_learning_progress = 0.0;
};

class AutotelicGenerator {
public:
    AutotelicGenerator() = default;

    /// 记录某次探索的学习进度 (预测误差的导数)
    void record_progress(const std::string& domain, double progress);

    /// 当系统无聊时调用，生成一个新的内在目标
    auto generate_goal() -> AutotelicGoal;

private:
    int ticks_since_last_input_ = 0;
    // 领域 -> 历史学习进度平滑值
    std::map<std::string, double> learning_progress_map_;
};

} // namespace ai_learning::learning
