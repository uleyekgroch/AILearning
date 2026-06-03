/**
 * @file simple_environment.hpp
 * @brief 简单网格环境 — 离散 2D 世界，Agent 可移动探索
 *
 * 实现 IEnvironment 接口。Agent 在 NxN 网格中移动，
 * 到达目标获得奖励。适合测试 autonomous_learn() 循环。
 */
#pragma once

#include "ai_learning/domain/environment/i_environment.hpp"

#include <random>
#include <vector>

namespace ai_learning::domain {

/// 简单网格世界环境
class SimpleEnvironment : public IEnvironment {
public:
    /// 构造
    /// @param width 网格宽度
    /// @param height 网格高度
    /// @param seed 随机种子
    SimpleEnvironment(int width = 8, int height = 8, int seed = 42);

    /// 获取当前观测：visual(16) + position(2)
    auto observe() const
        -> std::map<std::string, std::vector<float>> override;

    /// 执行动作（0=上 1=下 2=左 3=右），返回 {reward, done}
    auto step(int action)
        -> std::pair<double, bool> override;

    /// 重置环境（随机放置 Agent 和目标）
    auto reset()
        -> std::map<std::string, std::vector<float>> override;

    /// 按发展阶段调整难度
    void configure_for_stage(const std::string& stage) override;

    /// 获取 Agent 当前位置
    [[nodiscard]] auto agent_pos() const -> std::pair<int, int> {
        return {agent_x_, agent_y_};
    }

    /// 获取目标位置
    [[nodiscard]] auto target_pos() const -> std::pair<int, int> {
        return {target_x_, target_y_};
    }

    /// 获取网格尺寸
    [[nodiscard]] auto grid_size() const -> std::pair<int, int> {
        return {width_, height_};
    }

    /// 总步数
    [[nodiscard]] auto step_count() const -> int { return steps_; }

private:
    int width_;
    int height_;
    int agent_x_;
    int agent_y_;
    int target_x_;
    int target_y_;
    int steps_;
    int max_steps_;
    mutable std::mt19937 rng_;

    /// 生成 visual 观测：Agent 周围 4x4 的简单编码
    [[nodiscard]] auto make_visual() const -> std::vector<float>;

    /// 生成 position 观测：归一化坐标
    [[nodiscard]] auto make_position() const -> std::vector<float>;
};

}  // namespace ai_learning::domain
