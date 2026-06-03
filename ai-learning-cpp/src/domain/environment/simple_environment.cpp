/**
 * @file simple_environment.cpp
 * @brief 简单网格环境实现
 */

#include "ai_learning/domain/environment/simple_environment.hpp"

#include <algorithm>
#include <cmath>

namespace ai_learning::domain {

SimpleEnvironment::SimpleEnvironment(int width, int height, int seed)
    : width_(width), height_(height),
      agent_x_(0), agent_y_(0),
      target_x_(0), target_y_(0),
      steps_(0), max_steps_(width * height * 4),
      rng_(seed) {
    reset();
}

auto SimpleEnvironment::observe() const
    -> std::map<std::string, std::vector<float>> {
    return {
        {"visual", make_visual()},
        {"position", make_position()},
    };
}

auto SimpleEnvironment::step(int action)
    -> std::pair<double, bool> {
    // 0=上 1=下 2=左 3=右
    switch (action) {
        case 0: agent_y_ = std::max(0, agent_y_ - 1); break;
        case 1: agent_y_ = std::min(height_ - 1, agent_y_ + 1); break;
        case 2: agent_x_ = std::max(0, agent_x_ - 1); break;
        case 3: agent_x_ = std::min(width_ - 1, agent_x_ + 1); break;
        default: break;
    }
    ++steps_;

    // 计算奖励：距离目标的负距离
    double dx = static_cast<double>(agent_x_ - target_x_);
    double dy = static_cast<double>(agent_y_ - target_y_);
    double dist = std::sqrt(dx * dx + dy * dy);
    double max_dist = std::sqrt(
        static_cast<double>(width_ * width_ + height_ * height_));
    double reward = 1.0 - dist / max_dist;

    bool done = (agent_x_ == target_x_ && agent_y_ == target_y_)
                || steps_ >= max_steps_;

    // 到达目标额外奖励
    if (agent_x_ == target_x_ && agent_y_ == target_y_) {
        reward = 1.0;
    }

    return {reward, done};
}

auto SimpleEnvironment::reset()
    -> std::map<std::string, std::vector<float>> {
    std::uniform_int_distribution<int> x_dist(0, width_ - 1);
    std::uniform_int_distribution<int> y_dist(0, height_ - 1);

    agent_x_ = x_dist(rng_);
    agent_y_ = y_dist(rng_);

    // 确保目标不与 Agent 重叠
    do {
        target_x_ = x_dist(rng_);
        target_y_ = y_dist(rng_);
    } while (target_x_ == agent_x_ && target_y_ == agent_y_);

    steps_ = 0;
    return observe();
}

void SimpleEnvironment::configure_for_stage(const std::string& stage) {
    // 高级阶段使用更大的网格
    if (stage == "literacy" || stage == "complex") {
        max_steps_ = width_ * height_ * 8;
    } else if (stage == "two_word" || stage == "single_word") {
        max_steps_ = width_ * height_ * 4;
    } else {
        max_steps_ = width_ * height_ * 2;
    }
}

auto SimpleEnvironment::make_visual() const -> std::vector<float> {
    // 4x4 简单视觉：16 个值
    // 编码 Agent 周围的相对信息
    std::vector<float> visual(16, 0.0f);

    // 方向向量指向目标
    float dx = static_cast<float>(target_x_ - agent_x_);
    float dy = static_cast<float>(target_y_ - agent_y_);
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > 0.001f) {
        dx /= dist;
        dy /= dist;
    }

    // 方向编码
    visual[0] = dx;     // 目标 x 方向
    visual[1] = dy;     // 目标 y 方向
    visual[2] = dist / static_cast<float>(width_ + height_);  // 归一化距离
    visual[3] = static_cast<float>(steps_) / static_cast<float>(max_steps_);  // 时间进度

    // Agent 位置归一化
    visual[4] = static_cast<float>(agent_x_) / static_cast<float>(width_);
    visual[5] = static_cast<float>(agent_y_) / static_cast<float>(height_);

    // 边界距离
    visual[6] = static_cast<float>(agent_x_) / static_cast<float>(width_);
    visual[7] = 1.0f - visual[6];
    visual[8] = static_cast<float>(agent_y_) / static_cast<float>(height_);
    visual[9] = 1.0f - visual[8];

    // 是否到达目标
    visual[10] = (agent_x_ == target_x_ && agent_y_ == target_y_) ? 1.0f : 0.0f;

    return visual;
}

auto SimpleEnvironment::make_position() const -> std::vector<float> {
    return {
        static_cast<float>(agent_x_) / static_cast<float>(width_),
        static_cast<float>(agent_y_) / static_cast<float>(height_),
    };
}

}  // namespace ai_learning::domain
