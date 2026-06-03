/**
 * @file stdp_learning.cpp
 * @brief STDP 赫布学习实现
 */

#include "ai_learning/learning/stdp_learning.hpp"

#include <algorithm>
#include <cmath>

namespace ai_learning::learning {

STDP::STDP(double lr, double tau, double decay)
    : lr_(lr), tau_(tau), decay_(decay) {}

void STDP::strengthen(const std::string& pre, const std::string& post,
                      double timing_delta) {
    auto key = std::make_pair(pre, post);
    double& weight = connections_[key];

    double delta;
    if (timing_delta >= 0) {
        // LTP：pre 先于 post → 增强
        delta = lr_ * std::exp(-std::abs(timing_delta) / tau_);
        weight += delta;
    } else {
        // LTD：post 先于 pre → 减弱（幅度较小）
        delta = lr_ * 0.5 * std::exp(-std::abs(timing_delta) / tau_);
        weight -= delta;
    }

    // 裁剪到 [-1.0, 1.0]
    weight = std::clamp(weight, -1.0, 1.0);
}

auto STDP::get_weight(const std::string& pre, const std::string& post) const
    -> double {
    auto it = connections_.find(std::make_pair(pre, post));
    return (it != connections_.end()) ? it->second : 0.0;
}

auto STDP::get_strong_connections(
    const std::string& entity,
    double threshold) const -> std::vector<std::pair<std::string, double>> {
    std::vector<std::pair<std::string, double>> results;

    for (const auto& [key, weight] : connections_) {
        if (std::abs(weight) < threshold) continue;
        if (key.first == entity) {
            results.emplace_back(key.second, weight);
        } else if (key.second == entity) {
            results.emplace_back(key.first, weight);
        }
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return std::abs(b.second) < std::abs(a.second);
              });
    return results;
}

void STDP::decay_all() {
    for (auto& [key, weight] : connections_) {
        weight *= decay_;
    }
    // 移除极弱连接
    for (auto it = connections_.begin(); it != connections_.end(); ) {
        if (std::abs(it->second) < 0.01) {
            it = connections_.erase(it);
        } else {
            ++it;
        }
    }
}

auto STDP::connection_count() const -> size_t {
    return connections_.size();
}

}  // namespace ai_learning::learning
