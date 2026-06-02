/**
 * @file server.hpp
 * @brief REST HTTP 服务 — 暴露 Learner 能力为 HTTP API
 *
 * 设计原则：
 * - 持有一个 Learner 引用（不拥有）
 * - 路由注册集中管理
 * - 优雅关闭（信号处理）
 */
#pragma once

#include "server_config.hpp"

#include "ai_learning/core/learner.hpp"

#include <atomic>
#include <string>

namespace ai_learning::server {

/// REST HTTP 服务
///
/// 持有 Learner 引用，将 HTTP 请求路由到对应的 Learner 方法。
/// 单例设计：一个服务进程持有一个 Learner 实例。
class LearningServer {
public:
    /// 构造：接收 Learner 引用和配置
    explicit LearningServer(core::Learner& learner,
                            ServerConfig config = ServerConfig{});

    /// 启动 HTTP 服务（阻塞）
    auto run() -> void;

    /// 优雅关闭
    auto shutdown() -> void;

    /// 获取服务是否运行中
    [[nodiscard]] auto is_running() const -> bool;

private:
    /// 注册所有路由
    auto register_routes_() -> void;

    /// 注册核心 API 路由（Phase 1-2）
    auto register_core_routes_() -> void;

    /// 注册高级 API 路由（Phase 3-6）
    auto register_advanced_routes_() -> void;

    /// 注册系统路由（健康检查、统计等）
    auto register_system_routes_() -> void;

    core::Learner& learner_;
    ServerConfig config_;

    std::atomic<bool> running_{false};
};

}  // namespace ai_learning::server
