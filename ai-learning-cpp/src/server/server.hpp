/**
 * @file server.hpp
 * @brief REST HTTP + WebSocket 服务 — 暴露 Learner 能力为 HTTP/WS API
 *
 * 设计原则：
 * - 持有一个 Learner 引用（不拥有）
 * - 路由注册委托到 route_groups 中的各注册函数
 * - 优雅关闭（信号处理）
 * - WebSocket 实时事件推送
 * - 静态文件服务（Web Console）
 *
 * 路由实现文件：
 * - core_routes.cpp      系统 + 核心学习端点
 * - advanced_routes.cpp  Phase 3-6 高级认知端点
 * - society_routes.cpp   Phase 9 多 Agent 社会端点
 * - chat_routes.cpp      Phase 9 对话端点
 * - runtime_routes.cpp   Phase 9 运行时端点
 */
#pragma once

#include "server_config.hpp"
#include "event_adapter.hpp"
#include "route_groups.hpp"

#include "ai_learning/core/learner.hpp"

#include <crow.h>

#include <atomic>
#include <memory>
#include <string>

namespace ai_learning::server {

/// REST HTTP + WebSocket 服务
///
/// 持有 Learner 引用，将 HTTP 请求路由到对应的 Learner 方法。
/// WebSocket 端点提供实时学习事件推送。
class LearningServer {
public:
    /// 构造：接收 Learner 引用和配置
    explicit LearningServer(core::Learner& learner,
                            ServerConfig config = ServerConfig{});

    /// 启动 HTTP + WebSocket 服务（阻塞）
    auto run() -> void;

    /// 优雅关闭
    auto shutdown() -> void;

    /// 获取服务是否运行中
    [[nodiscard]] auto is_running() const -> bool;

private:
    /// 注册所有路由（委托到 route_groups）
    auto register_routes_(crow::SimpleApp& app) -> void;

    /// 注册静态文件路由（Phase 7.5 — Web Console）
    auto register_static_routes_(crow::SimpleApp& app) -> void;

    /// 注册 WebSocket 路由（Phase 7.4）
    auto register_ws_routes_(crow::SimpleApp& app) -> void;

    /// 启动心跳和统计推送的后台线程
    auto start_ws_background_tasks_() -> void;

    /// 停止后台线程
    auto stop_ws_background_tasks_() -> void;

    core::Learner& learner_;
    ServerConfig config_;

    std::atomic<bool> running_{false};

    /// 路由间共享状态（任务管理、懒初始化组件）
    std::unique_ptr<SharedState> shared_state_;

    /// Phase 7.4: WebSocket 事件管理
    EventAdapter event_adapter_;
    StatsPusher stats_pusher_;
    HeartbeatManager heartbeat_;

    /// 后台线程控制
    std::atomic<bool> ws_bg_running_{false};
};

}  // namespace ai_learning::server
