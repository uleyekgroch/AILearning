/**
 * @file server.hpp
 * @brief REST HTTP + WebSocket 服务 — 暴露 Learner 能力为 HTTP/WS API
 *
 * 设计原则：
 * - 持有一个 Learner 引用（不拥有）
 * - 路由注册集中管理
 * - 优雅关闭（信号处理）
 * - 异步任务管理（长时间操作不阻塞服务）
 * - Phase 7.4: WebSocket 实时事件推送
 */
#pragma once

#include "server_config.hpp"
#include "event_adapter.hpp"

#include "ai_learning/core/learner.hpp"

#include <crow.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace ai_learning::server {

/// 异步任务状态
struct AsyncTask {
    std::string task_id;
    std::string status;           // pending / running / completed / failed
    nlohmann::json result;        // 完成后的结果
    std::string error;            // 失败原因
};

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
    /// 注册所有路由
    auto register_routes_(crow::SimpleApp& app) -> void;

    /// 注册核心 API 路由（Phase 1-2）
    auto register_core_routes_(crow::SimpleApp& app) -> void;

    /// 注册高级 API 路由（Phase 3-6）
    auto register_advanced_routes_(crow::SimpleApp& app) -> void;

    /// 注册系统路由（健康检查、统计等）
    auto register_system_routes_(crow::SimpleApp& app) -> void;

    /// 注册静态文件路由（Phase 7.5 — Web Console）
    auto register_static_routes_(crow::SimpleApp& app) -> void;

    /// 注册 WebSocket 路由（Phase 7.4）
    auto register_ws_routes_(crow::SimpleApp& app) -> void;

    /// 启动心跳和统计推送的后台线程
    auto start_ws_background_tasks_() -> void;

    /// 停止后台线程
    auto stop_ws_background_tasks_() -> void;

    /// 生成唯一 task_id
    static auto generate_task_id_() -> std::string;

    core::Learner& learner_;
    ServerConfig config_;

    std::atomic<bool> running_{false};

    /// 异步任务管理
    std::mutex tasks_mutex_;
    std::map<std::string, AsyncTask> tasks_;

    /// Phase 7.4: WebSocket 事件管理
    EventAdapter event_adapter_;
    StatsPusher stats_pusher_;
    HeartbeatManager heartbeat_;

    /// 后台线程控制
    std::atomic<bool> ws_bg_running_{false};
};

}  // namespace ai_learning::server
