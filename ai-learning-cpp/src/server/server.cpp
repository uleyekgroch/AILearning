/**
 * @file server.cpp
 * @brief REST HTTP 服务实现 — 路由注册与请求处理
 *
 * Phase 7.1: 基础骨架 + 健康检查端点
 */

#include "server.hpp"

#include <crow.h>

#include <nlohmann/json.hpp>

#include <iostream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#endif

namespace ai_learning::server {

using json = nlohmann::json;

// ── 全局信号处理 ──────────────────────────────────────────────────

static std::atomic<bool> g_shutdown_requested{false};

#ifdef _WIN32
static BOOL WINAPI windows_signal_handler_(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT) {
        g_shutdown_requested = true;
        return TRUE;
    }
    return FALSE;
}
#else
static void posix_signal_handler_(int /*signal*/) {
    g_shutdown_requested = true;
}
#endif

// ── 构造 ────────────────────────────────────────────────────────

LearningServer::LearningServer(core::Learner& learner, ServerConfig config)
    : learner_(learner), config_(std::move(config)) {}

// ── 路由注册 ────────────────────────────────────────────────────

auto LearningServer::register_routes_() -> void {
    register_system_routes_();
    register_core_routes_();
    register_advanced_routes_();
}

auto LearningServer::register_system_routes_() -> void {
    // GET /api/health — 在 run() 中注册（需要 crow::SimpleApp）
}

auto LearningServer::register_core_routes_() -> void {
    // Phase 7.2 将在此添加核心学习端点
}

auto LearningServer::register_advanced_routes_() -> void {
    // Phase 7.3 将在此添加高级认知端点
}

// ── 启动 ────────────────────────────────────────────────────────

auto LearningServer::run() -> void {
    crow::SimpleApp app;

    // 注册健康检查端点
    CROW_ROUTE(app, "/api/health").methods("GET"_method)
    ([this]() -> crow::response {
        json body;
        body["status"] = "ok";
        body["version"] = config_.version;
        body["stage"] = learner_.stage();

        auto stats = learner_.get_stats();
        body["total_steps"] = stats.count("total_steps")
            ? static_cast<int>(stats.at("total_steps")) : 0;

        crow::response res{body.dump()};
        res.set_header("Content-Type", "application/json");
        return res;
    });

    // 设置日志级别
    app.loglevel(crow::LogLevel::Info);

    // 安装信号处理
#ifdef _WIN32
    SetConsoleCtrlHandler(windows_signal_handler_, TRUE);
#else
    signal(SIGINT, posix_signal_handler_);
    signal(SIGTERM, posix_signal_handler_);
#endif

    running_ = true;
    std::cout << "[Server] AILearning REST API starting on port "
              << config_.port << " (" << config_.threads << " threads)\n";

    // 在独立线程中监控关闭信号
    std::jthread monitor([this, &app]() {
        while (running_ && !g_shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (g_shutdown_requested) {
            std::cout << "\n[Server] Shutdown signal received, stopping...\n";
            app.stop();
            running_ = false;
        }
    });

    // 启动 HTTP 服务（阻塞，直到 app.stop() 被调用）
    app.port(config_.port).concurrency(
        static_cast<std::uint16_t>(config_.threads)).run();

    // 等待监控线程结束
    monitor.request_stop();
    if (monitor.joinable()) {
        monitor.join();
    }

    running_ = false;
    std::cout << "[Server] Stopped gracefully.\n";
}

// ── 关闭 ────────────────────────────────────────────────────────

auto LearningServer::shutdown() -> void {
    g_shutdown_requested = true;
}

auto LearningServer::is_running() const -> bool {
    return running_.load();
}

}  // namespace ai_learning::server
