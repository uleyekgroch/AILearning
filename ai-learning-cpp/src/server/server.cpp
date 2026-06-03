/**
 * @file server.cpp
 * @brief REST HTTP + WebSocket 服务 — 瘦编排器
 *
 * 职责：
 * - 构造、启动、关闭服务
 * - 信号处理（优雅关闭）
 * - 委托路由注册到各 route_groups 文件
 * - 静态文件服务（Web Console）
 * - WebSocket 路由 + 后台任务（心跳、统计推送）
 *
 * 路由实现已拆分到：
 * - core_routes.cpp      系统 + 核心学习端点
 * - advanced_routes.cpp  Phase 3-6 高级认知端点
 * - society_routes.cpp   Phase 9 多 Agent 社会端点
 * - chat_routes.cpp      Phase 9 对话端点
 * - runtime_routes.cpp   Phase 9 运行时端点
 */

#include "server.hpp"
#include "route_groups.hpp"
#include "dto.hpp"
#include "ai_learning/server/rate_limit_middleware.hpp"
#include "ai_learning/server/logger.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#endif

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

ai_learning::server::LearningServer::LearningServer(
    ai_learning::core::Learner& learner,
    ai_learning::server::ServerConfig config)
    : learner_(learner), config_(std::move(config)),
      shared_state_(std::make_unique<SharedState>()) {}

// ── 路由注册 ────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::register_routes_(::crow::SimpleApp& app) -> void {
    register_core_routes(app, learner_, config_, *shared_state_);
    register_advanced_routes(app, learner_, *shared_state_);
    register_society_routes(app, learner_, *shared_state_);
    register_chat_routes(app, learner_, *shared_state_);
    register_runtime_routes(app, learner_, *shared_state_);
    register_goals_routes(app, learner_, *shared_state_);
    register_openapi_routes(app, config_);
    register_metrics_routes(app, *shared_state_);
    register_static_routes_(app);
    register_ws_routes_(app);
}

// ── 静态文件路由（Phase 7.5 — Web Console）─────────────────────────

auto ai_learning::server::LearningServer::register_static_routes_(::crow::SimpleApp& app) -> void {
    const std::string web_dir = config_.static_dir.empty()
        ? std::string("web/")
        : config_.static_dir;

    static const auto read_file = [](const std::string& filepath) -> std::string {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) return {};
        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        return content;
    };

    static const auto get_mime = [](const std::string& path) -> std::string {
        auto pos = path.rfind('.');
        if (pos == std::string::npos) return "text/plain";
        std::string ext = path.substr(pos + 1);
        if (ext == "html" || ext == "htm") return "text/html; charset=utf-8";
        if (ext == "css") return "text/css; charset=utf-8";
        if (ext == "js")  return "application/javascript; charset=utf-8";
        if (ext == "json") return "application/json; charset=utf-8";
        if (ext == "png") return "image/png";
        if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
        if (ext == "svg") return "image/svg+xml";
        if (ext == "ico") return "image/x-icon";
        return "application/octet-stream";
    };

    // GET / — 重定向到 /console
    CROW_ROUTE(app, "/").methods("GET"_method)
    ([&web_dir](const ::crow::request&, ::crow::response& res) {
        std::string filepath = web_dir + "index.html";
        if (filepath.find("..") != std::string::npos) {
            res.code = 400;
            res.write("Bad Request");
            res.end();
            return;
        }
        std::string body = read_file(filepath);
        if (body.empty()) {
            res.code = 404;
            res.write("AILearning server running. Web console not found at " + filepath);
            res.end();
            return;
        }
        res.set_header("Content-Type", "text/html; charset=utf-8");
        res.write(body);
        res.end();
    });

    // GET /console — serve index.html explicitly
    CROW_ROUTE(app, "/console").methods("GET"_method)
    ([&web_dir](const ::crow::request&, ::crow::response& res) {
        std::string filepath = web_dir + "index.html";
        if (filepath.find("..") != std::string::npos) {
            res.code = 400;
            res.write("Bad Request");
            res.end();
            return;
        }
        std::string body = read_file(filepath);
        if (body.empty()) {
            res.code = 404;
            res.write("File not found");
            res.end();
            return;
        }
        res.set_header("Content-Type", "text/html; charset=utf-8");
        res.write(body);
        res.end();
    });

    // GET /web/<path> — serve static files from web/ directory
    CROW_ROUTE(app, "/web/<path>").methods("GET"_method)
    ([&web_dir](const ::crow::request&, ::crow::response& res, std::string file_path) {
        if (file_path.find("..") != std::string::npos) {
            res.code = 400;
            res.write("Bad Request");
            res.end();
            return;
        }
        std::string filepath = web_dir + file_path;
        std::string body = read_file(filepath);
        if (body.empty()) {
            res.code = 404;
            res.write("File not found: " + file_path);
            res.end();
            return;
        }
        res.set_header("Content-Type", get_mime(file_path));
        res.write(body);
        res.end();
    });

    std::cout << "[Server] Static files serving from: " << web_dir << "\n";
}

// ── WebSocket 路由 ──────────────────────────────────────────────────

auto ai_learning::server::LearningServer::register_ws_routes_(::crow::SimpleApp& app) -> void {

    // ── /ws/events — 所有学习事件实时推送 ──────────────────────────
    CROW_WEBSOCKET_ROUTE(app, "/ws/events")
    .onopen([this](::crow::websocket::connection& conn) {
        event_adapter_.add_connection(&conn);
        heartbeat_.add_connection(&conn);
        auto history = event_adapter_.get_history_json();
        if (!history.empty() && history != "[]") {
            try {
                conn.send_text(history);
            } catch (const std::exception& e) {
                std::cerr << "[WS/events] send history failed: " << e.what() << "\n";
            }
        }
    })
    .onclose([this](::crow::websocket::connection& conn,
                     const std::string& /*reason*/) {
        event_adapter_.remove_connection(&conn);
        heartbeat_.remove_connection(&conn);
    })
    .onmessage([this](::crow::websocket::connection& conn,
                       const std::string& data, bool is_binary) {
        heartbeat_.touch(&conn);

        if (!is_binary) {
            try {
                auto msg = json::parse(data);
                if (msg.contains("type") && msg["type"] == "ping") {
                    json pong;
                    pong["type"] = "pong";
                    pong["timestamp"] = std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count();
                    conn.send_text(pong.dump());
                }
            } catch (const std::exception&) {
                // 忽略无法解析的消息
            }
        }
    });

    // ── /ws/stats — 统计摘要推送 ──────────────────────────────────
    CROW_WEBSOCKET_ROUTE(app, "/ws/stats")
    .onopen([this](::crow::websocket::connection& conn) {
        stats_pusher_.add_connection(&conn);
        heartbeat_.add_connection(&conn);
        try {
            auto stats = learner_.get_stats();
            json msg;
            msg["type"] = "stats";
            msg["data"] = stats;
            conn.send_text(msg.dump());
        } catch (const std::exception& e) {
            std::cerr << "[WS/stats] initial send failed: " << e.what() << "\n";
        }
    })
    .onclose([this](::crow::websocket::connection& conn,
                     const std::string& /*reason*/) {
        stats_pusher_.remove_connection(&conn);
        heartbeat_.remove_connection(&conn);
    })
    .onmessage([this](::crow::websocket::connection& conn,
                       const std::string& /*data*/, bool /*is_binary*/) {
        heartbeat_.touch(&conn);
    });
}

// ── WebSocket 后台任务 ──────────────────────────────────────────────

auto ai_learning::server::LearningServer::start_ws_background_tasks_() -> void {
    ws_bg_running_ = true;

    // 心跳 + 超时检查线程（每 30 秒 ping 一次）
    std::jthread heartbeat_thread([this]() {
        while (ws_bg_running_) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            if (!ws_bg_running_) break;

            heartbeat_.send_pings();

            auto expired = heartbeat_.check_timeouts(std::chrono::seconds{60});
            for (auto* conn : expired) {
                std::cout << "[WS/heartbeat] Closing timed-out connection\n";
                try {
                    conn->close("timeout");
                } catch (const std::exception& e) {
                    std::cerr << "[WS/heartbeat] close failed: " << e.what() << "\n";
                }
                event_adapter_.remove_connection(conn);
                stats_pusher_.remove_connection(conn);
                heartbeat_.remove_connection(conn);
            }
        }
    });

    // 统计推送线程（每 1 秒推送一次）
    std::jthread stats_thread([this]() {
        while (ws_bg_running_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (!ws_bg_running_) break;

            try {
                auto stats = learner_.get_stats();
                json msg;
                msg["type"] = "stats";
                msg["timestamp"] = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count();
                msg["data"] = stats;
                stats_pusher_.push_stats(msg);
            } catch (const std::exception& e) {
                std::cerr << "[WS/stats] push failed: " << e.what() << "\n";
            }
        }
    });

    heartbeat_thread.detach();
    stats_thread.detach();
}

auto ai_learning::server::LearningServer::stop_ws_background_tasks_() -> void {
    ws_bg_running_ = false;
}

// ── 启动 ────────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::run() -> void {
    ::crow::SimpleApp app;

    register_routes_(app);

    // ── J.2: Rate limiting (available via RateLimiter middleware class)
    //         Production deployments should use reverse-proxy rate limiting
    //         (nginx, traefik, etc.). The RateLimiter class is provided for
    //         reference and can be integrated via Crow middleware if needed.
    {
        const char* env_rps = std::getenv("RATE_LIMIT_RPS");
        if (env_rps) {
            std::cout << "[Server] Note: RATE_LIMIT_RPS set to " << env_rps
                      << " — use a reverse proxy for production rate limiting\n";
        }
    }

    app.loglevel(::crow::LogLevel::Info);

#ifdef _WIN32
    SetConsoleCtrlHandler(windows_signal_handler_, TRUE);
#else
    signal(SIGINT, posix_signal_handler_);
    signal(SIGTERM, posix_signal_handler_);
#endif

    running_ = true;
    std::cout << "[Server] AILearning REST API starting on port "
              << config_.port << " (" << config_.threads << " threads)\n";

    start_ws_background_tasks_();

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

    app.port(config_.port).concurrency(
        static_cast<std::uint16_t>(config_.threads)).run();

    monitor.request_stop();
    if (monitor.joinable()) {
        monitor.join();
    }

    running_ = false;
    stop_ws_background_tasks_();

    std::cout << "[Server] Stopped gracefully.\n";
}

// ── 关闭 ────────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::shutdown() -> void {
    g_shutdown_requested = true;
}

auto ai_learning::server::LearningServer::is_running() const -> bool {
    return running_.load();
}
