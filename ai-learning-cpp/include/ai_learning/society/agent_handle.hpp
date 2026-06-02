/**
 * @file agent_handle.hpp
 * @brief Agent REST 客户端 — 封装对单个 Learner 服务的 HTTP 调用
 *
 * 设计原则：
 * - 每个 AgentHandle 代表一个远程 Learner 进程
 * - 通信仅通过 REST HTTP（无共享内存）
 * - 同步阻塞调用，简单可靠
 * - 使用 asio standalone HTTP 客户端
 */
#pragma once

#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif
#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <string>

namespace ai_learning::society {

/// Agent 状态
enum class AgentStatus { Active, Stopped };

/// 将 AgentStatus 转为字符串
auto to_string(AgentStatus s) -> std::string;

/// AgentHandle — 远程 Learner 服务的 REST 客户端
///
/// 封装 HTTP 调用，将 JSON 请求发送到指定端口的 Learner 服务。
class AgentHandle {
public:
    /// 构造：指定 agent ID、主机和端口
    AgentHandle(std::string agent_id, std::string host, int port);

    ~AgentHandle() = default;

    // 禁止拷贝（含 io_context）
    AgentHandle(const AgentHandle&) = delete;
    auto operator=(const AgentHandle&) -> AgentHandle& = delete;

    // 允许移动
    AgentHandle(AgentHandle&&) noexcept = default;
    auto operator=(AgentHandle&&) noexcept -> AgentHandle& = default;

    // ── REST API 方法 ──────────────────────────────────────────────

    /// 学习文本 → POST /api/learn/text
    auto learn(const std::string& text, const std::string& source = "text")
        -> nlohmann::json;

    /// 推理问答 → POST /api/reason
    auto reason(const std::string& question)
        -> nlohmann::json;

    /// 观察行为 → POST /api/observe-behavior
    auto observe_behavior(const nlohmann::json& observation)
        -> nlohmann::json;

    /// 获取统计 → GET /api/stats
    auto get_stats()
        -> nlohmann::json;

    /// 健康检查 → GET /api/health
    auto health() -> bool;

    // ── 访问器 ─────────────────────────────────────────────────────

    [[nodiscard]] auto id() const -> const std::string& { return agent_id_; }
    [[nodiscard]] auto host() const -> const std::string& { return host_; }
    [[nodiscard]] auto port() const -> int { return port_; }

private:
    /// 发送 HTTP GET 请求，返回 JSON 响应体
    auto http_get(const std::string& path)
        -> nlohmann::json;

    /// 发送 HTTP POST 请求，返回 JSON 响应体
    auto http_post(const std::string& path, const nlohmann::json& body)
        -> nlohmann::json;

    /// 解析 HTTP 响应为字符串
    auto read_response_(asio::ip::tcp::socket& socket)
        -> std::string;

    std::string agent_id_;
    std::string host_;
    int port_;
};

}  // namespace ai_learning::society
