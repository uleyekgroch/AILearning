/**
 * @file agent_handle.cpp
 * @brief Agent REST 客户端实现 — asio 同步 HTTP 调用
 *
 * 使用 asio standalone 进行简单的 HTTP GET/POST 请求。
 * 不依赖 libcurl 等外部库，保持轻量。
 */
#include "ai_learning/society/agent_handle.hpp"

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ai_learning::society {

// ── AgentStatus 转字符串 ──────────────────────────────────────────

auto to_string(AgentStatus s) -> std::string {
    switch (s) {
        case AgentStatus::Active:  return "active";
        case AgentStatus::Stopped: return "stopped";
    }
    return "unknown";
}

// ── 构造 ───────────────────────────────────────────────────────────

AgentHandle::AgentHandle(std::string agent_id, std::string host, int port)
    : agent_id_(std::move(agent_id))
    , host_(std::move(host))
    , port_(port) {}

// ── REST API 方法 ──────────────────────────────────────────────────

auto AgentHandle::learn(const std::string& text, const std::string& source)
    -> nlohmann::json {
    nlohmann::json body;
    body["text"] = text;
    body["source"] = source;
    return http_post("/api/learn/text", body);
}

auto AgentHandle::reason(const std::string& question)
    -> nlohmann::json {
    nlohmann::json body;
    body["question"] = question;
    return http_post("/api/reason", body);
}

auto AgentHandle::observe_behavior(const nlohmann::json& observation)
    -> nlohmann::json {
    return http_post("/api/observe-behavior", observation);
}

auto AgentHandle::get_stats()
    -> nlohmann::json {
    return http_get("/api/stats");
}

auto AgentHandle::health() -> bool {
    try {
        auto resp = http_get("/api/health");
        return resp.value("status", "") == "ok";
    } catch (...) {
        return false;
    }
}

// ── HTTP 客户端 ────────────────────────────────────────────────────

auto AgentHandle::http_get(const std::string& path)
    -> nlohmann::json {
    asio::io_context io;
    asio::ip::tcp::resolver resolver(io);
    auto endpoints = resolver.resolve(host_, std::to_string(port_));

    asio::ip::tcp::socket socket(io);
    asio::connect(socket, endpoints);

    // 构造 HTTP GET 请求
    std::string request =
        "GET " + path + " HTTP/1.1\r\n" +
        "Host: " + host_ + "\r\n" +
        "Connection: close\r\n" +
        "Accept: application/json\r\n" +
        "\r\n";

    asio::write(socket, asio::buffer(request));
    auto response_str = read_response_(socket);
    socket.close();

    // 提取 JSON body（跳过 HTTP 头部）
    auto body_start = response_str.find("\r\n\r\n");
    if (body_start == std::string::npos) {
        throw std::runtime_error("invalid HTTP response");
    }
    auto body = response_str.substr(body_start + 4);

    // 去除可能的 chunked encoding 长度前缀
    auto nl = body.find('\n');
    if (nl != std::string::npos && nl < body.size() - 1) {
        auto first_line = body.substr(0, nl);
        // 如果第一行是纯数字，说明是 chunked encoding
        bool is_chunk_size = !first_line.empty();
        for (char c : first_line) {
            if (c != '\r' && !std::isxdigit(static_cast<unsigned char>(c))) {
                is_chunk_size = false;
                break;
            }
        }
        if (is_chunk_size) {
            body = body.substr(nl + 1);
        }
    }

    // 去除尾部的 "0\r\n\r\n" chunked 结束标记
    auto end_marker = body.rfind("0\r\n");
    if (end_marker != std::string::npos && end_marker > body.size() - 8) {
        body = body.substr(0, end_marker);
    }

    return nlohmann::json::parse(body);
}

auto AgentHandle::http_post(const std::string& path, const nlohmann::json& body)
    -> nlohmann::json {
    asio::io_context io;
    asio::ip::tcp::resolver resolver(io);
    auto endpoints = resolver.resolve(host_, std::to_string(port_));

    asio::ip::tcp::socket socket(io);
    asio::connect(socket, endpoints);

    auto body_str = body.dump();

    // 构造 HTTP POST 请求
    std::string request =
        "POST " + path + " HTTP/1.1\r\n" +
        "Host: " + host_ + "\r\n" +
        "Connection: close\r\n" +
        "Content-Type: application/json\r\n" +
        "Content-Length: " + std::to_string(body_str.size()) + "\r\n" +
        "Accept: application/json\r\n" +
        "\r\n" +
        body_str;

    asio::write(socket, asio::buffer(request));
    auto response_str = read_response_(socket);
    socket.close();

    // 提取 JSON body
    auto body_start = response_str.find("\r\n\r\n");
    if (body_start == std::string::npos) {
        throw std::runtime_error("invalid HTTP response");
    }
    auto resp_body = response_str.substr(body_start + 4);

    // 去除 chunked encoding 长度前缀
    auto nl = resp_body.find('\n');
    if (nl != std::string::npos && nl < resp_body.size() - 1) {
        auto first_line = resp_body.substr(0, nl);
        bool is_chunk_size = !first_line.empty();
        for (char c : first_line) {
            if (c != '\r' && !std::isxdigit(static_cast<unsigned char>(c))) {
                is_chunk_size = false;
                break;
            }
        }
        if (is_chunk_size) {
            resp_body = resp_body.substr(nl + 1);
        }
    }

    // 去除尾部 chunked 结束标记
    auto end_marker = resp_body.rfind("0\r\n");
    if (end_marker != std::string::npos && end_marker > resp_body.size() - 8) {
        resp_body = resp_body.substr(0, end_marker);
    }

    return nlohmann::json::parse(resp_body);
}

auto AgentHandle::read_response_(asio::ip::tcp::socket& socket)
    -> std::string {
    std::string result;
    char buf[4096];
    asio::error_code ec;

    while (true) {
        auto len = socket.read_some(asio::buffer(buf), ec);
        if (ec == asio::error::eof) {
            break;
        }
        if (ec) {
            throw std::runtime_error("read error: " + ec.message());
        }
        result.append(buf, len);
    }
    return result;
}

}  // namespace ai_learning::society
