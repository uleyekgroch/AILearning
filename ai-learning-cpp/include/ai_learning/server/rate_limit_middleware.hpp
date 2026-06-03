/**
 * @file rate_limit_middleware.hpp
 * @brief Token bucket rate limiter — Crow 中间件
 *
 * 用法:
 *   app.use(rate_limiter);
 *
 * 配置（通过 ServerConfig 或环境变量）:
 *   RATE_LIMIT_RPS=100   — 每秒请求数上限
 *   RATE_LIMIT_BURST=200 — 突发请求桶大小
 */

#pragma once

#include <crow.h>

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ai_learning::server {

/// 单客户端限流桶
struct TokenBucket {
    double tokens = 0.0;
    std::chrono::steady_clock::time_point last_update;
};

/// Token bucket rate limiter for Crow
class RateLimiter {
public:
    explicit RateLimiter(double rps = 100.0, size_t burst = 200)
        : rps_(rps), burst_(burst) {}

    /// Check and optionally block the request. Returns true if allowed.
    auto check(const crow::request& req, crow::response& res) -> bool {
        std::string client = get_client_id_(req);

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();

        auto& bucket = buckets_[client];

        // 添加新令牌
        auto elapsed = std::chrono::duration<double>(now - bucket.last_update).count();
        bucket.tokens = std::min(
            static_cast<double>(burst_),
            bucket.tokens + elapsed * rps_);
        bucket.last_update = now;

        if (bucket.tokens < 1.0) {
            res.code = 429;
            res.write(R"({"error":"rate limit exceeded","retry_after":1})");
            res.set_header("Content-Type", "application/json");
            res.set_header("Retry-After", "1");
            res.end();
            return false;
        }

        bucket.tokens -= 1.0;
        return true;
    }

    /// 清理长期不活跃的客户端（可选，由后台任务调用）
    auto cleanup_inactive(std::chrono::seconds timeout = std::chrono::seconds(300)) -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        for (auto it = buckets_.begin(); it != buckets_.end();) {
            auto idle = std::chrono::duration_cast<std::chrono::seconds>(
                now - it->second.last_update).count();
            if (idle > timeout.count()) {
                it = buckets_.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    static auto get_client_id_(const crow::request& req) -> std::string {
        // 优先使用 X-Forwarded-For（代理后），fallback 到 remote_addr
        auto fwd = req.get_header_value("X-Forwarded-For");
        if (!fwd.empty()) {
            auto pos = fwd.find(',');
            return (pos == std::string::npos) ? fwd : fwd.substr(0, pos);
        }
        return req.remote_ip_address;
    }

    double rps_;
    size_t burst_;
    std::mutex mutex_;
    std::unordered_map<std::string, TokenBucket> buckets_;
};

}  // namespace ai_learning::server
