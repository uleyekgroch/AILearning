/**
 * @file metrics_collector.hpp
 * @brief 轻量指标收集器 — 推理延迟、请求计数、P50/P95/P99
 *
 * 线程安全：所有操作受 mutex 保护。
 * 零外部依赖，纯 C++20 标准库实现。
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace ai_learning::server {

/// 单条延迟样本
struct LatencySample {
    double ms = 0.0;               // 延迟毫秒
    std::string endpoint;          // 端点标识
    std::chrono::steady_clock::time_point timestamp;
};

/// 轻量指标收集器 — 环形缓冲区 + 滑动窗口百分位
class MetricsCollector {
public:
    explicit MetricsCollector(size_t max_samples = 10000)
        : max_samples_(max_samples) {}

    /// 记录一次请求延迟
    auto record_latency(double ms, const std::string& endpoint = "") -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.size() >= max_samples_) {
            samples_.erase(samples_.begin());
        }
        samples_.push_back({ms, endpoint, std::chrono::steady_clock::now()});
        total_requests_++;
    }

    /// 记录推理延迟（包装 convenience）
    auto record_inference(double ms) -> void {
        record_latency(ms, "inference");
    }

    /// 获取请求总数
    [[nodiscard]] auto total_requests() const -> size_t {
        return total_requests_.load();
    }

    /// 获取百分位延迟（P50/P95/P99），返回 {p50, p95, p99}
    [[nodiscard]] auto latency_percentiles() const -> std::array<double, 3> {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.empty()) return {0.0, 0.0, 0.0};

        std::vector<double> vals;
        vals.reserve(samples_.size());
        for (const auto& s : samples_) vals.push_back(s.ms);
        std::sort(vals.begin(), vals.end());

        auto percentile = [&](double p) -> double {
            size_t idx = static_cast<size_t>(p * (vals.size() - 1));
            return vals[idx];
        };

        return {percentile(0.50), percentile(0.95), percentile(0.99)};
    }

    /// 获取最近 N 秒的平均延迟
    [[nodiscard]] auto avg_latency_recent(std::chrono::seconds window = std::chrono::seconds(60)) const -> double {
        std::lock_guard<std::mutex> lock(mutex_);
        auto cutoff = std::chrono::steady_clock::now() - window;
        double sum = 0.0;
        size_t count = 0;
        for (const auto& s : samples_) {
            if (s.timestamp >= cutoff) {
                sum += s.ms;
                count++;
            }
        }
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    }

    /// 导出为 Prometheus 文本格式
    [[nodiscard]] auto to_prometheus() const -> std::string {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string out;
        out.reserve(512);

        auto [p50, p95, p99] = latency_percentiles();

        out += "# HELP ail_requests_total Total requests processed\n";
        out += "# TYPE ail_requests_total counter\n";
        out += "ail_requests_total " + std::to_string(total_requests_.load()) + "\n\n";

        out += "# HELP ail_latency_ms Inference latency in milliseconds\n";
        out += "# TYPE ail_latency_ms summary\n";
        out += "ail_latency_ms{quantile=\"0.5\"} " + std::to_string(p50) + "\n";
        out += "ail_latency_ms{quantile=\"0.95\"} " + std::to_string(p95) + "\n";
        out += "ail_latency_ms{quantile=\"0.99\"} " + std::to_string(p99) + "\n\n";

        out += "# HELP ail_latency_avg_ms Average latency over last 60s\n";
        out += "# TYPE ail_latency_avg_ms gauge\n";
        out += "ail_latency_avg_ms " + std::to_string(avg_latency_recent()) + "\n";

        return out;
    }

private:
    size_t max_samples_;
    std::vector<LatencySample> samples_;
    std::atomic<size_t> total_requests_{0};
    mutable std::mutex mutex_;
};

/// RAII 计时器 — 自动记录延迟到 MetricsCollector
class ScopedTimer {
public:
    explicit ScopedTimer(MetricsCollector& collector,
                         std::string endpoint = "")
        : collector_(collector), endpoint_(std::move(endpoint)),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        collector_.record_latency(ms, endpoint_);
    }

    // 禁止复制/移动
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    MetricsCollector& collector_;
    std::string endpoint_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace ai_learning::server
