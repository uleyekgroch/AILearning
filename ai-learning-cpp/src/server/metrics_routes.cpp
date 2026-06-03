/**
 * @file metrics_routes.cpp
 * @brief 指标路由 — Prometheus 格式 + JSON 摘要
 *
 * 端点:
 *   GET /api/metrics      — Prometheus text format
 *   GET /api/metrics/json — JSON 摘要（P50/P95/P99 + GPU）
 */

#include "route_groups.hpp"
#include "ai_learning/server/gpu_monitor.hpp"
#include "ai_learning/server/logger.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

void ai_learning::server::register_metrics_routes(
    crow::SimpleApp& app,
    const SharedState& state) {

    // GET /api/metrics — Prometheus text format
    CROW_ROUTE(app, "/api/metrics").methods("GET"_method)
    ([&state]() -> crow::response {
        auto text = state.metrics.to_prometheus();

        // 追加 GPU 显存指标（如果可用）
        auto gpu = GpuMonitor::memory_info();
        if (gpu.available) {
            text += "\n# HELP ail_gpu_memory_mb GPU memory usage in MB\n";
            text += "# TYPE ail_gpu_memory_mb gauge\n";
            text += "ail_gpu_memory_mb{type=\"total\"} " + std::to_string(gpu.total_mb) + "\n";
            text += "ail_gpu_memory_mb{type=\"used\"} " + std::to_string(gpu.used_mb) + "\n";
            text += "ail_gpu_memory_mb{type=\"free\"} " + std::to_string(gpu.free_mb) + "\n";
            text += "ail_gpu_device_info{name=\"" + gpu.device_name + "\",id=\"" +
                    std::to_string(gpu.device_id) + "\"} 1\n";
        }

        crow::response res{200, text};
        res.set_header("Content-Type", "text/plain; version=0.0.4");
        return res;
    });

    // GET /api/metrics/json — 人类可读的 JSON 摘要
    CROW_ROUTE(app, "/api/metrics/json").methods("GET"_method)
    ([&state]() -> crow::response {
        auto [p50, p95, p99] = state.metrics.latency_percentiles();

        json body;
        body["requests_total"] = state.metrics.total_requests();
        body["latency_ms"] = {
            {"p50", p50},
            {"p95", p95},
            {"p99", p99},
            {"avg_60s", state.metrics.avg_latency_recent()}
        };

        auto gpu = GpuMonitor::memory_info();
        if (gpu.available) {
            body["gpu"] = {
                {"available", true},
                {"device_name", gpu.device_name},
                {"device_id", gpu.device_id},
                {"memory_mb", {
                    {"total", gpu.total_mb},
                    {"used", gpu.used_mb},
                    {"free", gpu.free_mb}
                }}
            };
        } else {
            body["gpu"] = {{"available", false}};
        }

        crow::response res{200, body.dump(2)};
        res.set_header("Content-Type", "application/json");
        return res;
    });

    log_info("metrics", "registered /api/metrics and /api/metrics/json");
}
