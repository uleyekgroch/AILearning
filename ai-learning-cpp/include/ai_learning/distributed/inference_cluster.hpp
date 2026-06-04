/**
 * @file inference_cluster.hpp
 * @brief 分布式推理集群 — 多节点负载均衡
 *
 * 设计目标：
 *   - 将推理请求路由到多个后端节点
 *   - 支持健康检查和故障转移
 *   - 简单的轮询负载均衡（未来可扩展为加权/最少连接）
 *
 * 架构：
 *   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
 *   │   Node 1    │     │   Node 2    │     │   Node N    │
 *   │  (GPU 0)    │◄────┤  (GPU 1)    │◄────┤  (GPU 2)    │
 *   └─────────────┘     └─────────────┘     └─────────────┘
 *          ▲                                    ▲
 *          └────────────┬────────────────────────┘
 *                       │
 *              ┌────────┴────────┐
 *              │ Load Balancer │
 *              │  (Round Robin)  │
 *              └────────┬────────┘
 *                       │
 *                  Client Request
 *
 * 未来扩展：
 *   - gRPC 通信（当前为 REST/HTTP 占位）
 *   - 一致性哈希路由
 *   - 自动扩缩容
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ai_learning::distributed {

/// 推理节点状态
struct InferenceNode {
    std::string id;                    // 节点唯一标识
    std::string endpoint;              // 服务地址 (host:port)
    std::string gpu_info;              // GPU 信息
    int max_concurrent = 1;            // 最大并发请求数
    std::atomic<int> active_requests{0};  // 当前活跃请求数
    std::atomic<bool> healthy{true};      // 健康状态
    std::chrono::steady_clock::time_point last_heartbeat;

    InferenceNode() = default;
    InferenceNode(const InferenceNode& other) : id(other.id), endpoint(other.endpoint), gpu_info(other.gpu_info), max_concurrent(other.max_concurrent), active_requests(other.active_requests.load()), healthy(other.healthy.load()), last_heartbeat(other.last_heartbeat) {}
    InferenceNode(InferenceNode&& other) noexcept : id(std::move(other.id)), endpoint(std::move(other.endpoint)), gpu_info(std::move(other.gpu_info)), max_concurrent(other.max_concurrent), active_requests(other.active_requests.load()), healthy(other.healthy.load()), last_heartbeat(other.last_heartbeat) {}
    InferenceNode& operator=(const InferenceNode& other) {
        if (this != &other) {
            id = other.id;
            endpoint = other.endpoint;
            gpu_info = other.gpu_info;
            max_concurrent = other.max_concurrent;
            active_requests.store(other.active_requests.load());
            healthy.store(other.healthy.load());
            last_heartbeat = other.last_heartbeat;
        }
        return *this;
    }
    InferenceNode& operator=(InferenceNode&& other) noexcept {
        if (this != &other) {
            id = std::move(other.id);
            endpoint = std::move(other.endpoint);
            gpu_info = std::move(other.gpu_info);
            max_concurrent = other.max_concurrent;
            active_requests.store(other.active_requests.load());
            healthy.store(other.healthy.load());
            last_heartbeat = other.last_heartbeat;
        }
        return *this;
    }
};

/// 分布式推理集群 — 负载均衡器
class InferenceCluster {
public:
    InferenceCluster() = default;

    /// 注册新节点
    void register_node(const std::string& id,
                        const std::string& endpoint,
                        const std::string& gpu_info = "",
                        int max_concurrent = 1);

    /// 注销节点
    void unregister_node(const std::string& id);

    /// 获取下一个可用节点（轮询）
    /// @return 节点 ID，若无可用节点返回空字符串
    auto next_node() -> std::string;

    /// 报告节点心跳
    void heartbeat(const std::string& id);

    /// 标记节点健康状态
    void set_node_health(const std::string& id, bool healthy);

    /// 获取集群统计
    auto cluster_stats() const -> std::map<std::string, std::string>;

    /// 获取所有节点
    auto nodes() const -> std::vector<InferenceNode>;

    /// 节点数量
    [[nodiscard]] auto node_count() const -> size_t { return nodes_.size(); }

    /// 健康节点数量
    [[nodiscard]] auto healthy_node_count() const -> size_t;

private:
    mutable std::mutex mutex_;
    std::vector<InferenceNode> nodes_;
    std::atomic<size_t> next_index_{0};
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline void InferenceCluster::register_node(
    const std::string& id,
    const std::string& endpoint,
    const std::string& gpu_info,
    int max_concurrent) {
    std::lock_guard<std::mutex> lock(mutex_);
    // 检查是否已存在
    for (auto& n : nodes_) {
        if (n.id == id) {
            n.endpoint = endpoint;
            n.gpu_info = gpu_info;
            n.max_concurrent = max_concurrent;
            n.healthy = true;
            n.last_heartbeat = std::chrono::steady_clock::now();
            return;
        }
    }
    InferenceNode node;
    node.id = id;
    node.endpoint = endpoint;
    node.gpu_info = gpu_info;
    node.max_concurrent = max_concurrent;
    node.healthy = true;
    node.last_heartbeat = std::chrono::steady_clock::now();
    nodes_.push_back(std::move(node));
}

inline void InferenceCluster::unregister_node(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&id](const auto& n) { return n.id == id; }),
        nodes_.end());
}

inline auto InferenceCluster::next_node() -> std::string {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nodes_.empty()) return "";

    const size_t start = next_index_.fetch_add(1) % nodes_.size();
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const size_t idx = (start + i) % nodes_.size();
        auto& node = nodes_[idx];
        if (node.healthy.load() &&
            node.active_requests.load() < node.max_concurrent) {
            node.active_requests.fetch_add(1);
            return node.id;
        }
    }
    return "";  // 无可用节点
}

inline void InferenceCluster::heartbeat(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& n : nodes_) {
        if (n.id == id) {
            n.last_heartbeat = std::chrono::steady_clock::now();
            n.healthy = true;
            return;
        }
    }
}

inline void InferenceCluster::set_node_health(
    const std::string& id, bool healthy) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& n : nodes_) {
        if (n.id == id) {
            n.healthy = healthy;
            return;
        }
    }
}

inline auto InferenceCluster::cluster_stats() const
    -> std::map<std::string, std::string> {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::string> stats;
    stats["total_nodes"] = std::to_string(nodes_.size());
    size_t healthy = 0;
    int total_active = 0;
    for (const auto& n : nodes_) {
        if (n.healthy.load()) healthy++;
        total_active += n.active_requests.load();
    }
    stats["healthy_nodes"] = std::to_string(healthy);
    stats["active_requests"] = std::to_string(total_active);
    return stats;
}

inline auto InferenceCluster::nodes() const -> std::vector<InferenceNode> {
    std::lock_guard<std::mutex> lock(mutex_);
    return nodes_;
}

inline auto InferenceCluster::healthy_node_count() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& n : nodes_) {
        if (n.healthy.load()) count++;
    }
    return count;
}

}  // namespace ai_learning::distributed
