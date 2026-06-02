/**
 * @file event_adapter.hpp
 * @brief 事件适配器 — 将内部领域事件转换为 JSON 并推送到 WebSocket 客户端
 *
 * 职责：
 * - 将 DomainEvent 转换为 JSON 格式 {"type":"...","timestamp":"...","data":{...}}
 * - 维护最近 100 条事件的环形缓冲区
 * - 管理 WebSocket 连接集合，广播事件到所有客户端
 *
 * 设计原则：
 * - 实现 IEventSubscriber 接口，无缝接入现有事件系统
 * - 不修改 Learner 核心代码
 * - 线程安全：所有公共方法都通过 mutex 保护
 */
#pragma once

#include "ai_learning/domain/domain_events.hpp"

#include <crow.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace ai_learning::server {

/// 环形缓冲区容量
inline constexpr std::size_t kEventBufferSize = 100;

/// 事件适配器 — 将 DomainEvent 转为 JSON 并广播到 WebSocket 连接
///
/// 使用方式：
///   1. 创建 EventAdapter 实例
///   2. 通过 add_connection / remove_connection 管理 WS 连接
///   3. 调用 publish_event 将事件广播到所有连接
///   4. get_history 返回缓冲区内所有历史事件 JSON
class EventAdapter : public domain::IEventSubscriber {
public:
    EventAdapter() = default;
    ~EventAdapter() override = default;

    // 禁止拷贝（含 mutex）
    EventAdapter(const EventAdapter&) = delete;
    auto operator=(const EventAdapter&) -> EventAdapter& = delete;

    /// IEventSubscriber 接口实现 — 接收领域事件并广播
    void on_event(const domain::DomainEvent& event) override;

    /// 注册 WebSocket 连接
    void add_connection(crow::websocket::connection* conn);

    /// 移除 WebSocket 连接
    void remove_connection(crow::websocket::connection* conn);

    /// 获取历史事件 JSON 数组字符串（发送给新连接）
    [[nodiscard]] auto get_history_json() const -> std::string;

    /// 获取当前连接数
    [[nodiscard]] auto connection_count() const -> std::size_t;

    /// 手动发布一条自定义事件（用于推送统计摘要等）
    void publish_custom(const std::string& type, const std::string& data_json);

private:
    /// 将 DomainEvent 转为 JSON 字符串
    static auto event_to_json_(const domain::DomainEvent& event) -> std::string;

    /// 将 EventTime 格式化为 ISO 8601 字符串
    static auto format_timestamp_(const domain::EventTime& tp) -> std::string;

    /// 广播消息到所有连接
    void broadcast_(const std::string& message);

    /// 向环形缓冲区添加事件
    void buffer_event_(std::string event_json);

    mutable std::mutex mutex_;

    /// WebSocket 连接集合
    std::set<crow::websocket::connection*> connections_;

    /// 环形缓冲区（最多 kEventBufferSize 条）
    std::deque<std::string> event_buffer_;
};

/// 统计推送管理器 — 每秒向 /ws/stats 连接推送学习统计
class StatsPusher {
public:
    /// 获取统计数据的回调类型
    using StatsProvider = std::function<nlohmann::json()>;

    StatsPusher() = default;
    ~StatsPusher() = default;

    StatsPusher(const StatsPusher&) = delete;
    auto operator=(const StatsPusher&) -> StatsPusher& = delete;

    /// 注册连接
    void add_connection(crow::websocket::connection* conn);

    /// 移除连接
    void remove_connection(crow::websocket::connection* conn);

    /// 推送一次统计到所有连接
    void push_stats(const nlohmann::json& stats);

    /// 获取当前连接数
    [[nodiscard]] auto connection_count() const -> std::size_t;

private:
    mutable std::mutex mutex_;
    std::set<crow::websocket::connection*> connections_;
};

/// 心跳管理器 — 定期 ping，超时断开
class HeartbeatManager {
public:
    HeartbeatManager() = default;
    ~HeartbeatManager() = default;

    HeartbeatManager(const HeartbeatManager&) = delete;
    auto operator=(const HeartbeatManager&) -> HeartbeatManager& = delete;

    /// 注册连接并记录最后一次活动时间
    void add_connection(crow::websocket::connection* conn);

    /// 移除连接
    void remove_connection(crow::websocket::connection* conn);

    /// 更新连接活动时间（收到消息或 pong 时调用）
    void touch(crow::websocket::connection* conn);

    /// 向所有连接发送 ping
    void send_pings();

    /// 检查超时连接，返回需要断开的连接列表
    [[nodiscard]] auto check_timeouts(
        std::chrono::seconds timeout = std::chrono::seconds{60})
        -> std::vector<crow::websocket::connection*>;

private:
    mutable std::mutex mutex_;
    std::map<crow::websocket::connection*,
             std::chrono::steady_clock::time_point> connections_;
};

}  // namespace ai_learning::server
