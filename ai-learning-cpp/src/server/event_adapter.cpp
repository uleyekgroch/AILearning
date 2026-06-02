/**
 * @file event_adapter.cpp
 * @brief 事件适配器实现 — DomainEvent → JSON 转换与 WebSocket 广播
 */

#include "event_adapter.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace ai_learning::server {

using json = nlohmann::json;

// ── EventAdapter ────────────────────────────────────────────────

void EventAdapter::on_event(const domain::DomainEvent& event) {
    auto msg = event_to_json_(event);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_event_(msg);
    }
    broadcast_(msg);
}

void EventAdapter::add_connection(crow::websocket::connection* conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.insert(conn);
    std::cout << "[WS/events] Client connected. Total: "
              << connections_.size() << "\n";
}

void EventAdapter::remove_connection(crow::websocket::connection* conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.erase(conn);
    std::cout << "[WS/events] Client disconnected. Total: "
              << connections_.size() << "\n";
}

auto EventAdapter::get_history_json() const -> std::string {
    std::lock_guard<std::mutex> lock(mutex_);
    json arr = json::array();
    for (const auto& evt : event_buffer_) {
        arr.push_back(json::parse(evt));
    }
    return arr.dump();
}

auto EventAdapter::connection_count() const -> std::size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return connections_.size();
}

void EventAdapter::publish_custom(const std::string& type,
                                   const std::string& data_json) {
    json msg;
    msg["type"] = type;
    msg["timestamp"] = format_timestamp_(std::chrono::steady_clock::now());
    msg["data"] = json::parse(data_json);
    auto msg_str = msg.dump();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_event_(msg_str);
    }
    broadcast_(msg_str);
}

// ── 私有方法 ──────────────────────────────────────────────────

auto EventAdapter::event_to_json_(const domain::DomainEvent& event)
    -> std::string {
    json msg;

    std::visit([&msg](const auto& e) {
        using T = std::decay_t<decltype(e)>;
        json data;

        if constexpr (std::is_same_v<T, domain::EntityCreated>) {
            msg["type"] = "learn";
            data["subtype"] = "entity_created";
            data["entity_id"] = e.entity_id;
            data["entity_type"] = e.entity_type;
        } else if constexpr (std::is_same_v<T, domain::RelationAdded>) {
            msg["type"] = "learn";
            data["subtype"] = "relation_added";
            data["source_id"] = e.source_id;
            data["target_id"] = e.target_id;
            data["relation_type"] = e.relation_type;
            data["confidence"] = e.confidence;
        } else if constexpr (std::is_same_v<T, domain::KnowledgeLearned>) {
            msg["type"] = "learn";
            data["subtype"] = "knowledge_learned";
            data["content"] = e.content;
            data["source"] = e.source;
            data["confidence"] = e.confidence;
        } else if constexpr (std::is_same_v<T, domain::PredictionError>) {
            msg["type"] = "reason";
            data["subtype"] = "prediction_error";
            data["error_magnitude"] = e.error_magnitude;
            data["confidence"] = e.confidence;
        } else if constexpr (std::is_same_v<T, domain::StageAdvanced>) {
            msg["type"] = "stage_change";
            data["from_stage"] = e.from_stage;
            data["to_stage"] = e.to_stage;
        } else if constexpr (std::is_same_v<T, domain::VocabularyRecorded>) {
            msg["type"] = "learn";
            data["subtype"] = "vocabulary_recorded";
            data["symbol"] = e.symbol;
            data["success"] = e.success;
        } else if constexpr (std::is_same_v<T, domain::GrammarRuleExtracted>) {
            msg["type"] = "learn";
            data["subtype"] = "grammar_rule_extracted";
            data["pattern"] = e.pattern;
            data["confidence"] = e.confidence;
        } else if constexpr (std::is_same_v<T, domain::MemoryStored>) {
            msg["type"] = "remember";
            data["memory_id"] = e.memory_id;
            data["importance"] = e.importance;
        } else if constexpr (std::is_same_v<T, domain::MemoryConsolidated>) {
            msg["type"] = "remember";
            data["subtype"] = "memory_consolidated";
            data["consolidated_count"] = e.consolidated_count;
            data["forgotten_count"] = e.forgotten_count;
        }

        msg["timestamp"] = format_timestamp_(e.timestamp);
        msg["data"] = data;
    }, event);

    return msg.dump();
}

auto EventAdapter::format_timestamp_(const domain::EventTime& tp) -> std::string {
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        tp.time_since_epoch()).count();
    std::time_t t = static_cast<std::time_t>(secs);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

void EventAdapter::broadcast_(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* conn : connections_) {
        try {
            conn->send_text(message);
        } catch (const std::exception& e) {
            std::cerr << "[WS/events] send_text failed: " << e.what() << "\n";
        }
    }
}

void EventAdapter::buffer_event_(std::string event_json) {
    event_buffer_.push_back(std::move(event_json));
    while (event_buffer_.size() > kEventBufferSize) {
        event_buffer_.pop_front();
    }
}

// ── StatsPusher ─────────────────────────────────────────────────

void StatsPusher::add_connection(crow::websocket::connection* conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.insert(conn);
    std::cout << "[WS/stats] Client connected. Total: "
              << connections_.size() << "\n";
}

void StatsPusher::remove_connection(crow::websocket::connection* conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.erase(conn);
    std::cout << "[WS/stats] Client disconnected. Total: "
              << connections_.size() << "\n";
}

void StatsPusher::push_stats(const nlohmann::json& stats) {
    std::string msg = stats.dump();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* conn : connections_) {
        try {
            conn->send_text(msg);
        } catch (const std::exception& e) {
            std::cerr << "[WS/stats] send_text failed: " << e.what() << "\n";
        }
    }
}

auto StatsPusher::connection_count() const -> std::size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return connections_.size();
}

// ── HeartbeatManager ────────────────────────────────────────────

void HeartbeatManager::add_connection(crow::websocket::connection* conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_[conn] = std::chrono::steady_clock::now();
}

void HeartbeatManager::remove_connection(crow::websocket::connection* conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.erase(conn);
}

void HeartbeatManager::touch(crow::websocket::connection* conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = connections_.find(conn);
    if (it != connections_.end()) {
        it->second = std::chrono::steady_clock::now();
    }
}

void HeartbeatManager::send_pings() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [conn, _] : connections_) {
        try {
            conn->send_ping("");
        } catch (const std::exception& e) {
            std::cerr << "[WS/heartbeat] ping failed: " << e.what() << "\n";
        }
    }
}

auto HeartbeatManager::check_timeouts(std::chrono::seconds timeout)
    -> std::vector<crow::websocket::connection*> {
    auto now = std::chrono::steady_clock::now();
    std::vector<crow::websocket::connection*> expired;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [conn, last_active] : connections_) {
        if (now - last_active > timeout) {
            expired.push_back(conn);
        }
    }
    return expired;
}

}  // namespace ai_learning::server
