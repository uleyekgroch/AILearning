/**
 * @file domain_events.hpp
 * @brief 领域事件 — 所有状态变更的事件定义
 *
 * DDD 规则：所有状态变更必须通过领域事件，事件必须不可变。
 */
#pragma once

#include <chrono>
#include <string>
#include <variant>
#include <vector>

namespace ai_learning::domain {

/// 事件时间戳
using EventTime = std::chrono::steady_clock::time_point;

/// 基础事件 ID
using EventId = std::string;

// ── 知识领域事件 ────────────────────────────────────────────────

struct EntityCreated {
    EventId    event_id;
    EventTime  timestamp;
    std::string entity_id;
    std::string entity_type;
};

struct RelationAdded {
    EventId    event_id;
    EventTime  timestamp;
    std::string source_id;
    std::string target_id;
    std::string relation_type;
    double      confidence = 1.0;
};

struct KnowledgeLearned {
    EventId    event_id;
    EventTime  timestamp;
    std::string content;
    std::string source;
    double      confidence = 1.0;
};

// ── 学习领域事件 ────────────────────────────────────────────────

struct PredictionError {
    EventId    event_id;
    EventTime  timestamp;
    double     error_magnitude = 0.0;
    double     confidence      = 0.0;
};

struct StageAdvanced {
    EventId    event_id;
    EventTime  timestamp;
    std::string from_stage;
    std::string to_stage;
};

// ── 语言领域事件 ────────────────────────────────────────────────

struct VocabularyRecorded {
    EventId    event_id;
    EventTime  timestamp;
    std::string symbol;
    bool       success = false;
};

struct GrammarRuleExtracted {
    EventId    event_id;
    EventTime  timestamp;
    std::string pattern;
    double      confidence = 0.0;
};

// ── 记忆领域事件 ────────────────────────────────────────────────

struct MemoryStored {
    EventId    event_id;
    EventTime  timestamp;
    std::string memory_id;
    double      importance = 0.5;
};

struct MemoryConsolidated {
    EventId    event_id;
    EventTime  timestamp;
    int         consolidated_count = 0;
    int         forgotten_count    = 0;
};

// ── 统一事件类型 ────────────────────────────────────────────────
using DomainEvent = std::variant<
    EntityCreated,
    RelationAdded,
    KnowledgeLearned,
    PredictionError,
    StageAdvanced,
    VocabularyRecorded,
    GrammarRuleExtracted,
    MemoryStored,
    MemoryConsolidated
>;

/// 事件发布器接口
class IEventPublisher {
public:
    virtual ~IEventPublisher() = default;
    virtual void publish(DomainEvent event) = 0;
};

/// 事件订阅器接口
class IEventSubscriber {
public:
    virtual ~IEventSubscriber() = default;
    virtual void on_event(const DomainEvent& event) = 0;
};

}  // namespace ai_learning::domain
