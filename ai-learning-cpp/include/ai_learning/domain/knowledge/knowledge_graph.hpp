/**
 * @file knowledge_graph.hpp
 * @brief 知识图谱 — 聚合根
 *
 * DDD 聚合根：知识图谱是事务边界。
 * - 内部对象（Entity、Relation）只能通过本类访问
 * - 聚合根之间通过 ID 引用，不直接引用对象
 * - 所有状态变更发布领域事件
 *
 * 存储 Entity（节点）和 Relation（边），支持查询、路径查找、聚合。
 * 对应 Python 版 KnowledgeGraph。
 */
#pragma once

#include "ai_learning/domain/knowledge/entity.hpp"
#include "ai_learning/domain/knowledge/relation.hpp"
#include "ai_learning/domain/domain_events.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_learning::domain::knowledge {

/// 聚合统计结果
struct AggregateResult {
    std::string property_name;
    std::map<std::string, int> value_counts;
};

/// 路径查询结果
struct PathResult {
    std::vector<std::string> node_ids;
    bool found = false;
};

class KnowledgeGraph {
public:
    explicit KnowledgeGraph(
        domain::IEventPublisher* publisher = nullptr)
        : publisher_(publisher) {}

    // ── 增删（发布事件） ────────────────────────────────────────

    /// 添加实体（已存在则合并属性）。返回实体 ID。
    auto add_entity(Entity entity) -> const std::string&;

    /// 添加关系（已存在则加强）。
    void add_relation(Relation relation);

    /// 删除实体及其所有关系。
    void remove_entity(const std::string& entity_id);

    /// 获取或创建实体
    auto get_or_create(const std::string& entity_id,
                       const std::string& entity_type) -> const Entity&;

    // ── 查询 ────────────────────────────────────────────────────

    [[nodiscard]] auto get_entity(const std::string& id) const
        -> std::optional<std::reference_wrapper<const Entity>>;

    [[nodiscard]] auto has_entity(const std::string& id) const -> bool;

    /// 获取与某实体有关系的所有实体
    [[nodiscard]] auto get_related(
        const std::string& entity_id,
        const std::string& relation_type = "") const
        -> std::vector<std::reference_wrapper<const Entity>>;

    /// 获取与实体相关的所有关系
    [[nodiscard]] auto get_relations_of(
        const std::string& entity_id,
        const std::string& direction = "both") const
        -> std::vector<std::reference_wrapper<const Relation>>;

    /// 按条件查询实体
    [[nodiscard]] auto query(
        const std::string& entity_type = "",
        const std::map<std::string, std::string>& properties = {},
        const std::string& tag = "") const
        -> std::vector<std::reference_wrapper<const Entity>>;

    /// BFS 查找两个实体之间的最短路径
    [[nodiscard]] auto find_path(const std::string& source_id,
                                 const std::string& target_id,
                                 int max_depth = 3) const -> PathResult;

    /// 获取 N 跳邻居
    [[nodiscard]] auto get_neighbors(const std::string& entity_id,
                                     int depth = 1) const
        -> std::map<std::string, std::reference_wrapper<const Entity>>;

    /// 聚合统计
    [[nodiscard]] auto aggregate(const std::string& entity_type,
                                 const std::string& property_name) const
        -> AggregateResult;

    // ── 统计 ────────────────────────────────────────────────────

    [[nodiscard]] auto entity_count() const -> int {
        return static_cast<int>(entities_.size());
    }
    [[nodiscard]] auto relation_count() const -> int {
        return static_cast<int>(relations_.size());
    }

    [[nodiscard]] auto type_distribution() const
        -> std::map<std::string, int>;

    /// 获取所有实体 ID
    [[nodiscard]] auto get_all_entity_ids() const
        -> std::vector<std::string> {
        std::vector<std::string> ids;
        ids.reserve(entities_.size());
        for (const auto& [id, _] : entities_) ids.push_back(id);
        return ids;
    }

    /// 获取实体属性
    [[nodiscard]] auto get_entity_properties(
        const std::string& id) const
        -> std::map<std::string, std::string> {
        auto it = entities_.find(id);
        if (it == entities_.end()) return {};
        return it->second.properties();
    }

    // ── 持久化 ──────────────────────────────────────────────────

    struct State {
        std::map<std::string, Entity>   entities;
        std::vector<Relation>           relations;
    };

    [[nodiscard]] auto save_state() const -> State;
    void load_state(const State& state);
    void clear();

private:
    void rebuild_indices_();
    void publish_event_(DomainEvent event) const;

    // 数据存储
    std::map<std::string, Entity> entities_;
    std::vector<Relation>         relations_;

    // 索引
    std::unordered_map<std::string, std::vector<int>> outgoing_;  // entity_id → relation 索引
    std::unordered_map<std::string, std::vector<int>> incoming_;
    std::unordered_map<std::string, std::unordered_set<std::string>> type_index_;
    std::unordered_map<std::string, std::unordered_set<std::string>> tag_index_;

    // 事件发布器（可选，不拥有）
    domain::IEventPublisher* publisher_;
};

}  // namespace ai_learning::domain::knowledge
