/**
 * @file knowledge_graph.cpp
 * @brief 知识图谱聚合根实现
 */

#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

#include <chrono>

namespace ai_learning::domain::knowledge {

// ── 增删 ────────────────────────────────────────────────────────

auto KnowledgeGraph::add_entity(Entity entity) -> const std::string& {
    auto id = entity.id();  // 值拷贝，避免 move 后悬空

    if (entities_.contains(id)) {
        // 合并属性
        auto& existing = entities_.at(id);
        for (const auto& [k, v] : entity.properties()) {
            existing = existing.with_property(k, v);
        }
        for (const auto& tag : entity.tags()) {
            if (!existing.has_tag(tag)) {
                existing = existing.with_tag(tag);
                tag_index_[tag].insert(id);
            }
        }
    } else {
        auto type = entity.type();
        type_index_[type].insert(id);
        for (const auto& tag : entity.tags()) {
            tag_index_[tag].insert(id);
        }
        entities_.emplace(id, std::move(entity));
        publish_event_(EntityCreated{
            "evt_" + id, std::chrono::steady_clock::now(), id,
            entities_.at(id).type()});
    }
    // 注意：返回的是 entities_ 中 key 的引用，生命周期由 map 管理
    return entities_.find(id)->first;
}

void KnowledgeGraph::add_relation(Relation relation) {
    auto src = relation.source_id();  // 值拷贝，避免 move 后悬空
    auto tgt = relation.target_id();
    auto typ = relation.type();

    // 检查是否已存在相同 source-target-type 的关系
    for (int idx : outgoing_[src]) {
        auto& existing = relations_[idx];
        if (existing.target_id() == tgt && existing.type() == typ) {
            existing = existing.strengthen();
            return;
        }
    }

    int new_idx = static_cast<int>(relations_.size());
    relations_.push_back(std::move(relation));
    outgoing_[src].push_back(new_idx);
    incoming_[tgt].push_back(new_idx);

    publish_event_(RelationAdded{
        "evt_rel_" + std::to_string(new_idx),
        std::chrono::steady_clock::now(), std::move(src), std::move(tgt), std::move(typ), 1.0});
}

void KnowledgeGraph::remove_entity(const std::string& entity_id) {
    if (!entities_.contains(entity_id)) return;

    const auto& entity = entities_.at(entity_id);
    type_index_[entity.type()].erase(entity_id);
    for (const auto& tag : entity.tags()) {
        tag_index_[tag].erase(entity_id);
    }

    entities_.erase(entity_id);

    // 删除相关关系并重建索引
    relations_.erase(
        std::remove_if(relations_.begin(), relations_.end(),
            [&entity_id](const Relation& r) {
                return r.source_id() == entity_id ||
                       r.target_id() == entity_id;
            }),
        relations_.end());
    rebuild_indices_();
}

auto KnowledgeGraph::get_or_create(const std::string& entity_id,
                                   const std::string& entity_type)
    -> const Entity& {
    if (!entities_.contains(entity_id)) {
        add_entity(Entity(entity_id, entity_type));
    }
    return entities_.at(entity_id);
}

// ── 查询 ────────────────────────────────────────────────────────

auto KnowledgeGraph::get_entity(const std::string& id) const
    -> std::optional<std::reference_wrapper<const Entity>> {
    auto it = entities_.find(id);
    if (it != entities_.end()) return std::cref(it->second);
    return std::nullopt;
}

auto KnowledgeGraph::has_entity(const std::string& id) const -> bool {
    return entities_.contains(id);
}

auto KnowledgeGraph::get_related(
    const std::string& entity_id,
    const std::string& relation_type) const
    -> std::vector<std::reference_wrapper<const Entity>> {
    std::vector<std::reference_wrapper<const Entity>> result;
    auto it = outgoing_.find(entity_id);
    if (it == outgoing_.end()) return result;

    for (int idx : it->second) {
        const auto& rel = relations_[idx];
        if (!relation_type.empty() && rel.type() != relation_type) continue;
        auto eit = entities_.find(rel.target_id());
        if (eit != entities_.end()) {
            result.push_back(std::cref(eit->second));
        }
    }
    return result;
}

auto KnowledgeGraph::get_relations_of(
    const std::string& entity_id,
    const std::string& direction) const
    -> std::vector<std::reference_wrapper<const Relation>> {
    std::vector<std::reference_wrapper<const Relation>> result;

    if (direction == "out" || direction == "both") {
        auto it = outgoing_.find(entity_id);
        if (it != outgoing_.end()) {
            for (int idx : it->second) {
                result.push_back(std::cref(relations_[idx]));
            }
        }
    }
    if (direction == "in" || direction == "both") {
        auto it = incoming_.find(entity_id);
        if (it != incoming_.end()) {
            for (int idx : it->second) {
                result.push_back(std::cref(relations_[idx]));
            }
        }
    }
    return result;
}

auto KnowledgeGraph::query(
    const std::string& entity_type,
    const std::map<std::string, std::string>& properties,
    const std::string& tag) const
    -> std::vector<std::reference_wrapper<const Entity>> {
    std::unordered_set<std::string> candidates;

    // 按类型筛选
    if (!entity_type.empty()) {
        auto it = type_index_.find(entity_type);
        if (it != type_index_.end()) candidates = it->second;
    }
    // 按标签筛选
    if (!tag.empty()) {
        auto it = tag_index_.find(tag);
        if (it != tag_index_.end()) {
            if (candidates.empty()) {
                candidates = it->second;
            } else {
                std::unordered_set<std::string> intersection;
                for (const auto& id : candidates) {
                    if (it->second.contains(id)) intersection.insert(id);
                }
                candidates = std::move(intersection);
            }
        }
    }
    if (candidates.empty() && entity_type.empty() && tag.empty()) {
        for (const auto& [id, _] : entities_) candidates.insert(id);
    }

    std::vector<std::reference_wrapper<const Entity>> result;
    for (const auto& eid : candidates) {
        auto it = entities_.find(eid);
        if (it == entities_.end()) continue;
        if (properties.empty() || it->second.matches(properties)) {
            result.push_back(std::cref(it->second));
        }
    }
    return result;
}

auto KnowledgeGraph::find_path(const std::string& source_id,
                               const std::string& target_id,
                               int max_depth) const -> PathResult {
    if (!entities_.contains(source_id) || !entities_.contains(target_id)) {
        return {{}, false};
    }
    if (source_id == target_id) {
        return {{source_id}, true};
    }

    std::unordered_set<std::string> visited{source_id};
    std::queue<std::pair<std::string, std::vector<std::string>>> queue;
    queue.push({source_id, {source_id}});

    while (!queue.empty()) {
        auto [current, path] = std::move(queue.front());
        queue.pop();

        if (static_cast<int>(path.size()) > max_depth + 1) break;

        auto it = outgoing_.find(current);
        if (it == outgoing_.end()) continue;

        for (int idx : it->second) {
            const auto& next_id = relations_[idx].target_id();
            if (next_id == target_id) {
                path.push_back(next_id);
                return {path, true};
            }
            if (!visited.contains(next_id)) {
                visited.insert(next_id);
                auto new_path = path;
                new_path.push_back(next_id);
                queue.push({next_id, std::move(new_path)});
            }
        }
    }
    return {{}, false};
}

auto KnowledgeGraph::get_neighbors(const std::string& entity_id,
                                   int depth) const
    -> std::map<std::string, std::reference_wrapper<const Entity>> {
    std::map<std::string, std::reference_wrapper<const Entity>> result;
    std::unordered_set<std::string> frontier{entity_id};

    for (int d = 0; d < depth; ++d) {
        std::unordered_set<std::string> next_frontier;
        for (const auto& eid : frontier) {
            // 出边
            auto oit = outgoing_.find(eid);
            if (oit != outgoing_.end()) {
                for (int idx : oit->second) {
                    const auto& tid = relations_[idx].target_id();
                    if (tid != entity_id && !result.contains(tid)) {
                        auto eit = entities_.find(tid);
                        if (eit != entities_.end()) {
                            result.emplace(tid, std::cref(eit->second));
                            next_frontier.insert(tid);
                        }
                    }
                }
            }
            // 入边
            auto iit = incoming_.find(eid);
            if (iit != incoming_.end()) {
                for (int idx : iit->second) {
                    const auto& sid = relations_[idx].source_id();
                    if (sid != entity_id && !result.contains(sid)) {
                        auto eit = entities_.find(sid);
                        if (eit != entities_.end()) {
                            result.emplace(sid, std::cref(eit->second));
                            next_frontier.insert(sid);
                        }
                    }
                }
            }
        }
        frontier = std::move(next_frontier);
    }
    return result;
}

auto KnowledgeGraph::aggregate(const std::string& entity_type,
                               const std::string& property_name) const
    -> AggregateResult {
    AggregateResult result{property_name, {}};
    auto it = type_index_.find(entity_type);
    if (it == type_index_.end()) return result;

    for (const auto& eid : it->second) {
        auto eit = entities_.find(eid);
        if (eit == entities_.end()) continue;
        const auto& val = eit->second.get_property(property_name);
        if (!val.empty()) result.value_counts[val]++;
    }
    return result;
}

// ── 统计 ────────────────────────────────────────────────────────

auto KnowledgeGraph::type_distribution() const
    -> std::map<std::string, int> {
    std::map<std::string, int> dist;
    for (const auto& [type, ids] : type_index_) {
        dist[type] = static_cast<int>(ids.size());
    }
    return dist;
}

// ── 持久化 ──────────────────────────────────────────────────────

auto KnowledgeGraph::save_state() const -> State {
    return {entities_, relations_};
}

void KnowledgeGraph::load_state(const State& state) {
    clear();
    for (const auto& [id, entity] : state.entities) {
        entities_.emplace(id, entity);
        type_index_[entity.type()].insert(id);
        for (const auto& tag : entity.tags()) {
            tag_index_[tag].insert(id);
        }
    }
    for (const auto& rel : state.relations) {
        int idx = static_cast<int>(relations_.size());
        relations_.push_back(rel);
        outgoing_[rel.source_id()].push_back(idx);
        incoming_[rel.target_id()].push_back(idx);
    }
}

void KnowledgeGraph::clear() {
    entities_.clear();
    relations_.clear();
    outgoing_.clear();
    incoming_.clear();
    type_index_.clear();
    tag_index_.clear();
}

// ── 内部 ────────────────────────────────────────────────────────

void KnowledgeGraph::rebuild_indices_() {
    outgoing_.clear();
    incoming_.clear();
    for (int i = 0; i < static_cast<int>(relations_.size()); ++i) {
        outgoing_[relations_[i].source_id()].push_back(i);
        incoming_[relations_[i].target_id()].push_back(i);
    }
}

void KnowledgeGraph::publish_event_(DomainEvent event) const {
    if (auto pub = publisher_.lock()) {
        pub->publish(std::move(event));
    }
}

}  // namespace ai_learning::domain::knowledge
