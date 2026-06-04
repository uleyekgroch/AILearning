/**
 * @file text_learner.cpp
 * @brief 文本学习者实现
 */

#include "ai_learning/learning/text_learner.hpp"
#include "ai_learning/domain/domain_events.hpp"

#include <algorithm>
#include <sstream>
#include <chrono>

namespace ai_learning::learning {

using namespace domain::knowledge;

// ── 构造 ────────────────────────────────────────────────────────

TextLearner::TextLearner(KnowledgeGraph& kg)
    : kg_(kg) {
    stats_["total_learned"] = 0;
    stats_["verified"] = 0;
    stats_["failed"] = 0;
}

// ── 核心学习管线 ────────────────────────────────────────────────

auto TextLearner::learn_from_text(const std::string& text,
                                   const std::string& source)
    -> TextLearnResult {
    TextLearnResult result;

    // 1. 提取实体
    result.entities = KnowledgeExtractor::extract_entities(text);

    // 1b. 直接将提取到的实体注入 KG（extract_entities 已做质量过滤）
    for (const auto& entity_name : result.entities) {
        kg_.add_entity(Entity(entity_name, "concept", {}, 0.7, source));
    }

    // 2. 提取关系
    result.triples = KnowledgeExtractor::extract_triples(text, result.entities);

    // 3. 注入知识图谱（带矛盾检测）
    for (const auto& triple : result.triples) {
        if (triple.subject.empty() || triple.object.empty()) continue;

        auto conflict = check_contradiction_(
            triple.subject, triple.relation, triple.object);

        if (conflict) {
            resolve_contradiction_(triple.subject, triple.relation,
                                   triple.object, *conflict, source);
        } else {
            kg_.add_entity(Entity(triple.subject, "concept", {}, 0.9, source));
            kg_.add_entity(Entity(triple.object, "concept", {}, 0.9, source));
            kg_.add_relation(Relation(triple.subject, triple.object,
                                       triple.relation, 0.9));
        }
    }

    // 4. 因果提取
    result.causal_links = KnowledgeExtractor::extract_causal_links(text);
    for (const auto& link : result.causal_links) {
        kg_.add_entity(Entity(link.cause, "event", {}, 0.9, source));
        kg_.add_entity(Entity(link.effect, "event", {}, 0.9, source));
        kg_.add_relation(Relation(link.cause, link.effect, "导致", 0.9));
    }

    // 5. 数值提取
    result.numerical_facts = KnowledgeExtractor::extract_numerical_facts(text);
    for (const auto& fact : result.numerical_facts) {
        auto fact_id = fact.attribute + "_" + std::to_string(fact.value);
        kg_.add_entity(Entity(fact_id, "numerical",
                               {{"value", std::to_string(fact.value)},
                                {"unit", fact.unit}},
                               0.8, source));
    }

    // 6. STDP 学习
    update_stdp_connections_(result.entities);

    // 7. 海马记忆存储
    store_hippocampal_episode_(text, result.entities, result.triples);

    // 8. 验证
    auto [passed, score] = verify_knowledge_(text, result.entities, result.triples);
    result.verification_passed = passed;
    result.verification_score = score;

    // 9. 统计
    stats_["total_learned"]++;
    if (passed) {
        stats_["verified"]++;
    } else {
        stats_["failed"]++;
    }

    return result;
}

// ── 思考/回答 ────────────────────────────────────────────────────

auto TextLearner::think(const std::string& question) const
    -> std::string {
    // 路径 0: 常识库查询
    auto common = query_commonsense_(question);
    if (!common.empty()) return common;

    // 路径 1: STDP 连接推理
    auto q_entities = KnowledgeExtractor::extract_entities(question);
    if (!q_entities.empty()) {
        for (const auto& entity : q_entities) {
            auto related = query_stdp_(entity, 3);
            if (!related.empty()) {
                std::string answer = entity + "与";
                for (size_t i = 0; i < std::min(related.size(), size_t(3)); ++i) {
                    if (i > 0) answer += "、";
                    answer += related[i].first;
                }
                answer += "有较强的时序关联";
                return answer;
            }
        }
    }

    // 路径 2: 海马记忆检索
    for (const auto& entity : q_entities) {
        auto memory = query_hippocampal_(entity);
        if (memory && memory->size() > 5) {
            return "根据记忆，" + memory->substr(0, 50);
        }
    }

    // 路径 3: 知识图谱推理
    auto kg_answer = reason_from_kg_(question);
    if (!kg_answer.empty()) return kg_answer;

    return "抱歉，我暂时不知道答案";
}

// ── 私有方法 ─────────────────────────────────────────────────────

auto TextLearner::verify_knowledge_(
    const std::string& /*text*/,
    const std::vector<std::string>& /*entities*/,
    const std::vector<Triple>& triples) const
    -> std::pair<bool, double> {
    if (triples.empty()) return {true, 1.0};

    int passed = 0;
    for (const auto& t : triples) {
        auto rels = kg_.get_relations_of(t.subject);
        bool found = false;
        for (const auto& r : rels) {
            if (r.get().target_id() == t.object) {
                found = true;
                break;
            }
        }
        if (found) passed++;
    }

    auto score = static_cast<double>(passed) / static_cast<double>(triples.size());
    return {passed == static_cast<int>(triples.size()), score};
}

auto TextLearner::check_contradiction_(
    const std::string& subject,
    const std::string& relation,
    const std::string& obj) const
    -> std::optional<std::map<std::string, std::string>> {
    auto rels = kg_.get_relations_of(subject);
    for (const auto& r : rels) {
        if (r.get().type() == relation && r.get().target_id() != obj) {
            return std::map<std::string, std::string>{
                {"existing_obj", r.get().target_id()},
                {"new_obj", obj},
            };
        }
    }
    return std::nullopt;
}

void TextLearner::resolve_contradiction_(
    const std::string& subject,
    const std::string& relation,
    const std::string& obj,
    const std::map<std::string, std::string>& /*conflict*/,
    const std::string& source) {
    // 简单策略：保留新的（可扩展为来源可靠性比较）
    kg_.add_entity(Entity(subject, "concept", {}, 0.9, source));
    kg_.add_entity(Entity(obj, "concept", {}, 0.9, source));
    kg_.add_relation(Relation(subject, obj, relation, 0.9));
}

void TextLearner::update_stdp_connections_(
    const std::vector<std::string>& entities) {
    for (size_t i = 0; i + 1 < entities.size(); ++i) {
        auto key = std::make_pair(entities[i], entities[i + 1]);
        if (!stdp_connections_.contains(key)) {
            stdp_connections_[key] = 0.1f;
        }
        stdp_connections_[key] = std::min(1.0f,
            stdp_connections_[key] + stdp_lr_);
    }
}

auto TextLearner::query_stdp_(const std::string& entity, int top_k) const
    -> std::vector<std::pair<std::string, float>> {
    std::vector<std::pair<std::string, float>> results;

    for (const auto& [pair, weight] : stdp_connections_) {
        if (pair.first == entity && weight > 0.2f) {
            results.emplace_back(pair.second, weight);
        } else if (pair.second == entity && weight > 0.2f) {
            results.emplace_back(pair.first, weight);
        }
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }
    return results;
}

void TextLearner::store_hippocampal_episode_(
    const std::string& text,
    const std::vector<std::string>& entities,
    const std::vector<Triple>& triples) {
    HippocampalEpisode ep;
    ep.entities = entities;
    ep.triples = triples;
    ep.context = text;

    auto idx = static_cast<int>(hippocampal_episodes_.size());
    for (const auto& e : entities) {
        hippocampal_index_[e].push_back(idx);
    }
    hippocampal_episodes_.push_back(std::move(ep));

    // 容量管理
    while (static_cast<int>(hippocampal_episodes_.size()) > hippocampal_capacity_) {
        hippocampal_episodes_.erase(hippocampal_episodes_.begin());
        // 简化：重建索引（可优化）
        hippocampal_index_.clear();
        for (int i = 0; i < static_cast<int>(hippocampal_episodes_.size()); ++i) {
            for (const auto& e : hippocampal_episodes_[i].entities) {
                hippocampal_index_[e].push_back(i);
            }
        }
    }
}

auto TextLearner::query_hippocampal_(const std::string& entity) const
    -> std::optional<std::string> {
    auto it = hippocampal_index_.find(entity);
    if (it == hippocampal_index_.end() || it->second.empty()) return std::nullopt;

    auto idx = it->second.back();
    if (idx >= 0 && idx < static_cast<int>(hippocampal_episodes_.size())) {
        const auto& ep = hippocampal_episodes_[idx];
        if (!ep.context.empty() && ep.context.size() > 5) {
            return ep.context;
        }
    }
    return std::nullopt;
}

auto TextLearner::reason_from_kg_(const std::string& question) const
    -> std::string {
    auto entities = KnowledgeExtractor::extract_entities(question);
    if (entities.empty()) return "";

    auto& entity = entities[0];
    if (!kg_.has_entity(entity)) return "";

    auto rels = kg_.get_relations_of(entity);
    if (rels.empty()) return "";

    std::string answer = entity;
    for (const auto& r : rels) {
        answer += " " + r.get().type() + " " + r.get().target_id();
        if (answer.size() > 100) break;
    }
    return answer;
}

auto TextLearner::query_commonsense_(const std::string& question) const
    -> std::string {
    // 基于"自学知识图谱"的常识式作答（零大模型）。
    // 不再恒返回空：识别问句意图（定义/因果/位置），在 KG 里检索对应类型的
    // 关系，组合成自然语句。这与"原句回放"和"三元组裸拼接"不同——它按问句意图
    // 挑选关系类型并重新组织表达，因此放在 think() 最前面，可优先于海马原句回放。
    auto contains = [&](const char* kw) {
        return question.find(kw) != std::string::npos;
    };

    const bool is_definition =
        contains("是什么") || contains("什么是") || contains("是谁") ||
        contains("定义") || contains("是一种");
    const bool is_causal =
        contains("为什么") || contains("为何") || contains("原因") ||
        contains("会怎样") || contains("结果");
    const bool is_location = contains("在哪") || contains("哪里") || contains("位于");

    if (!is_definition && !is_causal && !is_location) return "";

    auto entities = KnowledgeExtractor::extract_entities(question);
    for (const auto& entity : entities) {
        if (!kg_.has_entity(entity)) continue;
        auto rels = kg_.get_relations_of(entity);
        if (rels.empty()) continue;

        // 按意图优先匹配的关系类型
        std::vector<std::string> want;
        if (is_definition) want = {"是", "属于", "称为", "叫做", "包括", "包含"};
        else if (is_causal) want = {"导致", "产生", "构成"};
        else /* location */ want = {"位于"};

        for (const auto& w : want) {
            for (const auto& r : rels) {
                if (r.get().type() != w) continue;
                const auto& obj = r.get().target_id();
                if (obj.empty() || obj == entity) continue;

                if (is_definition) {
                    if (w == "称为" || w == "叫做") {
                        return entity + " 又称 " + obj + "（基于已学知识）";
                    }
                    return entity + " " + w + " " + obj + "（基于已学知识）";
                }
                if (is_causal) {
                    return entity + " 会 " + w + " " + obj + "（基于已学知识）";
                }
                return entity + " 位于 " + obj + "（基于已学知识）";
            }
        }
    }
    return "";
}

}  // namespace ai_learning::learning
