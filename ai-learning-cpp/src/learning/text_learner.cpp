/**
 * @file text_learner.cpp
 * @brief 文本学习者实现
 */

#include "ai_learning/learning/text_learner.hpp"
#include "ai_learning/domain/domain_events.hpp"
#include "ai_learning/utils/utf8.hpp"

#include <algorithm>
#include <map>
#include <set>
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

    // 0. 累积原始语料，供 train_dependency_parser() 训练无监督分词/解析。
    //    设容量上限，超出淘汰最旧，避免长跑（如 REST 服务）无界增长。
    corpus_raw_.push_back(text);
    while (static_cast<int>(corpus_raw_.size()) > corpus_capacity_) {
        corpus_raw_.erase(corpus_raw_.begin());
    }

    // 1. 提取实体
    result.entities = KnowledgeExtractor::extract_entities(text);

    // 1b. 直接将提取到的实体注入 KG（extract_entities 已做质量过滤）
    for (const auto& entity_name : result.entities) {
        kg_.add_entity(Entity(entity_name, "concept", {}, 0.7, source));
    }

    // 2. 提取关系：已训练时走「分词→依存树→论元」，否则退回(实体接地+窗口)
    if (parse_enabled_) {
        result.triples = extract_triples_parsed_(text);
    }
    if (result.triples.empty()) {
        result.triples =
            KnowledgeExtractor::extract_triples(text, result.entities);
    }

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

    // 10. 写入自我模型（学习闭环：每次学习都更新「我在学什么/我了解什么」）
    update_self_model_(text, source, result);

    return result;
}

// ── 自我模型：学习闭环写入 + 自我指涉问答 ───────────────────────

void TextLearner::update_self_model_(const std::string& text,
                                     const std::string& source,
                                     const TextLearnResult& result) {
    using consciousness::AutobiographicalMemory;
    using consciousness::CoreExperience;
    using consciousness::SelfBelief;

    // (1) 当下体验：我正在学习、在想什么、感觉如何（按是否验证通过）
    CoreExperience exp;
    exp.what_am_i_doing = "学习文本知识";
    exp.what_am_i_thinking =
        result.entities.empty() ? "新输入" : result.entities.front();
    exp.what_am_i_feeling = result.verification_passed ? "理解了" : "还有点困惑";
    exp.attention_focus = result.entities.empty() ? 0.3 : 0.7;
    self_model_.experience_now(exp);

    // (2) 自我信念：每次学习强化「持续学习」；每个学到的实体强化「了解X」。
    //     update_self_belief 为慢更新(0.9/0.1)，反复学习同一领域→确信度上升，
    //     这正是「自我效能随经验增长」的可量化闭环。
    self_model_.update_self_belief(
        SelfBelief{"持续学习", 1.0, source, true});
    int n = 0;
    for (const auto& e : result.entities) {
        if (n++ >= 5) break;  // 每句至多强化前几个实体，避免噪声淹没
        self_model_.update_self_belief(
            SelfBelief{"了解" + e, 1.0, utils::utf8_truncate(text, 20), true});
    }

    // (3) 自传记忆：验证通过且抽到三元组的，记一条带「教训」的情景记忆
    if (result.verification_passed && !result.triples.empty()) {
        const auto& t = result.triples.front();
        AutobiographicalMemory mem;
        mem.narrative = utils::utf8_truncate(text, 20);
        mem.importance = result.verification_score;
        mem.lesson_learned = t.subject + t.relation + t.object;
        self_model_.remember(mem);
    }
}

auto TextLearner::answer_self_referential_(const std::string& q) const
    -> std::string {
    const bool self_q =
        q.find("你") != std::string::npos || q.find("自己") != std::string::npos;
    if (!self_q) return "";  // 非自我指涉问题，交回常规路径

    // 「你了解/知道/会 X 吗」：在自我信念里找主题（trait 形如「了解X」），
    // 命中且确信度足够→正面回答并给出自评确信度，否则坦诚不了解。
    const bool ask_know = q.find("了解") != std::string::npos ||
                          q.find("知道") != std::string::npos ||
                          q.find("会") != std::string::npos;
    const bool ask_what = q.find("什么") != std::string::npos ||
                          q.find("啥") != std::string::npos;

    if (ask_know && !ask_what) {
        // 按 UTF-8 码点切分（无监督实体多为碎片，按「字覆盖」而非整词匹配更稳）
        auto to_cps = [](const std::string& s) {
            std::vector<std::string> cps;
            for (size_t i = 0; i < s.size();) {
                unsigned char c = static_cast<unsigned char>(s[i]);
                size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                cps.push_back(s.substr(i, len));
                i += len;
            }
            return cps;
        };

        // 已学会的「字」→ 最高自评确信度（来自正向「了解X」信念）
        const std::string prefix = "了解";
        std::map<std::string, double> learned;
        for (const auto& b : self_model_.self_concept()) {
            if (!b.is_positive || b.confidence <= 0.3 ||
                b.trait.size() <= prefix.size() ||
                b.trait.compare(0, prefix.size(), prefix) != 0)
                continue;
            for (const auto& cp : to_cps(b.trait.substr(prefix.size())))
                learned[cp] = std::max(learned[cp], b.confidence);
        }

        // 从问句中剥离疑问/指代虚词，剩下的 CJK 字即为被问主题
        static const std::set<std::string> stop = {
            "你", "自", "己", "我", "了", "解", "知", "道", "会", "吗", "嘛",
            "呢", "的", "是", "请", "问", "什", "么", "啥", "，", "。",
            "？", "?", " "};
        std::string topic;
        std::vector<std::string> topic_cps;
        for (const auto& cp : to_cps(q)) {
            if (cp.size() < 3 || stop.count(cp)) continue;  // 仅留 CJK 内容字
            topic += cp;
            topic_cps.push_back(cp);
        }

        if (!topic_cps.empty()) {
            double min_conf = 1.0;
            bool all_covered = true;
            for (const auto& cp : topic_cps) {
                auto it = learned.find(cp);
                if (it == learned.end()) { all_covered = false; break; }
                min_conf = std::min(min_conf, it->second);
            }
            if (all_covered) {
                return "我了解" + topic + "（自评确信度" +
                       std::to_string(static_cast<int>(min_conf * 100)) + "%）";
            }
            return "关于" + topic + "，我目前还不太了解，需要继续学习";
        }
        return "你想问的，我目前还不太了解，需要继续学习";
    }

    // 「你学到了什么/你知道些什么」：汇报自我认识
    if (ask_what) return self_model_.reflect_on_self();

    // 「你是谁」等：当下自我同一性
    return self_model_.who_am_i_now();
}

// ── 无监督分词 + 依存解析：训练与抽取 ───────────────────────────

auto TextLearner::train_dependency_parser() -> bool {
    if (corpus_raw_.size() < 8) return false;  // 语料过少不训练

    segmenter_.fit(corpus_raw_);

    // 用学到的分词把语料切成词序列，喂给 DMV 依存归纳器
    std::vector<std::vector<std::string>> seg_corpus;
    for (const auto& raw : corpus_raw_) {
        for (auto& run : segmenter_.segment_runs(raw)) {
            if (run.size() >= 2) seg_corpus.push_back(std::move(run));
        }
    }
    if (seg_corpus.size() < 4) return false;

    dep_parser_.train(seg_corpus);
    parse_enabled_ = dep_parser_.is_trained();
    return parse_enabled_;
}

auto TextLearner::extract_triples_parsed_(const std::string& text) const
    -> std::vector<Triple> {
    std::vector<Triple> out;
    if (!parse_enabled_) return out;

    // KnowledgeExtractor::relation_patterns() 为 (relation, cue)；
    // extract_triples_from_parse 需 (cue, relation)，此处交换
    std::vector<std::pair<std::string, std::string>> cues;
    for (const auto& [rel, cue] : KnowledgeExtractor::relation_patterns()) {
        cues.emplace_back(cue, rel);
    }

    for (const auto& run : segmenter_.segment_runs(text)) {
        if (run.size() < 3) continue;  // 太短无主谓宾结构
        auto parsed = dep_parser_.parse(run);
        for (auto& t : extract_triples_from_parse(parsed, cues)) {
            out.push_back(std::move(t));
        }
    }
    return out;
}

// ── 思考/回答 ────────────────────────────────────────────────────

auto TextLearner::think(const std::string& question) const
    -> std::string {
    // 路径 -1: 自我指涉问答（你是谁/你了解X吗），由在线更新的自我模型作答
    auto self_ans = answer_self_referential_(question);
    if (!self_ans.empty()) return self_ans;

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
