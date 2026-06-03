/**
 * @file unified_engine.cpp
 * @brief 统一推理引擎实现
 */

#include "ai_learning/reasoning/unified_engine.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/domain/knowledge/relation.hpp"
#include "ai_learning/learning/statistical_learner.hpp"
#include "ai_learning/memory/episodic_memory.hpp"

#include <algorithm>
#include <functional>
#include <sstream>

namespace ai_learning::reasoning {

using namespace domain::knowledge;

UnifiedReasoningEngine::UnifiedReasoningEngine(KnowledgeGraph& kg)
    : kg_(kg) {}

void UnifiedReasoningEngine::set_statistical_learner(
    learning::StatisticalLearner* sl) {
    stat_learner_ = sl;
}

void UnifiedReasoningEngine::set_episodic_memory(
    memory::EpisodicMemory* em) {
    episodic_ = em;
}

// ── 主推理 ───────────────────────────────────────────────────────

auto UnifiedReasoningEngine::reason(const std::string& question) const
    -> std::vector<ReasoningResult> {
    std::vector<ReasoningResult> results;

    auto keywords = extract_keywords(question);
    // 补充：检查每个 UTF-8 字符和 ASCII 单词是否是已知实体
    {
        std::set<std::string> seen(keywords.begin(), keywords.end());
        for (size_t i = 0; i < question.size(); ) {
            auto uc = static_cast<unsigned char>(question[i]);
            int byte_len = 1;
            if (uc >= 0xE0) byte_len = 3;
            else if (uc >= 0xC0) byte_len = 2;
            if (uc < 0x80 && ((question[i] >= 'a' && question[i] <= 'z') ||
                              (question[i] >= 'A' && question[i] <= 'Z'))) {
                std::string word;
                while (i < question.size() &&
                       ((question[i] >= 'a' && question[i] <= 'z') ||
                        (question[i] >= 'A' && question[i] <= 'Z'))) {
                    word += question[i++];
                }
                if (!seen.contains(word)) { keywords.push_back(word); seen.insert(word); }
            } else {
                if (i + byte_len <= question.size()) {
                    auto ch = question.substr(i, byte_len);
                    if (!seen.contains(ch)) { keywords.push_back(ch); seen.insert(ch); }
                }
                i += byte_len;
            }
        }
    }

    auto direct = direct_query(question);
    results.insert(results.end(), direct.begin(), direct.end());

    auto causal = causal_reasoning(question);
    results.insert(results.end(), causal.begin(), causal.end());

    auto inductive = inductive_reasoning(question);
    results.insert(results.end(), inductive.begin(), inductive.end());

    auto analogical = analogical_reasoning(question);
    results.insert(results.end(), analogical.begin(), analogical.end());

    auto counterfactual = counterfactual_reasoning(question);
    results.insert(results.end(), counterfactual.begin(), counterfactual.end());

    auto probabilistic = probabilistic_reasoning(question);
    results.insert(results.end(), probabilistic.begin(), probabilistic.end());

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.confidence > b.confidence;
              });

    return results;
}

// ── 直接查询 ─────────────────────────────────────────────────────

auto UnifiedReasoningEngine::direct_query(const std::string& question) const
    -> std::vector<ReasoningResult> {
    std::vector<ReasoningResult> results;
    auto keywords = extract_keywords(question);

    // 补充：检查问题中的每个 UTF-8 字符是否是已知实体
    {
        std::set<std::string> seen;
        for (const auto& kw : keywords) seen.insert(kw);

        // 按 UTF-8 字符遍历
        for (size_t i = 0; i < question.size(); ) {
            auto uc = static_cast<unsigned char>(question[i]);
            int byte_len = 1;
            if (uc >= 0xE0) byte_len = 3;
            else if (uc >= 0xC0) byte_len = 2;
            if (i + byte_len <= question.size()) {
                auto ch = question.substr(i, byte_len);
                if (!seen.contains(ch) && kg_.has_entity(ch)) {
                    keywords.push_back(ch);
                    seen.insert(ch);
                }
            }
            i += byte_len;
        }
        // 也检查 ASCII 单词
        for (size_t i = 0; i < question.size(); ) {
            if ((question[i] >= 'A' && question[i] <= 'Z') ||
                (question[i] >= 'a' && question[i] <= 'z')) {
                std::string word;
                while (i < question.size() &&
                       ((question[i] >= 'a' && question[i] <= 'z') ||
                        (question[i] >= 'A' && question[i] <= 'Z'))) {
                    word += question[i++];
                }
                if (!seen.contains(word) && kg_.has_entity(word)) {
                    keywords.push_back(word);
                    seen.insert(word);
                }
            } else {
                ++i;
            }
        }
    }

    for (const auto& kw : keywords) {
        if (!kg_.has_entity(kw)) continue;

        auto relations = kg_.get_relations_of(kw, "out");
        for (const auto& ref : relations) {
            const auto& rel = ref.get();
            ReasoningResult r;
            r.content = kw + " " + rel.type() + " " + rel.target_id();
            r.confidence = rel.confidence();
            r.method = "direct";
            r.evidence = {kw, rel.target_id()};
            results.push_back(r);
        }

        auto in_rels = kg_.get_relations_of(kw, "in");
        for (const auto& ref : in_rels) {
            const auto& rel = ref.get();
            ReasoningResult r;
            r.content = rel.source_id() + " " + rel.type() + " " + kw;
            r.confidence = rel.confidence();
            r.method = "direct";
            r.evidence = {rel.source_id(), kw};
            results.push_back(r);
        }
    }

    return results;
}

// ── 因果推理 ─────────────────────────────────────────────────────

auto UnifiedReasoningEngine::causal_reasoning(const std::string& question) const
    -> std::vector<ReasoningResult> {
    std::vector<ReasoningResult> results;
    auto keywords = extract_keywords(question);

    for (const auto& kw : keywords) {
        if (!kg_.has_entity(kw)) continue;

        auto out_rels = kg_.get_relations_of(kw, "out");
        for (const auto& ref : out_rels) {
            const auto& rel = ref.get();
            auto rtype = rel.type();
            bool is_causal = rtype.find("导致") != std::string::npos ||
                             rtype.find("引起") != std::string::npos ||
                             rtype.find("因果") != std::string::npos;

            if (!is_causal) continue;

            // 继续查找下游因果链
            auto next_rels = kg_.get_relations_of(rel.target_id(), "out");
            for (const auto& nref : next_rels) {
                const auto& next = nref.get();
                auto ntype = next.type();
                if (ntype.find("导致") != std::string::npos ||
                    ntype.find("引起") != std::string::npos) {
                    ReasoningResult r;
                    r.content = kw + " -> " + rel.target_id() + " -> " + next.target_id();
                    r.confidence = rel.confidence() * next.confidence() * 0.8;
                    r.method = "causal";
                    r.evidence = {kw, rel.target_id(), next.target_id()};
                    r.reasoning_chain = {
                        "因果链: " + kw + " " + rtype + " " + rel.target_id(),
                        "因果链: " + rel.target_id() + " " + ntype + " " + next.target_id(),
                    };
                    results.push_back(r);
                }
            }
        }
    }

    return results;
}

// ── 归纳推理 ─────────────────────────────────────────────────────

auto UnifiedReasoningEngine::inductive_reasoning(
    const std::string& question) const
    -> std::vector<ReasoningResult> {
    std::vector<ReasoningResult> results;
    if (!episodic_) return results;

    auto keywords = extract_keywords(question);
    auto all = episodic_->get_recent(50);

    std::vector<std::string> related_memories;
    for (const auto& mem : all) {
        for (const auto& kw : keywords) {
            for (const auto& [k, v] : mem.metadata) {
                if (v.find(kw) != std::string::npos) {
                    related_memories.push_back(k + "=" + v);
                    break;
                }
            }
        }
    }

    if (related_memories.size() >= 2) {
        auto patterns = find_common_patterns(related_memories);
        for (const auto& pattern : patterns) {
            ReasoningResult r;
            r.content = "归纳: " + pattern;
            r.confidence = 0.6;
            r.method = "inductive";
            r.evidence = related_memories;
            r.reasoning_chain = {
                "从 " + std::to_string(related_memories.size()) + " 个实例中归纳"
            };
            results.push_back(r);
        }
    }

    return results;
}

// ── 类比推理 ─────────────────────────────────────────────────────

auto UnifiedReasoningEngine::analogical_reasoning(
    const std::string& question) const
    -> std::vector<ReasoningResult> {
    std::vector<ReasoningResult> results;
    auto keywords = extract_keywords(question);

    for (const auto& kw : keywords) {
        if (!kg_.has_entity(kw)) continue;

        auto out_rels = kg_.get_relations_of(kw, "out");
        for (const auto& ref : out_rels) {
            const auto& rel = ref.get();
            if (!kg_.has_entity(rel.target_id())) continue;

            auto target_rels = kg_.get_relations_of(rel.target_id(), "out");
            for (const auto& tref : target_rels) {
                const auto& t_rel = tref.get();
                if (t_rel.type() == rel.type()) {
                    ReasoningResult r;
                    r.content = "类比: " + kw + " 之于 " + rel.target_id()
                              + " 如同 " + rel.target_id() + " 之于 " + t_rel.target_id();
                    r.confidence = rel.confidence() * t_rel.confidence() * 0.5;
                    r.method = "analogical";
                    r.evidence = {kw, rel.target_id(), t_rel.target_id()};
                    r.reasoning_chain = {"关系映射: " + std::string(rel.type())};
                    results.push_back(r);
                }
            }
        }
    }

    return results;
}

// ── 反事实推理 ───────────────────────────────────────────────────

auto UnifiedReasoningEngine::counterfactual_reasoning(
    const std::string& question) const
    -> std::vector<ReasoningResult> {
    std::vector<ReasoningResult> results;
    auto keywords = extract_keywords(question);

    for (const auto& kw : keywords) {
        if (!kg_.has_entity(kw)) continue;

        auto out_rels = kg_.get_relations_of(kw, "out");
        for (const auto& ref : out_rels) {
            const auto& rel = ref.get();
            auto rtype = rel.type();
            if (rtype.find("导致") != std::string::npos ||
                rtype.find("因果") != std::string::npos) {
                ReasoningResult r;
                r.content = "反事实: 如果 " + kw + " 不"
                          + rtype + rel.target_id()
                          + ", 则可能无法观察到相关现象";
                r.confidence = rel.confidence() * 0.4;
                r.method = "counterfactual";
                r.evidence = {kw, rel.target_id()};
                r.reasoning_chain = {
                    "假设: " + kw + " 不" + rtype + rel.target_id(),
                    "推断: 缺少因果链将导致下游变化"
                };
                results.push_back(r);
            }
        }
    }

    return results;
}

// ── 概率推理 ─────────────────────────────────────────────────────

auto UnifiedReasoningEngine::probabilistic_reasoning(
    const std::string& question) const
    -> std::vector<ReasoningResult> {
    std::vector<ReasoningResult> results;
    if (!stat_learner_) return results;

    auto keywords = extract_keywords(question);

    for (const auto& kw : keywords) {
        auto info = stat_learner_->get_concept_info(kw);
        if (!info) continue;

        auto related = stat_learner_->get_related(kw, 5);
        for (const auto& [other, strength] : related) {
            ReasoningResult r;
            r.content = "统计关联: " + kw + " <-> " + other
                      + " (PMI=" + std::to_string(strength) + ")";
            r.confidence = std::min(strength / 5.0, 0.9);
            r.method = "probabilistic";
            r.evidence = {kw, other};
            results.push_back(r);
        }

        auto predictions = stat_learner_->predict_next(kw, 3);
        for (const auto& [pred, prob] : predictions) {
            ReasoningResult r;
            r.content = "预测: " + kw + " 之后可能是 " + pred
                      + " (P=" + std::to_string(prob) + ")";
            r.confidence = prob * 0.7;
            r.method = "probabilistic";
            r.evidence = {kw, pred};
            results.push_back(r);
        }
    }

    return results;
}

// ── 辅助方法 ─────────────────────────────────────────────────────

auto UnifiedReasoningEngine::extract_keywords(const std::string& text)
    -> std::vector<std::string> {
    // 1. 先分割为 UTF-8 字符 tokens
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size(); ) {
        auto uc = static_cast<unsigned char>(text[i]);
        int byte_len = 1;
        if (uc >= 0xE0) byte_len = 3;       // 3-byte UTF-8 (CJK)
        else if (uc >= 0xC0) byte_len = 2;  // 2-byte UTF-8
        if (i + byte_len <= text.size()) {
            chars.push_back(text.substr(i, byte_len));
        }
        i += byte_len;
    }

    // 2. 停用词
    static const std::vector<std::string> stops = {
        "的", "了", "是", "在", "有", "不", "会", "能", "和", "与",
        "这", "那", "个", "为", "上", "下", "来", "去", "到", "很",
        "也", "就", "要", "把", "被", "让", "给", "从", "向", "比"
    };
    auto is_stop = [&](const std::string& s) {
        return std::find(stops.begin(), stops.end(), s) != stops.end();
    };

    // 3. 提取 2-4 个字符的窗口作为关键词
    std::vector<std::string> keywords;
    for (int n = 4; n >= 2; --n) {
        for (size_t i = 0; i + static_cast<size_t>(n) <= chars.size(); ++i) {
            std::string gram;
            bool all_cjk = true;
            for (int j = 0; j < n; ++j) {
                gram += chars[i + j];
                auto uc = static_cast<unsigned char>(chars[i + j][0]);
                if (uc < 0x80) all_cjk = false;
            }
            if (!all_cjk) continue;
            // 跳过以停用词开头或结尾的
            if (is_stop(chars[i]) || is_stop(chars[i + n - 1])) continue;
            keywords.push_back(gram);
        }
    }

    // 4. 也添加单个 CJK 字符（非停用词）和 ASCII 单词
    for (const auto& ch : chars) {
        auto uc = static_cast<unsigned char>(ch[0]);
        if (uc >= 0x80 && !is_stop(ch)) {
            // 单个 CJK 字符
            keywords.push_back(ch);
        }
    }
    // ASCII 单词
    for (size_t i = 0; i < text.size(); ) {
        if ((text[i] >= 'A' && text[i] <= 'Z') ||
            (text[i] >= 'a' && text[i] <= 'z')) {
            std::string word;
            while (i < text.size() &&
                   ((text[i] >= 'a' && text[i] <= 'z') ||
                    (text[i] >= 'A' && text[i] <= 'Z'))) {
                word += text[i++];
            }
            keywords.push_back(word);
        } else {
            ++i;
        }
    }

    // 去重
    std::set<std::string> seen;
    std::vector<std::string> unique;
    for (const auto& kw : keywords) {
        if (!seen.contains(kw)) {
            seen.insert(kw);
            unique.push_back(kw);
        }
    }
    return unique;
}

auto UnifiedReasoningEngine::find_common_patterns(
    const std::vector<std::string>& memories)
    -> std::vector<std::string> {
    std::map<std::string, int> freq;
    for (const auto& mem : memories) {
        for (size_t i = 0; i + 2 <= mem.size(); ++i) {
            freq[mem.substr(i, 2)]++;
        }
    }

    std::vector<std::string> patterns;
    for (const auto& [word, count] : freq) {
        if (count >= 2) {
            patterns.push_back(word + " (出现" + std::to_string(count) + "次)");
        }
    }
    return patterns;
}

}  // namespace ai_learning::reasoning
