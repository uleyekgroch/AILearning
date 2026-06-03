/**
 * @file statistical_learner.cpp
 * @brief 统计学习器实现
 */

#include "ai_learning/learning/statistical_learner.hpp"
#include "ai_learning/utils/utf8.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace ai_learning::learning {

StatisticalLearner::StatisticalLearner(int max_ngram, int min_freq,
                                         double min_pmi, int max_concepts)
    : max_ngram_(max_ngram),
      min_freq_(min_freq),
      min_pmi_(min_pmi),
      max_concepts_(max_concepts) {
}

// ── 观察 ─────────────────────────────────────────────────────────

auto StatisticalLearner::observe(const std::string& text)
    -> std::map<std::string, std::vector<std::string>> {
    std::map<std::string, std::vector<std::string>> result;
    result["new_concepts"] = {};
    result["new_relations"] = {};

    if (text.empty() || text.size() < 4) return result;

    ++text_count_;
    int text_id = text_count_;

    // 1. 分词为字符 token 序列
    auto tokens = segment(text);
    total_chars_ += static_cast<int>(tokens.size());

    // 2. 统计 n-gram 频率（token 级别）
    std::set<std::string> all_ngrams;
    int n_tokens = static_cast<int>(tokens.size());
    for (int n = 1; n <= max_ngram_ && n <= n_tokens; ++n) {
        for (int i = 0; i + n <= n_tokens; ++i) {
            // 拼接 n 个 token
            std::string gram;
            for (int j = 0; j < n; ++j) {
                gram += tokens[i + j];
            }
            ngram_freq_[gram]++;
            ngram_contexts_[gram].insert(text_id);
            if (n == 1) {
                char_freq_[gram]++;
            }
            if (n >= 2 && is_valid_ngram(gram)) {
                all_ngrams.insert(gram);
            }
        }
    }

    // 3. 统计共现（限制对数避免 O(n²) 膨胀）
    std::vector<std::string> ngram_list(all_ngrams.begin(), all_ngrams.end());
    if (ngram_list.size() > 30) {
        ngram_list.resize(30);
    }
    for (size_t i = 0; i < ngram_list.size(); ++i) {
        for (size_t j = i + 1; j < ngram_list.size(); ++j) {
            const auto& a = ngram_list[i];
            const auto& b = ngram_list[j];
            // 避免包含关系
            if (a.find(b) != std::string::npos || b.find(a) != std::string::npos)
                continue;
            cooccurrence_[a][b]++;
            cooccurrence_[b][a]++;
        }
    }

    // 4. 更新转移概率（token 级别）
    if (tokens.size() >= 2) {
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            transitions_[tokens[i]][tokens[i + 1]]++;
            transition_totals_[tokens[i]]++;
        }
    }

    // 5. 检查概念涌现
    result["new_concepts"] = check_emergence(all_ngrams);

    // 6. 检查关系涌现
    auto new_rels = check_relation_emergence();
    for (const auto& [a, b] : new_rels) {
        result["new_relations"].push_back(a + "-" + b);
    }

    return result;
}

auto StatisticalLearner::observe_tokens(const std::vector<std::string>& tokens)
    -> std::map<std::string, std::vector<std::string>> {
    std::map<std::string, std::vector<std::string>> result;
    result["new_concepts"] = {};
    result["new_relations"] = {};

    if (tokens.empty()) return result;

    ++text_count_;
    int text_id = text_count_;

    total_chars_ += static_cast<int>(tokens.size());

    // 统计 n-gram 频率（token 级别）
    std::set<std::string> all_ngrams;
    int n_tokens = static_cast<int>(tokens.size());
    for (int n = 1; n <= max_ngram_ && n <= n_tokens; ++n) {
        for (int i = 0; i + n <= n_tokens; ++i) {
            std::string gram;
            for (int j = 0; j < n; ++j) {
                gram += tokens[i + j];
            }
            ngram_freq_[gram]++;
            ngram_contexts_[gram].insert(text_id);
            if (n == 1) {
                char_freq_[gram]++;
            }
            if (n >= 2 && is_valid_ngram(gram)) {
                all_ngrams.insert(gram);
            }
        }
    }

    // 共现统计
    std::vector<std::string> ngram_list(all_ngrams.begin(), all_ngrams.end());
    if (ngram_list.size() > 30) {
        ngram_list.resize(30);
    }
    for (size_t i = 0; i < ngram_list.size(); ++i) {
        for (size_t j = i + 1; j < ngram_list.size(); ++j) {
            const auto& a = ngram_list[i];
            const auto& b = ngram_list[j];
            if (a.find(b) != std::string::npos || b.find(a) != std::string::npos)
                continue;
            cooccurrence_[a][b]++;
            cooccurrence_[b][a]++;
        }
    }

    // 转移概率
    if (tokens.size() >= 2) {
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            transitions_[tokens[i]][tokens[i + 1]]++;
            transition_totals_[tokens[i]]++;
        }
    }

    // 检查概念和关系涌现
    result["new_concepts"] = check_emergence(all_ngrams);

    auto new_rels = check_relation_emergence();
    for (const auto& [a, b] : new_rels) {
        result["new_relations"].push_back(a + "-" + b);
    }

    return result;
}

// ── 查询 ─────────────────────────────────────────────────────────

auto StatisticalLearner::get_emergent_concepts(int min_freq,
                                                  double min_pmi) const
    -> std::vector<ConceptCandidate> {
    std::vector<ConceptCandidate> result;
    for (const auto& [_, c] : concepts_) {
        if (c.frequency >= min_freq && c.pmi >= min_pmi) {
            result.push_back(c);
        }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.pmi > b.pmi; });
    return result;
}

auto StatisticalLearner::get_emergent_relations(double min_pmi_val) const
    -> std::vector<CooccurrenceRelation> {
    std::vector<CooccurrenceRelation> result;
    for (const auto& [_, r] : relations_) {
        if (r.pmi >= min_pmi_val) {
            result.push_back(r);
        }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.pmi > b.pmi; });
    return result;
}

auto StatisticalLearner::predict_next(const std::string& context,
                                        int top_k) const
    -> std::vector<std::pair<std::string, double>> {
    if (context.empty()) return {};

    // 取最后一个 token
    auto tokens = segment(context);
    if (tokens.empty()) return {};

    auto last = tokens.back();
    auto it = transitions_.find(last);
    if (it == transitions_.end()) return {};

    auto total = static_cast<double>(transition_totals_.at(last));
    std::vector<std::pair<std::string, double>> probs;
    for (const auto& [next, count] : it->second) {
        probs.emplace_back(next, static_cast<double>(count) / total);
    }
    std::sort(probs.begin(), probs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (static_cast<int>(probs.size()) > top_k) {
        probs.resize(top_k);
    }
    return probs;
}

auto StatisticalLearner::get_concept_info(const std::string& concept_name) const
    -> const ConceptCandidate* {
    auto it = concepts_.find(concept_name);
    return it != concepts_.end() ? &it->second : nullptr;
}

auto StatisticalLearner::get_related(const std::string& concept_name,
                                       int top_k) const
    -> std::vector<std::pair<std::string, double>> {
    auto it = cooccurrence_.find(concept_name);
    if (it == cooccurrence_.end()) return {};

    std::vector<std::pair<std::string, double>> related;
    for (const auto& [other, count] : it->second) {
        auto pmi = compute_pair_pmi(concept_name, other);
        if (pmi > 0) {
            related.emplace_back(other, pmi);
        }
    }
    std::sort(related.begin(), related.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (static_cast<int>(related.size()) > top_k) {
        related.resize(top_k);
    }
    return related;
}

auto StatisticalLearner::get_surprise(const std::string& text) const -> double {
    if (text.empty() || text.size() < 2) return 0.0;

    auto tokens = segment(text);
    if (tokens.size() < 2) return 0.0;

    double total_surprise = 0.0;
    int count = 0;

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        auto it = transitions_.find(tokens[i]);
        if (it == transitions_.end()) {
            total_surprise += 5.0;
        } else {
            auto total_it = transition_totals_.find(tokens[i]);
            if (total_it == transition_totals_.end()) continue;
            auto total = static_cast<double>(total_it->second);
            auto next_it = it->second.find(tokens[i + 1]);
            double prob = (next_it != it->second.end())
                              ? static_cast<double>(next_it->second) / total
                              : 0.001;
            total_surprise += -std::log(prob);
        }
        ++count;
    }

    return count > 0 ? total_surprise / count : 0.0;
}

auto StatisticalLearner::get_stats() const -> std::map<std::string, double> {
    return {
        {"text_count", static_cast<double>(text_count_)},
        {"total_chars", static_cast<double>(total_chars_)},
        {"ngram_types", static_cast<double>(ngram_freq_.size())},
        {"concept_count", static_cast<double>(concepts_.size())},
        {"relation_count", static_cast<double>(relations_.size())},
    };
}

// ── 内部方法 ─────────────────────────────────────────────────────

auto StatisticalLearner::segment(const std::string& text) const
    -> std::vector<std::string> {
    // UTF-8 字符级分词：每个中文字符或连续英文单词为一个 token
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        auto uc = static_cast<unsigned char>(text[i]);
        auto byte_len = utils::utf8_char_len(uc);
        if (byte_len == 3 && uc >= 0xE4 && uc <= 0xE9) {
            // CJK 字符：3 字节 UTF-8
            auto ch = text.substr(i, 3);
            tokens.push_back(ch);
            i += 3;
        } else if ((text[i] >= 'a' && text[i] <= 'z') ||
                   (text[i] >= 'A' && text[i] <= 'Z')) {
            // 英文单词
            std::string word;
            while (i < text.size() &&
                   ((text[i] >= 'a' && text[i] <= 'z') ||
                    (text[i] >= 'A' && text[i] <= 'Z'))) {
                word += text[i];
                ++i;
            }
            tokens.push_back(word);
        } else {
            ++i;  // 跳过其他字符
        }
    }
    return tokens;
}

auto StatisticalLearner::is_valid_ngram(const std::string& gram) const -> bool {
    return !gram.empty();
}

auto StatisticalLearner::compute_pmi(const std::string& gram) const -> double {
    auto it = ngram_freq_.find(gram);
    if (it == ngram_freq_.end() || it->second < 1) return 0.0;

    // 拆分为单个 token（按 UTF-8 字符拆分）
    std::vector<std::string> chars;
    for (size_t i = 0; i < gram.size(); ) {
        auto uc = static_cast<unsigned char>(gram[i]);
        int byte_len = 1;
        if (uc >= 0xE0) byte_len = 3;       // 3-byte UTF-8 (CJK)
        else if (uc >= 0xC0) byte_len = 2;  // 2-byte UTF-8
        chars.push_back(gram.substr(i, byte_len));
        i += byte_len;
    }

    if (static_cast<int>(chars.size()) < 2) return 0.0;

    double p_gram = static_cast<double>(it->second) / total_chars_;
    if (p_gram <= 0) return 0.0;

    double p_indep = 1.0;
    for (const auto& ci : chars) {
        auto cit = char_freq_.find(ci);
        if (cit == char_freq_.end() || cit->second < 1) return 0.0;
        p_indep *= static_cast<double>(cit->second) / total_chars_;
    }
    if (p_indep <= 0) return 0.0;

    return std::log(p_gram / p_indep) / std::log(2.0);
}

auto StatisticalLearner::check_emergence(const std::set<std::string>& new_ngrams)
    -> std::vector<std::string> {
    std::vector<std::string> newly_emerged;

    for (const auto& gram : new_ngrams) {
        auto freq_it = ngram_freq_.find(gram);
        if (freq_it == ngram_freq_.end()) continue;
        int freq = freq_it->second;

        auto ctx_it = ngram_contexts_.find(gram);
        int contexts = ctx_it != ngram_contexts_.end()
                           ? static_cast<int>(ctx_it->second.size())
                           : 0;

        if (freq < min_freq_) continue;

        // 已存在 → 更新
        auto concept_it = concepts_.find(gram);
        if (concept_it != concepts_.end()) {
            concept_it->second.frequency = freq;
            concept_it->second.total_contexts = contexts;
            concept_it->second.last_seen_idx = text_count_;
            continue;
        }

        // 新候选 → 检查 PMI
        double pmi = compute_pmi(gram);
        if (pmi <= 0) continue;

        // 涌现！
        ConceptCandidate candidate;
        candidate.text = gram;
        candidate.frequency = freq;
        candidate.total_contexts = contexts;
        candidate.pmi = pmi;
        candidate.last_seen_idx = text_count_;
        concepts_[gram] = candidate;
        newly_emerged.push_back(gram);
    }

    // 容量限制
    if (static_cast<int>(concepts_.size()) > max_concepts_) {
        std::vector<std::pair<std::string, ConceptCandidate>> sorted(
            concepts_.begin(), concepts_.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.frequency > b.second.frequency;
                  });
        concepts_.clear();
        for (int i = 0; i < max_concepts_ && i < static_cast<int>(sorted.size()); ++i) {
            concepts_[sorted[i].first] = sorted[i].second;
        }
    }

    return newly_emerged;
}

auto StatisticalLearner::check_relation_emergence()
    -> std::vector<std::pair<std::string, std::string>> {
    std::vector<std::pair<std::string, std::string>> new_relations;

    for (const auto& [a, others] : cooccurrence_) {
        // 只处理已涌现的概念之间的关系
        if (concepts_.find(a) == concepts_.end()) continue;

        for (const auto& [b, count] : others) {
            if (count < min_freq_) continue;
            if (concepts_.find(b) == concepts_.end()) continue;

            // 规范化 key
            auto key = (a < b) ? (a + "|" + b) : (b + "|" + a);
            if (relations_.find(key) != relations_.end()) continue;

            double pmi = compute_pair_pmi(a, b);
            if (pmi < min_pmi_) continue;

            CooccurrenceRelation rel;
            rel.a = a;
            rel.b = b;
            rel.count = count;
            rel.pmi = pmi;
            relations_[key] = rel;
            new_relations.emplace_back(a, b);
        }
    }
    return new_relations;
}

auto StatisticalLearner::compute_pair_pmi(const std::string& a,
                                            const std::string& b) const -> double {
    auto it_a = ngram_freq_.find(a);
    auto it_b = ngram_freq_.find(b);
    if (it_a == ngram_freq_.end() || it_b == ngram_freq_.end()) return 0.0;

    auto co_it = cooccurrence_.find(a);
    if (co_it == cooccurrence_.end()) return 0.0;
    auto b_it = co_it->second.find(b);
    if (b_it == co_it->second.end()) return 0.0;

    double p_ab = static_cast<double>(b_it->second) / text_count_;
    double p_a = static_cast<double>(it_a->second) / total_chars_;
    double p_b = static_cast<double>(it_b->second) / total_chars_;

    if (p_ab <= 0 || p_a <= 0 || p_b <= 0) return 0.0;

    return std::log(p_ab / (p_a * p_b)) / std::log(2.0);
}

}  // namespace ai_learning::learning
