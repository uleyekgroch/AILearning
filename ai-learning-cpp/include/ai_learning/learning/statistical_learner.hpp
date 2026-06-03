/**
 * @file statistical_learner.hpp
 * @brief 统计学习器 — 从反复体验中让概念涌现
 *
 * 核心机制：
 * 1. N-gram 频率统计 → 概念涌现（PMI > 0 + 频率 >= min_freq）
 * 2. 共现统计 → 关系涌现（PMI 衡量绑定强度）
 * 3. 转移概率 → 序列预测
 */
#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 概念候选
struct ConceptCandidate {
    std::string text;
    int frequency = 0;
    int total_contexts = 0;       // 出现在多少不同文本中
    double pmi = 0.0;            // 点互信息
    int last_seen_idx = 0;

    [[nodiscard]] auto is_emergent() const -> bool {
        return frequency >= 3 && total_contexts >= 2 && pmi > 0.0;
    }

    [[nodiscard]] auto confidence() const -> double {
        return std::min(pmi / 5.0, 1.0);
    }
};

/// 共现关系
struct CooccurrenceRelation {
    std::string a;
    std::string b;
    int count = 0;
    double pmi = 0.0;

    [[nodiscard]] auto strength() const -> double {
        return std::min(pmi / 5.0, 1.0);
    }
};

/// 统计学习器
class StatisticalLearner {
public:
    explicit StatisticalLearner(int max_ngram = 4,
                                 int min_freq = 3,
                                 double min_pmi = 1.0,
                                 int max_concepts = 5000);

    /// 观察一条文本，更新统计并返回涌现信息
    auto observe(const std::string& text)
        -> std::map<std::string, std::vector<std::string>>;

    /// 观察预分词的 tokens（跳过内部分词）
    auto observe_tokens(const std::vector<std::string>& tokens)
        -> std::map<std::string, std::vector<std::string>>;

    /// 获取已涌现的概念列表
    [[nodiscard]] auto get_emergent_concepts(
        int min_freq = 0, double min_pmi = 0.0) const
        -> std::vector<ConceptCandidate>;

    /// 获取已涌现的关系列表
    [[nodiscard]] auto get_emergent_relations(double min_pmi = 0.0) const
        -> std::vector<CooccurrenceRelation>;

    /// 预测下一个可能的片段
    [[nodiscard]] auto predict_next(const std::string& context,
                                      int top_k = 5) const
        -> std::vector<std::pair<std::string, double>>;

    /// 获取概念信息
    [[nodiscard]] auto get_concept_info(const std::string& concept_name) const
        -> const ConceptCandidate*;

    /// 获取相关概念
    [[nodiscard]] auto get_related(const std::string& concept_name,
                                     int top_k = 10) const
        -> std::vector<std::pair<std::string, double>>;

    /// 计算文本的意外度（surprise）
    [[nodiscard]] auto get_surprise(const std::string& text) const -> double;

    /// 获取统计摘要
    [[nodiscard]] auto get_stats() const -> std::map<std::string, double>;

private:
    /// 分词为片段（中文单字 + 英文单词）
    auto segment(const std::string& text) const -> std::vector<std::string>;

    /// 判断 n-gram 是否有效（非纯虚词）
    [[nodiscard]] auto is_valid_ngram(const std::string& gram) const -> bool;

    /// 计算单个 n-gram 的 PMI
    [[nodiscard]] auto compute_pmi(const std::string& gram) const -> double;

    /// 检查概念涌现
    auto check_emergence(const std::set<std::string>& new_ngrams)
        -> std::vector<std::string>;

    /// 检查关系涌现
    auto check_relation_emergence() -> std::vector<std::pair<std::string, std::string>>;

    /// 计算一对概念的 PMI
    [[nodiscard]] auto compute_pair_pmi(const std::string& a,
                                          const std::string& b) const -> double;

    // 配置
    int max_ngram_;
    int min_freq_;
    double min_pmi_;
    int max_concepts_;

    // 核心统计
    std::map<std::string, int> char_freq_;
    std::map<std::string, int> ngram_freq_;
    std::map<std::string, std::set<int>> ngram_contexts_;

    // 共现统计
    std::map<std::string, std::map<std::string, int>> cooccurrence_;

    // 转移概率
    std::map<std::string, std::map<std::string, int>> transitions_;
    std::map<std::string, int> transition_totals_;

    // 已涌现的概念和关系
    std::map<std::string, ConceptCandidate> concepts_;
    std::map<std::string, CooccurrenceRelation> relations_;

    // 文本计数
    int text_count_ = 0;
    int total_chars_ = 0;
};

}  // namespace ai_learning::learning
