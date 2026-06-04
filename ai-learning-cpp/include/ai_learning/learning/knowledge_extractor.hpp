/**
 * @file knowledge_extractor.hpp
 * @brief 知识提取器 — 从文本中提取实体、关系、因果、数值
 *
 * 对应 Python 版 learner.py 中的 _extract_*_from_repr 系列方法。
 * 拆分为独立类，遵循 SRP。
 */
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 提取的三元组 (subject, relation, object, confidence)
struct Triple {
    std::string subject;
    std::string relation;
    std::string object;
    double confidence = 0.8;
};

/// 提取的因果关系
struct CausalLink {
    std::string cause;
    std::string effect;
};

/// 提取的数值事实
struct NumericalFact {
    std::string attribute;
    double value;
    std::string unit;
};

/// 文本学习结果
struct TextLearnResult {
    std::vector<std::string> entities;
    std::vector<Triple> triples;
    std::vector<CausalLink> causal_links;
    std::vector<NumericalFact> numerical_facts;
    bool verification_passed = false;
    double verification_score = 0.0;
};

/// 知识提取器（纯函数式，无状态）
///
/// 使用正则 + 模式匹配从中文文本中提取结构化知识。
class KnowledgeExtractor {
public:
    /// 从文本提取实体（中文 2-6 字词）
    static auto extract_entities(const std::string& text)
        -> std::vector<std::string>;

    /// 从文本提取关系三元组
    static auto extract_triples(const std::string& text,
                                 const std::vector<std::string>& entities)
        -> std::vector<Triple>;

    /// 从文本提取因果关系
    static auto extract_causal_links(const std::string& text)
        -> std::vector<CausalLink>;

    /// 从文本提取数值事实
    static auto extract_numerical_facts(const std::string& text)
        -> std::vector<NumericalFact>;

    /// 从文本中提取关键词
    static auto extract_keywords(const std::string& text)
        -> std::vector<std::string>;

    /// 关系模式表 (relation, 表层线索词)，供句法树论元抽取复用线索词集合
    static auto relation_patterns()
        -> const std::vector<std::pair<std::string, std::string>>&;

private:
    /// 中文词提取正则
    static auto extract_chinese_words_(const std::string& text, int min_len, int max_len)
        -> std::vector<std::string>;

    /// 关系模式匹配
    static const std::vector<std::pair<std::string, std::string>>& relation_patterns_();
};

}  // namespace ai_learning::learning
