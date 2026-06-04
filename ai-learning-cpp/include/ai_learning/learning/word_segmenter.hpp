/**
 * @file word_segmenter.hpp
 * @brief 无监督中文分词（零预训练 / 零词典 / 零标注）
 *
 * 方法：新词发现 + 最大概率切分
 *   1) 在语料的连续 CJK 串内统计所有 ≤L 字的子串；
 *   2) 候选词筛选用两个统计量：
 *      - 内部凝固度 cohesion = min_k log( c(w)·T / (c(left)·c(right)) )
 *        （子串各处切分都"抱团"远高于随机 → 是一个整体）；
 *      - 边界自由度 freedom = min( H(左邻字分布), H(右邻字分布) )
 *        （左右都能接多种字 → 不是总黏在别的词上的碎片）。
 *   3) 保留满足频次/凝固度/自由度阈值的多字词 + 全部单字（兜底），
 *      构成一元词模型；
 *   4) 对新句子用 Viterbi DP 求 argmax ∏ P(word) 的切分。
 *
 * 全程仅用语料自身统计，不读任何预训练权重 / 外部词典。
 */
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_learning::learning {

struct SegmenterConfig {
    int max_word_len = 4;        ///< 候选词最大字数 L
    int min_count = 3;           ///< 候选多字词最低频
    double min_cohesion = 2.0;   ///< 内部凝固度阈值（自然对数）
    double min_freedom = 0.4;    ///< 左右邻接熵下界（min(左,右)）
};

/// 无监督分词器
class WordSegmenter {
public:
    explicit WordSegmenter(SegmenterConfig cfg = {}) : cfg_(cfg) {}

    /// 在原始（无需空格）中文语料上学习词表与一元模型
    void fit(const std::vector<std::string>& corpus);

    /// 把一段文本切成词序列（非 CJK 字符视为硬边界并丢弃）
    [[nodiscard]] auto segment(const std::string& text) const
        -> std::vector<std::string>;

    /// 按"连续 CJK 串"切成多段，每段独立切词（更贴合按小句解析）
    [[nodiscard]] auto segment_runs(const std::string& text) const
        -> std::vector<std::vector<std::string>>;

    [[nodiscard]] auto trained() const -> bool { return trained_; }

    /// 学到的多字词数量（不含单字兜底）
    [[nodiscard]] auto multichar_word_count() const -> std::size_t {
        return multichar_count_;
    }

    /// 某个词是否在学到的词表里（含单字）
    [[nodiscard]] auto in_lexicon(const std::string& w) const -> bool {
        return logp_.count(w) > 0;
    }

private:
    SegmenterConfig cfg_;
    bool trained_ = false;
    std::size_t multichar_count_ = 0;
    double oov_logp_ = -30.0;
    std::unordered_map<std::string, double> logp_;  ///< 词 → log 概率

    /// Viterbi 最大概率切分（输入为单字序列）
    [[nodiscard]] auto viterbi_(const std::vector<std::string>& chars) const
        -> std::vector<std::string>;
};

}  // namespace ai_learning::learning
