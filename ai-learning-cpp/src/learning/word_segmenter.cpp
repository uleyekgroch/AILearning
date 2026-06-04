/**
 * @file word_segmenter.cpp
 * @brief 无监督中文分词实现（新词发现 + 最大概率 Viterbi 切分）
 */

#include "ai_learning/learning/word_segmenter.hpp"
#include "ai_learning/utils/utf8.hpp"

#include <algorithm>
#include <cmath>

namespace ai_learning::learning {

namespace {

// 把 UTF-8 单字解码成码点（仅需判断 CJK；非法/短串返回 0）
auto codepoint_of(const std::string& ch) -> uint32_t {
    if (ch.empty()) return 0;
    auto c0 = static_cast<unsigned char>(ch[0]);
    if (c0 < 0x80) return c0;
    if ((c0 & 0xE0) == 0xC0 && ch.size() >= 2)
        return ((c0 & 0x1Fu) << 6) |
               (static_cast<unsigned char>(ch[1]) & 0x3Fu);
    if ((c0 & 0xF0) == 0xE0 && ch.size() >= 3)
        return ((c0 & 0x0Fu) << 12) |
               ((static_cast<unsigned char>(ch[1]) & 0x3Fu) << 6) |
               (static_cast<unsigned char>(ch[2]) & 0x3Fu);
    if ((c0 & 0xF8) == 0xF0 && ch.size() >= 4)
        return ((c0 & 0x07u) << 18) |
               ((static_cast<unsigned char>(ch[1]) & 0x3Fu) << 12) |
               ((static_cast<unsigned char>(ch[2]) & 0x3Fu) << 6) |
               (static_cast<unsigned char>(ch[3]) & 0x3Fu);
    return 0;
}

auto is_cjk_char(const std::string& ch) -> bool {
    uint32_t cp = codepoint_of(ch);
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||  // CJK 基本区
           (cp >= 0x3400 && cp <= 0x4DBF);     // 扩展 A
}

// 把文本切成"连续 CJK 串"，每串为单字向量
auto cjk_runs(const std::string& text) -> std::vector<std::vector<std::string>> {
    std::vector<std::vector<std::string>> runs;
    std::vector<std::string> cur;
    utils::utf8_foreach(text, [&](const std::string& ch, size_t) {
        if (is_cjk_char(ch)) {
            cur.push_back(ch);
        } else if (!cur.empty()) {
            runs.push_back(std::move(cur));
            cur.clear();
        }
    });
    if (!cur.empty()) runs.push_back(std::move(cur));
    return runs;
}

auto join(const std::vector<std::string>& chars, int lo, int hi) -> std::string {
    std::string s;
    for (int i = lo; i < hi; ++i) s += chars[i];
    return s;
}

auto entropy(const std::unordered_map<std::string, long>& dist) -> double {
    long total = 0;
    for (const auto& [k, v] : dist) total += v;
    if (total == 0) return 0.0;
    double h = 0.0;
    for (const auto& [k, v] : dist) {
        double p = static_cast<double>(v) / static_cast<double>(total);
        if (p > 0.0) h -= p * std::log(p);
    }
    return h;
}

}  // namespace

void WordSegmenter::fit(const std::vector<std::string>& corpus) {
    const int L = std::max(2, cfg_.max_word_len);
    auto all_runs = std::vector<std::vector<std::string>>{};
    for (const auto& line : corpus)
        for (auto& r : cjk_runs(line)) all_runs.push_back(std::move(r));

    std::unordered_map<std::string, long> count;       // 子串频次（含单字）
    std::unordered_map<std::string, std::unordered_map<std::string, long>> lctx;
    std::unordered_map<std::string, std::unordered_map<std::string, long>> rctx;
    long T = 0;  // 字 token 总数

    for (const auto& r : all_runs) {
        int m = static_cast<int>(r.size());
        T += m;
        for (int start = 0; start < m; ++start)
            for (int len = 1; len <= L && start + len <= m; ++len) {
                std::string w = join(r, start, start + len);
                ++count[w];
                if (len >= 2) {
                    lctx[w][start > 0 ? r[start - 1] : "^"]++;
                    rctx[w][start + len < m ? r[start + len] : "$"]++;
                }
            }
    }
    if (T == 0) {
        trained_ = true;
        return;
    }

    // 词表：全部单字（兜底）+ 满足凝固度/自由度/频次的多字词
    std::unordered_map<std::string, long> lex;
    for (const auto& [w, c] : count) {
        int len = static_cast<int>(utils::utf8_chars(w).size());
        if (len == 1) {
            lex[w] = c;
            continue;
        }
        if (c < cfg_.min_count) continue;

        auto chars = utils::utf8_chars(w);
        // 内部凝固度：所有切分点都需"抱团"
        double cohesion = 1e18;
        for (int k = 1; k < len; ++k) {
            std::string left = join(chars, 0, k);
            std::string right = join(chars, k, len);
            long cl = count[left], cr = count[right];
            if (cl == 0 || cr == 0) {
                cohesion = -1e18;
                break;
            }
            double ratio = (static_cast<double>(c) * static_cast<double>(T)) /
                           (static_cast<double>(cl) * static_cast<double>(cr));
            cohesion = std::min(cohesion, std::log(ratio));
        }
        if (cohesion < cfg_.min_cohesion) continue;

        // 边界自由度：左右邻接熵都要够大
        double freedom = std::min(entropy(lctx[w]), entropy(rctx[w]));
        if (freedom < cfg_.min_freedom) continue;

        lex[w] = c;
    }

    // 一元模型：logp(word) = log((count+0.5)/(Z + 0.5·|lex|))
    long Z = 0;
    for (const auto& [w, c] : lex) Z += c;
    double denom = static_cast<double>(Z) + 0.5 * static_cast<double>(lex.size());
    logp_.clear();
    multichar_count_ = 0;
    for (const auto& [w, c] : lex) {
        logp_[w] = std::log((static_cast<double>(c) + 0.5) / denom);
        if (utils::utf8_chars(w).size() >= 2) ++multichar_count_;
    }
    // 未登录（非单字、不在词表）切片的惩罚分，远低于任何单字
    oov_logp_ = std::log(0.5 / denom) - 5.0;
    trained_ = true;
}

auto WordSegmenter::viterbi_(const std::vector<std::string>& chars) const
    -> std::vector<std::string> {
    int n = static_cast<int>(chars.size());
    const int L = std::max(2, cfg_.max_word_len);
    std::vector<double> best(n + 1, -1e18);
    std::vector<int> bp(n + 1, -1);
    best[0] = 0.0;
    for (int i = 1; i <= n; ++i)
        for (int len = 1; len <= L && len <= i; ++len) {
            int j = i - len;
            if (best[j] <= -1e17) continue;
            std::string w = join(chars, j, i);
            auto it = logp_.find(w);
            // 单字一定在词表；多字未登录给惩罚分（允许但不鼓励）
            double lp = (it != logp_.end()) ? it->second
                                            : (len == 1 ? -1e18 : oov_logp_);
            double cand = best[j] + lp;
            if (cand > best[i]) {
                best[i] = cand;
                bp[i] = j;
            }
        }
    std::vector<std::string> words;
    for (int i = n; i > 0;) {
        int j = bp[i];
        if (j < 0) {
            j = i - 1;  // 理论不该发生，单字兜底
        }
        words.push_back(join(chars, j, i));
        i = j;
    }
    std::reverse(words.begin(), words.end());
    return words;
}

auto WordSegmenter::segment_runs(const std::string& text) const
    -> std::vector<std::vector<std::string>> {
    std::vector<std::vector<std::string>> out;
    for (const auto& r : cjk_runs(text)) {
        if (!trained_) {
            out.push_back(r);  // 未训练：退化为单字序列
        } else {
            out.push_back(viterbi_(r));
        }
    }
    return out;
}

auto WordSegmenter::segment(const std::string& text) const
    -> std::vector<std::string> {
    std::vector<std::string> flat;
    for (auto& seg : segment_runs(text))
        for (auto& w : seg) flat.push_back(std::move(w));
    return flat;
}

}  // namespace ai_learning::learning
