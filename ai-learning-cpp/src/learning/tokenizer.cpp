/**
 * @file tokenizer.cpp
 * @brief 语无关分词器实现
 */

#include "ai_learning/learning/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace ai_learning::learning {

// ── 停用词表 ──────────────────────────────────────────────────────

static const std::unordered_set<std::string>& cn_stopwords() {
    static const std::unordered_set<std::string> sw = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    };
    return sw;
}

static const std::unordered_set<std::string>& en_stopwords() {
    static const std::unordered_set<std::string> sw = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "no", "if", "then", "that", "this",
        "it", "its", "he", "she", "they", "them", "we", "us", "you",
        "i", "me", "my", "his", "her", "their", "our", "your", "am",
        "so", "than", "too", "very", "just", "also", "into", "about",
        "up", "out", "as",
    };
    return sw;
}

// ── 辅助函数 ──────────────────────────────────────────────────────

static bool is_cn_char(const std::string& s) {
    return s.size() == 3 &&
           static_cast<unsigned char>(s[0]) >= 0x80;
}

static std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out += static_cast<char>(std::tolower(c));
    }
    return out;
}

// ── 中文分词 ──────────────────────────────────────────────────────

static std::vector<std::string> tokenize_chinese(const std::string& text) {
    std::vector<std::string> chars;
    std::string buffer;
    bool in_ascii = false;

    auto flush_ascii = [&]() {
        if (in_ascii && !buffer.empty()) {
            if (buffer.size() > 1 &&
                cn_stopwords().count(buffer) == 0 &&
                en_stopwords().count(to_lower(buffer)) == 0) {
                chars.push_back(buffer);
            }
            buffer.clear();
            in_ascii = false;
        }
    };

    for (size_t i = 0; i < text.size(); ) {
        auto uc = static_cast<unsigned char>(text[i]);
        if (uc >= 0x80) {
            flush_ascii();
            // 非 ASCII 字节：收集到完整 UTF-8 字符
            int byte_len = 1;
            if (uc >= 0xF0) byte_len = 4;
            else if (uc >= 0xE0) byte_len = 3;
            else if (uc >= 0xC0) byte_len = 2;

            if (i + byte_len <= text.size()) {
                auto ch = text.substr(i, byte_len);
                if (cn_stopwords().count(ch) == 0) {
                    chars.push_back(ch);
                }
            }
            i += byte_len;
        } else {
            if (!buffer.empty() && !in_ascii) {
                // 切换到 ASCII 模式前清空非 ASCII 缓冲
            }
            in_ascii = true;
            if (std::isalnum(uc) || text[i] == '_') {
                buffer += text[i];
            } else {
                if (!buffer.empty() && buffer.size() > 1 &&
                    cn_stopwords().count(buffer) == 0 &&
                    en_stopwords().count(to_lower(buffer)) == 0) {
                    chars.push_back(buffer);
                }
                buffer.clear();
                in_ascii = false;
            }
            ++i;
        }
    }
    // 处理尾部缓冲
    if (!buffer.empty()) {
        if (buffer.size() > 1 &&
            cn_stopwords().count(buffer) == 0 &&
            en_stopwords().count(to_lower(buffer)) == 0) {
            chars.push_back(buffer);
        }
    }

    // 生成 bigram 和 trigram（仅连续中文字符）
    std::vector<std::string> tokens;
    tokens.reserve(chars.size() * 3);
    for (const auto& t : chars) tokens.push_back(t);

    int n = static_cast<int>(chars.size());
    for (int i = 0; i < n; ++i) {
        if (!is_cn_char(chars[i])) continue;
        if (i + 1 < n && is_cn_char(chars[i + 1])) {
            auto bigram = chars[i] + chars[i + 1];
            if (cn_stopwords().count(bigram) == 0) {
                tokens.push_back(bigram);
            }
        }
        if (i + 2 < n && is_cn_char(chars[i + 1]) && is_cn_char(chars[i + 2])) {
            auto trigram = chars[i] + chars[i + 1] + chars[i + 2];
            if (cn_stopwords().count(trigram) == 0) {
                tokens.push_back(trigram);
            }
        }
    }

    return tokens;
}

// ── 英文分词 ──────────────────────────────────────────────────────

static std::vector<std::string> tokenize_english(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;

    for (size_t i = 0; i < text.size(); ++i) {
        auto uc = static_cast<unsigned char>(text[i]);
        if (std::isalnum(uc) || text[i] == '_' || text[i] == '\'') {
            word += text[i];
        } else {
            if (!word.empty()) {
                auto lowered = to_lower(word);
                // 过滤停用词和长度 1 的词
                if (lowered.size() > 1 && en_stopwords().count(lowered) == 0) {
                    tokens.push_back(lowered);
                }
                word.clear();
            }
        }
    }
    // 处理尾部
    if (!word.empty()) {
        auto lowered = to_lower(word);
        if (lowered.size() > 1 && en_stopwords().count(lowered) == 0) {
            tokens.push_back(lowered);
        }
    }

    return tokens;
}

// ── 自动检测 ──────────────────────────────────────────────────────

static Language detect_language(const std::string& text) {
    int cjk_bytes = 0;
    int ascii_bytes = 0;
    int scanned = 0;

    for (size_t i = 0; i < text.size() && scanned < 100; ++i) {
        if (std::isspace(static_cast<unsigned char>(text[i]))) continue;
        auto uc = static_cast<unsigned char>(text[i]);
        if (uc >= 0x80) {
            cjk_bytes++;
        } else {
            ascii_bytes++;
        }
        scanned++;
    }

    if (scanned == 0) return Language::kEnglish;
    return (static_cast<double>(cjk_bytes) / scanned > 0.3)
        ? Language::kChinese
        : Language::kEnglish;
}

// ── 公开接口 ──────────────────────────────────────────────────────

auto tokenize(const std::string& text, Language lang) -> std::vector<std::string> {
    if (text.empty()) return {};

    if (lang == Language::kAuto) {
        lang = detect_language(text);
    }

    switch (lang) {
        case Language::kChinese: return tokenize_chinese(text);
        case Language::kEnglish: return tokenize_english(text);
        default: return tokenize_chinese(text);
    }
}

}  // namespace ai_learning::learning
