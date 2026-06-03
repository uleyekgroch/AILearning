/**
 * @file knowledge_extractor.cpp
 * @brief 知识提取器实现 — 中文文本知识提取
 *
 * MSVC 兼容：不使用 \uXXXX regex，改用手动 UTF-8 解析。
 */

#include "ai_learning/learning/knowledge_extractor.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace ai_learning::learning {

namespace {

/// 虚词过滤 — 大幅扩充的中文停用词表
auto is_stop_word(const std::string& word) -> bool {
    // 按长度分层检查，避免 O(N) 全表扫描
    int len = static_cast<int>(word.size()) / 3;  // 中文字数

    // 单字（绝大多数为虚词/量词/代词）
    static const std::unordered_set<std::string> stop1 = {
        "的", "了", "是", "在", "有", "和", "与", "或",
        "而", "但", "被", "把", "让", "给", "从", "到",
        "以", "也", "不", "就", "都", "还", "要", "会",
        "能", "可", "我", "你", "他", "她", "它", "们",
        "这", "那", "一", "二", "三", "四", "五", "六",
        "七", "八", "九", "十", "百", "千", "万", "亿",
        "个", "只", "种", "条", "件", "次", "位", "名",
        "年", "月", "日", "时", "分", "秒",
        "上", "下", "中", "里", "外", "前", "后", "左",
        "右", "大", "小", "多", "少", "长", "短", "高",
        "低", "新", "旧", "好", "坏", "最", "很", "更",
        "已", "又", "再", "将", "则", "其", "此", "些",
        "每", "各", "该", "本", "某", "无", "非", "如",
        "之", "于", "及", "等", "为", "着", "过", "地",
        "得",
    };

    // 双字停用词（仅保留虚词、代词、通用动词 — 不包含知识实体名词）
    static const std::unordered_set<std::string> stop2 = {
        // 虚词/连词
        "的是", "了的", "是在", "有了", "和与", "而被",
        "因为", "所以", "如果", "那么", "但是", "而且",
        "不过", "然后", "虽然", "因此", "然而", "或者",
        "以及", "其中", "之后", "之前", "以来",
        "关于", "对于", "由于", "通过", "进行", "可以",
        "应该", "需要", "已经", "正在", "将会", "可能",
        "能够", "应当", "必须", "同时", "另外", "此外",
        // 方位/代词
        "之间", "上面", "下面", "里面", "外面", "前面",
        "后面", "这里", "那里", "这个", "那个", "这些",
        "那些", "什么", "怎么", "如何", "为什么", "哪",
        "没有", "不是", "也要", "还会", "就要", "都要",
        // 常见动词（不是知识实体）
        "使用", "包括", "称为", "叫做", "认为", "成为",
        "开始", "发展", "产生", "影响", "存在", "提供",
        "支持", "表示", "出现", "建立", "形成", "实现",
        "具有", "包含", "属于", "导致", "构成", "位于",
        "得到", "获得", "带来", "造成", "引起", "使得",
        "根据", "按照",
        // 时间词
        "目前", "现在", "当时", "后来",
        // 泛指/量词
        "第一", "一种", "一个", "一样", "一般",
        "一定", "一些", "许多", "部分",
        // 泛用名词（过于宽泛，几乎不携带领域知识）
        "方面", "问题", "情况", "时期", "方式",
        // 片段/碎片
        "的一", "的是", "了的", "是在", "有了",
        "中的", "了一", "学家", "学的", "是一",
        "等地", "等人", "等的", "华人民", "算机",
        "中华", "人民",
    };

    // 三字停用词
    static const std::unordered_set<std::string> stop3 = {
        "是由于", "是因为", "被认为", "被称为", "被称为",
        "被称为", "被称为", "被称作", "被称为",
        "也可以", "还可以", "不仅是", "不仅要", "不仅包括",
        "就是说", "也就是说", "也就是说",
        "一部分", "大多数", "大多数",
    };

    switch (len) {
    case 1: return stop1.contains(word);
    case 2: return stop2.contains(word);
    case 3: return stop3.contains(word);
    default: return false;
    }
}

/// 判断一个词是否像知识实体（而非通用词）
auto is_candidate_entity(const std::string& word) -> bool {
    int len = static_cast<int>(word.size()) / 3;
    // 单字几乎不可能是好的知识实体
    if (len <= 1) return false;
    // 检查停用词
    if (is_stop_word(word)) return false;
    return true;
}

/// 判断字节是否为中文字符的 UTF-8 首字节
/// CJK Unified Ideographs: U+4E00..U+9FFF → E4 B8 80 .. E9 BF BF
auto is_cjk_first_byte(unsigned char c) -> bool {
    return c >= 0xE4 && c <= 0xE9;
}

/// 判断是否为 UTF-8 连续字节 (10xxxxxx)
auto is_cont(unsigned char c) -> bool {
    return (c & 0xC0) == 0x80;
}

/// 安全检查 pos 处是否为一个中文字符起始（3 字节）
auto is_cjk_at(const std::string& s, size_t pos) -> bool {
    if (pos + 2 >= s.size()) return false;
    return is_cjk_first_byte(static_cast<unsigned char>(s[pos])) &&
           is_cont(static_cast<unsigned char>(s[pos + 1])) &&
           is_cont(static_cast<unsigned char>(s[pos + 2]));
}

/// 获取中文字符串的字数（每字 3 字节 UTF-8）
auto cjk_count(const std::string& s) -> int {
    return static_cast<int>(s.size()) / 3;
}

/// 从连续中文序列中提取子串（按字符索引）
auto cjk_substr(const std::string& seq, int start_char, int char_count)
    -> std::string {
    auto byte_start = static_cast<size_t>(start_char) * 3;
    auto byte_len = static_cast<size_t>(char_count) * 3;
    if (byte_start >= seq.size()) return "";
    byte_len = std::min(byte_len, seq.size() - byte_start);
    return seq.substr(byte_start, byte_len);
}

/// 将文本拆分为连续中文序列和非中文间隔
auto extract_chinese_sequences(const std::string& text)
    -> std::vector<std::string> {
    std::vector<std::string> result;
    std::string current;

    for (size_t i = 0; i < text.size(); ) {
        if (is_cjk_at(text, i)) {
            current += text[i];
            current += text[i + 1];
            current += text[i + 2];
            i += 3;
        } else {
            if (!current.empty()) {
                result.push_back(std::move(current));
                current.clear();
            }
            ++i;
        }
    }
    if (!current.empty()) {
        result.push_back(std::move(current));
    }
    return result;
}

/// 从中文序列中提取 n-gram（按中文字数）
auto extract_ngrams(const std::string& seq, int min_n, int max_n)
    -> std::vector<std::string> {
    std::vector<std::string> result;
    int total = cjk_count(seq);

    for (int n = min_n; n <= std::min(max_n, total); ++n) {
        for (int start = 0; start + n <= total; ++start) {
            result.push_back(cjk_substr(seq, start, n));
        }
    }
    return result;
}
/// 从位置 pos 向左提取最多 max_chars 个中文字符
auto extract_left_cjk(const std::string& text, size_t pos, int max_chars)
    -> std::string {
    std::string result;
    int count = 0;

    // pos 是 size_t（无符号），pos - 3 会下溢。用 int 安全处理
    auto ipos = static_cast<int>(pos);

    while (ipos >= 3 && count < max_chars) {
        if (is_cjk_at(text, static_cast<size_t>(ipos - 3))) {
            result = text.substr(static_cast<size_t>(ipos - 3), 3) + result;
            ipos -= 3;
            ++count;
        } else {
            break;
        }
    }
    return result;
}

/// 从位置 pos 向右提取最多 max_chars 个中文字符
auto extract_right_cjk(const std::string& text, size_t pos, int max_chars)
    -> std::string {
    std::string result;
    int count = 0;
    size_t p = pos;

    while (p + 2 < text.size() && count < max_chars) {
        if (is_cjk_at(text, p)) {
            result += text.substr(p, 3);
            p += 3;
            ++count;
        } else {
            break;
        }
    }
    return result;
}

}  // namespace

// ── 中文实体提取 ─────────────────────────────────────────────────

auto KnowledgeExtractor::extract_entities(const std::string& text)
    -> std::vector<std::string> {
    // 改进：只取 2-4 字的 n-gram（中文知识实体基本在 2-4 字范围内）
    // 减少碎片生成，结合停用词过滤
    return extract_chinese_words_(text, 2, 4);
}

auto KnowledgeExtractor::extract_chinese_words_(
    const std::string& text, int min_len, int max_len)
    -> std::vector<std::string> {
    auto sequences = extract_chinese_sequences(text);

    // 统计全文 n-gram 频率
    std::unordered_map<std::string, int> freq;
    for (const auto& seq : sequences) {
        auto ngrams = extract_ngrams(seq, min_len, max_len);
        for (const auto& word : ngrams) {
            if (is_candidate_entity(word)) {
                freq[word]++;
            }
        }
    }

    // 过滤策略：按原文中文字符数判断文本长度
    // 短文本（< 40 个中文字符）接受频率 >= 1 的词，长文本要求 >= 2
    int total_cjk_chars = 0;
    for (const auto& seq : sequences) {
        total_cjk_chars += cjk_count(seq);
    }
    int min_freq = (total_cjk_chars < 40) ? 1 : 2;

    std::vector<std::string> result;
    for (const auto& [word, count] : freq) {
        if (count >= min_freq) {
            result.push_back(word);
        }
    }

    // 按频率降序排序（高频实体更有意义）
    std::sort(result.begin(), result.end(),
              [&freq](const std::string& a, const std::string& b) {
                  return freq[a] > freq[b];
              });

    // 限制每篇文章最多返回 50 个实体（防止 KG 膨胀）
    if (result.size() > 50) {
        result.resize(50);
    }

    return result;
}

// ── 关系提取 ─────────────────────────────────────────────────────

auto KnowledgeExtractor::extract_triples(
    const std::string& text,
    [[maybe_unused]] const std::vector<std::string>& entities)
    -> std::vector<Triple> {
    std::vector<Triple> triples;

    for (const auto& pattern : relation_patterns_()) {
        auto pos = text.find(pattern.second);
        while (pos != std::string::npos) {
            auto subject = extract_left_cjk(text, pos, 6);
            auto right_start = pos + pattern.second.size();
            auto object = extract_right_cjk(text, right_start, 6);

            if (cjk_count(subject) >= 2 && cjk_count(object) >= 2) {
                Triple t;
                t.subject = subject;
                t.relation = pattern.first;
                t.object = object;
                t.confidence = 0.8;
                triples.push_back(std::move(t));
            }

            pos = text.find(pattern.second, right_start);
        }
    }

    return triples;
}

// ── 因果提取 ─────────────────────────────────────────────────────

auto KnowledgeExtractor::extract_causal_links(const std::string& text)
    -> std::vector<CausalLink> {
    std::vector<CausalLink> links;

    // 因为...所以... 模式
    auto because_pos = text.find("因为");
    auto so_pos = text.find("所以");
    if (because_pos != std::string::npos &&
        so_pos != std::string::npos && so_pos > because_pos) {
        CausalLink cl;
        auto cause_start = because_pos + 6;  // "因为" = 6 UTF-8 bytes
        cl.cause = extract_right_cjk(text, cause_start, 10);
        // 截断到 "所以" 之前
        if (so_pos < cause_start + cl.cause.size()) {
            auto max_chars = static_cast<int>((so_pos - cause_start) / 3);
            cl.cause = cjk_substr(cl.cause, 0, max_chars);
        }

        auto effect_start = so_pos + 6;  // "所以" = 6 UTF-8 bytes
        cl.effect = extract_right_cjk(text, effect_start, 6);

        if (!cl.cause.empty() && !cl.effect.empty()) {
            links.push_back(std::move(cl));
        }
    }

    // "导致"/"引起"/"使得" 模式
    static const std::vector<std::string> cause_words = {"导致", "引起", "使得"};
    for (const auto& cw : cause_words) {
        auto cw_pos = text.find(cw);
        while (cw_pos != std::string::npos) {
            CausalLink cl;
            cl.cause = extract_left_cjk(text, cw_pos, 6);
            auto right_start = cw_pos + cw.size();  // cw.size() = 字节数 (UTF-8)
            cl.effect = extract_right_cjk(text, right_start, 6);

            if (!cl.cause.empty() && !cl.effect.empty()) {
                links.push_back(std::move(cl));
            }

            cw_pos = text.find(cw, right_start);
        }
    }

    return links;
}

// ── 数值提取 ─────────────────────────────────────────────────────

auto KnowledgeExtractor::extract_numerical_facts(const std::string& text)
    -> std::vector<NumericalFact> {
    std::vector<NumericalFact> facts;

    struct Pattern {
        std::string unit_marker;
        std::string attr;
        std::string unit;
    };

    std::vector<Pattern> patterns = {
        {"度", "温度", "摄氏度"},
        {"米", "长度", "米"},
        {"千克", "重量", "千克"},
        {"年", "时间", "年"},
        {"秒", "时间", "秒"},
    };

    for (const auto& p : patterns) {
        auto pos = text.find(p.unit_marker);
        while (pos != std::string::npos) {
            // 向左扫描数字
            std::string num_str;
            size_t i = pos;
            while (i > 0) {
                auto c = static_cast<unsigned char>(text[i - 1]);
                if ((c >= '0' && c <= '9') || c == '.') {
                    num_str = std::string(1, text[i - 1]) + num_str;
                    --i;
                } else {
                    break;
                }
            }

            if (!num_str.empty()) {
                try {
                    NumericalFact f;
                    f.attribute = p.attr;
                    f.value = std::stod(num_str);
                    f.unit = p.unit;
                    facts.push_back(std::move(f));
                } catch (...) {
                    // 忽略解析错误
                }
            }

            pos = text.find(p.unit_marker, pos + p.unit_marker.size());
        }
    }

    return facts;
}

// ── 关键词提取 ───────────────────────────────────────────────────

auto KnowledgeExtractor::extract_keywords(const std::string& text)
    -> std::vector<std::string> {
    return extract_entities(text);
}

// ── 关系模式表 ───────────────────────────────────────────────────

auto KnowledgeExtractor::relation_patterns_()
    -> const std::vector<std::pair<std::string, std::string>>& {
    static const std::vector<std::pair<std::string, std::string>> patterns = {
        {"是", "是"},
        {"属于", "属于"},
        {"位于", "位于"},
        {"包含", "包含"},
        {"包括", "包括"},
        {"产生", "产生"},
        {"导致", "导致"},
        {"构成", "构成"},
        {"叫做", "叫做"},
        {"称为", "称为"},
    };
    return patterns;
}

}  // namespace ai_learning::learning
