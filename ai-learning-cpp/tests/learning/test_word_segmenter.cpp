/**
 * @file test_word_segmenter.cpp
 * @brief WordSegmenter（无监督分词）+ 依存树论元抽取兜底 + 关系模式表 单测 — TDD
 */

#include <catch2/catch_test_macros.hpp>

#include "ai_learning/learning/dependency_parser.hpp"
#include "ai_learning/learning/knowledge_extractor.hpp"
#include "ai_learning/learning/word_segmenter.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace ai_learning::learning;

namespace {

/// 生成「内容词出现在多种左右上下文」的无空格中文语料（边界熵足够大），
/// 与 segment_eval 同构，保证 fit() 能稳定学到真实词、拒绝虚词拼接。
auto make_raw_corpus() -> std::vector<std::string> {
    std::vector<std::string> persons = {"学生", "老师", "医生", "工人", "农民"};
    std::vector<std::string> subjects = {"数学", "物理", "化学", "历史", "地理"};
    std::vector<std::string> animals = {"猫", "狗", "老虎", "兔子"};
    std::vector<std::string> mods = {"聪明", "勤奋", "优秀", "可爱",
                                     "重要", "有趣", "年轻"};
    std::vector<std::string> verbs = {"喜欢", "学习", "研究", "讨厌"};
    std::vector<std::string> pronouns = {"我", "他", "她", "你"};
    std::vector<std::string> degree = {"很", "非常", "比较", "特别"};

    std::vector<std::vector<std::string>> g;
    auto add = [&](std::vector<std::string> ws) { g.push_back(std::move(ws)); };
    for (const auto& p : persons) {
        add({p, "是", "人类"});
        add({"人类", "包括", p});
    }
    for (const auto& s : subjects) {
        add({s, "是", "学科"});
        add({s, "是", "知识"});
        add({"学科", "包括", s});
    }
    for (const auto& a : animals) {
        add({a, "是", "动物"});
        add({"动物", "包括", a});
    }
    for (const auto& m : mods) {
        for (const auto& p : persons) add({m, "的", p, "是", "人类"});
        for (const auto& s : subjects) add({m, "的", s, "是", "学科"});
        for (const auto& a : animals) add({m, "的", a, "是", "动物"});
    }
    for (const auto& p : persons)
        for (const auto& d : degree)
            for (const auto& m : mods) add({p, d, m});
    for (const auto& s : subjects)
        for (const auto& d : degree) add({s, d, "重要"});
    for (const auto& pr : pronouns)
        for (const auto& v : verbs)
            for (const auto& s : subjects) add({pr, v, s});
    for (const auto& p : persons)
        for (const auto& v : verbs)
            for (const auto& s : subjects) add({p, v, s});
    for (const auto& pr : pronouns)
        for (const auto& v : verbs)
            for (const auto& a : animals) add({pr, v, a});
    {
        auto base = g;
        for (int rep = 0; rep < 3; ++rep)
            for (auto& s : base) g.push_back(s);
    }

    std::vector<std::string> raw;
    raw.reserve(g.size());
    for (const auto& ws : g) {
        std::string j;
        for (const auto& w : ws) j += w;
        raw.push_back(j);
    }
    return raw;
}

auto contains(const std::vector<std::string>& v, const std::string& x) -> bool {
    return std::find(v.begin(), v.end(), x) != v.end();
}

auto cfg2() -> SegmenterConfig {
    SegmenterConfig c;
    c.max_word_len = 2;  // 语料词汇以 1~2 字为主
    c.min_count = 3;
    c.min_cohesion = 1.5;
    c.min_freedom = 0.3;
    return c;
}

/// 含 3~6 字技术复合词 + 多上下文虚词的无空格语料（默认配置 L=6 用）。
auto make_compound_corpus() -> std::vector<std::string> {
    std::vector<std::string> base = {
        "人工智能改变世界",   "我了解人工智能",     "人工智能很强大",
        "他害怕人工智能",     "人工智能需要数据",   "计算机科学很有趣",
        "我喜欢计算机科学",   "他研究计算机科学",   "计算机科学包含很多方向",
        "机器学习是热门方向", "我在学机器学习",     "机器学习应用广泛",
        "公司使用机器学习",   "深度学习效果很好",   "他擅长深度学习",
        "深度学习改变了行业", "自然语言处理很难",   "我研究自然语言处理",
        "自然语言处理有用",   "这是一本书",         "那是一只猫",
        "他是医生",           "水是透明的",         "花是红的",
        "天空是蓝的",         "我吃了饭",           "他走了",
        "下雨了",             "花开了",             "桌子上的书",
        "老师的话",           "妈妈的爱",           "他在家里",
        "猫在沙发上",         "书在桌子上",         "鸟在天上飞",
        "小鸟在唱歌",         "大海很辽阔",         "孩子在公园玩",
        "苹果很甜",           "香蕉是黄的",         "橙子很酸",
        "美丽的花朵",         "红色的苹果",         "聪明的孩子",
        "勤劳的农民",         "遥远的地方",         "温暖的阳光",
        "高大的树木",         "干净的房间",
    };
    std::vector<std::string> c;
    for (int rep = 0; rep < 8; ++rep)
        for (const auto& s : base) c.push_back(s);
    return c;
}

}  // namespace

TEST_CASE("WordSegmenter: 未训练/空输入安全", "[segmenter]") {
    WordSegmenter seg;
    REQUIRE(seg.segment("").empty());

    // 未训练：单字兜底，绝不漏切（每个 CJK 字成一个 token）
    auto w = seg.segment("学生");
    REQUIRE(w.size() == 2);
    REQUIRE(seg.multichar_word_count() == 0);
}

TEST_CASE("WordSegmenter: 新词发现入词表、虚词拼接被拒", "[segmenter]") {
    WordSegmenter seg(cfg2());
    seg.fit(make_raw_corpus());

    REQUIRE(seg.multichar_word_count() > 0);

    SECTION("真实词进入词表") {
        REQUIRE(seg.in_lexicon("学生"));
        REQUIRE(seg.in_lexicon("数学"));
        REQUIRE(seg.in_lexicon("动物"));
        REQUIRE(seg.in_lexicon("老师"));
    }
    SECTION("虚词黏连不进词表") {
        REQUIRE_FALSE(seg.in_lexicon("是人"));
        REQUIRE_FALSE(seg.in_lexicon("的学"));
        REQUIRE_FALSE(seg.in_lexicon("是动"));
    }
}

TEST_CASE("WordSegmenter: Viterbi 切分还原已知词", "[segmenter]") {
    WordSegmenter seg(cfg2());
    seg.fit(make_raw_corpus());

    auto words = seg.segment("聪明的学生是人类");
    REQUIRE(contains(words, "学生"));
    REQUIRE(contains(words, "人类"));
    // 关系线索「是」应被切为独立单字，而非黏进相邻词
    REQUIRE(contains(words, "是"));
    // 拼接还原原串（无字符丢失/重复）
    std::string joined;
    for (const auto& w : words) joined += w;
    REQUIRE(joined == "聪明的学生是人类");
}

TEST_CASE("WordSegmenter: segment_runs 按非 CJK 边界分段", "[segmenter]") {
    WordSegmenter seg(cfg2());
    seg.fit(make_raw_corpus());

    auto runs = seg.segment_runs("学生, 数学");
    REQUIRE(runs.size() == 2);  // 逗号+空格切成两段
}

TEST_CASE("WordSegmenter: 3+ 字复合词发现，偏移碎片被拒", "[segmenter]") {
    WordSegmenter seg;  // 默认 max_word_len=6
    seg.fit(make_compound_corpus());

    SECTION("真复合词进词表（3~6 字）") {
        REQUIRE(seg.in_lexicon("人工智能"));
        REQUIRE(seg.in_lexicon("机器学习"));
        REQUIRE(seg.in_lexicon("深度学习"));
        REQUIRE(seg.in_lexicon("计算机科学"));
        REQUIRE(seg.in_lexicon("自然语言处理"));
    }
    SECTION("偏移碎片被边界自由度拒绝") {
        // 这些是真复合词的滑动子窗口，一侧总被同一字紧贴 → 左/右熵≈0
        REQUIRE_FALSE(seg.in_lexicon("工智能"));
        REQUIRE_FALSE(seg.in_lexicon("器学习"));
        REQUIRE_FALSE(seg.in_lexicon("算机科"));
        REQUIRE_FALSE(seg.in_lexicon("然语言"));
    }
    SECTION("切分保留完整复合词，不切碎") {
        auto w = seg.segment("他在学机器学习");
        REQUIRE(contains(w, "机器学习"));
        // 拼接无损
        std::string j;
        for (const auto& x : w) j += x;
        REQUIRE(j == "他在学机器学习");
    }
}

TEST_CASE("WordSegmenter: 虚词自动发现，内容词素不误判", "[segmenter]") {
    WordSegmenter seg;
    seg.fit(make_compound_corpus());

    SECTION("高频且左右皆自由的单字判为虚词") {
        REQUIRE(seg.is_function_word("的"));
        REQUIRE(seg.is_function_word("是"));
        REQUIRE(seg.is_function_word("在"));
        REQUIRE(seg.function_word_count() >= 3);
    }
    SECTION("高频但一侧上下文受限的内容词素不判为虚词") {
        // 学/科/机 在 科学/机器… 里一侧紧贴固定字 → min(左熵,右熵) 偏低
        REQUIRE_FALSE(seg.is_function_word("学"));
        REQUIRE_FALSE(seg.is_function_word("科"));
        REQUIRE_FALSE(seg.is_function_word("机"));
    }
    SECTION("3+ 字复合词不以虚词开头/结尾（边界过滤仅作用于 len>=3）") {
        // 设计约定：len<=2 完全保持既有行为，虚词边界过滤只对 len>=3 生效，
        // 因为长复合词才是本次要保护、不被跨词虚词污染的目标。
        for (const auto& w : seg.multichar_words()) {
            auto cs = ai_learning::utils::utf8_chars(w);
            if (cs.size() < 3) continue;
            REQUIRE_FALSE(seg.is_function_word(cs.front()));
            REQUIRE_FALSE(seg.is_function_word(cs.back()));
        }
    }
}

TEST_CASE("WordSegmenter: 默认配置不改变短词(len<=2)既有行为", "[segmenter]") {
    // 回归保护：len≤2 完全走原路径；用 cfg2(L=2) 与默认(L=6) 对同一 2 字词料
    // 比较——2 字词的收录不应因放开 L 而改变。
    auto corpus = make_raw_corpus();
    WordSegmenter s2(cfg2());
    s2.fit(corpus);
    SegmenterConfig c6;
    c6.min_count = 3;
    c6.min_cohesion = 1.5;
    c6.min_freedom = 0.3;  // 同 cfg2 但 max_word_len=6（默认）
    WordSegmenter s6(c6);
    s6.fit(corpus);
    for (const auto& w : {"学生", "数学", "动物", "老师"}) {
        REQUIRE(s2.in_lexicon(w) == s6.in_lexicon(w));
    }
    for (const auto& j : {"是人", "的学", "是动"}) {
        REQUIRE_FALSE(s6.in_lexicon(j));
    }
}

TEST_CASE("KnowledgeExtractor::relation_patterns 非空且含「是」", "[extractor]") {
    const auto& pats = KnowledgeExtractor::relation_patterns();
    REQUIRE_FALSE(pats.empty());
    bool has_shi = false;
    for (const auto& [rel, cue] : pats)
        if (rel == "是" || cue == "是") has_shi = true;
    REQUIRE(has_shi);
}

TEST_CASE("extract_triples_from_parse: 边界中心词失败时相邻词兜底", "[parse]") {
    // 手工构造一棵「边界中心词搜索必然失败」的树：
    //   谓词在 p=2（"是"）；左区间每个词都挂在左区间内部（head∈[0,p-1]），
    //   右区间每个词都挂在右区间内部（head>p）→ 主/宾边界中心词都找不到，
    //   触发兜底：主=p-1、宾=p+1，并取其依存子树作为论元短语。
    ParsedSentence ps;
    ps.words = {"甲甲", "乙乙", "是", "丙丙", "丁丁"};
    ps.classes = std::vector<int>(ps.words.size(), 0);
    ps.heads = {1, 0, -1, 4, 3};  // 左区间互挂、右区间互挂

    auto triples = extract_triples_from_parse(ps, {{"是", "是"}});
    REQUIRE_FALSE(triples.empty());
    REQUIRE(triples.front().relation == "是");
    // 论元来自兜底的相邻词子树（位于谓词两侧、≥2 个 CJK 字）
    REQUIRE(triples.front().subject.find("乙乙") != std::string::npos);
    REQUIRE(triples.front().object.find("丙丙") != std::string::npos);
}
