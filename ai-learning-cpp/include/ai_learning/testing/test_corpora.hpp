#pragma once
/**
 * @file test_corpora.hpp
 * @brief Shared test/eval corpus generators (DRY: single source of truth).
 *
 * Used by both segment_eval.cpp and Catch2 unit tests.
 */

#include <string>
#include <vector>

namespace ai_learning::testing {

/// Gold word-sequence corpus: each entry is a sentence split into gold words.
/// Designed so each content word appears in diverse left/right contexts
/// (boundary entropy high enough for unsupervised segmentation to learn).
inline auto make_gold_corpus() -> std::vector<std::vector<std::string>> {
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

    // Repeat to accumulate statistics
    {
        auto base = g;
        for (int rep = 0; rep < 3; ++rep)
            for (auto& s : base) g.push_back(s);
    }
    return g;
}

/// Convert gold word sequences to raw (no-space) strings.
inline auto gold_to_raw(const std::vector<std::vector<std::string>>& gold)
    -> std::vector<std::string> {
    std::vector<std::string> raw;
    raw.reserve(gold.size());
    for (const auto& ws : gold) {
        std::string j;
        for (const auto& w : ws) j += w;
        raw.push_back(j);
    }
    return raw;
}

/// Compound corpus (3~6 char technical terms + function-word contexts, 8x repeated).
inline auto make_compound_corpus() -> std::vector<std::string> {
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

}  // namespace ai_learning::testing
