/**
 * @file segment_eval.cpp
 * @brief 客观评测：无监督分词 + 「分词→依存解析→论元抽取」全链路 vs 硬窗口基线。
 *
 * 指标（零大模型、可复现）：
 *   A. 分词质量：在自动生成的语料上学词，对测试句做最大概率切分，
 *      与 gold 词边界比较，报告词级 P/R/F1。
 *   B. 新词发现：常见词（学生/数学/动物…）进入学到的词表，
 *      而虚词拼接（是人/的学…）不进词表。
 *   C. 端到端抽取：原始无空格中文 → 分词 → 依存树 → 谓词论元，
 *      与旧「find()+硬截6字窗口」对比主/宾语词边界完整性。
 */

#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/dependency_parser.hpp"
#include "ai_learning/learning/knowledge_extractor.hpp"
#include "ai_learning/learning/text_learner.hpp"
#include "ai_learning/learning/word_segmenter.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

using ai_learning::learning::DependencyGrammarInducer;
using ai_learning::learning::DepParseConfig;
using ai_learning::learning::extract_triples_from_parse;
using ai_learning::learning::KnowledgeExtractor;
using ai_learning::learning::SegmenterConfig;
using ai_learning::learning::WordSegmenter;

int g_passed = 0, g_total = 0;
void check(const std::string& name, bool ok, const std::string& detail = "") {
    ++g_total;
    if (ok) ++g_passed;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
}

auto cjk_chars(const std::string& s) -> std::vector<std::string> {
    std::vector<std::string> r;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0) ? 2
                              : ((c & 0xF0) == 0xE0)        ? 3
                                                            : 4;
        r.push_back(s.substr(i, len));
        i += len;
    }
    return r;
}

// 词序列 → gold 字级跨度集合（用于 P/R/F1）
auto spans_of(const std::vector<std::string>& words)
    -> std::vector<std::pair<int, int>> {
    std::vector<std::pair<int, int>> sp;
    int pos = 0;
    for (const auto& w : words) {
        int len = static_cast<int>(cjk_chars(w).size());
        sp.emplace_back(pos, pos + len);
        pos += len;
    }
    return sp;
}

}  // namespace

auto main() -> int {
    std::cout << "=== 无监督分词 + 端到端抽取评测 (零大模型) ===\n\n";

    // ── 用模板自动生成语料（同时拿到 gold 分词）──────────────────
    std::vector<std::string> persons = {"学生", "老师", "医生", "工人", "农民"};
    std::vector<std::string> subjects = {"数学", "物理", "化学", "历史", "地理"};
    std::vector<std::string> animals = {"猫", "狗", "老虎", "兔子"};
    std::vector<std::string> cats_person = {"人类"};
    std::vector<std::string> cats_subject = {"学科", "知识"};
    std::vector<std::string> cats_animal = {"动物"};
    std::vector<std::string> mods = {"聪明", "勤奋", "优秀", "可爱",
                                     "重要", "有趣", "年轻"};
    std::vector<std::string> verbs = {"喜欢", "学习", "研究", "讨厌"};

    std::vector<std::string> pronouns = {"我", "他", "她", "你"};
    std::vector<std::string> degree = {"很", "非常", "比较", "特别"};

    std::vector<std::vector<std::string>> gold_corpus;  // 每句的 gold 词序列
    auto add = [&](std::vector<std::string> ws) {
        gold_corpus.push_back(std::move(ws));
    };
    // 关键：让每个内容词出现在多种左右上下文里（边界熵才够大），
    // 这正是真实语言的特征；过于规整的模板会让词总黏在同一邻字而被误判为碎片。

    // X 是 类 / 类 包括 X（让"类"既在右边界又有左右变化）
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
    // 修饰 的 X 是 类
    for (const auto& m : mods) {
        for (const auto& p : persons) add({m, "的", p, "是", "人类"});
        for (const auto& s : subjects) add({m, "的", s, "是", "学科"});
        for (const auto& a : animals) add({m, "的", a, "是", "动物"});
    }
    // 主语 + 程度 + 修饰（让 mod 右侧不总是"的"，出现在句尾/不同邻字）
    for (const auto& p : persons)
        for (const auto& d : degree)
            for (const auto& m : mods) add({p, d, m});
    for (const auto& s : subjects)
        for (const auto& d : degree) add({s, d, "重要"});
    // 代词 + 动词 + 宾语（让动词/宾语左侧多样）
    for (const auto& pr : pronouns)
        for (const auto& v : verbs)
            for (const auto& s : subjects) add({pr, v, s});
    for (const auto& p : persons)
        for (const auto& v : verbs)
            for (const auto& s : subjects) add({p, v, s});
    for (const auto& pr : pronouns)
        for (const auto& v : verbs)
            for (const auto& a : animals) add({pr, v, a});
    // 重复若干轮累积统计
    {
        auto base = gold_corpus;
        for (int rep = 0; rep < 3; ++rep)
            for (auto& s : base) gold_corpus.push_back(s);
    }

    // 原始无空格语料
    std::vector<std::string> raw_corpus;
    raw_corpus.reserve(gold_corpus.size());
    for (const auto& ws : gold_corpus) {
        std::string j;
        for (const auto& w : ws) j += w;
        raw_corpus.push_back(j);
    }

    SegmenterConfig scfg;
    scfg.max_word_len = 2;  // 本语料词汇以 1~2 字为主（中文最常见）
    scfg.min_count = 3;
    scfg.min_cohesion = 1.5;
    scfg.min_freedom = 0.3;
    WordSegmenter seg(scfg);
    seg.fit(raw_corpus);
    std::cout << "  学到多字词数: " << seg.multichar_word_count() << "\n";

    // ── A. 分词 P/R/F1（在测试句上）──────────────────────────────
    std::vector<std::vector<std::string>> tests = {
        {"聪明", "的", "学生", "是", "人类"},
        {"勤奋", "的", "医生", "喜欢", "数学"},
        {"可爱", "的", "猫", "是", "动物"},
        {"老师", "研究", "物理"},
        {"重要", "的", "化学", "是", "学科"},
        {"年轻", "的", "工人", "学习", "历史"},
    };
    long correct = 0, pred_n = 0, gold_n = 0;
    for (const auto& g : tests) {
        std::string raw;
        for (const auto& w : g) raw += w;
        auto pred = seg.segment(raw);
        auto gs = spans_of(g);
        auto ps = spans_of(pred);
        gold_n += static_cast<long>(gs.size());
        pred_n += static_cast<long>(ps.size());
        for (const auto& s : ps)
            for (const auto& t : gs)
                if (s == t) {
                    ++correct;
                    break;
                }
    }
    double P = pred_n ? static_cast<double>(correct) / pred_n : 0;
    double R = gold_n ? static_cast<double>(correct) / gold_n : 0;
    double F1 = (P + R > 0) ? 2 * P * R / (P + R) : 0;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "P=%.2f R=%.2f F1=%.2f", P, R, F1);
    check("分词词级 F1 >= 0.80", F1 >= 0.80, buf);

    // ── B. 新词发现 / 拒绝虚词拼接 ───────────────────────────────
    bool words_ok = seg.in_lexicon("学生") && seg.in_lexicon("数学") &&
                    seg.in_lexicon("动物") && seg.in_lexicon("老师");
    bool junk_ok = !seg.in_lexicon("是人") && !seg.in_lexicon("的学") &&
                   !seg.in_lexicon("是动");
    check("真实词入词表 (学生/数学/动物/老师)", words_ok);
    check("虚词拼接不入词表 (是人/的学/是动)", junk_ok);

    // ── C. 端到端抽取 vs 硬窗口（原始无空格输入）────────────────
    DepParseConfig pcfg;
    pcfg.num_classes = 6;
    pcfg.em_iters = 25;
    pcfg.class_iters = 20;
    DependencyGrammarInducer parser(pcfg);
    // 用分词后的语料训练解析器
    std::vector<std::vector<std::string>> seg_corpus;
    for (const auto& raw : raw_corpus) {
        auto runs = seg.segment_runs(raw);
        for (auto& r : runs)
            if (r.size() >= 2) seg_corpus.push_back(std::move(r));
    }
    parser.train(seg_corpus);
    std::vector<std::pair<std::string, std::string>> cues = {{"是", "是"}};

    struct Case {
        std::string raw;
        std::string gold_subj_head;
        std::string gold_obj;
    };
    std::vector<Case> cases = {
        {"聪明的学生是人类", "学生", "人类"},
        {"勤奋而且优秀的医生是人类", "医生", "人类"},
        {"非常重要的数学是学科", "数学", "学科"},
        {"可爱的猫是动物", "猫", "动物"},
        {"年轻有趣的老师是人类", "老师", "人类"},
    };
    int parse_obj_ok = 0, parse_subj_contains = 0, parse_aligned = 0,
        win_aligned = 0;
    for (const auto& c : cases) {
        auto words = seg.segment(c.raw);
        auto ps = parser.parse(words);
        auto pt = extract_triples_from_parse(ps, cues);
        std::string s, o;
        if (!pt.empty()) {
            s = pt.front().subject;
            o = pt.front().object;
        }
        if (o == c.gold_obj) ++parse_obj_ok;
        if (s.find(c.gold_subj_head) != std::string::npos) ++parse_subj_contains;
        // 词边界完整：解析法主语为分词序列的整词拼接
        bool aligned = false;
        {
            std::string acc;
            for (size_t i = 0; i < words.size() && !aligned; ++i) {
                acc.clear();
                for (size_t j = i; j < words.size(); ++j) {
                    acc += words[j];
                    if (acc == s) {
                        aligned = true;
                        break;
                    }
                    if (acc.size() >= s.size()) break;
                }
            }
        }
        if (aligned && !s.empty()) ++parse_aligned;

        // 窗口基线
        auto wt = KnowledgeExtractor::extract_triples(c.raw, words);
        std::string ws, wo;
        for (const auto& t : wt)
            if (t.relation == "是") {
                ws = t.subject;
                wo = t.object;
                break;
            }
        bool waligned = false;
        {
            std::string acc;
            for (size_t i = 0; i < words.size() && !waligned; ++i) {
                acc.clear();
                for (size_t j = i; j < words.size(); ++j) {
                    acc += words[j];
                    if (acc == ws) {
                        waligned = true;
                        break;
                    }
                    if (acc.size() >= ws.size()) break;
                }
            }
        }
        if (waligned && !ws.empty()) ++win_aligned;

        std::cout << "    原句: " << c.raw << "  | 解析(主='" << s << "' 宾='"
                  << o << "')  窗口(主='" << ws << "' 宾='" << wo << "')\n";
    }
    int N = static_cast<int>(cases.size());
    check("端到端: 宾语还原 == 全对", parse_obj_ok == N,
          std::to_string(parse_obj_ok) + "/" + std::to_string(N));
    check("端到端: 主语含正确中心词 == 全对", parse_subj_contains == N,
          std::to_string(parse_subj_contains) + "/" + std::to_string(N));
    check("端到端: 主语词边界完整 且 >= 窗口",
          parse_aligned == N && parse_aligned >= win_aligned,
          "解析aligned=" + std::to_string(parse_aligned) + "/" +
              std::to_string(N) + " 窗口aligned=" + std::to_string(win_aligned));

    // ── D. 在线管线对接：TextLearner 真实启用「分词→依存树→论元」抽取 ──
    std::cout << "\n[D] TextLearner 在线管线（默认窗口 → 训练后切换句法树）\n";
    ai_learning::domain::knowledge::KnowledgeGraph kg;
    ai_learning::learning::TextLearner learner(kg);
    for (const auto& line : raw_corpus) learner.learn_from_text(line, "eval");

    bool before = learner.dependency_parsing_enabled();
    bool trained = learner.train_dependency_parser();
    bool after = learner.dependency_parsing_enabled();
    check("训练前默认关闭、训练后启用句法树抽取",
          !before && trained && after);

    auto res = learner.learn_from_text("聪明的学生是人类", "eval");
    std::string ls, lo;
    for (const auto& t : res.triples)
        if (t.relation == "是") {
            ls = t.subject;
            lo = t.object;
            break;
        }
    // 句法树应取出完整短语主语「聪明的学生」(>2 字)而非窗口碎片
    bool phrase = lo == "人类" && ls.find("学生") != std::string::npos &&
                  cjk_chars(ls).size() >= 3;
    check("在线抽取得到词边界完整短语主语", phrase,
          "主='" + ls + "' 宾='" + lo + "'");

    std::cout << "\n=== 客观分数: " << g_passed << " / " << g_total << " ===\n";

    bool hard = F1 >= 0.80 && words_ok && junk_ok && parse_obj_ok == N &&
                parse_subj_contains == N && trained && after && phrase;
    return hard ? 0 : 1;
}
