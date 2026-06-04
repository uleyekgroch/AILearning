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
#include "ai_learning/testing/test_corpora.hpp"

#include <algorithm>
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
    auto gold_corpus = ai_learning::testing::make_gold_corpus();
    auto raw_corpus = ai_learning::testing::gold_to_raw(gold_corpus);

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

    // ── E. 3+ 字复合词发现 + 虚词消歧（默认配置 L=6，零词典）──────────
    std::cout << "\n[E] 3+ 字复合词 + 虚词消歧（默认配置 max_word_len=6）\n";
    auto comp_corpus = ai_learning::testing::make_compound_corpus();

    WordSegmenter cseg;  // 默认配置：max_word_len=6 + 虚词自动发现
    cseg.fit(comp_corpus);

    bool comp4 = cseg.in_lexicon("人工智能") && cseg.in_lexicon("机器学习") &&
                 cseg.in_lexicon("深度学习");
    bool comp56 = cseg.in_lexicon("计算机科学") &&
                  cseg.in_lexicon("自然语言处理");
    // 偏移碎片（真复合词的滑动子窗口）必须被边界自由度拒绝
    bool frag_rejected = !cseg.in_lexicon("工智能") &&
                         !cseg.in_lexicon("器学习") &&
                         !cseg.in_lexicon("算机科");
    bool fw_detected = cseg.is_function_word("的") &&
                       cseg.is_function_word("是") && cseg.is_function_word("在");
    // 内容词素（学/科/机）虽高频但一侧上下文受限，不应被误判为虚词
    bool content_not_fw = !cseg.is_function_word("学") &&
                          !cseg.is_function_word("科") &&
                          !cseg.is_function_word("机");
    // 切分时复合词应整体保留，不被切碎
    auto cs = cseg.segment("他在学机器学习");
    bool seg_keep =
        std::find(cs.begin(), cs.end(), std::string("机器学习")) != cs.end();

    check("3+ 字复合词进词表 (人工智能/机器学习/深度学习)", comp4);
    check("5~6 字复合词进词表 (计算机科学/自然语言处理)", comp56);
    check("偏移碎片被拒 (工智能/器学习/算机科)", frag_rejected);
    check("虚词自动发现 (的/是/在)", fw_detected,
          "fw_count=" + std::to_string(cseg.function_word_count()));
    check("内容词素不被误判为虚词 (学/科/机)", content_not_fw);
    check("切分保留完整复合词 (机器学习)", seg_keep);

    std::cout << "\n=== 客观分数: " << g_passed << " / " << g_total << " ===\n";

    bool hard = F1 >= 0.80 && words_ok && junk_ok && parse_obj_ok == N &&
                parse_subj_contains == N && trained && after && phrase &&
                comp4 && comp56 && frag_rejected && fw_detected &&
                content_not_fw && seg_keep;
    return hard ? 0 : 1;
}
