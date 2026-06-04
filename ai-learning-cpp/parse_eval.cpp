/**
 * @file parse_eval.cpp
 * @brief 客观评测：无监督依存归纳的解码正确性 + 谓词论元抽取 vs 硬窗口基线。
 *
 * 三组客观指标（零大模型、可复现）：
 *   A. 解码正确性：Eisner 投影 Viterbi 的输出树，与「暴力枚举所有投影树取最大
 *      得分」逐句一致（用同一 score_tree 评分）。这是算法正确性的硬证明。
 *   B. 词类归纳：分布式聚类把同类词聚到一起（学生/老师、数学/物理）。
 *   C. 论元抽取：在依存树上取谓词的真实论元子树，对比旧的「find()+硬截6字窗口」，
 *      在带修饰语/虚词的句子上更干净地还原 (主语, 关系, 宾语)。
 */

#include "ai_learning/learning/dependency_parser.hpp"
#include "ai_learning/learning/knowledge_extractor.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

using ai_learning::learning::DependencyGrammarInducer;
using ai_learning::learning::DepParseConfig;
using ai_learning::learning::extract_triples_from_parse;
using ai_learning::learning::KnowledgeExtractor;
using ai_learning::learning::ParsedSentence;

int g_passed = 0;
int g_total = 0;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    ++g_total;
    if (ok) ++g_passed;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
}

// ── 暴力枚举所有合法投影依存树，返回最大 score_tree ──────────────────
bool reaches_root(const std::vector<int>& heads, int d) {
    int steps = 0;
    int cur = d;
    int n = static_cast<int>(heads.size());
    while (cur >= 0) {
        if (++steps > n + 1) return false;  // 成环
        cur = heads[cur];
    }
    return true;
}

bool is_projective(const std::vector<int>& heads) {
    int n = static_cast<int>(heads.size());
    // 每个节点子树须为连续区间
    for (int r = 0; r < n; ++r) {
        std::vector<char> in(n, 0);
        in[r] = 1;
        bool changed = true;
        while (changed) {
            changed = false;
            for (int d = 0; d < n; ++d)
                if (!in[d] && heads[d] >= 0 && in[heads[d]]) {
                    in[d] = 1;
                    changed = true;
                }
        }
        int lo = n, hi = -1, cnt = 0;
        for (int d = 0; d < n; ++d)
            if (in[d]) {
                lo = std::min(lo, d);
                hi = std::max(hi, d);
                ++cnt;
            }
        if (cnt != hi - lo + 1) return false;
    }
    return true;
}

double brute_best(const DependencyGrammarInducer& parser,
                  const std::vector<int>& cls) {
    int n = static_cast<int>(cls.size());
    std::vector<int> heads(n, -1);
    double best = -1e18;
    std::vector<int> choices;
    choices.push_back(-1);
    for (int k = 0; k < n; ++k) choices.push_back(k);

    std::function<void(int)> rec = [&](int d) {
        if (d == n) {
            for (int k = 0; k < n; ++k)
                if (!reaches_root(heads, k)) return;
            if (!is_projective(heads)) return;
            best = std::max(best, parser.score_tree(cls, heads));
            return;
        }
        for (int h : choices) {
            if (h == d) continue;
            heads[d] = h;
            rec(d + 1);
        }
        heads[d] = -1;
    };
    rec(0);
    return best;
}

std::vector<int> classes_of(const DependencyGrammarInducer& parser,
                            const std::vector<std::string>& ws) {
    std::vector<int> c;
    c.reserve(ws.size());
    for (const auto& w : ws) c.push_back(parser.class_of(w));
    return c;
}

}  // namespace

auto main() -> int {
    std::cout << "=== 客观依存归纳评测 (零大模型, 无监督) ===\n\n";

    // 训练语料：含「是/属于/包含」关系 + 修饰语/虚词，便于学到名词/谓词/修饰类
    std::vector<std::vector<std::string>> corpus = {
        {"学生", "是", "人类"},
        {"老师", "是", "人类"},
        {"医生", "是", "人类"},
        {"数学", "是", "学科"},
        {"物理", "是", "学科"},
        {"化学", "是", "学科"},
        {"聪明", "的", "学生", "是", "人类"},
        {"优秀", "的", "老师", "是", "人类"},
        {"重要", "的", "数学", "是", "学科"},
        {"有趣", "的", "物理", "是", "学科"},
        {"学生", "学习", "数学"},
        {"老师", "教", "物理"},
        {"猫", "是", "动物"},
        {"狗", "是", "动物"},
        {"可爱", "的", "猫", "是", "动物"},
        // 长左文修饰：用于暴露"硬截6字窗口"会切碎词，而依存取完整中心短语
        {"非常", "聪明", "而且", "勤奋", "的", "学生", "是", "人类"},
        {"非常", "聪明", "而且", "勤奋", "的", "老师", "是", "人类"},
        {"来自", "南方", "的", "高大", "的", "医生", "是", "人类"},
    };
    {  // 重复语料以累积共现统计（模拟反复接触）
        const std::vector<std::vector<std::string>> base = corpus;
        for (int rep = 0; rep < 5; ++rep)
            for (const auto& s : base) corpus.push_back(s);
    }

    DepParseConfig cfg;
    cfg.num_classes = 6;
    cfg.em_iters = 25;
    cfg.class_iters = 20;
    DependencyGrammarInducer parser(cfg);
    parser.train(corpus);

    // ── A. 解码正确性：Eisner == 暴力最优（逐句）──────────────────
    std::vector<std::vector<std::string>> dec_tests = {
        {"学生", "是", "人类"},
        {"聪明", "的", "学生", "是", "人类"},
        {"重要", "的", "数学", "是", "学科"},
        {"可爱", "的", "猫", "是", "动物"},
        {"老师", "教", "物理"},
    };
    int dec_ok = 0;
    for (const auto& ws : dec_tests) {
        auto cls = classes_of(parser, ws);
        auto ps = parser.parse(ws);
        double got = parser.score_tree(cls, ps.heads);
        double best = brute_best(parser, cls);
        if (std::abs(got - best) < 1e-6) ++dec_ok;
    }
    check("Eisner 解码 == 暴力投影最优 (逐句一致)",
          dec_ok == static_cast<int>(dec_tests.size()),
          std::to_string(dec_ok) + "/" + std::to_string(dec_tests.size()));

    // ── B. 词类归纳（分布式伪词性）：可复现的稳健分布信号 ─────────
    int c_stu = parser.class_of("学生");
    int c_tea = parser.class_of("老师");
    int c_doc = parser.class_of("医生");
    int c_de = parser.class_of("的");
    // B1: 同类人名词聚到一起（学生/老师/医生）——分布上左右邻居相近
    check("人名词聚类: 学生/老师/医生 同类",
          c_stu == c_tea && c_tea == c_doc,
          "学生=" + std::to_string(c_stu) + " 老师=" + std::to_string(c_tea) +
              " 医生=" + std::to_string(c_doc));
    // B2: 结构助词"的"自成一类，与内容词分离（功能词/内容词分布迥异）
    check("功能词分离: 助词'的' 与 内容词'学生' 不同类", c_de != c_stu,
          "的=" + std::to_string(c_de) + " 学生=" + std::to_string(c_stu));
    // B3: 聚类非退化（用到多个类，不是把所有词塞进一类）
    std::set<int> used;
    std::vector<std::string> probe = {"学生", "老师", "医生", "数学", "物理",
                                      "化学", "人类", "动物", "学科",
                                      "猫",   "狗",   "是",   "的"};
    for (const auto& w : probe) used.insert(parser.class_of(w));
    check("聚类非退化: 词类数 >= 3", used.size() >= 3,
          "实际类数=" + std::to_string(used.size()));

    // ── C. 论元抽取 vs 硬窗口基线 ─────────────────────────────────
    struct Case {
        std::vector<std::string> words;
        std::string gold_subj_head;
        std::string gold_obj_head;
    };
    std::vector<Case> cases = {
        {{"学生", "是", "人类"}, "学生", "人类"},
        {{"聪明", "的", "学生", "是", "人类"}, "学生", "人类"},
        {{"重要", "的", "数学", "是", "学科"}, "数学", "学科"},
        {{"可爱", "的", "猫", "是", "动物"}, "猫", "动物"},
        {{"有趣", "的", "物理", "是", "学科"}, "物理", "学科"},
        // 长左文：硬截6字窗口会切碎词
        {{"非常", "聪明", "而且", "勤奋", "的", "学生", "是", "人类"},
         "学生", "人类"},
        {{"来自", "南方", "的", "高大", "的", "医生", "是", "人类"},
         "医生", "人类"},
    };
    std::vector<std::pair<std::string, std::string>> cues = {{"是", "是"}};

    // 子串是否为句子中连续整词序列的拼接（词边界完整、未切碎词）
    auto word_boundary_aligned = [](const std::string& s,
                                    const std::vector<std::string>& ws) {
        if (s.empty()) return false;
        for (size_t i = 0; i < ws.size(); ++i) {
            std::string acc;
            for (size_t j = i; j < ws.size(); ++j) {
                acc += ws[j];
                if (acc == s) return true;
                if (acc.size() >= s.size()) break;
            }
        }
        return false;
    };

    int parse_obj_ok = 0, parse_subj_contains = 0;
    int win_obj_ok = 0, parse_subj_aligned = 0, win_subj_aligned = 0;
    for (const auto& c : cases) {
        // 解析法
        auto ps = parser.parse(c.words);
        auto pt = extract_triples_from_parse(ps, cues);
        std::string ps_subj, ps_obj;
        if (!pt.empty()) {
            ps_subj = pt.front().subject;
            ps_obj = pt.front().object;
        }
        if (ps_obj == c.gold_obj_head) ++parse_obj_ok;
        if (ps_subj.find(c.gold_subj_head) != std::string::npos)
            ++parse_subj_contains;
        if (word_boundary_aligned(ps_subj, c.words)) ++parse_subj_aligned;

        // 窗口基线（在无空格原文上跑旧抽取器）
        std::string joined;
        for (const auto& w : c.words) joined += w;
        auto wt = KnowledgeExtractor::extract_triples(joined, c.words);
        std::string w_subj, w_obj;
        for (const auto& t : wt)
            if (t.relation == "是") {
                w_subj = t.subject;
                w_obj = t.object;
                break;
            }
        if (w_obj == c.gold_obj_head) ++win_obj_ok;
        if (word_boundary_aligned(w_subj, c.words)) ++win_subj_aligned;

        std::cout << "    句: ";
        for (const auto& w : c.words) std::cout << w;
        std::cout << "  | 解析(主='" << ps_subj << "' 宾='" << ps_obj
                  << "')  窗口(主='" << w_subj << "' 宾='" << w_obj << "')\n";
    }
    int N = static_cast<int>(cases.size());
    check("解析法: 宾语头还原正确率 == 全对", parse_obj_ok == N,
          std::to_string(parse_obj_ok) + "/" + std::to_string(N));
    check("解析法: 主语含正确中心词 == 全对", parse_subj_contains == N,
          std::to_string(parse_subj_contains) + "/" + std::to_string(N));
    // 词边界完整性：解析法主语全部为整词序列且严格优于硬窗口（窗口在长左文切碎词）
    check("解析法主语词边界完整(整词序列) 且 > 硬窗口",
          parse_subj_aligned == N && parse_subj_aligned > win_subj_aligned,
          "解析aligned=" + std::to_string(parse_subj_aligned) + "/" +
              std::to_string(N) +
              " 窗口aligned=" + std::to_string(win_subj_aligned));

    std::cout << "\n=== 客观分数: " << g_passed << " / " << g_total << " ===\n";

    // 硬性能力：解码正确 + 宾语全对 + 主语全部命中中心词。
    bool hard_ok = (dec_ok == static_cast<int>(dec_tests.size())) &&
                   parse_obj_ok == N && parse_subj_contains == N;
    if (!hard_ok) {
        std::cout << "硬性能力回归失败(解码错误 或 论元抽取未达标)\n";
        return 1;
    }
    return 0;
}
