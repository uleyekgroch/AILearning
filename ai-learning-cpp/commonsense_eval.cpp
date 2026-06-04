/**
 * @file commonsense_eval.cpp
 * @brief 客观评测：常识库填充（报告 3.4b）。
 *
 * 此前 query_commonsense_ 对没有直接 KG 三元组的概念恒返回空，think() 因此
 * 落到海马「原句回放」。本评测验证：常识查询现在在 KG 无直接命中时，用系统
 * 自己学到的分布语义空间（PPMI 共现，零大模型）给出常识式联想作答。
 *
 * 指标（零大模型、可复现）：
 *   A. KG 直接命中优先：定义性问题且 KG 有 是/属于 关系 → 返回 KG 事实
 *      （含「基于已学知识」），而非语义兜底。
 *   B. 常识库不再恒空：对只在语料中共现、KG 无定义关系的概念问「什么是X」，
 *      返回自学语义联想（含「通常与」+ 同类概念），不再「不知道」。
 *   C. 语义联想命中同类：苹果↔香蕉/橙子 的共现近邻被正确召回。
 *   D. 无幻觉：从未学过的概念仍坦诚「不知道」（不凭空联想）。
 *   E. 不劫持非意图问题：不含定义/因果/位置意图词的问题不会触发语义兜底。
 */

#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/text_learner.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

using ai_learning::domain::knowledge::KnowledgeGraph;
using ai_learning::learning::TextLearner;

int g_passed = 0, g_total = 0;
void check(const std::string& name, bool ok, const std::string& detail = "") {
    ++g_total;
    if (ok) ++g_passed;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
}

auto has(const std::string& s, const std::string& sub) -> bool {
    return s.find(sub) != std::string::npos;
}

// 共现语料：三种水果共享同一组内容词上下文（甜/水果/好吃/营养/健康/多汁/新鲜），
// 但全程不含「是/属于」等定义关系线索 → KG 不会为水果建立定义三元组，从而强制
// 走自学语义联想兜底；上下文一致使水果彼此分布相似度最高、稳定盖过单个属性词。
// 另给「人工智能」一条带「是」关系的句子作为 KG 直接命中对照。
auto corpus() -> const std::vector<std::string>& {
    static const std::vector<std::string> c = {
        "苹果 甜 水果 好吃 营养 健康 多汁 新鲜",
        "香蕉 甜 水果 好吃 营养 健康 多汁 新鲜",
        "橙子 甜 水果 好吃 营养 健康 多汁 新鲜",
        "人工智能是计算机科学的分支",
    };
    return c;
}

}  // namespace

int main() {
    std::cout << "=== 常识库填充 客观评测 (零大模型，纯自学语义) ===\n\n";

    KnowledgeGraph kg;
    TextLearner learner(kg);
    constexpr int kEpochs = 30;
    for (int e = 0; e < kEpochs; ++e) {
        for (const auto& line : corpus()) {
            learner.learn_from_text(line, "eval_corpus");
        }
    }

    // ── A. KG 直接命中优先 ────────────────────────────────────────
    auto ai = learner.think("什么是人工智能");
    check("A KG 直接命中优先（定义事实，非语义兜底）",
          has(ai, "基于已学知识") && !has(ai, "基于已学语义共现"), ai);

    // ── B. 常识库不再恒空（语义联想兜底）─────────────────────────
    auto apple = learner.think("什么是苹果");
    check("B 常识库不再恒空（返回语义联想，非『不知道』/原句回放）",
          has(apple, "通常与") && has(apple, "基于已学语义共现") &&
              !has(apple, "抱歉") && !has(apple, "根据记忆"),
          apple);

    // ── C. 语义联想召回同类概念 ───────────────────────────────────
    check("C 语义近邻含同类水果（香蕉/橙子）",
          has(apple, "香蕉") || has(apple, "橙子"), apple);

    // ── D. 无幻觉：未学概念坦诚不知道 ─────────────────────────────
    auto mars = learner.think("什么是火星");
    check("D 未学概念→坦诚不知道（无幻觉联想）",
          has(mars, "抱歉"), mars);

    // ── E. 不劫持非意图问题 ───────────────────────────────────────
    // 不含定义/因果/位置意图词 → query_commonsense_ 直接返回空，不触发语义兜底。
    auto plain = learner.think("苹果香蕉");
    check("E 非意图问题不触发语义兜底",
          !has(plain, "基于已学语义共现"), plain);

    std::cout << "\n=== 客分: " << g_passed << " / " << g_total << " ===\n";
    return g_passed == g_total ? 0 : 1;
}
