/**
 * @file selfmodel_eval.cpp
 * @brief 客观评测：自我模型（SelfModel）真正接入学习闭环（报告 3.3）。
 *
 * 此前 SelfModel 是「陈列端点」——有完整 API 却不进核心学习循环。
 * 本评测验证它现在随 learn_from_text 在线更新，并驱动自我指涉问答：
 *
 * 指标（零大模型、可复现）：
 *   A. 学习前：无自我信念；问「你了解X吗」如实回答不了解（不凭空声称）。
 *   B. 学习驱动：学习后自我信念数 > 0、自传记忆 > 0。
 *   C. 自我效能随经验单调上升：多学一倍语料，「持续学习」确信度严格增大。
 *   D. 自我指涉问答：学过的主题→「我了解…」，没学过的→「不太了解」。
 *   E. 不劫持常规问答：不含「你/自己」的定义性问题仍走原路径。
 */

#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/text_learner.hpp"

#include <iostream>
#include <string>

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

// 「持续学习」特质的当前自评确信度（不存在则 0）
auto learn_confidence(const TextLearner& learner) -> double {
    for (const auto& b : learner.self_model().self_concept())
        if (b.trait == "持续学习") return b.confidence;
    return 0.0;
}

}  // namespace

int main() {
    std::cout << "=== SelfModel 接入学习闭环 客观评测 ===\n\n";

    // ── A. 学习前：自我未成形 ─────────────────────────────────────
    {
        KnowledgeGraph kg;
        TextLearner learner(kg);
        check("A1 学习前无自我信念", learner.self_model().stats().self_beliefs == 0);
        auto ans = learner.think("你了解数学吗");
        check("A2 学习前如实回答不了解",
              ans.find("不太了解") != std::string::npos, ans);
    }

    // ── B/C. 学习驱动自我信念 + 自我效能单调上升 ───────────────────
    double conf_few = 0.0, conf_many = 0.0;
    {
        KnowledgeGraph kg;
        TextLearner learner(kg);
        for (int i = 0; i < 4; ++i) {
            learner.learn_from_text("数学是一门学科");
            learner.learn_from_text("数学很重要");
        }
        conf_few = learn_confidence(learner);
        check("B1 学习后形成自我信念",
              learner.self_model().stats().self_beliefs > 0,
              "beliefs=" +
                  std::to_string(learner.self_model().stats().self_beliefs));
        check("B2 验证通过的学习积累自传记忆",
              learner.self_model().stats().autobiographical_memories > 0,
              "mem=" + std::to_string(
                           learner.self_model().stats().autobiographical_memories));

        for (int i = 0; i < 8; ++i) {
            learner.learn_from_text("数学是一门学科");
            learner.learn_from_text("数学很重要");
        }
        conf_many = learn_confidence(learner);
        check("C 自我效能随经验单调上升",
              conf_many > conf_few,
              "few=" + std::to_string(conf_few) +
                  " many=" + std::to_string(conf_many));
    }

    // ── D. 自我指涉问答：学过的肯定，没学过的坦诚 ─────────────────
    {
        KnowledgeGraph kg;
        TextLearner learner(kg);
        for (int i = 0; i < 10; ++i) {
            learner.learn_from_text("数学是一门学科");
            learner.learn_from_text("数学很重要");
        }
        auto known = learner.think("你了解数学吗");
        check("D1 学过的主题→正面回答且点名主题",
              known.find("我了解") != std::string::npos &&
                  known.find("数学") != std::string::npos,
              known);
        auto unknown = learner.think("你了解量子物理吗");
        check("D2 没学过的主题→坦诚不了解",
              unknown.find("不太了解") != std::string::npos, unknown);
        auto who = learner.think("你是谁");
        check("D3 你是谁→由自我模型作答(含学习者身份)",
              who.find("学习") != std::string::npos, who);
    }

    // ── E. 不劫持常规问答 ─────────────────────────────────────────
    {
        KnowledgeGraph kg;
        TextLearner learner(kg);
        learner.learn_from_text("人工智能是计算机科学的一个分支");
        auto ans = learner.think("什么是人工智能");
        check("E 非自我问题不被自我路径劫持",
              ans.find("自评确信度") == std::string::npos &&
                  ans.find("还不太了解") == std::string::npos);
    }

    std::cout << "\n=== 客分: " << g_passed << " / " << g_total << " ===\n";
    return g_passed == g_total ? 0 : 1;
}
