/**
 * @file semantic_eval.cpp
 * @brief 客观语义评测 — 用固定语料 + 可复现指标，量化"系统自己学到了多少语义"。
 *
 * 目的：取代主观的"与人类相似度 80%"自评，给出可复现、可回归的客观分数。
 * 全程不依赖任何外部大模型（LLM/预训练嵌入），只用系统自学的：
 *   分布语义(PPMI 共现) + 预测编码引擎(神经↔符号桥)。
 *
 * 评测思想（分布语义假设 Firth 1957）：
 *   语义相近的概念，其上下文分布相近。若系统真学到了语义，则：
 *   - 同类概念（数学/物理/化学）相似度应高于跨类（数学 vs 苹果）；
 *   - most_similar(数学) 的近邻里应出现同类概念；
 *   - 预测编码桥 semantic_associate() 能从概念向量联想回符号；
 *   - think() 不再对已学概念回答"不知道"。
 */

#include "ai_learning/core/learner.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

using ai_learning::core::Learner;
using ai_learning::core::LearnerConfig;

int g_passed = 0;
int g_total = 0;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    ++g_total;
    if (ok) ++g_passed;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
}

/// 固定结构化语料：两个语义簇，簇内共享上下文，簇间不共享。
const std::vector<std::string>& corpus() {
    static const std::vector<std::string> c = {
        // 学科簇：数学/物理/化学 共享 研究/学科/理论/科学
        "数学 是 研究 数量 的 学科 理论 科学",
        "物理 是 研究 物质 的 学科 理论 科学",
        "化学 是 研究 物质 的 学科 理论 科学",
        "数学 学科 理论 推导 严谨 科学 研究",
        "物理 学科 理论 实验 严谨 科学 研究",
        "化学 学科 理论 实验 严谨 科学 研究",
        // 水果簇：苹果/香蕉/橙子 共享 水果/甜/吃/营养
        "苹果 是 甜 的 水果 好吃 营养 健康",
        "香蕉 是 甜 的 水果 好吃 营养 健康",
        "橙子 是 甜 的 水果 好吃 营养 健康",
        "苹果 水果 甜 多汁 营养 健康 好吃",
        "香蕉 水果 甜 软糯 营养 健康 好吃",
        "橙子 水果 甜 多汁 营养 健康 好吃",
    };
    return c;
}

}  // namespace

auto main() -> int {
    std::cout << "=== 客观语义评测 (零大模型，纯自学) ===\n\n";

    // 配置：开启自学语义管线（默认关闭以兼容旧行为，这里显式打开）
    LearnerConfig config;
    config.obs_dim = 64;
    config.action_dim = 8;
    config.embedding_dim = 64;                 // 必须与 obs_dim 一致
    config.embedding_learning_enabled = true;  // 开启分布语义 + SGNS 自学
    config.embedding_predictive_learning = true;  // 开启预测编码预测学习
    config.ds_min_freq = 2;                    // 小语料降低阈值
    config.statistical_min_freq = 2;
    config.embedding_min_count = 1;

    Learner learner(config);

    // 训练：重复读语料以累积共现统计（模拟反复接触）
    constexpr int kEpochs = 40;
    for (int e = 0; e < kEpochs; ++e) {
        for (const auto& line : corpus()) {
            learner.learn_from_text(line, "eval_corpus");
        }
    }

    auto& ds = learner.distributional_semantics();

    // 指标 1：分布语义空间确实被自学填充
    bool has_core = ds.has_cpt("数学") && ds.has_cpt("物理") &&
                    ds.has_cpt("苹果");
    check("分布语义空间已自学填充(数学/物理/苹果均有向量)", has_core);

    // 指标 2：同类相似度 > 跨类相似度（学到了类别结构）
    double s_math_phys = ds.similarity("数学", "物理").similarity;
    double s_math_chem = ds.similarity("数学", "化学").similarity;
    double s_math_apple = ds.similarity("数学", "苹果").similarity;
    double s_math_banana = ds.similarity("数学", "香蕉").similarity;
    double within = (s_math_phys + s_math_chem) / 2.0;
    double across = (s_math_apple + s_math_banana) / 2.0;
    check("同类相似度 > 跨类相似度 (数学~物理/化学 > 数学~苹果/香蕉)",
          within > across,
          "within=" + std::to_string(within) +
              " across=" + std::to_string(across));

    // 指标 3：most_similar(数学) 的"概念近邻"(过滤同源碎片)含同类概念
    auto neigh_math = ds.most_similar("数学", 5, /*exclude_ngram_overlap=*/true);
    bool math_ok = false;
    std::string math_list;
    for (const auto& r : neigh_math) {
        math_list += r.cpt_b + " ";
        if (r.cpt_b == "物理" || r.cpt_b == "化学") math_ok = true;
    }
    check("most_similar(数学) 概念近邻含 物理/化学", math_ok, math_list);

    // 指标 4：most_similar(苹果) 的"概念近邻"(过滤同源碎片)含同类概念
    auto neigh_apple = ds.most_similar("苹果", 5, /*exclude_ngram_overlap=*/true);
    bool apple_ok = false;
    std::string apple_list;
    for (const auto& r : neigh_apple) {
        apple_list += r.cpt_b + " ";
        if (r.cpt_b == "香蕉" || r.cpt_b == "橙子") apple_ok = true;
    }
    check("most_similar(苹果) 概念近邻含 香蕉/橙子", apple_ok, apple_list);

    // 指标 5：神经↔符号桥可用（预测编码从概念向量联想回符号）
    auto assoc = learner.semantic_associate("数学", 3);
    std::string assoc_list;
    for (const auto& a : assoc) assoc_list += a.first + " ";
    check("神经↔符号桥 semantic_associate(数学) 非空", !assoc.empty(),
          assoc_list);

    // 指标 6：think() 对已学概念不再回答"不知道"
    auto ans = learner.think("数学");
    bool think_ok = !ans.empty() && ans != "抱歉，我暂时不知道答案";
    check("think(数学) 给出有意义回答(非'不知道')", think_ok, ans);

    std::cout << "\n=== 客观分数: " << g_passed << " / " << g_total << " ===\n";

    // 硬性能力（指标1/5）必须通过，否则视为回归失败；相似度类指标作为质量分。
    bool hard_ok = has_core && !assoc.empty();
    if (!hard_ok) {
        std::cout << "硬性能力回归失败(分布语义未填充 或 神经符号桥失效)\n";
        return 1;
    }
    return 0;
}
