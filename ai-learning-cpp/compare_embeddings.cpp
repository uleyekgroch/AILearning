/**
 * @file compare_embeddings.cpp
 * @brief 对比 Qwen2.5-3B 与 bge-small-zh-v1.5 的 embedding 质量
 *
 * 用法: ./compare_embeddings /path/to/qwen.gguf /path/to/bge-small-zh.gguf
 */

#include "ai_learning/language/llama_cpp_embedding_provider.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static auto cosine_similarity(const std::vector<float>& a,
                              const std::vector<float>& b) -> double {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

struct TestPair {
    std::string a;
    std::string b;
    std::string relation;  // expected: "similar" or "different"
};

static const std::vector<TestPair> kTestPairs = {
    // 相似对 (期望高相似度)
    {"人工智能", "机器学习", "similar"},
    {"深度学习", "神经网络", "similar"},
    {"医生", "医院", "similar"},
    {"飞机", "机场", "similar"},
    {"北京", "上海", "similar"},  // 都是中国城市
    // 不相似对 (期望低相似度)
    {"人工智能", "苹果", "different"},
    {"深度学习", "香蕉", "different"},
    {"医生", "桌子", "different"},
    {"飞机", "米饭", "different"},
    {"北京", "馒头", "different"},
};

static auto evaluate_provider(const std::string& name,
                              ai_learning::language::LlamaCppEmbeddingProvider& provider)
    -> std::pair<double, double> {  // (similar_mean, different_mean)
    std::cout << "\n--- " << name << " (dim=" << provider.dim() << ") ---\n";

    std::vector<double> similar_scores;
    std::vector<double> different_scores;

    for (const auto& pair : kTestPairs) {
        auto va = provider.embed(pair.a);
        auto vb = provider.embed(pair.b);
        double sim = cosine_similarity(va, vb);

        std::cout << std::fixed << std::setprecision(4)
                  << "  [" << pair.a << "] vs [" << pair.b << "]: "
                  << sim << " (" << pair.relation << ")\n";

        if (pair.relation == "similar") {
            similar_scores.push_back(sim);
        } else {
            different_scores.push_back(sim);
        }
    }

    double sim_mean = 0.0;
    for (auto s : similar_scores) sim_mean += s;
    sim_mean /= similar_scores.size();

    double diff_mean = 0.0;
    for (auto s : different_scores) diff_mean += s;
    diff_mean /= different_scores.size();

    std::cout << "  => Similar pairs mean:   " << sim_mean << "\n";
    std::cout << "  => Different pairs mean: " << diff_mean << "\n";
    std::cout << "  => Discrimination gap:   " << (sim_mean - diff_mean) << "\n";

    return {sim_mean, diff_mean};
}

auto main(int argc, char* argv[]) -> int {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " /path/to/qwen.gguf /path/to/bge-small-zh.gguf\n";
        return 1;
    }

    std::string qwen_path = argv[1];
    std::string bge_path = argv[2];

    std::cout << std::string(70, '=') << "\n";
    std::cout << "Embedding Quality Comparison\n";
    std::cout << std::string(70, '=') << "\n";

    // --- Qwen baseline ---
    double qwen_sim = 0.0, qwen_diff = 0.0;
    try {
        ai_learning::language::LlamaCppEmbeddingProvider qwen(qwen_path, 512);
        std::tie(qwen_sim, qwen_diff) = evaluate_provider("Qwen2.5-3B-Instruct", qwen);
    } catch (const std::exception& e) {
        std::cerr << "Qwen load failed: " << e.what() << "\n";
        return 1;
    }

    // --- bge-small-zh ---
    double bge_sim = 0.0, bge_diff = 0.0;
    try {
        ai_learning::language::LlamaCppEmbeddingProvider bge(bge_path, 512);
        std::tie(bge_sim, bge_diff) = evaluate_provider("bge-small-zh-v1.5", bge);
    } catch (const std::exception& e) {
        std::cerr << "BGE load failed: " << e.what() << "\n";
        return 1;
    }

    // --- Summary ---
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Summary\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "                    Qwen2.5-3B    bge-small-zh\n";
    std::cout << "Similar mean:       " << qwen_sim << "      " << bge_sim << "\n";
    std::cout << "Different mean:     " << qwen_diff << "      " << bge_diff << "\n";
    std::cout << "Gap (sim - diff):   " << (qwen_sim - qwen_diff)
              << "      " << (bge_sim - bge_diff) << "\n";

    double qwen_gap = qwen_sim - qwen_diff;
    double bge_gap = bge_sim - bge_diff;

    if (bge_gap > qwen_gap) {
        std::cout << "\n[PASS] bge-small-zh has better discrimination (gap +"
                  << (bge_gap - qwen_gap) << ")\n";
    } else {
        std::cout << "\n[INFO] Qwen baseline gap=" << qwen_gap
                  << ", bge gap=" << bge_gap << "\n";
    }

    return 0;
}
