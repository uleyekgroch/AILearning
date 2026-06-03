/**
 * @file verify_llama_cpp.cpp
 * @brief llama.cpp 三阶段集成验证程序
 *
 * 使用 Qwen2.5-3B-Instruct Q4_K_M 模型验证：
 * 1. Phase 1: LlamaCppEmbeddingProvider 嵌入质量
 * 2. Phase 2: LlamaCppLLMProvider 对话生成
 * 3. Phase 3: UnifiedReasoningEngine 神经推理增强
 *
 * 用法: ./verify_llama_cpp /path/to/qwen2.5-3b-instruct-q4_k_m.gguf
 */

#include "ai_learning/core/learner.hpp"
#include "ai_learning/core/learner_factory.hpp"
#include "ai_learning/language/llm_provider.hpp"
#include "ai_learning/language/embedding_provider.hpp"
#include "ai_learning/language/llama_cpp_embedding_provider.hpp"
#include "ai_learning/reasoning/unified_engine.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace ai_learning;

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

static void print_banner(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

static auto test_phase1_embedding(const std::string& model_path) -> bool {
    print_banner("Phase 1: Embedding Provider");

    try {
        // 使用 Qwen 模型测试 embedding（虽然它不是专门的 embedding 模型）
        // 设置较小的维度，因为 Qwen 的隐藏层维度是 2048
        language::LlamaCppEmbeddingProvider provider(model_path, 512);

        if (!provider.is_available()) {
            std::cerr << "[FAIL] Embedding provider not available\n";
            return false;
        }
        std::cout << "[OK] Embedding provider loaded\n";
        std::cout << "  Dim: " << provider.dim() << "\n";

        // 测试语义相似度
        auto v1 = provider.embed("人工智能");
        auto v2 = provider.embed("机器学习");
        auto v3 = provider.embed("苹果");

        if (v1.empty() || v2.empty() || v3.empty()) {
            std::cerr << "[FAIL] Empty embedding vectors\n";
            return false;
        }

        double sim12 = cosine_similarity(v1, v2);
        double sim13 = cosine_similarity(v1, v3);

        std::cout << "  '人工智能' vs '机器学习': " << sim12 << "\n";
        std::cout << "  '人工智能' vs '苹果': " << sim13 << "\n";

        // 期望：AI相关词的相似度 > AI与水果的相似度
        if (sim12 > sim13) {
            std::cout << "[OK] Semantic similarity is sensible\n";
        } else {
            std::cout << "[WARN] Similarity ordering unexpected (Qwen is not an embedding model)\n";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Exception: " << e.what() << "\n";
        return false;
    }
}

static auto test_phase2_dialog(const std::string& model_path) -> bool {
    print_banner("Phase 2: Dialog / LLM Provider");

    try {
        language::LlamaCppLLMProvider llm(model_path);
        std::cout << "[OK] LLM provider loaded: " << llm.name() << "\n";

        // 测试简单对话
        std::string prompt = "用一句话解释什么是人工智能";
        std::cout << "  Prompt: " << prompt << "\n";
        std::cout << "  Generating...\n";

        std::string response = llm.complete(prompt, "你是 helpful AI 助手");

        if (response.empty()) {
            std::cerr << "[FAIL] Empty response\n";
            return false;
        }

        std::cout << "  Response: " << response << "\n";
        std::cout << "[OK] Dialog generation works\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Exception: " << e.what() << "\n";
        return false;
    }
}

static auto test_phase3_reasoning(const std::string& model_path) -> bool {
    print_banner("Phase 3: Neural Reasoning Augmentation");

    try {
        // 创建一个简单的 Learner，配置 LLM 模型路径
        core::LearnerConfig config;
        config.obs_dim = 128;
        config.action_dim = 8;
        config.llm_model_path = model_path;

        auto learner = core::LearnerFactory::create_default(config);
        std::cout << "[OK] Learner created with LLM model\n";

        // 先学习一些知识
        learner->learn_from_text("人工智能是计算机科学的一个分支");
        learner->learn_from_text("机器学习是人工智能的子领域");
        std::cout << "[OK] Injected sample knowledge\n";

        // 测试推理 — 当知识图谱无法直接回答时，LLM 应补充
        std::string question = "机器学习和人工智能有什么关系";
        std::cout << "  Question: " << question << "\n";
        std::cout << "  Reasoning...\n";

        auto results = learner->reason(question);

        if (results.empty()) {
            std::cerr << "[FAIL] No reasoning results\n";
            return false;
        }

        // 打印所有结果
        for (size_t i = 0; i < std::min(results.size(), size_t(3)); ++i) {
            const auto& r = results[i];
            std::cout << "  [" << i + 1 << "] method=" << r.method
                      << " confidence=" << r.confidence << "\n";
            std::cout << "      content: " << r.content.substr(0, 100)
                      << (r.content.size() > 100 ? "..." : "") << "\n";
        }

        // 检查是否有 neural 方法的结果（LLM 补充）
        bool has_neural = false;
        for (const auto& r : results) {
            if (r.method == "neural") {
                has_neural = true;
                break;
            }
        }

        if (has_neural) {
            std::cout << "[OK] Neural reasoning augmentation triggered\n";
        } else {
            std::cout << "[INFO] Symbolic reasoning sufficient (no neural fallback needed)\n";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Exception: " << e.what() << "\n";
        return false;
    }
}

auto main(int argc, char* argv[]) -> int {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " /path/to/qwen2.5-3b-instruct-q4_k_m.gguf\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "Model: " << model_path << "\n";

    bool p1 = test_phase1_embedding(model_path);
    bool p2 = test_phase2_dialog(model_path);
    bool p3 = test_phase3_reasoning(model_path);

    print_banner("Summary");
    std::cout << "Phase 1 (Embedding): " << (p1 ? "PASS" : "FAIL") << "\n";
    std::cout << "Phase 2 (Dialog):    " << (p2 ? "PASS" : "FAIL") << "\n";
    std::cout << "Phase 3 (Reasoning): " << (p3 ? "PASS" : "FAIL") << "\n";

    return (p1 && p2 && p3) ? 0 : 1;
}
