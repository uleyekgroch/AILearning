/**
 * @file test_gpu_offload.cpp
 * @brief 测试 llama.cpp GPU offload 功能
 *
 * 用法: ./test_gpu_offload /path/to/model.gguf [n_gpu_layers]
 */

#include "ai_learning/language/llm_provider.hpp"

#include <iostream>
#include <string>
#include <chrono>

auto main(int argc, char* argv[]) -> int {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " /path/to/model.gguf [n_gpu_layers]\n";
        return 1;
    }

    std::string model_path = argv[1];
    int n_gpu_layers = (argc >= 3) ? std::stoi(argv[2]) : 0;

    std::cout << "Model: " << model_path << "\n";
    std::cout << "GPU layers: " << n_gpu_layers << "\n\n";

    try {
        ai_learning::language::LlamaCppLLMProvider llm(model_path, n_gpu_layers);
        std::cout << "[OK] LLM loaded: " << llm.name() << "\n\n";

        std::string prompt = "用一句话解释什么是人工智能";
        std::cout << "Prompt: " << prompt << "\n";

        auto start = std::chrono::steady_clock::now();
        std::string response = llm.complete(prompt, "你是 helpful AI 助手");
        auto end = std::chrono::steady_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Response: " << response.substr(0, 200)
                  << (response.size() > 200 ? "..." : "") << "\n";
        std::cout << "Time: " << ms << " ms\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
