/**
 * @file llama_cpp_embedding_provider.hpp
 * @brief llama.cpp 预训练模型嵌入提供者
 *
 * 可选依赖：仅在定义 AI_LEARNING_WITH_LLAMA_CPP 时可用。
 *
 * 用法：
 *   LlamaCppEmbeddingProvider provider("/path/to/bge-small-zh-v1.5.q4_k_m.gguf");
 *   auto vec = provider.embed("人工智能");
 *
 * 模型推荐：
 *   - 中文：bge-small-zh-v1.5 (Q4_K_M, ~300MB)
 *   - 英文：gte-base (Q4_K_M, ~500MB)
 */
#pragma once

#include "embedding_provider.hpp"

#include <memory>
#include <mutex>
#include <string>

namespace ai_learning::language {

#ifdef AI_LEARNING_WITH_LLAMA_CPP

/// llama.cpp 嵌入提供者
class LlamaCppEmbeddingProvider : public IEmbeddingProvider {
public:
    /// 构造并加载模型
    /// @param model_path .gguf 模型文件路径
    /// @param embedding_dim 期望的嵌入维度（模型输出可能不同，会截断/补零）
    explicit LlamaCppEmbeddingProvider(const std::string& model_path,
                                        int embedding_dim = 512,
                                        int n_gpu_layers = 0);
    ~LlamaCppEmbeddingProvider() override;

    // 禁止拷贝（持有 llama 上下文）
    LlamaCppEmbeddingProvider(const LlamaCppEmbeddingProvider&) = delete;
    LlamaCppEmbeddingProvider& operator=(const LlamaCppEmbeddingProvider&) = delete;

    // 允许移动
    LlamaCppEmbeddingProvider(LlamaCppEmbeddingProvider&&) noexcept;
    LlamaCppEmbeddingProvider& operator=(LlamaCppEmbeddingProvider&&) noexcept;

    auto embed(const std::string& text) -> std::vector<float> override;
    auto embed_batch(const std::vector<std::string>& texts)
        -> std::vector<std::vector<float>> override;

    [[nodiscard]] auto dim() const -> int override { return embedding_dim_; }
    [[nodiscard]] auto is_available() const -> bool override { return ctx_ != nullptr; }

private:
    void* model_ = nullptr;   // opaque: llama_model*
    void* ctx_ = nullptr;       // opaque: llama_context*
    int embedding_dim_ = 512;
    int model_n_vocab_ = 0;
    mutable std::mutex mutex_;  // llama_context is not thread-safe

    /// 内部编码单条文本
    auto embed_impl_(const std::string& text) -> std::vector<float>;
};

#else

// 未启用 llama.cpp 时的占位符（编译时错误提示）
class LlamaCppEmbeddingProvider : public IEmbeddingProvider {
public:
    explicit LlamaCppEmbeddingProvider(const std::string&, int = 512) {
        throw std::runtime_error(
            "LlamaCppEmbeddingProvider requires AI_LEARNING_WITH_LLAMA_CPP "
            "to be enabled at CMake configure time. "
            "Re-run: cmake -DAI_LEARNING_WITH_LLAMA_CPP=ON ..");
    }
    auto embed(const std::string&) -> std::vector<float> override { return {}; }
    auto embed_batch(const std::vector<std::string>&)
        -> std::vector<std::vector<float>> override { return {}; }
    [[nodiscard]] auto dim() const -> int override { return 0; }
    [[nodiscard]] auto is_available() const -> bool override { return false; }
};

#endif  // AI_LEARNING_WITH_LLAMA_CPP

}  // namespace ai_learning::language
