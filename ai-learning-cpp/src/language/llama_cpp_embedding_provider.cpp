/**
 * @file llama_cpp_embedding_provider.cpp
 * @brief llama.cpp 嵌入提供者实现
 *
 * 可选依赖：仅在定义 AI_LEARNING_WITH_LLAMA_CPP 时编译。
 */

#include "ai_learning/language/llama_cpp_embedding_provider.hpp"

#include <cmath>
#include <stdexcept>

#ifdef AI_LEARNING_WITH_LLAMA_CPP

// llama.cpp 头文件
#include <llama.h>

namespace {
inline auto to_model(void* p) -> llama_model* { return static_cast<llama_model*>(p); }
inline auto to_ctx(void* p)   -> llama_context* { return static_cast<llama_context*>(p); }
} // namespace

namespace ai_learning::language {

// ── 构造 / 析构 ─────────────────────────────────────────────────

LlamaCppEmbeddingProvider::LlamaCppEmbeddingProvider(
    const std::string& model_path, int embedding_dim)
    : embedding_dim_(embedding_dim) {

    // 模型参数
    auto mparams = llama_model_default_params();
    // embedding 模型通常不需要 GPU offload，保持 CPU 推理
    mparams.n_gpu_layers = 0;

    model_ = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error(
            "Failed to load llama.cpp model: " + model_path);
    }

    model_n_vocab_ = llama_n_vocab(to_model(model_));

    // 上下文参数
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;           // 小上下文足够 embedding
    cparams.n_batch = 512;         // 单次最大 batch
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;  // mean pooling
    cparams.embeddings = true;     // 启用 embedding 模式

    ctx_ = llama_new_context_with_model(to_model(model_), cparams);
    if (!ctx_) {
        llama_free_model(to_model(model_));
        model_ = nullptr;
        throw std::runtime_error(
            "Failed to initialize llama.cpp context");
    }
}

LlamaCppEmbeddingProvider::~LlamaCppEmbeddingProvider() {
    if (ctx_) {
        llama_free(to_ctx(ctx_));
        ctx_ = nullptr;
    }
    if (model_) {
        llama_free_model(to_model(model_));
        model_ = nullptr;
    }
}

// ── 移动语义 ───────────────────────────────────────────────────

LlamaCppEmbeddingProvider::LlamaCppEmbeddingProvider(
    LlamaCppEmbeddingProvider&& other) noexcept
    : model_(other.model_),
      ctx_(other.ctx_),
      embedding_dim_(other.embedding_dim_),
      model_n_vocab_(other.model_n_vocab_) {
    other.model_ = nullptr;
    other.ctx_ = nullptr;
}

LlamaCppEmbeddingProvider& LlamaCppEmbeddingProvider::operator=(
    LlamaCppEmbeddingProvider&& other) noexcept {
    if (this != &other) {
        if (ctx_) llama_free(to_ctx(ctx_));
        if (model_) llama_free_model(to_model(model_));
        model_ = other.model_;
        ctx_ = other.ctx_;
        embedding_dim_ = other.embedding_dim_;
        model_n_vocab_ = other.model_n_vocab_;
        other.model_ = nullptr;
        other.ctx_ = nullptr;
    }
    return *this;
}

// ── 嵌入接口 ─────────────────────────────────────────────────────

auto LlamaCppEmbeddingProvider::embed(const std::string& text)
    -> std::vector<float> {
    if (!ctx_) return std::vector<float>(embedding_dim_, 0.0f);
    return embed_impl_(text);
}

auto LlamaCppEmbeddingProvider::embed_batch(
    const std::vector<std::string>& texts)
    -> std::vector<std::vector<float>> {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(embed(text));
    }
    return results;
}

// ── 内部实现 ───────────────────────────────────────────────────

auto LlamaCppEmbeddingProvider::embed_impl_(const std::string& text)
    -> std::vector<float> {

    // Tokenize
    std::vector<llama_token> tokens(512);
    const int n_tokens = llama_tokenize(
        to_model(model_),
        text.c_str(),
        static_cast<int>(text.size()),
        tokens.data(),
        static_cast<int>(tokens.size()),
        true,   // add_special
        false   // parse_special
    );
    if (n_tokens < 0) {
        return std::vector<float>(embedding_dim_, 0.0f);
    }
    tokens.resize(n_tokens);

    // 获取模型嵌入维度
    const int model_embd_dim = llama_n_embd(to_model(model_));

    // Decode (single batch)
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    batch.logits[n_tokens - 1] = 1;  // 只在最后位置取 embedding
    batch.n_tokens = n_tokens;

    const int decode_ok = llama_decode(to_ctx(ctx_), batch);
    llama_batch_free(batch);

    if (decode_ok != 0) {
        return std::vector<float>(embedding_dim_, 0.0f);
    }

    // 获取 embeddings (mean pooling)
    const float* emb = llama_get_embeddings_seq(to_ctx(ctx_), 0);
    if (!emb) {
        // 回退到上下文级 embedding
        emb = llama_get_embeddings(to_ctx(ctx_));
    }
    if (!emb) {
        return std::vector<float>(embedding_dim_, 0.0f);
    }

    // 归一化 + 维度对齐
    std::vector<float> result;
    result.reserve(embedding_dim_);
    double norm_sq = 0.0;
    for (int i = 0; i < model_embd_dim; ++i) {
        float v = emb[i];
        result.push_back(v);
        norm_sq += v * v;
    }

    // L2 归一化
    if (norm_sq > 1e-12) {
        const float inv_norm = 1.0f / static_cast<float>(std::sqrt(norm_sq));
        for (auto& v : result) v *= inv_norm;
    }

    // 截断或补零到目标维度
    if (static_cast<int>(result.size()) > embedding_dim_) {
        result.resize(embedding_dim_);
    } else if (static_cast<int>(result.size()) < embedding_dim_) {
        result.resize(embedding_dim_, 0.0f);
    }

    return result;
}

}  // namespace ai_learning::language

#else

// llama.cpp 未启用：空实现（头文件中已有编译错误提示）
namespace ai_learning::language {}  // namespace

#endif  // AI_LEARNING_WITH_LLAMA_CPP
