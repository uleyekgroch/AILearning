/**
 * @file llm_provider.cpp
 * @brief LLM 提供者实现
 */

#include "ai_learning/language/llm_provider.hpp"

#include <cstdlib>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace ai_learning::language {

// ═══════════════════════════════════════════════════════════════════
// StubLLMProvider
// ═══════════════════════════════════════════════════════════════════

auto StubLLMProvider::complete(const std::string& prompt,
                               const std::string& /*system_prompt*/)
    -> std::string {
    // 基于关键词的简单模式匹配
    if (prompt.find("解释") != std::string::npos ||
        prompt.find("explain") != std::string::npos) {
        return "这是一个很好的问题。让我从基本概念开始解释："
               "每个知识点都可以分解为更小的组成部分，"
               "通过理解这些基础部分之间的关系，"
               "我们可以建立对整体概念的深入理解。";
    }

    if (prompt.find("为什么") != std::string::npos ||
        prompt.find("why") != std::string::npos) {
        return "这是一个关于因果关系的问题。"
               "根据已有的知识，我们可以从多个角度分析原因："
               "首先是直接原因，其次是潜在的系统性因素。"
               "建议通过实验来验证这些假设。";
    }

    if (prompt.find("怎么") != std::string::npos ||
        prompt.find("如何") != std::string::npos ||
        prompt.find("how") != std::string::npos) {
        return "关于这个问题，建议采用以下步骤："
               "首先明确目标和约束条件，"
               "然后分解为可执行的小任务，"
               "逐步验证每个步骤的正确性。";
    }

    if (prompt.find("学习") != std::string::npos ||
        prompt.find("教") != std::string::npos ||
        prompt.find("teach") != std::string::npos ||
        prompt.find("learn") != std::string::npos) {
        return "我已理解您传授的知识。"
               "让我将这些信息整合到我的知识体系中，"
               "建立相关的概念关联，并在需要时进行验证。";
    }

    // 默认响应
    return "我理解了。让我处理这些信息。";
}

// ═══════════════════════════════════════════════════════════════════
// OpenAICompatibleProvider
// ═══════════════════════════════════════════════════════════════════

OpenAICompatibleProvider::OpenAICompatibleProvider(
    const std::string& base_url,
    const std::string& api_key,
    const std::string& model)
    : base_url_(base_url), model_(model) {
    // API Key 优先级：构造参数 > 环境变量 > 硬编码默认值
    if (!api_key.empty()) {
        api_key_ = api_key;
    } else {
        const char* env_key = std::getenv("DASHSCOPE_API_KEY");
        api_key_ = (env_key != nullptr) ? env_key
                    : "sk-b68b1187aba542c3b2fec09cdc02634c";
    }
}

auto OpenAICompatibleProvider::complete(const std::string& prompt,
                                        const std::string& system_prompt)
    -> std::string {
    // 构建 OpenAI 兼容请求体
    nlohmann::json messages = nlohmann::json::array();

    if (!system_prompt.empty()) {
        messages.push_back({{"role", "system"}, {"content", system_prompt}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    nlohmann::json body = {
        {"model", model_},
        {"messages", messages}
    };

    std::string url = "https://" + base_url_ +
                      "/compatible-mode/v1/chat/completions";
    std::string raw = exec_curl_(url, body.dump());

    if (raw.empty()) {
        return "[LLM 请求失败：无响应]";
    }

    try {
        auto resp = nlohmann::json::parse(raw);
        if (resp.contains("choices") && !resp["choices"].empty()) {
            return resp["choices"][0]["message"]["content"].get<std::string>();
        }
        if (resp.contains("error")) {
            return "[LLM 错误：" +
                   resp["error"]["message"].get<std::string>() + "]";
        }
        return "[LLM 响应格式异常]";
    } catch (const nlohmann::json::exception&) {
        return "[LLM 响应解析失败]";
    }
}

auto OpenAICompatibleProvider::exec_curl_(const std::string& url,
                                          const std::string& body_json) const
    -> std::string {
    // 通过 curl 子进程执行 HTTPS POST，避免 OpenSSL 依赖
    // 将 JSON body 写入临时文件以避免 shell 转义问题
    std::string tmp_path = std::tmpnam(nullptr);

    // 写入请求体到临时文件
    std::FILE* tmp = std::fopen(tmp_path.c_str(), "w");
    if (!tmp) {
        return "";
    }
    std::fputs(body_json.c_str(), tmp);
    std::fclose(tmp);

    // 构造 curl 命令
    // -s 静默模式  -X POST  -d @file 从文件读取 body
    std::ostringstream cmd;
    cmd << "curl -s -X POST \"" << url << "\""
        << " -H \"Authorization: Bearer " << api_key_ << "\""
        << " -H \"Content-Type: application/json\""
        << " -d @" << tmp_path
        << " --max-time 30";

    // 执行并捕获输出
    std::string output;
    constexpr int kBufSize = 4096;
    char buffer[kBufSize];

#ifdef _WIN32
    auto pipe = _popen(cmd.str().c_str(), "r");
#else
    auto pipe = popen(cmd.str().c_str(), "r");
#endif

    if (pipe) {
        while (std::fgets(buffer, kBufSize, pipe) != nullptr) {
            output += buffer;
        }
#ifdef _WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }

    // 清理临时文件
    std::remove(tmp_path.c_str());

    return output;
}

// ═══════════════════════════════════════════════════════════════════
// LlamaCppLLMProvider（可选依赖）
// ═══════════════════════════════════════════════════════════════════

#ifdef AI_LEARNING_WITH_LLAMA_CPP

#include <llama.h>

namespace {

/// 构造 chat 格式的 prompt（简单的 system/user/assistant 格式）
auto build_chat_prompt(const std::string& user_prompt,
                       const std::string& system_prompt) -> std::string {
    if (system_prompt.empty()) {
        return "User: " + user_prompt + "\nAssistant: ";
    }
    return "System: " + system_prompt + "\n\nUser: " + user_prompt
           + "\nAssistant: ";
}

}  // namespace

LlamaCppLLMProvider::LlamaCppLLMProvider(const std::string& model_path)
    : max_tokens_(512), temperature_(0.8f) {
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only for compatibility

    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error(
            "Failed to load llama.cpp model: " + model_path);
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error(
            "Failed to initialize llama.cpp context");
    }
}

LlamaCppLLMProvider::~LlamaCppLLMProvider() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

LlamaCppLLMProvider::LlamaCppLLMProvider(LlamaCppLLMProvider&& other) noexcept
    : model_(other.model_), ctx_(other.ctx_),
      max_tokens_(other.max_tokens_), temperature_(other.temperature_) {
    other.model_ = nullptr;
    other.ctx_ = nullptr;
}

LlamaCppLLMProvider& LlamaCppLLMProvider::operator=(
    LlamaCppLLMProvider&& other) noexcept {
    if (this != &other) {
        if (ctx_) llama_free(ctx_);
        if (model_) llama_model_free(model_);
        model_ = other.model_;
        ctx_ = other.ctx_;
        max_tokens_ = other.max_tokens_;
        temperature_ = other.temperature_;
        other.model_ = nullptr;
        other.ctx_ = nullptr;
    }
    return *this;
}

auto LlamaCppLLMProvider::complete(const std::string& prompt,
                                   const std::string& system_prompt)
    -> std::string {
    if (!ctx_ || !model_) return "[llama.cpp not initialized]";
    return generate_(build_chat_prompt(prompt, system_prompt));
}

auto LlamaCppLLMProvider::generate_(const std::string& full_prompt)
    -> std::string {
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // ── Tokenize prompt ─────────────────────────────────────────
    const int n_ctx = llama_n_ctx(ctx_);
    std::vector<llama_token> prompt_tokens(n_ctx);
    const int n_prompt = llama_tokenize(
        vocab,
        full_prompt.c_str(),
        static_cast<int>(full_prompt.size()),
        prompt_tokens.data(),
        static_cast<int>(prompt_tokens.size()),
        true,   // add_special
        false   // parse_special
    );
    if (n_prompt < 0) {
        return "[tokenization failed]";
    }
    prompt_tokens.resize(n_prompt);

    // ── Decode prompt ───────────────────────────────────────────
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    batch.logits[n_prompt - 1] = 1;  // 只在最后位置计算 logits
    batch.n_tokens = n_prompt;

    if (llama_decode(ctx_, batch) != 0) {
        llama_batch_free(batch);
        return "[decode failed]";
    }
    llama_batch_free(batch);

    // ── Build sampler chain ─────────────────────────────────────
    llama_sampler* smpl = llama_sampler_chain_init({});
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature_));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    // ── Generation loop ─────────────────────────────────────────
    std::string result;
    int n_cur = n_prompt;

    for (int i = 0; i < max_tokens_; ++i) {
        const llama_token next = llama_sampler_sample(smpl, ctx_, -1);

        // Check end-of-generation
        if (llama_token_is_eog(vocab, next)) {
            break;
        }

        // Detokenize
        char buf[32];
        const int n = llama_token_to_piece(
            vocab, next, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }

        // Decode next token
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = next;
        b.pos[0] = n_cur++;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;

        if (llama_decode(ctx_, b) != 0) {
            llama_batch_free(b);
            break;
        }
        llama_batch_free(b);
    }

    llama_sampler_free(smpl);
    return result;
}

#endif  // AI_LEARNING_WITH_LLAMA_CPP

}  // namespace ai_learning::language
