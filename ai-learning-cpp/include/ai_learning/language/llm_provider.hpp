#pragma once
/**
 * @file llm_provider.hpp
 * @brief LLM 提供者接口 — 支持多种大语言模型后端
 *
 * 设计原则：
 * - 依赖倒置：上层通过 ILLMProvider 接口解耦具体实现
 * - 开闭原则：新增 LLM 后端只需实现接口，不修改已有代码
 *
 * 提供者：
 * 1. StubLLMProvider — 离线测试用，基于模式匹配返回固定响应
 * 2. OpenAICompatibleProvider — 调用 OpenAI 兼容 API（通义千问等）
 */

#include <stdexcept>
#include <string>

namespace ai_learning::language {

/// LLM 提供者接口
class ILLMProvider {
public:
    virtual ~ILLMProvider() = default;

    /// 发送 prompt 并获取补全响应
    /// @param prompt 用户输入
    /// @param system_prompt 系统提示词（可选）
    /// @return LLM 生成的文本
    virtual auto complete(const std::string& prompt,
                          const std::string& system_prompt = "")
        -> std::string = 0;

    /// 获取提供者名称（用于日志和调试）
    [[nodiscard]] virtual auto name() const -> std::string = 0;
};

/// 离线测试用桩实现 — 基于关键词模式匹配返回固定响应
///
/// 无需网络或 API Key，适合单元测试和离线开发。
/// 当未配置 API Key 时自动作为降级方案。
class StubLLMProvider final : public ILLMProvider {
public:
    auto complete(const std::string& prompt,
                  const std::string& system_prompt = "")
        -> std::string override;

    [[nodiscard]] auto name() const -> std::string override {
        return "stub";
    }
};

/// OpenAI 兼容 API 提供者 — 通过 curl 子进程调用 HTTPS API
///
/// 支持任何 OpenAI 兼容端点：
/// - 通义千问 (DashScope): dashscope.aliyuncs.com
/// - DeepSeek, OpenAI, 以及其他兼容服务
///
/// 使用 curl 子进程执行 HTTPS 请求，避免引入 OpenSSL 依赖。
class OpenAICompatibleProvider final : public ILLMProvider {
public:
    /// 构造 OpenAI 兼容提供者
    /// @param base_url API 基础 URL（不含协议前缀和路径）
    /// @param api_key API 密钥
    /// @param model 模型名称
    explicit OpenAICompatibleProvider(
        const std::string& base_url = "dashscope.aliyuncs.com",
        const std::string& api_key = "",
        const std::string& model = "qwen-plus-latest");

    auto complete(const std::string& prompt,
                  const std::string& system_prompt = "")
        -> std::string override;

    [[nodiscard]] auto name() const -> std::string override {
        return "openai_compatible";
    }

private:
    /// 执行 curl 命令并返回输出
    [[nodiscard]] auto exec_curl_(const std::string& url,
                                  const std::string& body_json) const
        -> std::string;

    std::string base_url_;
    std::string api_key_;
    std::string model_;
};

// ── llama.cpp 本地模型提供者（可选依赖）──────────────────────────

#ifdef AI_LEARNING_WITH_LLAMA_CPP

/// llama.cpp 本地模型 LLM 提供者
///
/// 在本地运行 .gguf 模型文件，无需网络或 API Key。
/// 适合离线部署和隐私敏感场景。
class LlamaCppLLMProvider final : public ILLMProvider {
public:
    explicit LlamaCppLLMProvider(const std::string& model_path);
    ~LlamaCppLLMProvider() override;

    // 禁止拷贝
    LlamaCppLLMProvider(const LlamaCppLLMProvider&) = delete;
    LlamaCppLLMProvider& operator=(const LlamaCppLLMProvider&) = delete;

    // 允许移动
    LlamaCppLLMProvider(LlamaCppLLMProvider&& other) noexcept;
    LlamaCppLLMProvider& operator=(LlamaCppLLMProvider&& other) noexcept;

    auto complete(const std::string& prompt,
                  const std::string& system_prompt = "")
        -> std::string override;

    [[nodiscard]] auto name() const -> std::string override {
        return "llama_cpp";
    }

private:
    void* model_ = nullptr;    // opaque: llama_model*
    void* ctx_ = nullptr;        // opaque: llama_context*
    int max_tokens_ = 512;
    float temperature_ = 0.8f;

    auto generate_(const std::string& full_prompt) -> std::string;
};

#else

/// 未启用 llama.cpp 时的占位符
class LlamaCppLLMProvider final : public ILLMProvider {
public:
    explicit LlamaCppLLMProvider(const std::string& /*model_path*/) {
        throw std::runtime_error(
            "LlamaCppLLMProvider requires AI_LEARNING_WITH_LLAMA_CPP. "
            "Re-run: cmake -DAI_LEARNING_WITH_LLAMA_CPP=ON ..");
    }
    auto complete(const std::string&, const std::string& = "")
        -> std::string override { return {}; }
    [[nodiscard]] auto name() const -> std::string override {
        return "llama_cpp_disabled";
    }
};

#endif  // AI_LEARNING_WITH_LLAMA_CPP

}  // namespace ai_learning::language
