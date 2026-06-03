/**
 * @file embedding_provider.hpp
 * @brief 嵌入提供者抽象接口
 *
 * 允许系统以插件方式接入不同 embedding 后端：
 * - 内置 SGNS（Skip-gram Negative Sampling）
 * - llama.cpp 预训练模型（如 bge-small-zh）
 * - 未来：OpenAI API、本地 ONNX 等
 */
#pragma once

#include <string>
#include <vector>

namespace ai_learning::language {

/// 文本嵌入提供者接口
class IEmbeddingProvider {
public:
    virtual ~IEmbeddingProvider() = default;

    /// 将单条文本编码为稠密向量
    virtual auto embed(const std::string& text) -> std::vector<float> = 0;

    /// 批量编码（可能比多次调用 embed 更高效）
    virtual auto embed_batch(const std::vector<std::string>& texts)
        -> std::vector<std::vector<float>> = 0;

    /// 嵌入维度
    [[nodiscard]] virtual auto dim() const -> int = 0;

    /// 后端是否可用（模型已加载且正常）
    [[nodiscard]] virtual auto is_available() const -> bool = 0;
};

}  // namespace ai_learning::language
