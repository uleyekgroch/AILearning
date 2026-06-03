/**
 * @file image_encoder.hpp
 * @brief 图像编码器接口 — 将图像转换为语义嵌入向量
 *
 * 当前实现为 Stub（占位符），返回基于图像内容的确定性向量。
 * 生产环境替换为 CLIP / LLaVA 等视觉-语言模型编码器。
 *
 * 替换路径：
 *   1. 集成 ONNX Runtime 或 llama.cpp 的 clip 模块
 *   2. 继承 IImageEncoder，实现 encode() 调用真实模型
 *   3. 在 LearnerFactory 中注入新的编码器实例
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai_learning::perception {

/// 图像编码结果
struct ImageEncodeResult {
    std::vector<float> embedding;       // 语义嵌入向量
    int width = 0;                      // 原始宽度
    int height = 0;                     // 原始高度
    std::string format;                 // 图像格式 (png, jpg, etc.)
    bool success = false;               // 编码是否成功
    std::string error;                  // 错误信息（若失败）
};

/// 图像编码器接口
class IImageEncoder {
public:
    virtual ~IImageEncoder() = default;

    /// 将图像字节数据编码为语义嵌入向量
    /// @param image_data 原始图像字节（PNG/JPG/BMP 等）
    /// @param width      图像宽度（可选，用于 stub 生成确定性向量）
    /// @param height     图像高度（可选）
    /// @return ImageEncodeResult 包含嵌入向量或错误信息
    virtual auto encode(const std::vector<uint8_t>& image_data,
                        int width = 0, int height = 0)
        -> ImageEncodeResult = 0;

    /// 获取输出嵌入维度
    [[nodiscard]] virtual auto embedding_dim() const -> int = 0;

    /// 获取编码器名称
    [[nodiscard]] virtual auto name() const -> std::string = 0;
};

// ═══════════════════════════════════════════════════════════════════
// StubImageEncoder — 占位符实现（无需外部依赖）
// ═══════════════════════════════════════════════════════════════════

/// 基于图像内容生成确定性伪嵌入（用于开发和测试）
class StubImageEncoder final : public IImageEncoder {
public:
    explicit StubImageEncoder(int dim = 512) : dim_(dim) {}

    auto encode(const std::vector<uint8_t>& image_data,
                int width, int height) -> ImageEncodeResult override {
        ImageEncodeResult result;
        result.width = width;
        result.height = height;
        result.format = "stub";
        result.success = true;

        // 基于图像内容生成确定性伪嵌入
        // 使用简单 hash + sin/cos 生成固定维度向量
        result.embedding.resize(dim_);
        uint32_t seed = 0;
        for (auto b : image_data) {
            seed = seed * 31 + static_cast<uint32_t>(b);
        }
        seed = seed ? seed : 1;  // 避免全零输入

        for (int i = 0; i < dim_; ++i) {
            float x = static_cast<float>(seed + i * 997) / static_cast<float>(UINT32_MAX);
            result.embedding[i] = std::sin(x * 6.28318530718f);  // [-1, 1]
        }

        return result;
    }

    [[nodiscard]] auto embedding_dim() const -> int override { return dim_; }
    [[nodiscard]] auto name() const -> std::string override { return "stub_image_encoder"; }

private:
    int dim_;
};

// ═══════════════════════════════════════════════════════════════════
// ClipImageEncoder — CLIP 集成占位符（文档说明如何接入）
// ═══════════════════════════════════════════════════════════════════

#ifdef AI_LEARNING_WITH_CLIP
// 当启用 CLIP 时，可接入 llama.cpp 的 clip 模块或 ONNX Runtime
// 示例：使用 llama.cpp 的 clip 编码器
// #include "clip.h"
// class ClipImageEncoder : public IImageEncoder { ... };
#endif

}  // namespace ai_learning::perception
