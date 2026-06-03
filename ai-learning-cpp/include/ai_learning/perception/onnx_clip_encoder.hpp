/**
 * @file onnx_clip_encoder.hpp
 * @brief ONNX Runtime CLIP 图像编码器 — 将图像编码为语义嵌入向量
 *
 * 编译要求:
 *   - 定义 AI_LEARNING_WITH_ONNX
 *   - 链接 onnxruntime
 *   - stb_image.h 用于图像解码（自动从 include/third_party/ 包含）
 *
 * 模型要求:
 *   - CLIP 图像编码器 ONNX 模型（ViT-B/32 或 ViT-L/14）
 *   - 下载: https://huggingface.co/microsoft/clip-vit-base-patch32/tree/main/onnx
 *
 * 启用步骤:
 *   1. 安装 ONNX Runtime (https://onnxruntime.ai/docs/install/)
 *   2. cmake -DAI_LEARNING_WITH_ONNX=ON ..
 *   3. 下载 CLIP ONNX 模型到 models/clip-vit-base-patch32.onnx
 */

#pragma once

#include "image_encoder.hpp"

#ifdef AI_LEARNING_WITH_ONNX

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

// 尝试包含 stb_image 进行图像解码
#if __has_include("stb_image.h")
#define AI_LEARNING_HAS_STB_IMAGE
#include "stb_image.h"
#endif

namespace ai_learning::perception {

/// CLIP 图像编码器 — ONNX Runtime 后端
class OnnxClipImageEncoder final : public IImageEncoder {
public:
    /// @param model_path ONNX 模型文件路径
    /// @param embedding_dim 输出嵌入维度（ViT-B/32=512, ViT-L/14=768）
    explicit OnnxClipImageEncoder(const std::string& model_path,
                                   int embedding_dim = 512);

    ~OnnxClipImageEncoder() override = default;

    /// 编码图像为语义嵌入向量
    /// @param image_data 原始图像字节（PNG/JPG，如果 stb_image 可用）
    ///                     或预处理后的 float 数组（如果无 stb_image）
    /// @param width      图像宽度（仅在无 stb_image 时需要）
    /// @param height     图像高度（仅在无 stb_image 时需要）
    auto encode(const std::vector<uint8_t>& image_data,
                int width = 0, int height = 0) -> ImageEncodeResult override;

    [[nodiscard]] auto embedding_dim() const -> int override { return embedding_dim_; }
    [[nodiscard]] auto name() const -> std::string override { return "onnx_clip_" + model_path_; }

private:
    std::string model_path_;
    int embedding_dim_;

    // ONNX Runtime
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    // 输入/输出名称（缓存）
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_dims_;
    std::vector<std::vector<int64_t>> output_dims_;

    bool initialized_ = false;
    std::string init_error_;

    auto init_session_() -> bool;
    auto preprocess_(const std::vector<uint8_t>& image_data,
                     int width, int height) -> std::vector<float>;
    auto l2_normalize_(std::vector<float>& vec) -> void;
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline OnnxClipImageEncoder::OnnxClipImageEncoder(
    const std::string& model_path, int embedding_dim)
    : model_path_(model_path),
      embedding_dim_(embedding_dim),
      env_(ORT_LOGGING_LEVEL_WARNING, "clip_image_encoder"),
      memory_info_(Ort::MemoryInfo::CreateCpu(
          OrtArenaAllocator, OrtMemTypeDefault)) {

    if (!init_session_()) {
        std::cerr << "[OnnxClipImageEncoder] init failed: " << init_error_ << "\n";
    }
}

inline auto OnnxClipImageEncoder::init_session_() -> bool {
    try {
        // 检查模型文件存在
        std::ifstream f(model_path_, std::ios::binary);
        if (!f.good()) {
            init_error_ = "model not found: " + model_path_;
            return false;
        }
        f.close();

        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 尝试 CUDA EP（如果可用）
#ifdef AI_LEARNING_WITH_CUDA
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        } catch (const Ort::Exception&) {
            // CUDA 不可用，回退到 CPU
        }
#endif

        session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options_);

        // 缓存输入/输出名称和维度
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        size_t num_outputs = session_->GetOutputCount();

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());
            // 释放所有权但保留指针（allocator 管理内存）
            (void)name.release();

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_dims_.push_back(tensor_info.GetShape());
        }

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
            (void)name.release();

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_dims_.push_back(tensor_info.GetShape());
        }

        initialized_ = true;
        return true;
    } catch (const Ort::Exception& e) {
        init_error_ = std::string("ONNX error: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        init_error_ = std::string("error: ") + e.what();
        return false;
    }
}

inline auto OnnxClipImageEncoder::preprocess_(
    const std::vector<uint8_t>& image_data,
    int width, int height) -> std::vector<float> {

    constexpr int target_size = 224;
    constexpr float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    constexpr float stddev[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    std::vector<float> input_tensor_values(1 * 3 * target_size * target_size);

#ifdef AI_LEARNING_HAS_STB_IMAGE
    // 使用 stb_image 解码图像
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(
        image_data.data(), static_cast<int>(image_data.size()),
        &w, &h, &channels, 3);

    if (!pixels) {
        // stb_image 解码失败，尝试将数据视为原始像素
        if (width > 0 && height > 0 && image_data.size() >= static_cast<size_t>(width * height * 3)) {
            w = width; h = height;
            pixels = const_cast<stbi_uc*>(image_data.data());
            channels = 3;
        } else {
            return {};  // 失败
        }
    }

    // Resize + Normalize (双线性插值近似)
    auto get_pixel = [&](int x, int y, int c) -> float {
        if (x < 0) x = 0; if (x >= w) x = w - 1;
        if (y < 0) y = 0; if (y >= h) y = h - 1;
        return static_cast<float>(pixels[(y * w + x) * 3 + c]) / 255.0f;
    };

    for (int y = 0; y < target_size; ++y) {
        for (int x = 0; x < target_size; ++x) {
            float src_x = static_cast<float>(x) * w / target_size;
            float src_y = static_cast<float>(y) * h / target_size;
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, w - 1);
            int y1 = std::min(y0 + 1, h - 1);
            float dx = src_x - x0;
            float dy = src_y - y0;

            for (int c = 0; c < 3; ++c) {
                float v00 = get_pixel(x0, y0, c);
                float v01 = get_pixel(x0, y1, c);
                float v10 = get_pixel(x1, y0, c);
                float v11 = get_pixel(x1, y1, c);
                float v = v00 * (1 - dx) * (1 - dy) +
                          v10 * dx * (1 - dy) +
                          v01 * (1 - dx) * dy +
                          v11 * dx * dy;
                // NCHW layout
                input_tensor_values[c * target_size * target_size + y * target_size + x] =
                    (v - mean[c]) / stddev[c];
            }
        }
    }

    if (pixels != image_data.data()) {
        stbi_image_free(pixels);
    }
#else
    // 无 stb_image：假设输入已经是 224x224 RGB float 数组
    if (width == target_size && height == target_size &&
        image_data.size() == sizeof(float) * 3 * target_size * target_size) {
        std::memcpy(input_tensor_values.data(), image_data.data(), image_data.size());
    } else {
        // 无法处理
        return {};
    }
#endif

    return input_tensor_values;
}

inline auto OnnxClipImageEncoder::l2_normalize_(std::vector<float>& vec) -> void {
    float sq_sum = 0.0f;
    for (float v : vec) sq_sum += v * v;
    float norm = std::sqrt(sq_sum);
    if (norm > 1e-8f) {
        for (auto& v : vec) v /= norm;
    }
}

inline auto OnnxClipImageEncoder::encode(
    const std::vector<uint8_t>& image_data,
    int width, int height) -> ImageEncodeResult {

    ImageEncodeResult result;
    result.format = "onnx_clip";

    if (!initialized_) {
        result.success = false;
        result.error = "encoder not initialized: " + init_error_;
        return result;
    }

    // 预处理
    auto input_tensor_values = preprocess_(image_data, width, height);
    if (input_tensor_values.empty()) {
        result.success = false;
        result.error = "image preprocessing failed (ensure stb_image is available or provide preprocessed tensor)";
        return result;
    }

    try {
        // 创建输入张量
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // 运行推理
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());

        // 提取输出
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        result.embedding.assign(output_data, output_data + output_count);
        l2_normalize_(result.embedding);
        result.success = true;

    } catch (const Ort::Exception& e) {
        result.success = false;
        result.error = std::string("ONNX inference error: ") + e.what();
    } catch (const std::exception& e) {
        result.success = false;
        result.error = std::string("inference error: ") + e.what();
    }

    return result;
}

}  // namespace ai_learning::perception

#else  // !AI_LEARNING_WITH_ONNX

// ONNX 未启用时提供空占位，避免编译错误
namespace ai_learning::perception {
class OnnxClipImageEncoder : public IImageEncoder {
public:
    explicit OnnxClipImageEncoder(const std::string&, int = 512) {}
    auto encode(const std::vector<uint8_t>&, int, int) -> ImageEncodeResult override {
        return ImageEncodeResult{ {}, 0, 0, "onnx_disabled", false,
            "ONNX Runtime not enabled. Build with -DAI_LEARNING_WITH_ONNX=ON" };
    }
    [[nodiscard]] auto embedding_dim() const -> int override { return 512; }
    [[nodiscard]] auto name() const -> std::string override { return "onnx_clip_disabled"; }
};
}  // namespace ai_learning::perception

#endif  // AI_LEARNING_WITH_ONNX
