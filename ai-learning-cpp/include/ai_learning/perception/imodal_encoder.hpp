/**
 * @file imodal_encoder.hpp
 * @brief 通用多模态编码器接口 — 图像、音频、文本统一抽象
 *
 * 设计原则：
 *   - IModalEncoder 是最高层抽象，模态无关
 *   - IImageEncoder / IAudioEncoder 是特化接口（向后兼容）
 *   - 每种编码器可独立实现，通过 MultiModalEncoder 聚合
 */

#pragma once

#include "image_encoder.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ai_learning::perception {

// ═══════════════════════════════════════════════════════════════════
// 通用模态类型
// ═══════════════════════════════════════════════════════════════════

enum class ModalityType { Image = 0, Audio = 1, Text = 2, Video = 3 };

inline auto modality_name(ModalityType t) -> const char* {
    switch (t) {
        case ModalityType::Image:  return "image";
        case ModalityType::Audio:  return "audio";
        case ModalityType::Text:   return "text";
        case ModalityType::Video:  return "video";
    }
    return "unknown";
}

// ═══════════════════════════════════════════════════════════════════
// 通用编码结果
// ═══════════════════════════════════════════════════════════════════

struct ModalEncodeResult {
    std::vector<float> embedding;       // 语义嵌入向量
    ModalityType modality = ModalityType::Image;
    std::string format;                 // 原始格式 (png, wav, mp3, ...)
    bool success = false;               // 编码是否成功
    std::string error;                  // 错误信息（若失败）
    std::map<std::string, std::string> meta;  // 模态特定元数据
};

// ═══════════════════════════════════════════════════════════════════
// 通用多模态编码器接口
// ═══════════════════════════════════════════════════════════════════

class IModalEncoder {
public:
    virtual ~IModalEncoder() = default;

    /// 将原始字节编码为语义嵌入
    /// @param raw_data   原始模态数据字节
    /// @param modality   模态类型
    /// @param meta       模态特定元数据（如 width/height/sample_rate）
    virtual auto encode(const std::vector<uint8_t>& raw_data,
                        ModalityType modality,
                        const std::map<std::string, int>& params = {})
        -> ModalEncodeResult = 0;

    /// 获取输出嵌入维度
    [[nodiscard]] virtual auto embedding_dim() const -> int = 0;

    /// 获取编码器名称
    [[nodiscard]] virtual auto name() const -> std::string = 0;

    /// 支持的模态列表
    [[nodiscard]] virtual auto supported_modalities() const
        -> std::vector<ModalityType> = 0;
};

// ═══════════════════════════════════════════════════════════════════
// 音频编码器接口
// ═══════════════════════════════════════════════════════════════════

struct AudioEncodeResult {
    std::vector<float> embedding;       // 语义嵌入向量
    int sample_rate = 0;                // 采样率
    int channels = 0;                   // 声道数
    float duration_sec = 0.0f;          // 时长（秒）
    std::string format;                 // 音频格式 (wav, mp3, etc.)
    bool success = false;               // 编码是否成功
    std::string error;                  // 错误信息
};

class IAudioEncoder {
public:
    virtual ~IAudioEncoder() = default;

    /// 将音频字节数据编码为语义嵌入
    /// @param audio_data 原始音频字节（WAV/MP3/FLAC 等）
    /// @param sample_rate 采样率（可选）
    /// @param channels    声道数（可选）
    virtual auto encode(const std::vector<uint8_t>& audio_data,
                        int sample_rate = 0, int channels = 0)
        -> AudioEncodeResult = 0;

    [[nodiscard]] virtual auto embedding_dim() const -> int = 0;
    [[nodiscard]] virtual auto name() const -> std::string = 0;
};

// ═══════════════════════════════════════════════════════════════════
// StubAudioEncoder — 占位符实现
// ═══════════════════════════════════════════════════════════════════

/// 基于音频内容生成确定性伪嵌入
class StubAudioEncoder final : public IAudioEncoder {
public:
    explicit StubAudioEncoder(int dim = 128) : dim_(dim) {}

    auto encode(const std::vector<uint8_t>& audio_data,
                int sample_rate, int channels) -> AudioEncodeResult override {
        AudioEncodeResult result;
        result.sample_rate = sample_rate;
        result.channels = channels;
        result.format = "stub";
        result.success = true;
        result.duration_sec = audio_data.empty() ? 0.0f
            : static_cast<float>(audio_data.size()) / (sample_rate * channels * 2.0f + 1.0f);

        // 基于音频内容生成确定性伪嵌入
        result.embedding.resize(dim_);
        uint32_t seed = 0;
        for (auto b : audio_data) {
            seed = seed * 31 + static_cast<uint32_t>(b);
        }
        seed = seed ? seed : 1;

        for (int i = 0; i < dim_; ++i) {
            float x = static_cast<float>(seed + i * 733) / static_cast<float>(UINT32_MAX);
            result.embedding[i] = std::sin(x * 6.28318530718f);  // [-1, 1]
        }

        return result;
    }

    [[nodiscard]] auto embedding_dim() const -> int override { return dim_; }
    [[nodiscard]] auto name() const -> std::string override { return "stub_audio_encoder"; }

private:
    int dim_;
};

// ═══════════════════════════════════════════════════════════════════
// MultiModalEncoder — 聚合多种编码器
// ═══════════════════════════════════════════════════════════════════

/// 媒体编码器聚合器 — 根据模态类型路由到对应编码器
class MediaEncoder : public IModalEncoder {
public:
    MediaEncoder() = default;

    /// 注册图像编码器
    void register_image(std::shared_ptr<class IImageEncoder> enc) {
        image_enc_ = std::move(enc);
    }

    /// 注册音频编码器
    void register_audio(std::shared_ptr<IAudioEncoder> enc) {
        audio_enc_ = std::move(enc);
    }

    auto encode(const std::vector<uint8_t>& raw_data,
                ModalityType modality,
                const std::map<std::string, int>& params = {})
        -> ModalEncodeResult override {
        ModalEncodeResult result;
        result.modality = modality;

        switch (modality) {
            case ModalityType::Image: {
                if (!image_enc_) {
                    result.error = "image encoder not registered";
                    return result;
                }
                auto img_result = image_enc_->encode(raw_data,
                    params.count("width") ? params.at("width") : 0,
                    params.count("height") ? params.at("height") : 0);
                result.embedding = std::move(img_result.embedding);
                result.format = img_result.format;
                result.success = img_result.success;
                result.error = img_result.error;
                result.meta["width"] = std::to_string(img_result.width);
                result.meta["height"] = std::to_string(img_result.height);
                break;
            }
            case ModalityType::Audio: {
                if (!audio_enc_) {
                    result.error = "audio encoder not registered";
                    return result;
                }
                auto aud_result = audio_enc_->encode(raw_data,
                    params.count("sample_rate") ? params.at("sample_rate") : 0,
                    params.count("channels") ? params.at("channels") : 0);
                result.embedding = std::move(aud_result.embedding);
                result.format = aud_result.format;
                result.success = aud_result.success;
                result.error = aud_result.error;
                result.meta["sample_rate"] = std::to_string(aud_result.sample_rate);
                result.meta["channels"] = std::to_string(aud_result.channels);
                result.meta["duration_sec"] = std::to_string(aud_result.duration_sec);
                break;
            }
            default:
                result.error = std::string("unsupported modality: ") + modality_name(modality);
                return result;
        }

        return result;
    }

    [[nodiscard]] auto embedding_dim() const -> int override {
        // 返回所有注册编码器中的最大维度
        int max_dim = 0;
        if (image_enc_) max_dim = std::max(max_dim, image_enc_->embedding_dim());
        if (audio_enc_) max_dim = std::max(max_dim, audio_enc_->embedding_dim());
        return max_dim;
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "multimodal_aggregator";
    }

    [[nodiscard]] auto supported_modalities() const
        -> std::vector<ModalityType> override {
        std::vector<ModalityType> types;
        if (image_enc_) types.push_back(ModalityType::Image);
        if (audio_enc_) types.push_back(ModalityType::Audio);
        return types;
    }

private:
    std::shared_ptr<class IImageEncoder> image_enc_;
    std::shared_ptr<IAudioEncoder> audio_enc_;
};

}  // namespace ai_learning::perception
