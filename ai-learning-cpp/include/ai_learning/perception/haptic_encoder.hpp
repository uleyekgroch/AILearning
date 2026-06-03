/**
 * @file haptic_encoder.hpp
 * @brief 触觉/本体感觉编码器 — 扩展多模态感知
 *
 * 理论基础：
 *   - Lederman & Klatzky (1987) Hand movements: 探索性程序
 *   - Gibson (1966) The Senses Considered as Perceptual Systems
 *   - Smith & Gasser (2005) 具身认知发展
 *
 * 编码模态：
 *   1. Tactile (触觉) — 表面纹理、温度、压力
 *   2. Proprioception (本体感觉) — 身体位置、运动、力度
 *   3. Kinesthetic (动觉) — 运动感觉
 *   4. Vestibular (前庭) — 平衡感、加速度
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::perception {

/// 触觉数据类型
enum class HapticModality { kTactile, kProprioception, kKinesthetic, kVestibular };

/// 触觉信号
struct TactileSignal {
    double pressure = 0.0;        ///< 压力 0~1
    double temperature = 25.0;    ///< 温度 (Celsius)
    double texture_roughness = 0.0;  ///< 纹理粗糙度 0~1
    double vibration = 0.0;       ///< 振动 0~1
    std::string contact_point;    ///< 接触点
};

/// 本体感觉信号
struct ProprioceptiveSignal {
    std::string joint_name;       ///< 关节名称
    double angle = 0.0;           ///< 角度 (radians)
    double angular_velocity = 0.0; ///< 角速度
    double force = 0.0;           ///< 施加力 0~1
    double tension = 0.0;         ///< 肌肉张力 0~1
};

/// 前庭信号
struct VestibularSignal {
    double linear_accel_x = 0.0;   ///< 线性加速度 X
    double linear_accel_y = 0.0;   ///< 线性加速度 Y
    double linear_accel_z = 0.0;   ///< 线性加速度 Z
    double angular_vel_x = 0.0;    ///< 角速度 X
    double angular_vel_y = 0.0;    ///< 角速度 Y
    double angular_vel_z = 0.0;    ///< 角速度 Z
};

/// 触觉编码结果
struct HapticEncodeResult {
    std::vector<float> embedding;   ///< 语义嵌入
    HapticModality modality;        ///< 模态类型
    double intensity = 0.0;         ///< 信号强度
    bool success = false;
    std::string error;
    std::map<std::string, double> meta;
};

/// 触觉编码器接口
class IHapticEncoder {
public:
    virtual ~IHapticEncoder() = default;

    virtual auto encode_tactile(const TactileSignal& signal)
        -> HapticEncodeResult = 0;

    virtual auto encode_proprioceptive(const ProprioceptiveSignal& signal)
        -> HapticEncodeResult = 0;

    virtual auto encode_vestibular(const VestibularSignal& signal)
        -> HapticEncodeResult = 0;

    /// 融合多种触觉信号为统一嵌入
    virtual auto fuse(const std::vector<HapticEncodeResult>& signals)
        -> std::vector<float> = 0;

    [[nodiscard]] virtual auto embedding_dim() const -> int = 0;
    [[nodiscard]] virtual auto name() const -> std::string = 0;
};

/// Stub触觉编码器 — 开发测试用
class StubHapticEncoder : public IHapticEncoder {
public:
    explicit StubHapticEncoder(int dim = 64) : dim_(dim) {}

    auto encode_tactile(const TactileSignal& signal)
        -> HapticEncodeResult override {
        HapticEncodeResult result;
        result.modality = HapticModality::kTactile;
        result.intensity = (signal.pressure + signal.vibration + signal.texture_roughness) / 3.0;
        result.embedding = generate_stub_embedding_(signal.pressure, signal.texture_roughness);
        result.success = true;
        result.meta["temperature"] = signal.temperature;
        return result;
    }

    auto encode_proprioceptive(const ProprioceptiveSignal& signal)
        -> HapticEncodeResult override {
        HapticEncodeResult result;
        result.modality = HapticModality::kProprioception;
        result.intensity = signal.force;
        result.embedding = generate_stub_embedding_(signal.angle, signal.force);
        result.success = true;
        result.meta["joint"] = 0.0;  // simplified
        return result;
    }

    auto encode_vestibular(const VestibularSignal& signal)
        -> HapticEncodeResult override {
        HapticEncodeResult result;
        result.modality = HapticModality::kVestibular;
        double mag = std::sqrt(signal.linear_accel_x * signal.linear_accel_x
            + signal.linear_accel_y * signal.linear_accel_y
            + signal.linear_accel_z * signal.linear_accel_z);
        result.intensity = std::min(1.0, mag);
        result.embedding = generate_stub_embedding_(
            signal.linear_accel_x, signal.angular_vel_z);
        result.success = true;
        return result;
    }

    auto fuse(const std::vector<HapticEncodeResult>& signals)
        -> std::vector<float> override {
        std::vector<float> fused(dim_, 0.0f);
        if (signals.empty()) return fused;

        for (const auto& s : signals) {
            for (size_t i = 0; i < std::min(s.embedding.size(), fused.size()); ++i) {
                fused[i] += s.embedding[i] * s.intensity;
            }
        }
        // 归一化
        double norm = 0.0;
        for (auto v : fused) norm += v * v;
        if (norm > 1e-8) {
            for (auto& v : fused) v /= std::sqrt(norm);
        }
        return fused;
    }

    [[nodiscard]] auto embedding_dim() const -> int override { return dim_; }
    [[nodiscard]] auto name() const -> std::string override { return "stub_haptic_encoder"; }

private:
    int dim_;

    auto generate_stub_embedding_(double seed_a, double seed_b) const
        -> std::vector<float> {
        std::vector<float> emb(dim_);
        uint32_t seed = static_cast<uint32_t>(std::abs(seed_a) * 10000)
                      + static_cast<uint32_t>(std::abs(seed_b) * 10000) * 31;
        seed = seed ? seed : 1;
        for (int i = 0; i < dim_; ++i) {
            float x = static_cast<float>(seed + i * 733)
                      / static_cast<float>(UINT32_MAX);
            emb[i] = std::sin(x * 6.28318530718f);
        }
        return emb;
    }
};

}  // namespace ai_learning::perception