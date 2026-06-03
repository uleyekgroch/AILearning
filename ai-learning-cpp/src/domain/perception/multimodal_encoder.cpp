/**
 * @file multimodal_encoder.cpp
 * @brief 多模态感知编码器实现
 */

#include "ai_learning/domain/perception/multimodal_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ai_learning::perception {

MultiModalEncoder::MultiModalEncoder(
    int obs_dim,
    const std::map<std::string, int>& modality_dims)
    : obs_dim_(obs_dim), modality_dims_(modality_dims) {

    // 初始化等权重
    double init_w = 1.0 / static_cast<double>(modality_dims_.size());
    for (const auto& [name, _] : modality_dims_) {
        modality_weights_[name] = init_w;
    }
}

auto MultiModalEncoder::encode(const domain::RawInput& raw_input) const
    -> std::vector<float> {
    // 1. 对每个模态：截断/填充到目标维度比例
    // 2. 乘以注意力权重
    // 3. 拼接后投影到 obs_dim

    std::vector<float> encoded;
    encoded.reserve(obs_dim_);

    for (const auto& [modality, dim] : modality_dims_) {
        float weight = 1.0f;
        if (modality_weights_.contains(modality)) {
            weight = static_cast<float>(modality_weights_.at(modality));
        }

        // 每个模态分配的维度比例
        int alloc = obs_dim_ / static_cast<int>(modality_dims_.size());

        if (raw_input.contains(modality)) {
            const auto& data = raw_input.at(modality);
            // 加权 + 截断/填充
            for (int i = 0; i < alloc; ++i) {
                float val = (i < static_cast<int>(data.size()))
                    ? data[i] * weight
                    : 0.0f;
                encoded.push_back(val);
            }
        } else {
            // 模态缺失，零填充
            for (int i = 0; i < alloc; ++i) {
                encoded.push_back(0.0f);
            }
        }
    }

    // 精确填充到 obs_dim_
    encoded.resize(obs_dim_, 0.0f);

    // L2 归一化
    float norm = 0.0f;
    for (auto v : encoded) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (auto& v : encoded) v /= norm;
    }

    return encoded;
}

auto MultiModalEncoder::get_modality_weights() const
    -> std::map<std::string, double> {
    return modality_weights_;
}

void MultiModalEncoder::update_weight(const std::string& modality,
                                       double delta) {
    if (!modality_weights_.contains(modality)) return;
    modality_weights_[modality] = std::max(0.01, modality_weights_[modality] + delta);
}

auto MultiModalEncoder::modalities() const -> std::vector<std::string> {
    std::vector<std::string> names;
    for (const auto& [name, _] : modality_dims_) {
        names.push_back(name);
    }
    return names;
}

}  // namespace ai_learning::perception
