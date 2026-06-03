/**
 * @file multimodal_encoder.hpp
 * @brief 多模态感知编码器 — 将视觉/听觉/位置输入编码为统一表示
 *
 * 实现注意力加权融合：各模态可学习权重，投影到固定维度 obs_dim。
 */
#pragma once

#include "ai_learning/domain/perception/i_perception.hpp"

#include <map>
#include <string>
#include <vector>

namespace ai_learning::perception {

/// 多模态注意力编码器
class MultiModalEncoder : public domain::IPerception {
public:
    /// 构造，指定目标维度和模态配置
    /// @param obs_dim 输出向量维度
    /// @param modality_dims 各模态输入维度，如 {{"visual",256},{"auditory",64}}
    MultiModalEncoder(
        int obs_dim,
        const std::map<std::string, int>& modality_dims = {
            {"visual", 16}, {"auditory", 4}, {"position", 2}
        });

    /// 编码原始多模态输入为统一向量
    auto encode(const domain::RawInput& raw_input) const
        -> std::vector<float> override;

    /// 获取当前模态注意力权重
    [[nodiscard]] auto get_modality_weights() const
        -> std::map<std::string, double> override;

    /// 更新模态权重（学习过程中调用）
    void update_weight(const std::string& modality, double delta);

    /// 获取输出维度
    [[nodiscard]] auto obs_dim() const -> int { return obs_dim_; }

    /// 获取已注册的模态列表
    [[nodiscard]] auto modalities() const -> std::vector<std::string>;

private:
    int obs_dim_;
    std::map<std::string, int> modality_dims_;
    std::map<std::string, double> modality_weights_;
};

}  // namespace ai_learning::perception
