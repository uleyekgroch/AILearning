/**
 * @file i_perception.hpp
 * @brief 感知领域接口 — 将原始多模态输入编码为内部表示
 *
 * DDD 限界上下文：感知
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain {

/// 多模态原始输入
using RawInput = std::map<std::string, std::vector<float>>;

class IPerception {
public:
    virtual ~IPerception() = default;

    /// 将原始输入（视觉/听觉/位置）编码为统一内部表示向量
    virtual auto encode(const RawInput& raw_input) const
        -> std::vector<float> = 0;

    /// 获取各模态的注意力权重
    virtual auto get_modality_weights() const
        -> std::map<std::string, double> = 0;
};

}  // namespace ai_learning::domain
