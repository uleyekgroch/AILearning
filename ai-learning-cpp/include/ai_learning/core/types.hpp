/**
 * @file types.hpp
 * @brief 核心类型别名 — 整个系统的公共类型定义
 *
 * 替代 Python 版的 torch.Tensor，使用 std::vector<float> 作为轻量张量。
 * 所有领域共享同一套类型，零 numpy/torch 暴露。
 */
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_learning::core {

// ── 张量替代 ────────────────────────────────────────────────────
/// 一维浮点张量（替代 torch::Tensor 的轻量实现）
using Tensor = std::vector<float>;

/// 二维矩阵 — 扁平行主序，内嵌形状信息
/// 用于消除 mat_vec/mat_vec_bias 传错 rows/cols 的风险
struct Matrix {
    std::vector<float> data;
    int rows = 0;
    int cols = 0;

    Matrix() = default;
    explicit Matrix(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}
    Matrix(std::vector<float> d, int r, int c)
        : data(std::move(d)), rows(r), cols(c) {}

    [[nodiscard]] auto operator()(int r, int c) const -> float {
        return data[r * cols + c];
    }
    auto operator()(int r, int c) -> float& {
        return data[r * cols + c];
    }

    [[nodiscard]] auto size() const -> size_t { return data.size(); }
    [[nodiscard]] auto empty() const -> bool { return data.empty(); }

    auto begin() -> auto { return data.begin(); }
    auto end() -> auto { return data.end(); }
    [[nodiscard]] auto begin() const -> auto { return data.begin(); }
    [[nodiscard]] auto end() const -> auto { return data.end(); }
};

/// 属性字典
using Properties = std::map<std::string, std::string>;

/// 元数据字典
using Metadata = std::unordered_map<std::string, std::string>;

/// 标签集合
using TagSet = std::unordered_set<std::string>;

/// 实体 ID 类型
using EntityId = std::string;

/// 关系 ID 类型
using RelationId = std::string;

/// 时间戳（毫秒 epoch）
using Timestamp = int64_t;

/// 统计字典
using Stats = std::map<std::string, double>;

}  // namespace ai_learning::core
