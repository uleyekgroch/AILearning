/**
 * @file pixel_image_encoder.hpp
 * @brief 从零的「感知式」图像编码器（零预训练、纯 CPU）
 *
 * 立场：与项目「拒绝大模型」一致——本编码器**不加载任何预训练权重**
 * （区别于可选的 CLIP/LLaVA）。它直接从原始灰度像素提取**真实的感知结构**：
 * 把图像下采样为 GxG 网格、取每格平均强度并做 L2 归一化。
 *
 * 与 StubImageEncoder（对字节做 hash → 伪随机向量、无感知结构）不同：
 * 视觉上相近的图像 → 网格强度相近 → 向量欧氏距离近，因此可被 GroundingModule
 * 聚成同一概念。这让符号（如「苹果」）能从**视觉经验**里获得接地，而非靠预训练。
 *
 * 输入约定：image_data 为 width*height 的 8 位灰度像素（行优先）。
 * 这是最小可行的「文本/图片」接地通路；后期可扩展彩色直方图、边缘、真实可交互环境。
 */

#pragma once

#include "ai_learning/perception/image_encoder.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ai_learning::perception {

/// 基于像素网格的从零感知编码器（实现 IImageEncoder，无外部依赖、无预训练）
class PixelGridImageEncoder final : public IImageEncoder {
public:
    /// @param grid 网格边长 G（输出维度 = G*G，默认 4 → 16 维，对齐 obs_dim=16）
    explicit PixelGridImageEncoder(int grid = 4) : grid_(grid > 0 ? grid : 4) {}

    auto encode(const std::vector<uint8_t>& image_data, int width, int height)
        -> ImageEncodeResult override {
        ImageEncodeResult result;
        result.width = width;
        result.height = height;
        result.format = "gray8";

        const int dim = grid_ * grid_;
        result.embedding.assign(dim, 0.0F);

        // 仅接受 width*height == 灰度像素数 的原始灰度图。
        if (width <= 0 || height <= 0 ||
            image_data.size() != static_cast<size_t>(width) *
                                     static_cast<size_t>(height)) {
            result.success = false;
            result.error = "expect raw gray8 of size width*height";
            return result;
        }

        // 1. 下采样为 GxG 网格，取每格平均强度（归一化到 [0,1]）
        std::vector<double> sum(dim, 0.0);
        std::vector<long> cnt(dim, 0);
        for (int y = 0; y < height; ++y) {
            int gy = std::min(grid_ - 1, y * grid_ / height);
            for (int x = 0; x < width; ++x) {
                int gx = std::min(grid_ - 1, x * grid_ / width);
                int cell = gy * grid_ + gx;
                sum[cell] += image_data[static_cast<size_t>(y) * width + x];
                cnt[cell] += 1;
            }
        }
        for (int i = 0; i < dim; ++i) {
            double mean = cnt[i] > 0 ? sum[i] / static_cast<double>(cnt[i]) : 0.0;
            result.embedding[i] = static_cast<float>(mean / 255.0);
        }

        // 2. L2 归一化：让"图案/明暗分布"决定距离，弱化整体亮度/尺度差异
        double norm = 0.0;
        for (float v : result.embedding) norm += static_cast<double>(v) * v;
        norm = std::sqrt(norm);
        if (norm > 1e-8) {
            for (float& v : result.embedding) {
                v = static_cast<float>(v / norm);
            }
        }

        result.success = true;
        return result;
    }

    [[nodiscard]] auto embedding_dim() const -> int override {
        return grid_ * grid_;
    }
    [[nodiscard]] auto name() const -> std::string override {
        return "pixel_grid_image_encoder";
    }

private:
    int grid_;
};

}  // namespace ai_learning::perception
