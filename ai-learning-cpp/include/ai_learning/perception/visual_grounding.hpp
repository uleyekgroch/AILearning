/**
 * @file visual_grounding.hpp
 * @brief 「图像 ↔ 文字符号」接地系统（零预训练）
 *
 * 把从零感知编码器（PixelGridImageEncoder）与既有的 GroundingModule 打通：
 *   learn_symbol(符号, 图像) → 编码图像 → 在感知空间聚类 → 把符号绑定到该聚类
 *   recognize(图像)        → 编码图像 → 找最近聚类 → 反查绑定到该聚类的符号
 *
 * 这条通路让「苹果」这类文字符号能从**视觉经验**获得接地，而不是依赖预训练视觉
 * 大模型。它复用 Learner 已持有的 GroundingModule（此前从未被使用——报告 3.3/6.3
 * 指出的「陈列品」问题），使该模块真正进入闭环。
 */

#pragma once

#include "ai_learning/language/grounding.hpp"
#include "ai_learning/perception/pixel_image_encoder.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ai_learning::perception {

class VisualGroundingSystem {
public:
    /// @param grounding 外部注入的接地模块（与 Learner 共享，统一感知空间）
    /// @param grid      像素网格边长，输出维度 = grid*grid
    explicit VisualGroundingSystem(language::GroundingModule& grounding,
                                   int grid = 4)
        : grounding_(grounding), encoder_(grid) {}

    /// 学习：把一张灰度图与文字符号绑定，返回其感知聚类 ID（失败返回 -1）
    auto learn_symbol(const std::string& symbol,
                      const std::vector<uint8_t>& gray, int w, int h) -> int {
        auto enc = encoder_.encode(gray, w, h);
        if (!enc.success) return -1;
        grounding_.ground_from_social(symbol, enc.embedding, "vision");
        return grounding_.ground_from_perception(enc.embedding);
    }

    /// 识别：给一张新图，返回最匹配的已学符号及相似度（无匹配返回 {"", 0}）
    /// 注：图像编码器接口 encode() 非 const，故此方法非 const（不修改接地状态）。
    [[nodiscard]] auto recognize(const std::vector<uint8_t>& gray, int w, int h)
        -> std::pair<std::string, float> {
        auto enc = encoder_.encode(gray, w, h);
        if (!enc.success) return {"", 0.0F};

        auto sims = grounding_.find_similar_concepts(enc.embedding, 5);
        // 从最近聚类向外，找第一个被某符号绑定的聚类
        for (const auto& sc : sims) {
            for (const auto& sym : grounding_.get_grounded_symbols()) {
                auto mapping = grounding_.get_symbol_meaning(sym);
                if (!mapping) continue;
                for (int cid : mapping->referent_clusters) {
                    if (cid == sc.cluster_id) {
                        return {sym, sc.similarity};
                    }
                }
            }
        }
        return {"", 0.0F};
    }

    [[nodiscard]] auto embedding_dim() const -> int {
        return encoder_.embedding_dim();
    }

private:
    language::GroundingModule& grounding_;
    PixelGridImageEncoder encoder_;
};

}  // namespace ai_learning::perception
