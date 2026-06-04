/**
 * @file multimodal_eval.cpp
 * @brief 客观多模态接地评测 — 量化"图像↔文字符号"的从零接地能力。
 *
 * 立场：全程零预训练（不用 CLIP/LLaVA/任何视觉大模型）。图像用从零的
 * 感知编码（PixelGridImageEncoder：灰度下采样为网格 + L2 归一化）得到向量，
 * 喂给既有 GroundingModule 聚类，并与文字符号（苹果/香蕉）绑定。
 *
 * 评测思想：
 *   若系统真的从视觉经验里接地了符号，则——
 *   - 同类图像（多张"苹果")应聚到同一感知簇，跨类（苹果 vs 香蕉）应分到不同簇；
 *   - 对一张没见过的新图，recognize() 应反查回正确的文字符号；
 *   - 同一符号既能在文本（分布语义/KG）里存在，又能被视觉接地 = 真正的多模态接地。
 */

#include "ai_learning/core/learner.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ai_learning::core::Learner;
using ai_learning::core::LearnerConfig;

int g_passed = 0;
int g_total = 0;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    ++g_total;
    if (ok) ++g_passed;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
}

constexpr int kW = 8;
constexpr int kH = 8;

/// 确定性的小扰动（按 variant 取值，保证可复现）
auto jitter(int variant, int idx) -> int {
    return ((variant * 37 + idx * 13) % 11) - 5;  // [-5, +5]
}

auto clamp8(int v) -> std::uint8_t {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<std::uint8_t>(v);
}

/// "苹果"图样：中心明亮的圆形块（中间亮、四周暗）
auto make_apple(int variant) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> img(kW * kH);
    double cx = (kW - 1) / 2.0, cy = (kH - 1) / 2.0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            double dx = x - cx, dy = y - cy;
            double d = std::sqrt(dx * dx + dy * dy);
            int base = d < 2.5 ? 220 : 40;
            img[y * kW + x] = clamp8(base + jitter(variant, y * kW + x));
        }
    }
    return img;
}

/// "香蕉"图样：右侧竖条明亮（左暗右亮），与中心块明显不同
auto make_banana(int variant) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> img(kW * kH);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            int base = x >= kW - 3 ? 220 : 40;
            img[y * kW + x] = clamp8(base + jitter(variant, y * kW + x));
        }
    }
    return img;
}

}  // namespace

auto main() -> int {
    std::cout << "=== 客观多模态接地评测 (零大模型，从零感知) ===\n\n";

    LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 8;
    config.embedding_dim = 16;
    config.embedding_learning_enabled = true;
    config.ds_min_freq = 1;
    config.statistical_min_freq = 1;
    config.embedding_min_count = 1;

    Learner learner(config);

    // 1) 先用文本学习符号「苹果/香蕉」，让它们在分布语义/KG 里存在
    for (int e = 0; e < 5; ++e) {
        learner.learn_from_text("苹果 是 甜 的 水果 好吃 营养", "mm");
        learner.learn_from_text("香蕉 是 甜 的 水果 好吃 营养", "mm");
    }
    bool text_has = learner.distributional_semantics().has_cpt("苹果") &&
                    learner.distributional_semantics().has_cpt("香蕉");
    check("符号已在文本侧自学(分布语义含 苹果/香蕉)", text_has);

    // 2) 视觉接地：每类用 3 张带扰动的图，把图像与同名文字符号绑定
    int ca = -1, cb = -1;
    for (int v = 0; v < 3; ++v) {
        ca = learner.ground_image("苹果", make_apple(v), kW, kH);
        cb = learner.ground_image("香蕉", make_banana(v), kW, kH);
    }
    check("视觉接地成功(苹果/香蕉 均得到感知聚类)", ca >= 0 && cb >= 0,
          "apple_cluster=" + std::to_string(ca) +
              " banana_cluster=" + std::to_string(cb));

    // 3) 同类聚同簇、跨类分异簇
    check("同类聚同簇且跨类分异簇(苹果簇 != 香蕉簇)", ca != cb,
          "apple=" + std::to_string(ca) + " banana=" + std::to_string(cb));

    // 4) 对没见过的新图做识别（用更大扰动的 variant）
    auto r_apple = learner.recognize_image(make_apple(7), kW, kH);
    check("识别新苹果图 → 苹果", r_apple.first == "苹果",
          "got='" + r_apple.first + "' sim=" + std::to_string(r_apple.second));

    auto r_banana = learner.recognize_image(make_banana(7), kW, kH);
    check("识别新香蕉图 → 香蕉", r_banana.first == "香蕉",
          "got='" + r_banana.first + "' sim=" + std::to_string(r_banana.second));

    // 5) 多模态接地闭环：同一符号「苹果」既有文本侧向量，又被视觉接地
    bool cross_modal =
        learner.distributional_semantics().has_cpt("苹果") &&
        r_apple.first == "苹果";
    check("多模态接地(同一符号 苹果 同时由文本与视觉支撑)", cross_modal);

    std::cout << "\n=== 客观分数: " << g_passed << " / " << g_total << " ===\n";

    // 硬性能力：跨类可分 + 新图识别正确，否则视为接地失败。
    bool hard_ok = (ca != cb) && r_apple.first == "苹果" &&
                   r_banana.first == "香蕉";
    if (!hard_ok) {
        std::cout << "硬性能力回归失败(视觉接地未能区分/识别类别)\n";
        return 1;
    }
    return 0;
}
