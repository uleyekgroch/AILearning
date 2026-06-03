#pragma once
/**
 * @file grounding.hpp
 * @brief 符号接地模块 — 从感知到抽象的渐进过程
 *
 * 三层接地：
 * 1. 感知聚类 → 原始概念
 * 2. 社会标注 → 语言符号
 * 3. 概念查询 → 相似概念检索
 */

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::language {

/// 感知聚类：将相似感知经验归类为概念
struct PerceptualCluster {
    std::vector<float> centroid;  ///< 聚类中心向量
    int count = 0;                ///< 聚类中的样本数
};

/// 符号映射：符号 → 概念的对应关系
struct SymbolMapping {
    std::vector<int> referent_clusters;  ///< 关联的聚类 ID
    float confidence = 1.0F;             ///< 接地置信度
    std::vector<std::string> contexts;   ///< 使用上下文
    int usage_count = 1;                 ///< 使用次数
};

/// 相似概念结果
struct SimilarConcept {
    int cluster_id;
    float similarity;  ///< 0~1，越大越相似
};

class GroundingModule {
public:
    explicit GroundingModule(int obs_dim = 16);

    /// 从感知经验建立概念，返回聚类 ID
    [[nodiscard]] auto ground_from_perception(
        const std::vector<float>& observation) -> int;

    /// 从社会交互接地符号
    void ground_from_social(const std::string& symbol,
                            const std::vector<float>& referent,
                            const std::string& context = "");

    /// 获取符号含义
    [[nodiscard]] auto get_symbol_meaning(
        const std::string& symbol) const -> std::optional<SymbolMapping>;

    /// 获取所有已接地的符号
    [[nodiscard]] auto get_grounded_symbols() const -> std::vector<std::string>;

    /// 找到与给定观测最相似的概念
    [[nodiscard]] auto find_similar_concepts(
        const std::vector<float>& observation,
        int top_k = 3) const -> std::vector<SimilarConcept>;

    /// 获取感知聚类数
    [[nodiscard]] auto cluster_count() const -> int {
        return static_cast<int>(clusters_.size());
    }

    /// 获取符号数
    [[nodiscard]] auto symbol_count() const -> int {
        return static_cast<int>(symbol_mappings_.size());
    }

private:
    int obs_dim_;
    std::map<int, PerceptualCluster> clusters_;
    std::map<std::string, SymbolMapping> symbol_mappings_;
    int cluster_counter_ = 0;

    /// 计算两个向量的欧氏距离
    [[nodiscard]] static auto euclidean_distance(
        const std::vector<float>& a,
        const std::vector<float>& b) -> float;

    /// 更新聚类中心（增量均值）
    static void update_centroid(std::vector<float>& centroid,
                                const std::vector<float>& new_obs,
                                int new_count);
};

}  // namespace ai_learning::language
