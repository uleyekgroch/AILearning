/**
 * @file distributional_semantics.hpp
 * @brief 分布语义引擎 — 从语料中自主涌现语义理解
 *
 * 核心原则：
 *   "You shall know a word by the company it keeps" — Firth, 1957
 *
 * 不使用任何外部 AI 模型。完全依靠：
 *   1. 共现统计 → 概念的分布表示（类似 GloVe）
 *   2. 余弦相似度 → 语义相似度
 *   3. PPMI 矩阵 → 正点互信息（比原始共现更好的语义信号）
 *   4. 因果结构 → WorldModel 的层次/因果约束
 *   5. 预测编码 → 概念表示的持续精化
 *
 * 与 Word2Vec/GloVe 的区别：
 *   - 它们是离线训练的静态模型
 *   - 我们是增量学习的动态模型（每读一条新文本就更新）
 *   - 它们的目标函数是预测/重建
 *   - 我们的目标函数是预测编码误差最小化（更接近人脑）
 *
 * 不使用 LLM embedding = 不借用任何外部知识 = 真正的自主学习
 */
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 概念的分布表示（稀疏向量）
struct ConceptVector {
    std::string name;                           ///< 概念名称
    std::map<std::string, double> dimensions;   ///< 维度:上下文概念 → PPMI权重
    int observation_count = 0;                  ///< 观察次数
    double total_weight = 0.0;                  ///< 总权重

    /// 向量模长
    [[nodiscard]] auto norm() const -> double;

    /// 稠密化：转为固定维度的浮点向量（取 top-N 维度）
    auto to_dense(int max_dims = 128) const -> std::vector<float>;
};

/// 语义相似度结果
struct SimilarityResult {
    std::string cpt_a;
    std::string cpt_b;
    double similarity = 0.0;       ///< 余弦相似度 -1~1
    double pmi = 0.0;              ///< 点互信息
    std::string explanation;       ///< 为什么相似
};

/// 类比结果
struct AnalogyResult {
    std::string a_is_to_b;
    std::string as_c_is_to_d;
    double confidence = 0.0;
    std::string reasoning;
};

/// 语义聚类
struct SemanticCluster {
    std::string label;             ///< 聚类标签（最有代表性的概念）
    std::vector<std::string> members;
    double cohesion = 0.0;         ///< 聚集度
    std::vector<std::string> common_contexts;  ///< 共同上下文
};

/// 语义学习统计
struct SemanticStats {
    int cpts_represented = 0;
    int total_dimensions = 0;
    int texts_processed = 0;
    double avg_vector_density = 0.0;  ///< 平均非零维度数
    int clusters_found = 0;
};

/// 分布语义引擎配置
struct DistributionalSemanticsConfig {
    int window_size = 5;               ///< 上下文窗口大小
    int min_cpt_freq = 3;              ///< 概念最低频率
    double ppmi_threshold = 0.5;       ///< PPMI 阈值（低于此值的维度被裁剪）
    int max_dimensions = 500;          ///< 每个概念的最大维度数
    int max_cpts = 10000;              ///< 最大概念数
    double similarity_threshold = 0.3; ///< 相似度阈值
    bool use_causal_prior = true;      ///< 是否利用 WorldModel 因果先验
};

/// 分布语义引擎
class DistributionalSemantics {
public:
    explicit DistributionalSemantics(
        const DistributionalSemanticsConfig& config = DistributionalSemanticsConfig{});

    // ── 语料摄入（增量学习） ────────────────────────────

    /// 从一段文本中学习（分词后增量更新分布表示）
    auto learn_from_tokens(const std::vector<std::string>& tokens)
        -> std::vector<std::string>;  ///< 返回新增的涌现概念

    /// 从原始文本学习（内置中文分词）
    auto learn_from_text(const std::string& text)
        -> std::vector<std::string>;

    /// 批量学习
    auto learn_batch(const std::vector<std::string>& texts)
        -> int;  ///< 返回总新增概念数

    // ── 语义查询 ──────────────────────────────────────

    /// 计算两个概念的语义相似度（余弦相似度）
    auto similarity(const std::string& cpt_a,
                    const std::string& cpt_b) const
        -> SimilarityResult;

    /// 找到最相似的概念
    auto most_similar(const std::string& cpt,
                      int top_k = 10) const
        -> std::vector<SimilarityResult>;

    /// 语义类比：A 之于 B，如同 C 之于 ?
    auto analogy(const std::string& a, const std::string& b,
                 const std::string& c, int top_k = 5) const
        -> std::vector<AnalogyResult>;

    /// 获取概念的所有语义邻居
    auto semantic_neighborhood(const std::string& cpt,
                                double threshold = 0.3) const
        -> std::vector<SimilarityResult>;

    // ── 概念表示 ──────────────────────────────────────

    /// 获取概念的分布向量
    auto get_vector(const std::string& cpt) const
        -> std::optional<ConceptVector>;

    /// 获取概念的稠密向量表示
    auto get_dense_vector(const std::string& cpt,
                          int dims = 128) const
        -> std::optional<std::vector<float>>;

    /// 概念是否存在
    [[nodiscard]] auto has_cpt(const std::string& cpt) const
        -> bool;

    // ── 语义聚类 ──────────────────────────────────────

    /// 发现语义聚类（基于连通分量 + 密度）
    auto discover_clusters(double threshold = 0.4) const
        -> std::vector<SemanticCluster>;

    // ── 因果先验（与 WorldModel 协同） ────────────────

    /// 注册因果知识（来自 WorldModel）
    void register_causal_link(const std::string& cause,
                               const std::string& effect,
                               double strength);

    /// 获取因果相关的概念
    auto causally_related(const std::string& cpt) const
        -> std::vector<std::pair<std::string, double>>;

    // ── 查询 ──────────────────────────────────────────

    /// 获取所有概念名
    [[nodiscard]] auto all_cpts() const -> std::vector<std::string>;

    /// 获取统计信息
    [[nodiscard]] auto stats() const -> SemanticStats;

    /// 获取配置
    [[nodiscard]] auto config() const -> const DistributionalSemanticsConfig& {
        return config_;
    }

private:
    DistributionalSemanticsConfig config_;

    /// 核心数据结构：概念 → 分布向量
    std::map<std::string, ConceptVector> vectors_;

    /// 共现计数（symmetric: A→B count = B→A count）
    std::map<std::string, std::map<std::string, int>> cooccurrence_;

    /// 概念频率
    std::map<std::string, int> cpt_freq_;

    /// 因果先验（来自 WorldModel）
    std::map<std::string, std::map<std::string, double>> causal_prior_;

    /// 全局维度表（所有出现过的上下文概念）
    std::set<std::string> all_dimensions_;

    /// 统计
    int texts_processed_ = 0;
    int total_tokens_ = 0;

    // ── 内部方法 ──────────────────────────────────────

    /// 中文分词（单字切分 + 停用词过滤）
    auto segment_(const std::string& text) const
        -> std::vector<std::string>;

    /// 更新共现矩阵
    void update_cooccurrence_(const std::vector<std::string>& tokens);

    /// 从共现矩阵计算 PPMI 并更新向量
    void rebuild_vectors_();

    /// 计算 PPMI（正点互信息）
    auto compute_ppmi_(const std::string& target,
                       const std::string& context) const
        -> double;

    /// 余弦相似度
    static auto cosine_similarity_(const ConceptVector& a,
                                    const ConceptVector& b)
        -> double;

    /// 停用词检查
    [[nodiscard]] auto is_stopword_(const std::string& token) const -> bool;
};

}  // namespace ai_learning::learning
