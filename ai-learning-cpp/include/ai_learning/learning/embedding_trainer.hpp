/**
 * @file embedding_trainer.hpp
 * @brief 稠密 Embedding 训练器 — 从零训练，CPU + CUDA 双路径
 *
 * 训练方法：Skip-gram with Negative Sampling (SGNS)
 *   目标：给定中心词 w，最大化上下文词 c 的概率
 *   loss = -log σ(v_c · v_w) - Σ_k log σ(-v_k · v_w)
 *
 * CUDA 加速点：
 *   - 批量点积计算（正/负样本评分）
 *   - 批量 SGD 向量更新
 *   - 大规模并行负采样
 *
 * 输出：每个概念 → 128维稠密 float 向量
 *   可直接喂入 PredictiveCodingEngine.learn(obs, action, actual)
 */
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 训练配置
struct EmbeddingTrainerConfig {
    int embedding_dim = 128;        ///< 嵌入维度（与 PC 引擎 obs_dim 对齐）
    int window_size = 5;            ///< 上下文窗口大小
    int neg_samples = 5;            ///< 每个正样本的负采样数
    double learning_rate = 0.025;   ///< 初始学习率
    double min_learning_rate = 0.001; ///< 最低学习率
    int epochs = 5;                 ///< 训练轮次
    int min_count = 2;              ///< 最低词频
    int batch_size = 512;           ///< 批大小（CUDA 用）
    int max_vocab = 50000;          ///< 最大词表大小
    int seed = 42;                  ///< 随机种子
    int report_interval = 10000;    ///< 训练进度报告间隔
};

/// 训练进度
struct TrainingProgress {
    int epoch = 0;
    int total_epochs = 0;
    long long pairs_processed = 0;
    long long total_pairs = 0;
    double loss = 0.0;
    double current_lr = 0.0;
    int vocab_size = 0;
    bool use_cuda = false;
    double elapsed_seconds = 0.0;
};

/// 训练结果
struct TrainingResult {
    int vocab_size = 0;
    int embedding_dim = 0;
    long long total_pairs = 0;
    double final_loss = 0.0;
    double elapsed_seconds = 0.0;
    bool used_cuda = false;
};

/// Embedding 查询结果
struct EmbeddingQuery {
    std::string word;
    std::vector<float> embedding;
    float similarity = 0.0f;
};

/// 稠密 Embedding 训练器
class EmbeddingTrainer {
public:
    explicit EmbeddingTrainer(
        const EmbeddingTrainerConfig& config = EmbeddingTrainerConfig{});

    // ── 语料构建 ──────────────────────────────────────

    /// 添加一条训练文本
    void add_text(const std::string& text);

    /// 添加预分词的 tokens（跳过内部分词）
    void add_tokens(const std::vector<std::string>& tokens);

    /// 批量添加文本
    void add_corpus(const std::vector<std::string>& texts);

    /// 清空语料
    void clear();

    // ── 训练 ────────────────────────────────────────

    /// 训练（自动选择 CPU 或 CUDA）
    auto train() -> TrainingResult;

    /// 设置进度回调
    using ProgressCallback = std::function<void(const TrainingProgress&)>;
    void set_progress_callback(ProgressCallback cb);

    // ── 查询 ────────────────────────────────────────

    /// 获取词向量
    auto get_embedding(const std::string& word) const
        -> std::optional<std::vector<float>>;

    /// 最相似词
    auto most_similar(const std::string& word, int top_k = 10) const
        -> std::vector<EmbeddingQuery>;

    /// 类比推理：A 之于 B ≈ C 之于 ?
    auto analogy(const std::string& a, const std::string& b,
                 const std::string& c, int top_k = 5) const
        -> std::vector<EmbeddingQuery>;

    /// 词是否存在
    [[nodiscard]] auto has_word(const std::string& word) const -> bool;

    /// 获取所有词
    [[nodiscard]] auto vocabulary() const -> std::vector<std::string>;

    /// 词表大小
    [[nodiscard]] auto vocab_size() const -> int;

    /// 是否已训练
    [[nodiscard]] auto is_trained() const -> bool;

    // ── 导出 ────────────────────────────────────────

    /// 导出整个词向量矩阵
    auto export_embeddings() const
        -> std::pair<std::vector<std::string>, std::vector<float>>;

    /// 从 DistributionalSemantics 导入概念
    void import_vocabulary(const std::vector<std::string>& words);

    // ── 序列化 ──────────────────────────────────────

    auto save(const std::string& path) const -> bool;
    auto load(const std::string& path) -> bool;

    [[nodiscard]] auto config() const -> const EmbeddingTrainerConfig& {
        return config_;
    }

private:
    EmbeddingTrainerConfig config_;

    // ── 词表 ──────────────────────────────────────
    std::map<std::string, int> word2idx_;
    std::vector<std::string> idx2word_;
    std::vector<int> word_freq_;
    int total_words_ = 0;

    // ── 原始语料缓冲 ──────────────────────────────
    std::vector<std::vector<std::string>> raw_docs_;
    std::map<std::string, int> raw_freq_;

    // ── 训练数据 ──────────────────────────────────
    std::vector<int> corpus_;           ///< 训练语料（token index 序列）
    std::vector<int> neg_table_;        ///< 负采样表
    static constexpr int NEG_TABLE_SIZE = 100000;

    // ── 词向量 ────────────────────────────────────
    std::vector<float> W_in_;           ///< 中心词向量 (vocab × dim)
    std::vector<float> W_out_;          ///< 上下文词向量 (vocab × dim)

    bool trained_ = false;
    std::mt19937 rng_;
    ProgressCallback progress_cb_;

    // ── 内部方法 ──────────────────────────────────

    void build_vocabulary_();
    void build_neg_table_();
    void init_embeddings_();
    void prepare_corpus_();

    static auto sigmoid_(float x) -> float;

    /// CPU 训练一个 epoch
    auto train_epoch_cpu_(int epoch, double base_lr, double& total_loss) -> long long;

    /// 中文分词（与 DistributionalSemantics 相同逻辑）
    auto segment_(const std::string& text) const -> std::vector<std::string>;

    /// 停用词检查
    [[nodiscard]] auto is_stopword_(const std::string& token) const -> bool;
};

}  // namespace ai_learning::learning
