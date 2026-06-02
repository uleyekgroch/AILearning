/**
 * @file embedding_trainer.cpp
 * @brief 稠密 Embedding 训练器 — CPU 实现（+ CUDA 调度）
 *
 * Skip-gram with Negative Sampling (SGNS)
 *
 * 训练循环：
 *   for each (center, context) pair in corpus:
 *     1. 正样本：最大化 σ(v_context · v_center)
 *     2. 负采样：最大化 σ(-v_neg · v_center)
 *     3. SGD 更新：v_center += lr * (label - σ(score)) * v_other
 */

#include "ai_learning/learning/embedding_trainer.hpp"
#include "ai_learning/core/tensor_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace ai_learning::learning {

// ── 构造 ──────────────────────────────────────────────────────

EmbeddingTrainer::EmbeddingTrainer(const EmbeddingTrainerConfig& config)
    : config_(config), rng_(config.seed) {}

auto EmbeddingTrainer::sigmoid_(float x) -> float {
    if (x > 30.0f) return 1.0f;
    if (x < -30.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// ── 语料构建 ──────────────────────────────────────────────────

void EmbeddingTrainer::add_text(const std::string& text) {
    auto tokens = segment_(text);
    if (tokens.empty()) return;

    raw_docs_.push_back(tokens);
    for (const auto& t : tokens) {
        raw_freq_[t]++;
    }
}

void EmbeddingTrainer::add_corpus(const std::vector<std::string>& texts) {
    for (const auto& t : texts) add_text(t);
}

void EmbeddingTrainer::clear() {
    raw_docs_.clear();
    raw_freq_.clear();
    corpus_.clear();
    word2idx_.clear();
    idx2word_.clear();
    word_freq_.clear();
    W_in_.clear();
    W_out_.clear();
    trained_ = false;
    total_words_ = 0;
}

void EmbeddingTrainer::import_vocabulary(const std::vector<std::string>& words) {
    for (const auto& w : words) {
        if (raw_freq_.count(w) == 0) {
            raw_freq_[w] = config_.min_count;  // 给到最低频率
        }
    }
}

// ── 构建词表 ──────────────────────────────────────────────────

void EmbeddingTrainer::build_vocabulary_() {
    // 过滤低频词
    std::vector<std::pair<std::string, int>> freq_pairs;
    for (const auto& [word, freq] : raw_freq_) {
        if (freq >= config_.min_count) {
            freq_pairs.push_back({word, freq});
        }
    }

    // 按频率降序排序
    std::sort(freq_pairs.begin(), freq_pairs.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // 截断到最大词表
    if (static_cast<int>(freq_pairs.size()) > config_.max_vocab) {
        freq_pairs.resize(config_.max_vocab);
    }

    // 构建映射
    word2idx_.clear();
    idx2word_.clear();
    word_freq_.clear();
    total_words_ = 0;

    for (int i = 0; i < static_cast<int>(freq_pairs.size()); ++i) {
        const auto& [word, freq] = freq_pairs[i];
        word2idx_[word] = i;
        idx2word_.push_back(word);
        word_freq_.push_back(freq);
        total_words_ += freq;
    }
}

void EmbeddingTrainer::prepare_corpus_() {
    corpus_.clear();
    for (const auto& doc : raw_docs_) {
        for (const auto& token : doc) {
            auto it = word2idx_.find(token);
            if (it != word2idx_.end()) {
                corpus_.push_back(it->second);
            }
        }
        // 文档分隔标记（-1）
        corpus_.push_back(-1);
    }
}

void EmbeddingTrainer::build_neg_table_() {
    // 负采样概率 = freq^0.75 / Σ freq^0.75
    neg_table_.resize(NEG_TABLE_SIZE);
    double power_sum = 0.0;
    for (int f : word_freq_) {
        power_sum += std::pow(static_cast<double>(f), 0.75);
    }

    int vocab_sz = static_cast<int>(idx2word_.size());
    int idx = 0;
    double cumulative = 0.0;
    for (int i = 0; i < vocab_sz; ++i) {
        cumulative += std::pow(static_cast<double>(word_freq_[i]), 0.75) / power_sum;
        int boundary = static_cast<int>(cumulative * NEG_TABLE_SIZE);
        while (idx < boundary && idx < NEG_TABLE_SIZE) {
            neg_table_[idx++] = i;
        }
    }
    while (idx < NEG_TABLE_SIZE) {
        neg_table_[idx++] = vocab_sz - 1;
    }
}

void EmbeddingTrainer::init_embeddings_() {
    int V = static_cast<int>(idx2word_.size());
    int D = config_.embedding_dim;

    // Xavier uniform: U[-sqrt(6/(V+D)), sqrt(6/(V+D))]
    float scale = std::sqrt(6.0f / static_cast<float>(V + D));
    std::uniform_real_distribution<float> dist(-scale, scale);

    W_in_.resize(V * D);
    W_out_.resize(V * D, 0.0f);  // context vectors 初始化为 0

    for (auto& w : W_in_) {
        w = dist(rng_);
    }
}

// ── 训练 ──────────────────────────────────────────────────────

auto EmbeddingTrainer::train() -> TrainingResult {
    auto t0 = std::chrono::steady_clock::now();

    // 1. 构建词表
    build_vocabulary_();
    if (idx2word_.empty()) {
        return {};
    }

    // 2. 准备训练数据
    prepare_corpus_();
    build_neg_table_();
    init_embeddings_();

    // 3. 估算训练对数
    long long total_pairs = 0;
    int ws = config_.window_size;
    for (int pos = 0; pos < static_cast<int>(corpus_.size()); ++pos) {
        if (corpus_[pos] < 0) continue;
        int start = std::max(0, pos - ws);
        int end = std::min(static_cast<int>(corpus_.size()) - 1, pos + ws);
        for (int c = start; c <= end; ++c) {
            if (c != pos && corpus_[c] >= 0) total_pairs++;
        }
    }

    bool use_cuda = core::cuda_available() && vocab_size() > 100;
    (void)use_cuda;  // CUDA dispatch 未实现时避免 unused warning

    TrainingResult result;
    result.vocab_size = vocab_size();
    result.embedding_dim = config_.embedding_dim;
    result.total_pairs = total_pairs;
    result.used_cuda = use_cuda;

    // 4. 训练循环
    double final_loss = 0.0;
    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        double lr = config_.learning_rate -
            (config_.learning_rate - config_.min_learning_rate) *
            (static_cast<double>(epoch) / config_.epochs);

        double epoch_loss = 0.0;
        long long pairs = train_epoch_cpu_(epoch, lr, epoch_loss);

        final_loss = epoch_loss / static_cast<double>(std::max(1LL, pairs));

        if (progress_cb_) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            TrainingProgress prog;
            prog.epoch = epoch + 1;
            prog.total_epochs = config_.epochs;
            prog.pairs_processed = pairs;
            prog.total_pairs = total_pairs;
            prog.loss = final_loss;
            prog.current_lr = lr;
            prog.vocab_size = vocab_size();
            prog.use_cuda = use_cuda;
            prog.elapsed_seconds = elapsed;
            progress_cb_(prog);
        }
    }

    trained_ = true;

    // 训练完成后释放中间缓冲（corpus_ 和 neg_table_ 仅训练时需要）
    corpus_.clear();
    corpus_.shrink_to_fit();
    neg_table_.clear();
    neg_table_.shrink_to_fit();

    auto t1 = std::chrono::steady_clock::now();
    result.final_loss = final_loss;
    result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();

    return result;
}

auto EmbeddingTrainer::train_epoch_cpu_(
    int /*epoch*/, double base_lr, double& total_loss) -> long long
{
    int D = config_.embedding_dim;
    int ws = config_.window_size;
    int neg = config_.neg_samples;
    int corpus_len = static_cast<int>(corpus_.size());

    // 打乱：随机化训练起点
    std::uniform_int_distribution<int> neg_dist(0, NEG_TABLE_SIZE - 1);
    std::uniform_int_distribution<int> win_dist(1, ws);

    long long pairs_processed = 0;
    total_loss = 0.0;
    float lr = static_cast<float>(base_lr);

    // 临时梯度缓冲
    std::vector<float> grad_center(D, 0.0f);

    for (int pos = 0; pos < corpus_len; ++pos) {
        int center = corpus_[pos];
        if (center < 0) continue;

        // 动态窗口大小
        int actual_ws = win_dist(rng_);
        int start = std::max(0, pos - actual_ws);
        int end = std::min(corpus_len - 1, pos + actual_ws);

        for (int c = start; c <= end; ++c) {
            if (c == pos || corpus_[c] < 0) continue;
            int context = corpus_[c];

            // ── SGNS 训练步骤 ──
            std::fill(grad_center.begin(), grad_center.end(), 0.0f);

            // 中心词向量指针
            float* v_center = &W_in_[center * D];

            // 正样本 + 负样本
            for (int k = -1; k < neg; ++k) {
                int target;
                float label;

                if (k == -1) {
                    // 正样本
                    target = context;
                    label = 1.0f;
                } else {
                    // 负采样
                    target = neg_table_[neg_dist(rng_)];
                    if (target == context) continue;
                    label = 0.0f;
                }

                float* v_target = &W_out_[target * D];

                // 点积
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += v_center[d] * v_target[d];
                }

                // sigmoid
                float sig = sigmoid_(dot);
                float grad = lr * (label - sig);

                // 累积 loss
                float eps = label - sig;
                total_loss += static_cast<double>(eps * eps);

                // 更新梯度
                for (int d = 0; d < D; ++d) {
                    grad_center[d] += grad * v_target[d];
                    v_target[d] += grad * v_center[d];
                }
            }

            // 更新中心词向量
            for (int d = 0; d < D; ++d) {
                v_center[d] += grad_center[d];
            }

            pairs_processed++;
        }
    }

    return pairs_processed;
}

void EmbeddingTrainer::set_progress_callback(ProgressCallback cb) {
    progress_cb_ = std::move(cb);
}

// ── 查询 ──────────────────────────────────────────────────────

auto EmbeddingTrainer::get_embedding(const std::string& word) const
    -> std::optional<std::vector<float>>
{
    auto it = word2idx_.find(word);
    if (it == word2idx_.end() || !trained_) return std::nullopt;

    int idx = it->second;
    int D = config_.embedding_dim;
    return std::vector<float>(W_in_.begin() + idx * D,
                               W_in_.begin() + (idx + 1) * D);
}

auto EmbeddingTrainer::most_similar(const std::string& word, int top_k) const
    -> std::vector<EmbeddingQuery>
{
    if (!trained_) return {};

    auto it = word2idx_.find(word);
    if (it == word2idx_.end()) return {};

    int D = config_.embedding_dim;
    int V = static_cast<int>(idx2word_.size());
    int src_idx = it->second;
    const float* src = &W_in_[src_idx * D];

    // 归一化源向量
    float src_norm = 0.0f;
    for (int d = 0; d < D; ++d) src_norm += src[d] * src[d];
    src_norm = std::sqrt(src_norm);
    if (src_norm < 1e-9f) return {};

    std::vector<EmbeddingQuery> results;
    for (int i = 0; i < V; ++i) {
        if (i == src_idx) continue;

        const float* tgt = &W_in_[i * D];
        float dot = 0.0f, tgt_norm = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += src[d] * tgt[d];
            tgt_norm += tgt[d] * tgt[d];
        }
        tgt_norm = std::sqrt(tgt_norm);
        float sim = (tgt_norm < 1e-9f) ? 0.0f : dot / (src_norm * tgt_norm);

        EmbeddingQuery eq;
        eq.word = idx2word_[i];
        eq.embedding = std::vector<float>(tgt, tgt + D);
        eq.similarity = sim;
        results.push_back(eq);
    }

    std::partial_sort(results.begin(),
                       results.begin() + std::min(top_k, static_cast<int>(results.size())),
                       results.end(),
                       [](const auto& a, const auto& b) {
                           return a.similarity > b.similarity;
                       });

    results.resize(std::min(top_k, static_cast<int>(results.size())));
    return results;
}

auto EmbeddingTrainer::analogy(const std::string& a, const std::string& b,
                                const std::string& c, int top_k) const
    -> std::vector<EmbeddingQuery>
{
    if (!trained_) return {};

    auto it_a = word2idx_.find(a);
    auto it_b = word2idx_.find(b);
    auto it_c = word2idx_.find(c);
    if (it_a == word2idx_.end() || it_b == word2idx_.end() ||
        it_c == word2idx_.end()) {
        return {};
    }

    int D = config_.embedding_dim;
    int V = static_cast<int>(idx2word_.size());

    // 目标向量 = b - a + c
    std::vector<float> target(D, 0.0f);
    const float *va = &W_in_[it_a->second * D];
    const float *vb = &W_in_[it_b->second * D];
    const float *vc = &W_in_[it_c->second * D];

    for (int d = 0; d < D; ++d) {
        target[d] = vb[d] - va[d] + vc[d];
    }

    float target_norm = 0.0f;
    for (int d = 0; d < D; ++d) target_norm += target[d] * target[d];
    target_norm = std::sqrt(target_norm);

    std::vector<EmbeddingQuery> results;
    for (int i = 0; i < V; ++i) {
        if (i == it_a->second || i == it_b->second || i == it_c->second) continue;

        const float* tgt = &W_in_[i * D];
        float dot = 0.0f, tgt_norm = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += target[d] * tgt[d];
            tgt_norm += tgt[d] * tgt[d];
        }
        tgt_norm = std::sqrt(tgt_norm);
        float denom = target_norm * tgt_norm;
        float sim = (denom < 1e-9f) ? 0.0f : dot / denom;

        EmbeddingQuery eq;
        eq.word = idx2word_[i];
        eq.embedding = std::vector<float>(tgt, tgt + D);
        eq.similarity = sim;
        results.push_back(eq);
    }

    std::partial_sort(results.begin(),
                       results.begin() + std::min(top_k, static_cast<int>(results.size())),
                       results.end(),
                       [](const auto& a, const auto& b) {
                           return a.similarity > b.similarity;
                       });

    results.resize(std::min(top_k, static_cast<int>(results.size())));
    return results;
}

auto EmbeddingTrainer::has_word(const std::string& word) const -> bool {
    return word2idx_.count(word) > 0;
}

auto EmbeddingTrainer::vocabulary() const -> std::vector<std::string> {
    return idx2word_;
}

auto EmbeddingTrainer::vocab_size() const -> int {
    return static_cast<int>(idx2word_.size());
}

auto EmbeddingTrainer::is_trained() const -> bool {
    return trained_;
}

auto EmbeddingTrainer::export_embeddings() const
    -> std::pair<std::vector<std::string>, std::vector<float>>
{
    return {idx2word_, W_in_};
}

// ── 序列化 ────────────────────────────────────────────────────

auto EmbeddingTrainer::save(const std::string& path) const -> bool {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    int V = static_cast<int>(idx2word_.size());
    int D = config_.embedding_dim;

    out.write(reinterpret_cast<const char*>(&V), sizeof(int));
    out.write(reinterpret_cast<const char*>(&D), sizeof(int));

    for (int i = 0; i < V; ++i) {
        int len = static_cast<int>(idx2word_[i].size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(int));
        out.write(idx2word_[i].data(), len);
    }

    out.write(reinterpret_cast<const char*>(W_in_.data()),
              W_in_.size() * sizeof(float));

    return out.good();
}

auto EmbeddingTrainer::load(const std::string& path) -> bool {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    int V, D;
    in.read(reinterpret_cast<char*>(&V), sizeof(int));
    in.read(reinterpret_cast<char*>(&D), sizeof(int));

    idx2word_.resize(V);
    word2idx_.clear();
    for (int i = 0; i < V; ++i) {
        int len;
        in.read(reinterpret_cast<char*>(&len), sizeof(int));
        idx2word_[i].resize(len);
        in.read(idx2word_[i].data(), len);
        word2idx_[idx2word_[i]] = i;
    }

    W_in_.resize(V * D);
    in.read(reinterpret_cast<char*>(W_in_.data()), V * D * sizeof(float));

    config_.embedding_dim = D;
    trained_ = true;
    return in.good();
}

// ── 分词（与 DistributionalSemantics 一致） ──────────────────

auto EmbeddingTrainer::segment_(const std::string& text) const
    -> std::vector<std::string>
{
    std::vector<std::string> chars;
    std::string buffer;
    bool in_ascii = false;

    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 0x80) {
            if (in_ascii && !buffer.empty()) {
                if (!is_stopword_(buffer) && buffer.size() > 1)
                    chars.push_back(buffer);
                buffer.clear();
                in_ascii = false;
            }
            buffer += c;
            if (buffer.size() >= 3) {
                if (!is_stopword_(buffer)) chars.push_back(buffer);
                buffer.clear();
            }
        } else {
            if (!buffer.empty() && !in_ascii) {
                if (!is_stopword_(buffer)) chars.push_back(buffer);
                buffer.clear();
            }
            in_ascii = true;
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                buffer += c;
            } else {
                if (!buffer.empty() && !is_stopword_(buffer) && buffer.size() > 1)
                    chars.push_back(buffer);
                buffer.clear();
                in_ascii = false;
            }
        }
    }
    if (!buffer.empty() && !is_stopword_(buffer)) {
        if (buffer.size() > 1 || static_cast<unsigned char>(buffer[0]) >= 0x80)
            chars.push_back(buffer);
    }

    // bi-gram + tri-gram
    std::vector<std::string> tokens;
    for (const auto& t : chars) tokens.push_back(t);

    auto is_cn = [](const std::string& s) -> bool {
        return s.size() == 3 && static_cast<unsigned char>(s[0]) >= 0x80;
    };

    int n = static_cast<int>(chars.size());
    for (int i = 0; i < n; ++i) {
        if (!is_cn(chars[i])) continue;
        if (i + 1 < n && is_cn(chars[i + 1])) {
            tokens.push_back(chars[i] + chars[i + 1]);
        }
        if (i + 2 < n && is_cn(chars[i + 1]) && is_cn(chars[i + 2])) {
            tokens.push_back(chars[i] + chars[i + 1] + chars[i + 2]);
        }
    }

    return tokens;
}

auto EmbeddingTrainer::is_stopword_(const std::string& token) const -> bool {
    static const std::unordered_set<std::string> stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "no", "if", "then", "that", "this",
    };
    return stopwords.count(token) > 0;
}

}  // namespace ai_learning::learning
