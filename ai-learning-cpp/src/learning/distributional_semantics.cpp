/**
 * @file distributional_semantics.cpp
 * @brief 分布语义引擎实现
 */

#include "ai_learning/learning/distributional_semantics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace ai_learning::learning {

// ── ConceptVector 方法 ────────────────────────────────────────

auto ConceptVector::norm() const -> double {
    double sum = 0.0;
    for (const auto& [_, w] : dimensions) {
        sum += w * w;
    }
    return std::sqrt(sum);
}

auto ConceptVector::to_dense(int max_dims) const -> std::vector<float> {
    // 取权重最高的 max_dims 个维度
    std::vector<std::pair<std::string, double>> sorted(
        dimensions.begin(), dimensions.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<float> dense;
    dense.reserve(max_dims);
    for (int i = 0; i < max_dims && i < static_cast<int>(sorted.size()); ++i) {
        dense.push_back(static_cast<float>(sorted[i].second));
    }
    dense.resize(max_dims, 0.0f);
    return dense;
}

// ── 构造 ──────────────────────────────────────────────────────

DistributionalSemantics::DistributionalSemantics(
    const DistributionalSemanticsConfig& config)
    : config_(config) {}

// ── 语料摄入 ──────────────────────────────────────────────────

auto DistributionalSemantics::learn_from_tokens(
    const std::vector<std::string>& tokens)
    -> std::vector<std::string>
{
    if (tokens.empty()) return {};

    texts_processed_++;
    total_tokens_ += static_cast<int>(tokens.size());

    // 更新概念频率
    std::set<std::string> unique_tokens(tokens.begin(), tokens.end());
    for (const auto& t : unique_tokens) {
        cpt_freq_[t]++;
    }

    // 更新共现矩阵
    update_cooccurrence_(tokens);

    // 检查哪些概念达到最低频率，重建它们的向量
    std::vector<std::string> emerged;
    for (const auto& [token, freq] : cpt_freq_) {
        if (freq >= config_.min_cpt_freq && vectors_.count(token) == 0) {
            emerged.push_back(token);
        }
    }

    // 增量更新向量（只更新涉及的维度）
    if (texts_processed_ % 10 == 0 || !emerged.empty()) {
        rebuild_vectors_();
    }

    return emerged;
}

auto DistributionalSemantics::learn_from_text(const std::string& text)
    -> std::vector<std::string>
{
    auto tokens = segment_(text);
    return learn_from_tokens(tokens);
}

auto DistributionalSemantics::learn_batch(
    const std::vector<std::string>& texts)
    -> int
{
    int total_new = 0;
    for (const auto& text : texts) {
        auto emerged = learn_from_text(text);
        total_new += static_cast<int>(emerged.size());
    }
    return total_new;
}

// ── 语义查询 ──────────────────────────────────────────────────

auto DistributionalSemantics::similarity(
    const std::string& cpt_a,
    const std::string& cpt_b) const
    -> SimilarityResult
{
    SimilarityResult result;
    result.cpt_a = cpt_a;
    result.cpt_b = cpt_b;

    auto it_a = vectors_.find(cpt_a);
    auto it_b = vectors_.find(cpt_b);

    if (it_a == vectors_.end() || it_b == vectors_.end()) {
        result.similarity = 0.0;
        result.explanation = "概念未在学习语料中出现";
        return result;
    }

    result.similarity = cosine_similarity_(it_a->second, it_b->second);

    // 计算直接 PMI
    result.pmi = compute_ppmi_(cpt_a, cpt_b);

    // 生成解释：找共同的上下文维度
    std::vector<std::string> common_dims;
    for (const auto& [dim, w_a] : it_a->second.dimensions) {
        auto it = it_b->second.dimensions.find(dim);
        if (it != it_b->second.dimensions.end() && w_a > 0.5 && it->second > 0.5) {
            common_dims.push_back(dim);
        }
    }

    if (!common_dims.empty()) {
        std::ostringstream oss;
        oss << "共同上下文: ";
        int count = 0;
        for (const auto& d : common_dims) {
            if (count > 0) oss << ", ";
            oss << d;
            if (++count >= 5) break;
        }
        if (static_cast<int>(common_dims.size()) > 5) {
            oss << " 等" << common_dims.size() << "个";
        }
        result.explanation = oss.str();
    } else {
        result.explanation = "无显著共同上下文";
    }

    return result;
}

auto DistributionalSemantics::most_similar(
    const std::string& cpt, int top_k) const
    -> std::vector<SimilarityResult>
{
    auto it = vectors_.find(cpt);
    if (it == vectors_.end()) return {};

    std::vector<SimilarityResult> results;
    for (const auto& [other, vec] : vectors_) {
        if (other == cpt) continue;

        SimilarityResult sr;
        sr.cpt_a = cpt;
        sr.cpt_b = other;
        sr.similarity = cosine_similarity_(it->second, vec);
        sr.pmi = compute_ppmi_(cpt, other);
        results.push_back(sr);
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

auto DistributionalSemantics::analogy(
    const std::string& a, const std::string& b,
    const std::string& c, int top_k) const
    -> std::vector<AnalogyResult>
{
    // 类比：A 之于 B ≈ C 之于 D
    // 向量运算: D ≈ B - A + C
    auto it_a = vectors_.find(a);
    auto it_b = vectors_.find(b);
    auto it_c = vectors_.find(c);

    if (it_a == vectors_.end() || it_b == vectors_.end() ||
        it_c == vectors_.end()) {
        return {};
    }

    // 构造目标向量: B - A + C
    std::map<std::string, double> target;
    for (const auto& [dim, w] : it_b->second.dimensions) {
        target[dim] += w;
    }
    for (const auto& [dim, w] : it_a->second.dimensions) {
        target[dim] -= w;
    }
    for (const auto& [dim, w] : it_c->second.dimensions) {
        target[dim] += w;
    }

    ConceptVector target_vec;
    target_vec.name = "target";
    target_vec.dimensions = target;

    std::vector<AnalogyResult> results;
    for (const auto& [d, vec] : vectors_) {
        if (d == a || d == b || d == c) continue;

        double sim = cosine_similarity_(target_vec, vec);
        if (sim > 0.0) {
            AnalogyResult ar;
            ar.a_is_to_b = a + " -> " + b;
            ar.as_c_is_to_d = c + " -> " + d;
            ar.confidence = sim;
            ar.reasoning = "向量类比: " + b + " - " + a + " + " + c + " ~= " + d;
            results.push_back(ar);
        }
    }

    std::partial_sort(results.begin(),
                       results.begin() + std::min(top_k, static_cast<int>(results.size())),
                       results.end(),
                       [](const auto& x, const auto& y) {
                           return x.confidence > y.confidence;
                       });

    results.resize(std::min(top_k, static_cast<int>(results.size())));
    return results;
}

auto DistributionalSemantics::semantic_neighborhood(
    const std::string& cpt, double threshold) const
    -> std::vector<SimilarityResult>
{
    auto it = vectors_.find(cpt);
    if (it == vectors_.end()) return {};

    std::vector<SimilarityResult> neighbors;
    for (const auto& [other, vec] : vectors_) {
        if (other == cpt) continue;
        double sim = cosine_similarity_(it->second, vec);
        if (sim >= threshold) {
            SimilarityResult sr;
            sr.cpt_a = cpt;
            sr.cpt_b = other;
            sr.similarity = sim;
            neighbors.push_back(sr);
        }
    }

    std::sort(neighbors.begin(), neighbors.end(),
        [](const auto& a, const auto& b) { return a.similarity > b.similarity; });

    return neighbors;
}

// ── 概念表示 ──────────────────────────────────────────────────

auto DistributionalSemantics::get_vector(const std::string& cpt) const
    -> std::optional<ConceptVector>
{
    auto it = vectors_.find(cpt);
    if (it != vectors_.end()) return it->second;
    return std::nullopt;
}

auto DistributionalSemantics::get_dense_vector(
    const std::string& cpt, int dims) const
    -> std::optional<std::vector<float>>
{
    auto it = vectors_.find(cpt);
    if (it == vectors_.end()) return std::nullopt;
    return it->second.to_dense(dims);
}

auto DistributionalSemantics::has_cpt(const std::string& cpt) const
    -> bool
{
    return vectors_.find(cpt) != vectors_.end();
}

// ── 语义聚类 ──────────────────────────────────────────────────

auto DistributionalSemantics::discover_clusters(double threshold) const
    -> std::vector<SemanticCluster>
{
    // 基于连通分量的简单聚类
    std::map<std::string, int> component;
    int comp_id = 0;

    for (const auto& [cpt_name, _] : vectors_) {
        if (component.count(cpt_name)) continue;

        // BFS
        std::vector<std::string> cluster_members;
        std::vector<std::string> queue = {cpt_name};
        component[cpt_name] = comp_id;

        while (!queue.empty()) {
            auto current = queue.back();
            queue.pop_back();
            cluster_members.push_back(current);

            auto it = vectors_.find(current);
            if (it == vectors_.end()) continue;

            for (const auto& [other, vec] : vectors_) {
                if (component.count(other)) continue;
                if (cosine_similarity_(it->second, vec) >= threshold) {
                    component[other] = comp_id;
                    queue.push_back(other);
                }
            }
        }

        comp_id++;
    }

    // 收集聚类结果
    std::map<int, std::vector<std::string>> clusters_map;
    for (const auto& [cpt_name, cid] : component) {
        clusters_map[cid].push_back(cpt_name);
    }

    std::vector<SemanticCluster> clusters;
    for (const auto& [cid, members] : clusters_map) {
        if (members.size() < 2) continue;  // 忽略孤立点

        SemanticCluster cluster;

        // 找最有代表性的概念（与所有其他成员的平均相似度最高）
        double best_avg = -1.0;
        for (const auto& m : members) {
            auto it_m = vectors_.find(m);
            if (it_m == vectors_.end()) continue;
            double avg = 0.0;
            for (const auto& other : members) {
                if (other == m) continue;
                auto it_o = vectors_.find(other);
                if (it_o != vectors_.end()) {
                    avg += cosine_similarity_(it_m->second, it_o->second);
                }
            }
            avg /= (members.size() - 1);
            if (avg > best_avg) {
                best_avg = avg;
                cluster.label = m;
            }
        }

        cluster.members = members;
        cluster.cohesion = best_avg;

        // 找共同上下文
        auto it_label = vectors_.find(cluster.label);
        if (it_label != vectors_.end()) {
            for (const auto& [dim, w] : it_label->second.dimensions) {
                if (w > 1.0) {
                    cluster.common_contexts.push_back(dim);
                }
            }
        }

        clusters.push_back(cluster);
    }

    // 按大小排序
    std::sort(clusters.begin(), clusters.end(),
        [](const auto& a, const auto& b) {
            return a.members.size() > b.members.size();
        });

    return clusters;
}

// ── 因果先验 ──────────────────────────────────────────────────

void DistributionalSemantics::register_causal_link(
    const std::string& cause,
    const std::string& effect,
    double strength)
{
    causal_prior_[cause][effect] = strength;
    causal_prior_[effect][cause] = strength * 0.5;  // 反向弱关联
}

auto DistributionalSemantics::causally_related(
    const std::string& cpt) const
    -> std::vector<std::pair<std::string, double>>
{
    auto it = causal_prior_.find(cpt);
    if (it == causal_prior_.end()) return {};

    std::vector<std::pair<std::string, double>> result(
        it->second.begin(), it->second.end());
    std::sort(result.begin(), result.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    return result;
}

// ── 查询 ──────────────────────────────────────────────────────

auto DistributionalSemantics::all_cpts() const
    -> std::vector<std::string>
{
    std::vector<std::string> result;
    result.reserve(vectors_.size());
    for (const auto& [cpt_name, _] : vectors_) {
        result.push_back(cpt_name);
    }
    return result;
}

auto DistributionalSemantics::stats() const -> SemanticStats
{
    double total_density = 0.0;
    for (const auto& [_, vec] : vectors_) {
        total_density += static_cast<double>(vec.dimensions.size());
    }
    double avg_density = vectors_.empty() ? 0.0
        : total_density / static_cast<double>(vectors_.size());

    return {
        static_cast<int>(vectors_.size()),
        static_cast<int>(all_dimensions_.size()),
        texts_processed_,
        avg_density,
        0,  // clusters discovered on demand
    };
}

// ── 内部方法 ──────────────────────────────────────────────────

auto DistributionalSemantics::segment_(const std::string& text) const
    -> std::vector<std::string>
{
    // Step 1: 提取所有 token（中文单字 + 英文单词 + 标点作为分隔符）
    std::vector<std::string> chars;  // 中文单字和英文单词

    std::string buffer;
    bool in_ascii = false;

    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);

        if (uc >= 0x80) {
            // 非ASCII（中文等多字节字符）
            if (in_ascii && !buffer.empty()) {
                if (!is_stopword_(buffer) && buffer.size() > 1) {
                    chars.push_back(buffer);
                }
                buffer.clear();
                in_ascii = false;
            }
            buffer += c;
            // 中文UTF-8：3字节一个字符
            if (buffer.size() >= 3) {
                if (!is_stopword_(buffer)) {
                    chars.push_back(buffer);
                }
                buffer.clear();
            }
        } else {
            // ASCII
            if (!buffer.empty() && !in_ascii) {
                if (!is_stopword_(buffer)) {
                    chars.push_back(buffer);
                }
                buffer.clear();
            }

            in_ascii = true;
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                buffer += c;
            } else {
                if (!buffer.empty() && !is_stopword_(buffer) && buffer.size() > 1) {
                    chars.push_back(buffer);
                }
                buffer.clear();
                in_ascii = false;
            }
        }
    }
    if (!buffer.empty() && !is_stopword_(buffer)) {
        if (buffer.size() > 1 || static_cast<unsigned char>(buffer[0]) >= 0x80) {
            chars.push_back(buffer);
        }
    }

    // Step 2: 生成 bi-gram 和 tri-gram（连续中文字符组成词组）
    std::vector<std::string> tokens;

    // 先加入所有单字 token
    for (const auto& t : chars) {
        tokens.push_back(t);
    }

    // 检测连续中文字符序列，生成 n-gram
    auto is_chinese = [](const std::string& s) -> bool {
        return s.size() == 3 &&
               static_cast<unsigned char>(s[0]) >= 0x80;
    };

    int n = static_cast<int>(chars.size());
    for (int i = 0; i < n; ++i) {
        // 只对中文字符做 n-gram
        if (!is_chinese(chars[i])) continue;

        // bi-gram: 两个字组成的词
        if (i + 1 < n && is_chinese(chars[i + 1])) {
            std::string bigram = chars[i] + chars[i + 1];
            if (!is_stopword_(bigram)) {
                tokens.push_back(bigram);
            }
        }

        // tri-gram: 三个字组成的词
        if (i + 2 < n && is_chinese(chars[i + 1]) && is_chinese(chars[i + 2])) {
            std::string trigram = chars[i] + chars[i + 1] + chars[i + 2];
            if (!is_stopword_(trigram)) {
                tokens.push_back(trigram);
            }
        }
    }

    return tokens;
}

void DistributionalSemantics::update_cooccurrence_(
    const std::vector<std::string>& tokens)
{
    int n = static_cast<int>(tokens.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j <= std::min(i + config_.window_size, n - 1); ++j) {
            const auto& a = tokens[i];
            const auto& b = tokens[j];
            if (a == b) continue;
            cooccurrence_[a][b]++;
            cooccurrence_[b][a]++;
        }
    }
}

void DistributionalSemantics::rebuild_vectors_()
{
    // 为每个达到最低频率的概念构建/更新分布向量
    for (const auto& [cpt_name, freq] : cpt_freq_) {
        if (freq < config_.min_cpt_freq) continue;

        auto& vec = vectors_[cpt_name];
        vec.name = cpt_name;
        vec.observation_count = freq;

        // 清空旧维度
        vec.dimensions.clear();
        vec.total_weight = 0.0;

        // 从共现矩阵构建 PPMI 向量
        auto it = cooccurrence_.find(cpt_name);
        if (it == cooccurrence_.end()) continue;

        for (const auto& [context, count] : it->second) {
            (void)count;  // count used inside compute_ppmi_
            double ppmi = compute_ppmi_(cpt_name, context);

            if (ppmi > config_.ppmi_threshold) {
                vec.dimensions[context] = ppmi;
                vec.total_weight += ppmi;
                all_dimensions_.insert(context);
            }
        }

        // 裁剪到最大维度数
        if (static_cast<int>(vec.dimensions.size()) > config_.max_dimensions) {
            std::vector<std::pair<std::string, double>> sorted(
                vec.dimensions.begin(), vec.dimensions.end());
            std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            vec.dimensions.clear();
            for (int i = 0; i < config_.max_dimensions; ++i) {
                vec.dimensions[sorted[i].first] = sorted[i].second;
            }
        }

        // 加入因果先验增强
        if (config_.use_causal_prior) {
            auto cp_it = causal_prior_.find(cpt_name);
            if (cp_it != causal_prior_.end()) {
                for (const auto& [related, strength] : cp_it->second) {
                    vec.dimensions["causal:" + related] = strength * 2.0;
                }
            }
        }
    }
}

auto DistributionalSemantics::compute_ppmi_(
    const std::string& target,
    const std::string& context) const
    -> double
{
    // PPMI = max(0, PMI)
    // PMI = log(P(target, context) / (P(target) * P(context)))
    auto it_target = cpt_freq_.find(target);
    auto it_context = cpt_freq_.find(context);
    if (it_target == cpt_freq_.end() || it_context == cpt_freq_.end()) {
        return 0.0;
    }

    auto it_cooc = cooccurrence_.find(target);
    if (it_cooc == cooccurrence_.end()) return 0.0;

    auto it_count = it_cooc->second.find(context);
    if (it_count == it_cooc->second.end()) return 0.0;

    double joint = static_cast<double>(it_count->second);
    double p_target = static_cast<double>(it_target->second);
    double p_context = static_cast<double>(it_context->second);
    double total = std::max(1, total_tokens_);

    double pmi = std::log2((joint * total) / (p_target * p_context));
    return std::max(0.0, pmi);  // PPMI
}

auto DistributionalSemantics::cosine_similarity_(
    const ConceptVector& a,
    const ConceptVector& b)
    -> double
{
    double dot = 0.0;
    for (const auto& [dim, wa] : a.dimensions) {
        auto it = b.dimensions.find(dim);
        if (it != b.dimensions.end()) {
            dot += wa * it->second;
        }
    }

    double norm_a = a.norm();
    double norm_b = b.norm();

    if (norm_a < 1e-9 || norm_b < 1e-9) return 0.0;
    return dot / (norm_a * norm_b);
}

auto DistributionalSemantics::is_stopword_(const std::string& token) const
    -> bool
{
    static const std::unordered_set<std::string> stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "no", "if", "then", "that", "this",
    };
    return stopwords.count(token) > 0;
}

}  // namespace ai_learning::learning
