/**
 * @file dependency_parser.cpp
 * @brief 无监督依存句法归纳 + 谓词论元抽取实现（零预训练、零标注）
 */

#include "ai_learning/learning/dependency_parser.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>

namespace ai_learning::learning {

namespace {

constexpr double kNeg = -1e18;
constexpr double kEps = 1e-9;

auto safe_log(double p) -> double {
    return std::log(p < 1e-12 ? 1e-12 : p);
}

auto dist_bucket(int dist, int buckets) -> int {
    int b = dist - 1;
    if (b < 0) b = 0;
    if (b >= buckets) b = buckets - 1;
    return b;
}

/// 统计一个汉字串里的 CJK 字符数（用于过滤过短论元）
auto cjk_count(const std::string& s) -> int {
    int n = 0;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            i += 1;
        } else if ((c >> 5) == 0x6) {
            i += 2;
        } else if ((c >> 4) == 0xE) {
            i += 3;
            ++n;
        } else if ((c >> 3) == 0x1E) {
            i += 4;
        } else {
            i += 1;
        }
    }
    return n;
}

}  // namespace

// ════════════════════════════════════════════════════════════════
// WordClassInducer
// ════════════════════════════════════════════════════════════════

void WordClassInducer::fit(
    const std::vector<std::vector<std::string>>& sentences) {
    // 1. 词频
    std::unordered_map<std::string, int> freq;
    for (const auto& s : sentences)
        for (const auto& w : s) freq[w]++;

    std::vector<std::string> vocab;
    for (const auto& [w, f] : freq)
        if (f >= cfg_.min_count) vocab.push_back(w);
    std::sort(vocab.begin(), vocab.end(),
              [&](const std::string& a, const std::string& b) {
                  if (freq[a] != freq[b]) return freq[a] > freq[b];
                  return a < b;  // 频次相同按字典序，保证可复现
              });
    if (vocab.empty()) return;

    // 2. 特征词 = 最高频的 F 个词
    const int feat = std::min<int>(40, static_cast<int>(vocab.size()));
    std::unordered_map<std::string, int> feat_idx;
    for (int i = 0; i < feat; ++i) feat_idx[vocab[i]] = i;

    // 3. 上下文向量：左邻接 [0,feat) + 右邻接 [feat,2feat)
    const int dim = 2 * feat;
    std::unordered_map<std::string, std::vector<double>> vec;
    for (const auto& w : vocab) vec[w] = std::vector<double>(dim, 0.0);
    for (const auto& s : sentences) {
        for (size_t i = 0; i < s.size(); ++i) {
            if (!vec.count(s[i])) continue;
            if (i > 0) {
                auto it = feat_idx.find(s[i - 1]);
                if (it != feat_idx.end()) vec[s[i]][it->second] += 1.0;
            }
            if (i + 1 < s.size()) {
                auto it = feat_idx.find(s[i + 1]);
                if (it != feat_idx.end()) vec[s[i]][feat + it->second] += 1.0;
            }
        }
    }
    // log(1+x) 压缩 + L2 归一化（→ 余弦 k-means）
    for (auto& [w, v] : vec) {
        double norm = 0.0;
        for (double& x : v) {
            x = std::log(1.0 + x);
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm > kEps)
            for (double& x : v) x /= norm;
    }

    // 4. 余弦 k-means：质心初始化用频次均匀分布的 K 个种子词（确定性）
    const int K = cfg_.num_classes;
    std::vector<std::vector<double>> cent(K, std::vector<double>(dim, 0.0));
    for (int k = 0; k < K; ++k) {
        int idx = static_cast<int>(
            (static_cast<long long>(k) * vocab.size()) / std::max(1, K));
        idx = std::min<int>(idx, static_cast<int>(vocab.size()) - 1);
        cent[k] = vec[vocab[idx]];
    }

    std::unordered_map<std::string, int> assign;
    for (int iter = 0; iter < cfg_.class_iters; ++iter) {
        // 分配
        for (const auto& w : vocab) {
            const auto& v = vec[w];
            int best = 0;
            double best_dot = -1e18;
            for (int k = 0; k < K; ++k) {
                double dot = 0.0;
                for (int d = 0; d < dim; ++d) dot += v[d] * cent[k][d];
                if (dot > best_dot) {
                    best_dot = dot;
                    best = k;
                }
            }
            assign[w] = best;
        }
        // 更新质心
        std::vector<std::vector<double>> nc(K, std::vector<double>(dim, 0.0));
        std::vector<int> cnt(K, 0);
        for (const auto& w : vocab) {
            int k = assign[w];
            const auto& v = vec[w];
            for (int d = 0; d < dim; ++d) nc[k][d] += v[d];
            cnt[k]++;
        }
        for (int k = 0; k < K; ++k) {
            if (cnt[k] == 0) continue;  // 空簇保留旧质心
            double norm = 0.0;
            for (double x : nc[k]) norm += x * x;
            norm = std::sqrt(norm);
            if (norm > kEps)
                for (int d = 0; d < dim; ++d) cent[k][d] = nc[k][d] / norm;
        }
    }

    word_class_ = std::map<std::string, int>(assign.begin(), assign.end());
    // OOV 归到最高频词所在类（通常是功能词类，作为合理默认）
    oov_class_ = assign.count(vocab[0]) ? assign[vocab[0]] : 0;
}

auto WordClassInducer::class_of(const std::string& word) const -> int {
    auto it = word_class_.find(word);
    return it != word_class_.end() ? it->second : oov_class_;
}

// ════════════════════════════════════════════════════════════════
// DependencyGrammarInducer
// ════════════════════════════════════════════════════════════════

void DependencyGrammarInducer::init_harmonic_(
    const std::vector<std::vector<int>>& corpus_classes) {
    const int K = cfg_.num_classes;
    p_attach_.assign(K, std::vector<std::vector<double>>(
                            2, std::vector<double>(K, kEps)));
    p_dist_.assign(2, std::vector<double>(kDistBuckets, kEps));
    p_root_.assign(K, kEps);

    for (const auto& cls : corpus_classes) {
        const int L = static_cast<int>(cls.size());
        for (int d = 0; d < L; ++d) p_root_[cls[d]] += 1.0 / (1.0 + d);
        for (int h = 0; h < L; ++h) {
            for (int d = 0; d < L; ++d) {
                if (h == d) continue;
                int dir = d > h ? 1 : 0;
                int dist = std::abs(d - h);
                double w = 1.0 / dist;  // 谐波：偏好近邻
                p_attach_[cls[h]][dir][cls[d]] += w;
                p_dist_[dir][dist_bucket(dist, kDistBuckets)] += w;
            }
        }
    }
    // 归一化
    for (int h = 0; h < K; ++h)
        for (int dir = 0; dir < 2; ++dir) {
            double s = 0.0;
            for (int c = 0; c < K; ++c) s += p_attach_[h][dir][c];
            for (int c = 0; c < K; ++c) p_attach_[h][dir][c] /= s;
        }
    for (int dir = 0; dir < 2; ++dir) {
        double s = 0.0;
        for (double x : p_dist_[dir]) s += x;
        for (double& x : p_dist_[dir]) x /= s;
    }
    double rs = 0.0;
    for (double x : p_root_) rs += x;
    for (double& x : p_root_) x /= rs;
}

auto DependencyGrammarInducer::arc_score_(const std::vector<int>& cls, int hpos,
                                          int dpos) const -> double {
    if (hpos < 0) return safe_log(p_root_[cls[dpos]]);
    int dir = dpos > hpos ? 1 : 0;
    int dist = std::abs(dpos - hpos);
    return safe_log(p_attach_[cls[hpos]][dir][cls[dpos]]) +
           safe_log(p_dist_[dir][dist_bucket(dist, kDistBuckets)]);
}

auto DependencyGrammarInducer::score_tree(const std::vector<int>& cls,
                                          const std::vector<int>& heads) const
    -> double {
    double s = 0.0;
    for (int d = 0; d < static_cast<int>(heads.size()); ++d)
        s += arc_score_(cls, heads[d], d);
    return s;
}

// 标准一阶投影 Eisner Viterbi 解码（ROOT 置于位置 0，词位 1..n）。
auto DependencyGrammarInducer::eisner_parse_(const std::vector<int>& cls) const
    -> std::vector<int> {
    const int n = static_cast<int>(cls.size());
    if (n == 0) return {};

    const int M = n + 1;  // 含 ROOT
    // s(h_idx, d_idx)：节点下标 0..n（0=ROOT）。ROOT 只能向右。
    auto s = [&](int h_idx, int d_idx) -> double {
        int hpos = h_idx - 1;  // -1 表示 ROOT
        int dpos = d_idx - 1;
        if (hpos < 0 && dpos < 0) return kNeg;
        return arc_score_(cls, hpos, dpos);
    };

    // ce/ie[i][j][dir]，dir: 0=Left(head=j), 1=Right(head=i)
    auto mk = [&]() {
        return std::vector<std::vector<std::array<double, 2>>>(
            M, std::vector<std::array<double, 2>>(M, {kNeg, kNeg}));
    };
    auto mki = [&]() {
        return std::vector<std::vector<std::array<int, 2>>>(
            M, std::vector<std::array<int, 2>>(M, {-1, -1}));
    };
    auto ce = mk();
    auto ie = mk();
    auto sce = mki();
    auto sie = mki();

    for (int i = 0; i < M; ++i) {
        ce[i][i][0] = 0.0;
        ce[i][i][1] = 0.0;
    }

    for (int m = 1; m < M; ++m) {
        for (int i = 0; i + m < M; ++i) {
            int j = i + m;
            // 不完整(trapezoid)
            // I_R：弧 i->j（head=i）
            {
                double best = kNeg;
                int arg = -1;
                for (int r = i; r < j; ++r) {
                    double v = ce[i][r][1] + ce[r + 1][j][0];
                    if (v <= kNeg / 2) continue;
                    v += s(i, j);
                    if (v > best) {
                        best = v;
                        arg = r;
                    }
                }
                ie[i][j][1] = best;
                sie[i][j][1] = arg;
            }
            // I_L：弧 j->i（head=j）。ROOT(0) 不能作依存项，故 i==0 时无效
            if (i > 0) {
                double best = kNeg;
                int arg = -1;
                for (int r = i; r < j; ++r) {
                    double v = ce[i][r][1] + ce[r + 1][j][0];
                    if (v <= kNeg / 2) continue;
                    v += s(j, i);
                    if (v > best) {
                        best = v;
                        arg = r;
                    }
                }
                ie[i][j][0] = best;
                sie[i][j][0] = arg;
            }
            // 完整(triangle)
            // C_R: max_{i<r<=j} ie[i][r][1] + ce[r][j][1]
            {
                double best = kNeg;
                int arg = -1;
                for (int r = i + 1; r <= j; ++r) {
                    if (ie[i][r][1] <= kNeg / 2 || ce[r][j][1] <= kNeg / 2)
                        continue;
                    double v = ie[i][r][1] + ce[r][j][1];
                    if (v > best) {
                        best = v;
                        arg = r;
                    }
                }
                ce[i][j][1] = best;
                sce[i][j][1] = arg;
            }
            // C_L: max_{i<=r<j} ce[i][r][0] + ie[r][j][0]
            {
                double best = kNeg;
                int arg = -1;
                for (int r = i; r < j; ++r) {
                    if (ce[i][r][0] <= kNeg / 2 || ie[r][j][0] <= kNeg / 2)
                        continue;
                    double v = ce[i][r][0] + ie[r][j][0];
                    if (v > best) {
                        best = v;
                        arg = r;
                    }
                }
                ce[i][j][0] = best;
                sce[i][j][0] = arg;
            }
        }
    }

    std::vector<int> heads(n, -1);
    // 递归回溯
    std::function<void(int, int, int)> dec_c;
    std::function<void(int, int, int)> dec_i;
    dec_c = [&](int i, int j, int dir) {
        if (i == j) return;
        int r = sce[i][j][dir];
        if (r < 0) return;
        if (dir == 1) {
            dec_i(i, r, 1);
            dec_c(r, j, 1);
        } else {
            dec_c(i, r, 0);
            dec_i(r, j, 0);
        }
    };
    dec_i = [&](int i, int j, int dir) {
        int r = sie[i][j][dir];
        if (r < 0) return;
        if (dir == 1) {            // 弧 i->j
            int dep = j - 1;
            heads[dep] = i - 1;    // i==0 → ROOT(-1)
        } else {                   // 弧 j->i
            int dep = i - 1;
            heads[dep] = j - 1;
        }
        dec_c(i, r, 1);
        dec_c(r + 1, j, 0);
    };
    dec_c(0, n, 1);
    return heads;
}

void DependencyGrammarInducer::train(
    const std::vector<std::vector<std::string>>& sentences) {
    classes_ = WordClassInducer(cfg_);
    classes_.fit(sentences);

    std::vector<std::vector<int>> corpus(sentences.size());
    for (size_t i = 0; i < sentences.size(); ++i) {
        corpus[i].reserve(sentences[i].size());
        for (const auto& w : sentences[i])
            corpus[i].push_back(classes_.class_of(w));
    }

    init_harmonic_(corpus);
    const int K = cfg_.num_classes;

    for (int it = 0; it < cfg_.em_iters; ++it) {
        auto a_cnt = std::vector<std::vector<std::vector<double>>>(
            K, std::vector<std::vector<double>>(2, std::vector<double>(K, kEps)));
        auto d_cnt =
            std::vector<std::vector<double>>(2, std::vector<double>(kDistBuckets, kEps));
        auto r_cnt = std::vector<double>(K, kEps);

        for (const auto& cls : corpus) {
            if (cls.empty()) continue;
            auto heads = eisner_parse_(cls);
            for (int d = 0; d < static_cast<int>(cls.size()); ++d) {
                int h = heads[d];
                if (h < 0) {
                    r_cnt[cls[d]] += 1.0;
                } else {
                    int dir = d > h ? 1 : 0;
                    a_cnt[cls[h]][dir][cls[d]] += 1.0;
                    d_cnt[dir][dist_bucket(std::abs(d - h), kDistBuckets)] += 1.0;
                }
            }
        }
        // M 步：归一化
        for (int h = 0; h < K; ++h)
            for (int dir = 0; dir < 2; ++dir) {
                double sm = 0.0;
                for (int c = 0; c < K; ++c) sm += a_cnt[h][dir][c];
                for (int c = 0; c < K; ++c) p_attach_[h][dir][c] = a_cnt[h][dir][c] / sm;
            }
        for (int dir = 0; dir < 2; ++dir) {
            double sm = 0.0;
            for (double x : d_cnt[dir]) sm += x;
            for (int b = 0; b < kDistBuckets; ++b) p_dist_[dir][b] = d_cnt[dir][b] / sm;
        }
        double rs = 0.0;
        for (double x : r_cnt) rs += x;
        for (int c = 0; c < K; ++c) p_root_[c] = r_cnt[c] / rs;
    }
    trained_ = true;
}

auto DependencyGrammarInducer::parse(const std::vector<std::string>& words) const
    -> ParsedSentence {
    ParsedSentence ps;
    ps.words = words;
    ps.classes.reserve(words.size());
    for (const auto& w : words) ps.classes.push_back(classes_.class_of(w));
    ps.heads = trained_ ? eisner_parse_(ps.classes)
                        : std::vector<int>(words.size(), -1);
    return ps;
}

// ════════════════════════════════════════════════════════════════
// 谓词论元抽取（在依存树上取论元子树）
// ════════════════════════════════════════════════════════════════

namespace {

/// 计算节点 idx 的子树连续跨度 [lo,hi]（投影树保证连续），并排除 exclude。
auto subtree_span(const ParsedSentence& p, int idx, int exclude)
    -> std::pair<int, int> {
    const int n = static_cast<int>(p.words.size());
    std::vector<char> in(n, 0);
    in[idx] = 1;
    // 反复纳入"head 已在集合中的节点"，直到稳定（n 很小）
    bool changed = true;
    while (changed) {
        changed = false;
        for (int d = 0; d < n; ++d) {
            if (in[d] || d == exclude) continue;
            if (p.heads[d] >= 0 && in[p.heads[d]]) {
                in[d] = 1;
                changed = true;
            }
        }
    }
    int lo = idx, hi = idx;
    for (int d = 0; d < n; ++d)
        if (in[d]) {
            lo = std::min(lo, d);
            hi = std::max(hi, d);
        }
    return {lo, hi};
}

auto join_span(const ParsedSentence& p, int lo, int hi, int exclude)
    -> std::string {
    std::string out;
    for (int k = lo; k <= hi; ++k) {
        if (k == exclude) continue;
        out += p.words[k];
    }
    return out;
}

}  // namespace

auto extract_triples_from_parse(
    const ParsedSentence& parsed,
    const std::vector<std::pair<std::string, std::string>>& relations)
    -> std::vector<Triple> {
    std::vector<Triple> triples;
    const int n = static_cast<int>(parsed.words.size());

    std::unordered_map<std::string, std::string> cue;
    for (const auto& [w, rel] : relations) cue[w] = rel;

    for (int p = 0; p < n; ++p) {
        auto it = cue.find(parsed.words[p]);
        if (it == cue.end()) continue;

        // 边界中心词法（与"谓词是否为树根"无关，更稳健）：
        // 主语 = 左区间 [0,p-1] 内"中心词落在区间外(挂到谓词/右侧/ROOT)"的词，
        //        取最靠近谓词者 → 即左侧名词短语的中心词；
        // 宾语 = 右区间 [p+1,n-1] 内"中心词落在区间外(挂到谓词/左侧/ROOT)"的词，
        //        取最靠近谓词者。
        int subj = -1;
        for (int d = p - 1; d >= 0; --d) {
            int h = parsed.heads[d];
            if (h < 0 || h >= p) { subj = d; break; }
        }
        int obj = -1;
        for (int d = p + 1; d < n; ++d) {
            int h = parsed.heads[d];
            if (h <= p) { obj = d; break; }
        }
        // 兜底：边界中心词找不到时（长句树结构有噪声），退为谓词紧邻词，
        // 保证有侧词就不漏抽（取其依存子树作为论元短语）。
        if (subj < 0 && p > 0) subj = p - 1;
        if (obj < 0 && p < n - 1) obj = p + 1;
        if (subj < 0 || obj < 0) continue;

        auto [slo, shi] = subtree_span(parsed, subj, p);
        auto [olo, ohi] = subtree_span(parsed, obj, p);
        // 论元子树裁剪到各自区间，不跨过谓词位置 p
        slo = std::max(slo, 0);
        shi = std::min(shi, p - 1);
        olo = std::max(olo, p + 1);
        ohi = std::min(ohi, n - 1);
        std::string s_str = join_span(parsed, slo, shi, p);
        std::string o_str = join_span(parsed, olo, ohi, p);

        if (cjk_count(s_str) >= 2 && cjk_count(o_str) >= 2) {
            Triple t;
            t.subject = s_str;
            t.relation = it->second;
            t.object = o_str;
            t.confidence = 0.8;  // 来自依存结构而非硬窗口
            triples.push_back(std::move(t));
        }
    }
    return triples;
}

}  // namespace ai_learning::learning
