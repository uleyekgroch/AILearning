/**
 * @file dependency_parser.hpp
 * @brief 无监督依存句法归纳 + 谓词论元抽取（零预训练、零标注、零句法器）
 *
 * 立场：与项目「拒绝大模型」一致——不读任何预训练权重，也不依赖标注树库。
 * 全部从自学语料的分布统计里**归纳**出句法结构：
 *
 *   1) WordClassInducer：用左右邻接共现 + PPMI 把词聚成「伪词类」(induced POS)，
 *      作为 DMV 的稀疏化骨架（参数定义在词类上而非词上）。
 *   2) DependencyGrammarInducer：一阶**边因子**投影依存模型（DMV 的诚实简化，
 *      源自 Klein & Manning 2004 思路）。用 Eisner 投影 Viterbi + Viterbi-EM 训练，
 *      学「中心词类→论元类」的方向概率 + 学到的距离分布（近邻偏好，近似价数），
 *      输出每句的投影依存树。边因子让 Eisner 解码可证明正确（无价数折叠歧义）。
 *   3) extract_triples_from_parse：在依存树上，把关系线索词(是/导致/位于…)当谓词，
 *      取它的**真实依存论元(主语/宾语子树)**，替代「find()+硬截6字窗口」。
 *
 * 诚实边界：小语料归纳出的树有噪声；这不是 PropBank 级精标 SRL，而是
 * 「无监督依存结构 + 核心论元(近似施事/受事)」。质量由客观评测量化，不夸大。
 *
 * 输入约定：句子为已按空白切好的词序列（与项目现有评测语料一致）。
 */

#pragma once

#include "ai_learning/learning/knowledge_extractor.hpp"

#include <map>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 解析后的句子：词、词类、以及每个词的中心词下标(head[i]，-1 表示挂到 ROOT)
struct ParsedSentence {
    std::vector<std::string> words;
    std::vector<int> classes;  ///< 每个词的伪词类 id
    std::vector<int> heads;    ///< heads[i]：第 i 个词的中心词下标；-1 = ROOT
};

/// 依存归纳配置
struct DepParseConfig {
    int num_classes = 8;   ///< 伪词类数 K
    int em_iters = 20;     ///< Viterbi-EM 轮数
    int class_iters = 15;  ///< 词类 k-means 轮数
    int min_count = 1;     ///< 词进入词表的最小频次
    unsigned seed = 42;    ///< 随机种子（k-means 初始化，保证可复现）
};

// ── 词类归纳（分布式伪词性）────────────────────────────────────────
class WordClassInducer {
public:
    explicit WordClassInducer(DepParseConfig cfg = {}) : cfg_(cfg) {}

    /// 从语料(已分词句子)归纳词类
    void fit(const std::vector<std::vector<std::string>>& sentences);

    /// 查词类（未登录词归为专用 OOV 类）
    [[nodiscard]] auto class_of(const std::string& word) const -> int;

    [[nodiscard]] auto num_classes() const -> int { return cfg_.num_classes; }

private:
    DepParseConfig cfg_;
    std::map<std::string, int> word_class_;
    int oov_class_ = 0;
};

// ── DMV 依存语法归纳器 ─────────────────────────────────────────────
class DependencyGrammarInducer {
public:
    explicit DependencyGrammarInducer(DepParseConfig cfg = {}) : cfg_(cfg) {}

    /// 训练：归纳词类 → DMV 谐波初始化 → Viterbi-EM
    void train(const std::vector<std::vector<std::string>>& sentences);

    /// 解析一句（已分词）为投影依存树
    [[nodiscard]] auto parse(const std::vector<std::string>& words) const
        -> ParsedSentence;

    [[nodiscard]] auto is_trained() const -> bool { return trained_; }

private:
    DepParseConfig cfg_;
    WordClassInducer classes_;
    bool trained_ = false;

    // 边因子概率参数（定义在词类上）。dir: 0=左, 1=右
    // p_attach_[head_class][dir][child_class]；p_dist_[dir][bucket]；p_root_[c]
    static constexpr int kDistBuckets = 4;  // 距离桶: 1,2,3,>=4
    std::vector<std::vector<std::vector<double>>> p_attach_;
    std::vector<std::vector<double>> p_dist_;
    std::vector<double> p_root_;

    void init_harmonic_(const std::vector<std::vector<int>>& corpus_classes);
    // Eisner 投影 Viterbi：给定类序列，返回 heads（-1=ROOT）
    [[nodiscard]] auto eisner_parse_(const std::vector<int>& cls) const
        -> std::vector<int>;
    // 弧得分(对数)：head 位置 hpos、dependent 位置 dpos（句内 0..n-1），
    // hpos<0 表示 ROOT。供解码与单测复用。
    [[nodiscard]] auto arc_score_(const std::vector<int>& cls, int hpos,
                                  int dpos) const -> double;

public:
    /// 给定类序列与一棵树(heads)，按同一模型计算对数得分（供测试/EM 校验）。
    [[nodiscard]] auto score_tree(const std::vector<int>& cls,
                                  const std::vector<int>& heads) const -> double;
    /// 暴露词类查询（供论元抽取/测试）
    [[nodiscard]] auto class_of(const std::string& w) const -> int {
        return classes_.class_of(w);
    }
};

// ── 基于依存树的谓词论元抽取（替代硬截窗口）─────────────────────────
/// 在依存树上，把每个关系线索词当谓词，取其左/右依存论元(子树跨度)组成三元组。
/// @param parsed       已解析句子
/// @param relations    关系线索词 → 关系名（如 {"是","是"}）
auto extract_triples_from_parse(
    const ParsedSentence& parsed,
    const std::vector<std::pair<std::string, std::string>>& relations)
    -> std::vector<Triple>;

}  // namespace ai_learning::learning
