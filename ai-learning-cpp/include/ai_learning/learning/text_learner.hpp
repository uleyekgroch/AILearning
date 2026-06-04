/**
 * @file text_learner.hpp
 * @brief 文本学习者 — 从文本学习知识的核心管线
 *
 * 对应 Python 版 learn_from_text() 的 720 行方法。
 * 拆分为独立类，遵循 SRP。
 *
 * 核心流程：
 * 1. 编码文本 → 2. 提取实体 → 3. 提取关系 → 4. 注入知识图谱
 * 5. 因果提取 → 6. 数值提取 → 7. 验证 → 8. 反馈
 */
#pragma once

#include "ai_learning/consciousness/self_model.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/dependency_parser.hpp"
#include "ai_learning/learning/knowledge_extractor.hpp"
#include "ai_learning/learning/word_segmenter.hpp"

#include <map>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 文本学习器配置
struct TextLearnerConfig {
    int obs_dim = 128;
    bool statistical_learning_enabled = true;
    bool statistical_use_as_primary = true;
};

/// 文本学习器
///
/// 从文本中提取结构化知识并注入知识图谱。
/// 不持有编码器状态（无状态变换），专注于提取逻辑。
class TextLearner {
public:
    explicit TextLearner(domain::knowledge::KnowledgeGraph& kg);

    /// 从文本学习 — 核心入口
    auto learn_from_text(const std::string& text,
                         const std::string& source = "text")
        -> TextLearnResult;

    /// 思考/回答问题
    auto think(const std::string& question) const
        -> std::string;

    /// 获取学习统计
    [[nodiscard]] auto stats() const -> const std::map<std::string, int>& {
        return stats_;
    }

    /// 用已累积语料（learn_from_text 自动累积）(重新)训练无监督分词器 +
    /// 依存解析器，并启用「分词→依存树→论元」抽取路径替代硬窗口。
    /// 喂入足够文本后调用；未调用时维持原有(实体接地+窗口)抽取，行为不变。
    /// @return 是否成功启用（语料不足时返回 false）
    auto train_dependency_parser() -> bool;

    /// 当前是否已启用基于句法树的论元抽取
    [[nodiscard]] auto dependency_parsing_enabled() const -> bool {
        return parse_enabled_;
    }

    /// 自我模型（随每次 learn_from_text 在线更新，供自我指涉问答）
    [[nodiscard]] auto self_model() const
        -> const consciousness::SelfModel& {
        return self_model_;
    }

private:
    /// 把本次学习写入自我模型：当下体验 + 自我信念(领域/实体) + 自传记忆。
    /// 这让 SelfModel 真正进入学习闭环，而非陈列端点。
    void update_self_model_(const std::string& text, const std::string& source,
                            const TextLearnResult& result);

    /// 自我指涉问答（你是谁/你了解X吗/你学到了什么）；非自我问题返回空。
    [[nodiscard]] auto answer_self_referential_(const std::string& q) const
        -> std::string;

    /// 基于依存树的论元抽取（已训练时使用，否则返回空交由窗口法兜底）
    [[nodiscard]] auto extract_triples_parsed_(const std::string& text) const
        -> std::vector<Triple>;

    /// 验证学到的知识
    auto verify_knowledge_(const std::string& text,
                           const std::vector<std::string>& entities,
                           const std::vector<Triple>& triples) const
        -> std::pair<bool, double>;

    /// 检查矛盾
    auto check_contradiction_(const std::string& subject,
                               const std::string& relation,
                               const std::string& obj) const
        -> std::optional<std::map<std::string, std::string>>;

    /// 解决矛盾
    void resolve_contradiction_(const std::string& subject,
                                 const std::string& relation,
                                 const std::string& obj,
                                 const std::map<std::string, std::string>& conflict,
                                 const std::string& source);

    /// STDP 赫布学习：增强连续出现的实体连接
    void update_stdp_connections_(const std::vector<std::string>& entities);

    /// STDP 推理：基于时序关联找相关概念
    auto query_stdp_(const std::string& entity, int top_k = 5) const
        -> std::vector<std::pair<std::string, float>>;

    /// 海马记忆存储
    void store_hippocampal_episode_(
        const std::string& text,
        const std::vector<std::string>& entities,
        const std::vector<Triple>& triples);

    /// 海马记忆检索
    auto query_hippocampal_(const std::string& entity) const
        -> std::optional<std::string>;

    /// 从知识图谱推理答案
    auto reason_from_kg_(const std::string& question) const
        -> std::string;

    /// 常识库查询（简单模式匹配）
    auto query_commonsense_(const std::string& question) const
        -> std::string;

    // 依赖
    domain::knowledge::KnowledgeGraph& kg_;

    // STDP 连接: (pre, post) → weight
    std::map<std::pair<std::string, std::string>, float> stdp_connections_;
    float stdp_lr_ = 0.01f;

    // 海马记忆
    struct HippocampalEpisode {
        std::vector<std::string> entities;
        std::vector<Triple> triples;
        std::string context;
    };
    std::vector<HippocampalEpisode> hippocampal_episodes_;
    std::map<std::string, std::vector<int>> hippocampal_index_;
    int hippocampal_capacity_ = 5000;

    // 统计
    std::map<std::string, int> stats_;

    // 无监督分词 + 依存解析（默认关闭；train_dependency_parser() 后启用）
    std::vector<std::string> corpus_raw_;  // learn_from_text 累积的原始文本
    int corpus_capacity_ = 10000;          // 上限，超出淘汰最旧（防无界增长）
    WordSegmenter segmenter_;
    DependencyGrammarInducer dep_parser_;
    bool parse_enabled_ = false;

    // 自我模型（意识级自我表征，随学习闭环在线更新）
    consciousness::SelfModel self_model_;
};

}  // namespace ai_learning::learning
