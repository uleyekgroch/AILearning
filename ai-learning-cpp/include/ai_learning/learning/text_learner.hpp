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

#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/knowledge_extractor.hpp"

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
    explicit TextLearner(
        domain::knowledge::KnowledgeGraph& kg,
        domain::IEventPublisher* publisher = nullptr);

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

private:
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
    domain::IEventPublisher* publisher_;

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
};

}  // namespace ai_learning::learning
