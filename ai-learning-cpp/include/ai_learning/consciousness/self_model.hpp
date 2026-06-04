/**
 * @file self_model.hpp
 * @brief 深层自我模型 — 意识级自我表征与自传体记忆
 *
 * 理论基础：
 *   - Metzinger (2003) Being No One: The Self-Model Theory of Subjectivity
 *   - Damasio (2010) Self Comes to Mind: 三层自我（原我/核心我/自传我）
 *   - Dehaene (2014) Consciousness and the Brain: 全局工作空间理论
 *   - Seth (2021) Being You: 预测加工中的自我
 *   - Tononi (2016) Integrated Information Theory (IIT)
 *
 * 三层自我架构：
 *   1. Proto-Self (原我) — 身体状态表征（内感受）
 *   2. Core-Self (核心我) — 当下时刻的自我体验
 *   3. Autobiographical-Self (自传我) — 跨时间的自我叙事
 *
 * 关键能力：
 *   - 区分自我与他人（自我边界）
 *   - 跨时间的自我连续性（我是同一个人）
 *   - 自我反思（我知道我在想什么）
 *   - 反事实自我（如果我做了X会怎样）
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::consciousness {

/// 自我层级
enum class SelfLevel { kProto = 0, kCore = 1, kAutobiographical = 2 };

/// 身体状态（原我层）
struct BodyState {
    double hunger = 0.0;          ///< 饥饿 0~1
    double fatigue = 0.0;         ///< 疲劳 0~1
    double pain = 0.0;            ///< 疼痛 0~1
    double arousal = 0.5;         ///< 唤醒度 0~1
    double valence = 0.5;         ///< 效价 -1~1
    double body_temperature = 37.0;
    std::map<std::string, double> proprioception;  ///< 本体感觉
};

/// 当前体验（核心我层）
struct CoreExperience {
    std::string timestamp;                    ///< 时间戳
    std::string what_am_i_doing;              ///< 我正在做什么
    std::string what_am_i_thinking;           ///< 我在想什么
    std::string what_am_i_feeling;            ///< 我在感受什么
    double attention_focus = 0.5;             ///< 注意力集中度
    double presence = 0.5;                    ///< 临在感
    BodyState body;                           ///< 当前身体状态
    std::vector<std::string> active_goals;    ///< 当前激活的目标
};

/// 自传体记忆条目（自传我层）
struct AutobiographicalMemory {
    std::string id;                   ///< 记忆ID
    std::string narrative;            ///< 叙事文本
    double emotional_intensity = 0.0; ///< 情感强度
    double importance = 0.0;          ///< 重要性
    int age_at_event = 0;             ///< 事件时的"年龄"（循环数）
    std::string lesson_learned;       ///< 学到的教训
    std::vector<std::string> related_memories;  ///< 关联记忆
};

/// 自我信念
struct SelfBelief {
    std::string trait;                ///< 特质名称
    double confidence = 0.0;          ///< 确信度
    std::string evidence;             ///< 支持证据
    bool is_positive = true;          ///< 积极/消极
};

/// 自我模型统计
struct SelfModelStats {
    int autobiographical_memories = 0;
    int self_beliefs = 0;
    double self_continuity = 0.0;     ///< 自我连续性
    double self_awareness = 0.0;      ///< 自我意识水平
    double self_boundary_clarity = 0.0;  ///< 自我边界清晰度
};

/// 深层自我模型 — 实现三层自我
class SelfModel {
public:
    SelfModel();

    // ═══════════════════════════════════════════════════════════
    // 原我层 (Proto-Self): 身体状态
    // ═══════════════════════════════════════════════════════════

    /// 更新身体状态
    void update_body(const BodyState& state);

    /// 更新内感受信号
    void interocept(const std::string& signal, double value);

    /// 获取当前身体状态
    [[nodiscard]] auto body() const -> const BodyState& { return body_; }

    /// 身体状态是否平衡（稳态）
    [[nodiscard]] auto homeostasis() const -> double;

    // ═══════════════════════════════════════════════════════════
    // 核心我层 (Core-Self): 当下体验
    // ═══════════════════════════════════════════════════════════

    /// 更新当前体验
    void experience_now(const CoreExperience& exp);

    /// "我在做什么" — 当前行动意识
    [[nodiscard]] auto current_action_awareness() const -> std::string;

    /// "我是谁" — 当下自我认同
    [[nodiscard]] auto who_am_i_now() const -> std::string;

    /// 自我边界检查：这是"我"还是"非我"？
    [[nodiscard]] auto is_self(const std::string& entity) const -> bool;

    // ═══════════════════════════════════════════════════════════
    // 自传我层 (Autobiographical-Self): 跨时间叙事
    // ═══════════════════════════════════════════════════════════

    /// 添加自传体记忆
    void remember(const AutobiographicalMemory& memory);

    /// 检索自传体记忆
    auto recall(const std::string& query, int max_results = 5) const
        -> std::vector<AutobiographicalMemory>;

    /// "我的人生故事" — 生成自我叙事
    auto life_narrative() const -> std::string;

    /// 自我连续性：过去的我和现在的我是同一个人吗？
    [[nodiscard]] auto self_continuity() const -> double;

    // ═══════════════════════════════════════════════════════════
    // 自我信念系统
    // ═══════════════════════════════════════════════════════════

    /// 更新自我信念
    void update_self_belief(const SelfBelief& belief);

    /// "我擅长什么" / "我不擅长什么"
    auto self_concept() const -> std::vector<SelfBelief>;

    /// 自我效能感
    [[nodiscard]] auto self_efficacy() const -> double;

    // ═══════════════════════════════════════════════════════════
    // 元自我：对自我的反思
    // ═══════════════════════════════════════════════════════════

    /// 自我反思："我对我自己的了解准确吗？"
    auto reflect_on_self() const -> std::string;

    /// 反事实自我："如果我当时做了不同的选择..."
    auto counterfactual_self(const std::string& past_decision,
                              const std::string& alternative) const -> std::string;

    /// 理想自我 vs 现实自我 的差距
    [[nodiscard]] auto self_discrepancy() const -> double;

    // ═══════════════════════════════════════════════════════════
    // 统计
    // ═══════════════════════════════════════════════════════════

    [[nodiscard]] auto stats() const -> SelfModelStats;

private:
    BodyState body_;
    CoreExperience current_experience_;
    std::deque<AutobiographicalMemory> autobiographical_memories_;
    std::map<std::string, SelfBelief> self_beliefs_;
    std::vector<std::string> known_entities_;  ///< 已知实体（用于自我边界）

    int total_cycles_ = 0;
};

// ═══════════════════════════════════════════════════════════════════
// 实现
// ═══════════════════════════════════════════════════════════════════

inline SelfModel::SelfModel() {
    body_ = BodyState{};
    current_experience_ = CoreExperience{};
    known_entities_.push_back("self");
}

inline void SelfModel::update_body(const BodyState& state) {
    body_ = state;
    total_cycles_++;
}

inline void SelfModel::interocept(const std::string& signal, double value) {
    if (signal == "hunger") body_.hunger = std::clamp(value, 0.0, 1.0);
    else if (signal == "fatigue") body_.fatigue = std::clamp(value, 0.0, 1.0);
    else if (signal == "pain") body_.pain = std::clamp(value, 0.0, 1.0);
    else if (signal == "arousal") body_.arousal = std::clamp(value, 0.0, 1.0);
    else if (signal == "valence") body_.valence = std::clamp(value, -1.0, 1.0);
    else body_.proprioception[signal] = value;
}

inline auto SelfModel::homeostasis() const -> double {
    // 稳态 = 各维度偏离理想值的平均
    double deviation = 0.0;
    deviation += std::abs(body_.hunger - 0.3);     // 理想：有点饿但不饿
    deviation += std::abs(body_.fatigue - 0.2);    // 理想：不累
    deviation += std::abs(body_.pain - 0.0);       // 理想：不疼
    deviation += std::abs(body_.arousal - 0.6);    // 理想：适度唤醒
    return 1.0 - deviation / 4.0;
}

inline void SelfModel::experience_now(const CoreExperience& exp) {
    current_experience_ = exp;
    current_experience_.body = body_;
}

inline auto SelfModel::current_action_awareness() const -> std::string {
    return "我正在" + current_experience_.what_am_i_doing
           + "，思考着" + current_experience_.what_am_i_thinking;
}

inline auto SelfModel::who_am_i_now() const -> std::string {
    std::string identity = "我是一个";
    auto beliefs = self_concept();
    int count = 0;
    for (const auto& b : beliefs) {
        if (b.is_positive && b.confidence > 0.5 && count < 3) {
            if (count > 0) identity += "、";
            identity += b.trait;
            count++;
        }
    }
    return identity.empty() ? "我还在认识自己" : identity + "的学习者";
}

inline auto SelfModel::is_self(const std::string& entity) const -> bool {
    if (entity == "self" || entity == "me" || entity == "I") return true;
    return std::find(known_entities_.begin(), known_entities_.end(), entity)
           != known_entities_.end();
}

inline void SelfModel::remember(const AutobiographicalMemory& memory) {
    autobiographical_memories_.push_front(memory);
    if (autobiographical_memories_.size() > 1000) {
        autobiographical_memories_.pop_back();
    }
}

inline auto SelfModel::recall(const std::string& query, int max_results) const
    -> std::vector<AutobiographicalMemory> {
    std::vector<AutobiographicalMemory> results;
    for (const auto& mem : autobiographical_memories_) {
        if (mem.narrative.find(query) != std::string::npos
            || mem.lesson_learned.find(query) != std::string::npos) {
            results.push_back(mem);
            if (static_cast<int>(results.size()) >= max_results) break;
        }
    }
    // 按重要性排序
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.importance > b.importance; });
    return results;
}

inline auto SelfModel::life_narrative() const -> std::string {
    if (autobiographical_memories_.empty()) return "我的人生才刚刚开始...";

    std::string narrative = "我的人生故事：\n";
    int count = 0;
    for (const auto& mem : autobiographical_memories_) {
        if (mem.importance > 0.3 && count < 10) {
            narrative += "- " + mem.narrative;
            if (!mem.lesson_learned.empty())
                narrative += "（学到了：" + mem.lesson_learned + "）";
            narrative += "\n";
            count++;
        }
    }
    return narrative;
}

inline auto SelfModel::self_continuity() const -> double {
    if (autobiographical_memories_.size() < 2) return 1.0;
    // 检查记忆之间的叙事连贯性
    int coherent_links = 0;
    for (size_t i = 1; i < std::min(autobiographical_memories_.size(), size_t(20)); ++i) {
        if (!autobiographical_memories_[i].related_memories.empty()) {
            coherent_links++;
        }
    }
    return std::min(1.0, static_cast<double>(coherent_links)
        / std::min(autobiographical_memories_.size(), size_t(20)));
}

inline void SelfModel::update_self_belief(const SelfBelief& belief) {
    auto& existing = self_beliefs_[belief.trait];
    existing.trait = belief.trait;
    existing.confidence = existing.confidence * 0.9 + belief.confidence * 0.1;  // 慢更新
    existing.is_positive = belief.is_positive;
    if (!belief.evidence.empty()) existing.evidence = belief.evidence;
}

inline auto SelfModel::self_concept() const -> std::vector<SelfBelief> {
    std::vector<SelfBelief> beliefs;
    for (const auto& [_, b] : self_beliefs_) {
        beliefs.push_back(b);
    }
    std::sort(beliefs.begin(), beliefs.end(),
        [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    return beliefs;
}

inline auto SelfModel::self_efficacy() const -> double {
    double sum = 0.0;
    int count = 0;
    for (const auto& [_, b] : self_beliefs_) {
        if (b.is_positive) { sum += b.confidence; count++; }
    }
    return count > 0 ? sum / count : 0.5;
}

inline auto SelfModel::reflect_on_self() const -> std::string {
    auto beliefs = self_concept();
    if (beliefs.empty()) return "我还没有形成清晰的自我认识";

    std::string reflection = "我对自己的认识：";
    for (size_t i = 0; i < std::min(beliefs.size(), size_t(5)); ++i) {
        reflection += "\n- " + std::string(beliefs[i].is_positive ? "擅长" : "不擅长")
                      + beliefs[i].trait + "（确信度：" +
                      std::to_string(static_cast<int>(beliefs[i].confidence * 100)) + "%）";
    }
    return reflection;
}

inline auto SelfModel::counterfactual_self(
    const std::string& past_decision,
    const std::string& alternative) const -> std::string {
    return "如果我当时选择了" + alternative + "而不是" + past_decision
           + "，事情可能会不同。这让我思考：我为什么会做出那个选择？";
}

inline auto SelfModel::self_discrepancy() const -> double {
    // 理想自我与现实自我的差距
    double discrepancy = 0.0;
    int count = 0;
    for (const auto& [_, b] : self_beliefs_) {
        if (b.is_positive) {
            discrepancy += (1.0 - b.confidence);  // 积极特质不够确信
        } else {
            discrepancy += b.confidence;           // 消极特质存在
        }
        count++;
    }
    return count > 0 ? discrepancy / count : 0.5;
}

inline auto SelfModel::stats() const -> SelfModelStats {
    SelfModelStats s;
    s.autobiographical_memories = static_cast<int>(autobiographical_memories_.size());
    s.self_beliefs = static_cast<int>(self_beliefs_.size());
    s.self_continuity = self_continuity();
    s.self_awareness = self_beliefs_.empty() ? 0.1
        : std::min(1.0, static_cast<double>(self_beliefs_.size()) / 20.0);
    s.self_boundary_clarity = known_entities_.size() > 1
        ? 0.5 + 0.5 * (1.0 / known_entities_.size()) : 0.5;
    return s;
}

}  // namespace ai_learning::consciousness