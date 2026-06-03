/**
 * @file continual_learner.cpp
 * @brief 持续终身学习引擎实现
 */

#include "ai_learning/learning/continual_learner.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>

namespace ai_learning::learning {

// ── 构造 ────────────────────────────────────────────────────────

ContinualLearner::ContinualLearner(const ContinualConfig& config)
    : config_(config) {}

// ── 知识保护 ──────────────────────────────────────────────────

auto ContinualLearner::register_knowledge(
    const std::string& knowledge_id,
    const std::string& domain,
    double confidence,
    int usage_count)
    -> KnowledgeProtection
{
    KnowledgeProtection prot;
    prot.knowledge_id = knowledge_id;
    prot.original_confidence = confidence;
    prot.current_confidence = confidence;

    // Fisher 重要性 = f(confidence, usage_count)
    // 使用次数越多、置信度越高 → 重要性越高
    prot.importance = confidence * (1.0 + std::log1p(usage_count) / 5.0);
    prot.importance = std::min(prot.importance, 1.0);

    prot.level = determine_protection_level_(prot.importance);
    prot.times_reinforced = 0;

    protection_registry_[knowledge_id] = prot;
    domain_index_[domain].push_back(knowledge_id);

    return prot;
}

auto ContinualLearner::assess_impact(
    const std::string& new_knowledge,
    const std::string& domain)
    -> std::vector<KnowledgeConflict>
{
    std::vector<KnowledgeConflict> conflicts;

    auto it = domain_index_.find(domain);
    if (it == domain_index_.end()) return conflicts;

    for (const auto& old_id : it->second) {
        auto prot_it = protection_registry_.find(old_id);
        if (prot_it == protection_registry_.end()) continue;

        double score = conflict_score_(old_id, new_knowledge);
        if (score > 0.3) {
            KnowledgeConflict conflict;
            conflict.old_knowledge = old_id;
            conflict.new_knowledge = new_knowledge;
            conflict.conflict_score = score;
            conflict.resolution = "";
            conflict.reason = "同领域知识冲突，分数=" + std::to_string(score);
            conflicts.push_back(conflict);
            conflicts_detected_++;
        }
    }

    return conflicts;
}

auto ContinualLearner::resolve_conflict(const KnowledgeConflict& conflict)
    -> KnowledgeConflict
{
    KnowledgeConflict resolved = conflict;

    auto old_prot = protection_registry_.find(conflict.old_knowledge);

    if (old_prot == protection_registry_.end()) {
        resolved.resolution = "override";
        resolved.reason = "旧知识未受保护";
        conflicts_resolved_++;
        return resolved;
    }

    // 基于保护等级决定
    switch (old_prot->second.level) {
    case ProtectionLevel::kHigh:
        resolved.resolution = "keep_old";
        resolved.reason = "核心知识不可覆写（保护等级: High）";
        break;

    case ProtectionLevel::kMedium:
        resolved.resolution = "merge";
        resolved.reason = "中等保护知识，尝试合并新旧知识";
        break;

    case ProtectionLevel::kLow:
        resolved.resolution = "branch";
        resolved.reason = "低保护知识，创建分支保留两个版本";
        break;

    default:
        resolved.resolution = "override";
        resolved.reason = "无保护知识，允许覆写";
        break;
    }

    conflicts_resolved_++;
    return resolved;
}

void ContinualLearner::update_importance(
    const std::string& knowledge_id,
    double new_evidence)
{
    auto it = protection_registry_.find(knowledge_id);
    if (it == protection_registry_.end()) return;

    // Bayesian 更新：重要性随证据调整
    auto& prot = it->second;
    prot.importance = prot.importance * 0.8 + new_evidence * 0.2;
    prot.importance = std::clamp(prot.importance, 0.0, 1.0);
    prot.level = determine_protection_level_(prot.importance);
}

// ── 经验回放 ──────────────────────────────────────────────────

void ContinualLearner::add_replay_entry(const ReplayEntry& entry)
{
    replay_buffer_.push_back(entry);
    trim_replay_buffer_();
}

auto ContinualLearner::replay(int count) -> int
{
    auto samples = sample_replay(count);
    int replayed = static_cast<int>(samples.size());

    // 更新回放计数
    for (auto& entry : replay_buffer_) {
        for (const auto& sample : samples) {
            if (sample.observation == entry.observation) {
                entry.replay_count++;
            }
        }
    }

    total_replays_ += replayed;
    return replayed;
}

auto ContinualLearner::sample_replay(int count) const
    -> std::vector<ReplayEntry>
{
    if (replay_buffer_.empty()) return {};

    // 优先级采样：重要 + 久未回放
    std::vector<double> scores = priority_scores_();

    // 加权随机采样
    std::vector<ReplayEntry> samples;
    if (scores.empty()) return samples;

    double total = std::accumulate(scores.begin(), scores.end(), 0.0);
    if (total <= 0.0) return samples;

    // 使用确定性采样避免随机性在测试中造成不稳定
    // 选择分数最高的 top-k
    std::vector<size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

    int to_sample = std::min(count, static_cast<int>(replay_buffer_.size()));
    for (int i = 0; i < to_sample; ++i) {
        samples.push_back(replay_buffer_[indices[i]]);
    }

    return samples;
}

// ── 遗忘检测 ──────────────────────────────────────────────────

auto ContinualLearner::detect_forgetting() const
    -> std::vector<ForgettingAlert>
{
    std::vector<ForgettingAlert> alerts;

    for (const auto& [id, prot] : protection_registry_) {
        double degradation = prot.original_confidence - prot.current_confidence;

        if (degradation > config_.forgetting_threshold) {
            ForgettingAlert alert;
            alert.knowledge_id = id;
            alert.original_confidence = prot.original_confidence;
            alert.current_confidence = prot.current_confidence;
            alert.degradation = degradation;
            alert.protection = prot.level;

            if (prot.level == ProtectionLevel::kHigh) {
                alert.recommendation = "立即回放巩固（核心知识退化）";
            } else if (prot.level == ProtectionLevel::kMedium) {
                alert.recommendation = "安排回放复习";
            } else {
                alert.recommendation = "可选择性回顾";
            }

            alerts.push_back(alert);
        }
    }

    return alerts;
}

auto ContinualLearner::retention_rate() const -> double
{
    if (protection_registry_.empty()) return 1.0;

    double total_retention = 0.0;
    for (const auto& [id, prot] : protection_registry_) {
        double ratio = prot.original_confidence > 0.0
            ? prot.current_confidence / prot.original_confidence
            : 1.0;
        total_retention += std::clamp(ratio, 0.0, 1.0);
    }

    return total_retention / protection_registry_.size();
}

auto ContinualLearner::retention_rate(const std::string& domain) const
    -> double
{
    auto it = domain_index_.find(domain);
    if (it == domain_index_.end() || it->second.empty()) return 1.0;

    double total = 0.0;
    for (const auto& id : it->second) {
        auto pit = protection_registry_.find(id);
        if (pit != protection_registry_.end()) {
            double ratio = pit->second.original_confidence > 0.0
                ? pit->second.current_confidence / pit->second.original_confidence
                : 1.0;
            total += std::clamp(ratio, 0.0, 1.0);
        }
    }

    return total / it->second.size();
}

// ── 学习协调 ──────────────────────────────────────────────────

auto ContinualLearner::prepare_for_new_learning(
    const std::string& new_domain,
    const std::vector<std::string>& new_knowledge)
    -> std::vector<KnowledgeConflict>
{
    std::vector<KnowledgeConflict> all_conflicts;

    // 检查与同领域知识的冲突
    for (const auto& nk : new_knowledge) {
        auto conflicts = assess_impact(nk, new_domain);
        for (auto& c : conflicts) {
            auto resolved = resolve_conflict(c);
            all_conflicts.push_back(resolved);
        }
    }

    // 跨领域检查（仅高保护知识）
    for (const auto& [id, prot] : protection_registry_) {
        if (prot.level == ProtectionLevel::kHigh) {
            for (const auto& nk : new_knowledge) {
                double score = conflict_score_(id, nk);
                if (score > 0.5) {
                    KnowledgeConflict conflict;
                    conflict.old_knowledge = id;
                    conflict.new_knowledge = nk;
                    conflict.conflict_score = score;
                    conflict.resolution = "keep_old";
                    conflict.reason = "跨领域冲突，核心知识优先保护";
                    all_conflicts.push_back(conflict);
                }
            }
        }
    }

    return all_conflicts;
}

void ContinualLearner::post_learning_update(
    const std::string& learned_domain,
    double performance)
{
    // 根据新学习表现调整置信度
    auto it = domain_index_.find(learned_domain);
    if (it == domain_index_.end()) return;

    for (const auto& id : it->second) {
        auto pit = protection_registry_.find(id);
        if (pit == protection_registry_.end()) continue;

        auto& prot = pit->second;

        // 记录置信度历史
        confidence_history_[id].push_back(prot.current_confidence);

        // 如果新学习表现差，可能意味着旧知识被干扰
        if (performance < 0.3) {
            prot.current_confidence *= 0.95;  // 轻微退化
        }

        // 如果新学习表现好，巩固旧知识
        if (performance > 0.7) {
            prot.times_reinforced++;
            prot.current_confidence = std::min(
                prot.current_confidence * 1.02, 1.0);
        }
    }
}

// ── 查询 ──────────────────────────────────────────────────────

auto ContinualLearner::get_protection(const std::string& knowledge_id) const
    -> std::optional<KnowledgeProtection>
{
    auto it = protection_registry_.find(knowledge_id);
    if (it != protection_registry_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto ContinualLearner::protected_knowledge() const
    -> std::vector<KnowledgeProtection>
{
    std::vector<KnowledgeProtection> result;
    for (const auto& [id, prot] : protection_registry_) {
        if (prot.level != ProtectionLevel::kNone) {
            result.push_back(prot);
        }
    }
    return result;
}

auto ContinualLearner::stats() const -> ContinualLearningStats
{
    ContinualLearningStats s;
    s.total_knowledge_protected = static_cast<int>(
        protection_registry_.size());

    for (const auto& [id, prot] : protection_registry_) {
        if (prot.level != ProtectionLevel::kNone) {
            s.total_knowledge_protected++;
        }
    }

    s.total_replays = total_replays_;
    s.conflicts_detected = conflicts_detected_;
    s.conflicts_resolved = conflicts_resolved_;
    s.forgetting_alerts = forgetting_alerts_;
    s.avg_knowledge_retention = retention_rate();
    return s;
}

// ── 内部方法 ──────────────────────────────────────────────────

auto ContinualLearner::conflict_score_(
    const std::string& old_k,
    const std::string& new_k) const -> double
{
    // 简单的文本冲突检测：共同关键词越多 → 冲突越大
    // 在完整实现中需要语义分析
    std::set<std::string> old_words, new_words;

    // 按字符分割（简单分词）
    for (size_t i = 0; i + 1 < old_k.size(); ++i) {
        old_words.insert(old_k.substr(i, 2));
    }
    for (size_t i = 0; i + 1 < new_k.size(); ++i) {
        new_words.insert(new_k.substr(i, 2));
    }

    int intersection = 0;
    for (const auto& w : old_words) {
        if (new_words.count(w)) intersection++;
    }

    int union_size = static_cast<int>(old_words.size() + new_words.size() - intersection);
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
}

auto ContinualLearner::determine_protection_level_(double importance) const
    -> ProtectionLevel
{
    if (importance >= 0.8) return ProtectionLevel::kHigh;
    if (importance >= config_.importance_threshold) return ProtectionLevel::kMedium;
    if (importance >= 0.2) return ProtectionLevel::kLow;
    return ProtectionLevel::kNone;
}

auto ContinualLearner::priority_scores_() const -> std::vector<double>
{
    std::vector<double> scores;
    scores.reserve(replay_buffer_.size());

    for (const auto& entry : replay_buffer_) {
        // 优先级 = 重要性 × (1 + 时间衰减)
        double time_factor = 1.0 / (1.0 + entry.replay_count);
        scores.push_back(entry.importance * time_factor);
    }

    return scores;
}

void ContinualLearner::trim_replay_buffer_()
{
    while (static_cast<int>(replay_buffer_.size()) > config_.replay_buffer_size) {
        // 移除最不重要且已回放最多的条目
        auto min_it = std::min_element(
            replay_buffer_.begin(), replay_buffer_.end(),
            [](const ReplayEntry& a, const ReplayEntry& b) {
                double score_a = a.importance / (1.0 + a.replay_count);
                double score_b = b.importance / (1.0 + b.replay_count);
                return score_a < score_b;
            });
        replay_buffer_.erase(min_it);
    }
}

}  // namespace ai_learning::learning
