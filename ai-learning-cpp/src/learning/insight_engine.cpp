/**
 * @file insight_engine.cpp
 * @brief 知识重组/顿悟引擎实现
 */

#include "ai_learning/learning/insight_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <set>

namespace ai_learning::learning {

InsightEngine::InsightEngine(const InsightConfig& config)
    : config_(config) {}

// ── 知识管理 ──────────────────────────────────────────────────

void InsightEngine::register_element(const KnowledgeElement& element)
{
    elements_[element.id] = element;
}

void InsightEngine::add_constraint(const MentalConstraint& constraint)
{
    constraints_[constraint.id] = constraint;
}

// ── 重组触发 ──────────────────────────────────────────────────

auto InsightEngine::try_insight(const std::string& problem_context)
    -> std::optional<InsightEvent>
{
    total_attempts_++;

    // 策略 1: 尝试远程联想（找最不相关的元素对）
    if (elements_.size() >= 2) {
        // 寻找关联分数最低（但非零）的元素对 — 最有可能产生顿悟
        std::string best_a, best_b;
        double lowest_assoc = 1.0;

        for (const auto& [id_a, _] : elements_) {
            for (const auto& [id_b, _] : elements_) {
                if (id_a >= id_b) continue;
                double score = association_score_(id_a, id_b);
                if (score > 0.0 && score < lowest_assoc) {
                    lowest_assoc = score;
                    best_a = id_a;
                    best_b = id_b;
                }
            }
        }

        if (!best_a.empty() && !best_b.empty()) {
            auto insight = remote_association(best_a, best_b);
            if (insight.has_value()) return insight;
        }
    }

    // 策略 2: 尝试释放最弱的约束
    for (const auto& [id, constraint] : constraints_) {
        if (constraint.is_active && constraint.strength < config_.constraint_release_threshold) {
            auto insight = release_constraint(id, problem_context);
            if (insight.has_value()) return insight;
        }
    }

    // 策略 3: 视角转换
    static const std::vector<std::string> viewpoints = {
        "相反视角", "微观视角", "宏观视角", "类比视角", "历史视角"
    };
    for (const auto& viewpoint : viewpoints) {
        auto insight = perspective_shift(problem_context, viewpoint);
        if (insight.has_value() && insight->creativity_score > config_.novelty_threshold) {
            return insight;
        }
    }

    return std::nullopt;
}

auto InsightEngine::remote_association(const std::string& element_a,
                                        const std::string& element_b)
    -> std::optional<InsightEvent>
{
    auto it_a = elements_.find(element_a);
    auto it_b = elements_.find(element_b);
    if (it_a == elements_.end() || it_b == elements_.end()) return std::nullopt;

    total_attempts_++;

    const auto& a = it_a->second;
    const auto& b = it_b->second;

    // 检查是否有共同特征（潜在的隐藏联系）
    std::set<std::string> features_a, features_b;
    for (const auto& [k, v] : a.features) features_a.insert(k);
    for (const auto& [k, v] : b.features) features_b.insert(k);

    std::vector<std::string> shared_features;
    for (const auto& f : features_a) {
        if (features_b.count(f)) shared_features.push_back(f);
    }

    // 检查连接的重叠
    std::set<std::string> connections_a(a.connections.begin(), a.connections.end());
    std::set<std::string> connections_b(b.connections.begin(), b.connections.end());
    std::vector<std::string> shared_connections;
    for (const auto& c : connections_a) {
        if (connections_b.count(c)) shared_connections.push_back(c);
    }

    // 如果没有共同点，通过特征值的互补性发现联系
    double complementarity = 0.0;
    for (const auto& [key, val_a] : a.features) {
        auto it = b.features.find(key);
        if (it != b.features.end()) {
            // 互补性：一个高一个低
            complementarity += std::abs(val_a - it->second);
        }
    }

    // 构建顿悟事件
    InsightEvent insight;
    insight.id = next_id_();
    insight.trigger = "remote_association";
    insight.elements_combined = {element_a, element_b};

    std::ostringstream old_oss;
    old_oss << "[" << a.content << "] 与 [" << b.content << "] 无关";
    insight.old_perspective = old_oss.str();

    std::ostringstream new_oss;
    new_oss << "[" << a.content << "] 和 [" << b.content << "] 通过";
    if (!shared_features.empty()) {
        new_oss << " 共同特征 ";
        for (const auto& f : shared_features) new_oss << f << " ";
    } else {
        new_oss << " 互补性 ";
    }
    new_oss << "关联";
    insight.new_perspective = new_oss.str();

    insight.surprise_level = 1.0 - association_score_(element_a, element_b);
    insight.confidence = (shared_features.size() * 0.2 +
                          shared_connections.size() * 0.15 +
                          complementarity * 0.1);
    insight.confidence = std::min(insight.confidence, 1.0);

    // 评估创造性
    auto creativity = assess_creativity(insight);
    insight.creativity_score = creativity.novelty;
    insight.utility_score = creativity.utility;

    if (insight.confidence >= config_.confidence_threshold) {
        insight.verified = verify_insight(insight);
        insights_.push_back(insight);
        successful_insights_++;
        return insight;
    }

    return std::nullopt;
}

auto InsightEngine::release_constraint(const std::string& constraint_id,
                                        const std::string& problem)
    -> std::optional<InsightEvent>
{
    auto it = constraints_.find(constraint_id);
    if (it == constraints_.end()) return std::nullopt;

    total_attempts_++;

    auto& constraint = it->second;
    if (!constraint.is_active) return std::nullopt;

    // 释放约束
    constraint.is_active = false;
    released_constraints_.push_back(constraint_id);
    constraints_released_++;

    // 构建顿悟事件
    InsightEvent insight;
    insight.id = next_id_();
    insight.trigger = "constraint_release";
    insight.constraint_released = constraint_id;

    insight.old_perspective = "受约束: " + constraint.description;
    insight.new_perspective = "释放约束 [" + constraint.description +
                              "] 后，问题 [" + problem + "] 有了新视角";

    // 计算释放约束的收益
    insight.surprise_level = constraint.strength * 0.8;
    insight.confidence = 0.5 + (1.0 - constraint.strength) * 0.3;
    insight.creativity_score = constraint.strength;
    insight.utility_score = 0.5;

    auto creativity = assess_creativity(insight);
    insight.creativity_score = creativity.novelty;
    insight.utility_score = creativity.utility;

    insight.verified = verify_insight(insight);
    insights_.push_back(insight);
    successful_insights_++;

    return insight;
}

auto InsightEngine::perspective_shift(const std::string& problem,
                                       const std::string& new_viewpoint)
    -> std::optional<InsightEvent>
{
    if (elements_.empty()) return std::nullopt;

    total_attempts_++;

    InsightEvent insight;
    insight.id = next_id_();
    insight.trigger = "perspective_shift";

    // 收集与问题相关的元素
    std::vector<std::string> relevant;
    for (const auto& [id, elem] : elements_) {
        if (problem.find(elem.domain) != std::string::npos ||
            elem.content.find(problem) != std::string::npos) {
            relevant.push_back(id);
        }
    }

    if (relevant.empty()) {
        // 取所有元素的子集
        int count = std::min(3, static_cast<int>(elements_.size()));
        int i = 0;
        for (const auto& [id, _] : elements_) {
            relevant.push_back(id);
            if (++i >= count) break;
        }
    }

    insight.elements_combined = relevant;
    insight.old_perspective = "原视角: " + problem;
    insight.new_perspective = new_viewpoint + " — " + problem;

    // 计算新颖度
    insight.surprise_level = novelty_score_(insight);
    insight.confidence = 0.4;
    insight.creativity_score = insight.surprise_level;
    insight.utility_score = 0.3;

    // 只有足够新颖的视角转换才作为顿悟
    if (insight.creativity_score >= config_.novelty_threshold * 0.5) {
        insight.verified = verify_insight(insight);
        insights_.push_back(insight);
        successful_insights_++;
        return insight;
    }

    return std::nullopt;
}

// ── 重组方案 ──────────────────────────────────────────────────

auto InsightEngine::propose_reorganization(const std::string& domain)
    -> std::vector<ReorganizationPlan>
{
    std::vector<ReorganizationPlan> plans;

    // Plan 1: 同领域元素重新排列
    std::vector<std::string> domain_elements;
    for (const auto& [id, elem] : elements_) {
        if (elem.domain == domain) domain_elements.push_back(id);
    }

    if (domain_elements.size() >= 2) {
        ReorganizationPlan plan;
        plan.elements_to_rearrange = domain_elements;
        plan.new_structure = domain + " 元素的重新排列";
        plan.method = "constraint_release";
        plan.estimated_novelty = 0.5;
        plans.push_back(plan);
    }

    // Plan 2: 跨领域联想
    std::set<std::string> domains;
    for (const auto& [id, elem] : elements_) {
        if (elem.domain != domain) domains.insert(elem.domain);
    }

    for (const auto& other_domain : domains) {
        ReorganizationPlan plan;
        plan.elements_to_rearrange = {domain, other_domain};
        plan.new_structure = domain + " + " + other_domain + " 跨域关联";
        plan.method = "remote_association";
        plan.estimated_novelty = 0.7;
        plans.push_back(plan);
    }

    return plans;
}

auto InsightEngine::execute_reorganization(const ReorganizationPlan& plan)
    -> InsightEvent
{
    total_attempts_++;

    InsightEvent insight;
    insight.id = next_id_();
    insight.trigger = plan.method;
    insight.elements_combined = plan.elements_to_rearrange;
    insight.old_perspective = "原始结构";
    insight.new_perspective = plan.new_structure;
    insight.confidence = 0.5;
    insight.creativity_score = plan.estimated_novelty;
    insight.surprise_level = plan.estimated_novelty;

    auto creativity = assess_creativity(insight);
    insight.creativity_score = creativity.novelty;
    insight.utility_score = creativity.utility;
    insight.verified = verify_insight(insight);

    insights_.push_back(insight);
    successful_insights_++;

    return insight;
}

// ── 验证与评估 ──────────────────────────────────────────────────

auto InsightEngine::verify_insight(const InsightEvent& insight) -> bool
{
    // 简单验证：置信度超过阈值 + 有实质内容
    return insight.confidence >= config_.confidence_threshold &&
           !insight.new_perspective.empty() &&
           insight.new_perspective != insight.old_perspective;
}

auto InsightEngine::assess_creativity(const InsightEvent& insight) const
    -> CreativityAssessment
{
    CreativityAssessment assessment;

    // 新颖度：与已有顿悟的差异
    assessment.novelty = novelty_score_(insight);

    // 惊讶度
    assessment.surprise = surprise_score_(insight);

    // 实用性：置信度 × 0.5 + 元素数 × 0.1
    assessment.utility = insight.confidence * 0.5 +
                         std::min(insight.elements_combined.size() * 0.1, 0.5);

    // 优雅度：简洁性
    assessment.elegance = insight.new_perspective.size() < 100 ? 0.8 : 0.4;

    // 综合
    assessment.overall = assessment.novelty * 0.3 +
                         assessment.utility * 0.3 +
                         assessment.surprise * 0.2 +
                         assessment.elegance * 0.2;

    // 判定
    if (assessment.overall > 0.7) {
        assessment.verdict = "breakthrough";
    } else if (assessment.overall > 0.4) {
        assessment.verdict = "incremental";
    } else {
        assessment.verdict = "trivial";
    }

    return assessment;
}

// ── 查询 ──────────────────────────────────────────────────────

auto InsightEngine::stats() const -> std::map<std::string, double>
{
    return {
        {"total_attempts", static_cast<double>(total_attempts_)},
        {"successful_insights", static_cast<double>(successful_insights_)},
        {"constraints_released", static_cast<double>(constraints_released_)},
        {"elements_count", static_cast<double>(elements_.size())},
        {"constraints_count", static_cast<double>(constraints_.size())},
        {"insight_rate", total_attempts_ > 0
            ? static_cast<double>(successful_insights_) / total_attempts_ : 0.0},
    };
}

// ── 内部方法 ──────────────────────────────────────────────────

auto InsightEngine::association_score_(const std::string& id_a,
                                        const std::string& id_b) const
    -> double
{
    auto it_a = elements_.find(id_a);
    auto it_b = elements_.find(id_b);
    if (it_a == elements_.end() || it_b == elements_.end()) return 0.0;

    const auto& a = it_a->second;
    const auto& b = it_b->second;

    // 直接连接
    for (const auto& conn : a.connections) {
        if (conn == id_b) return 1.0;
    }

    // 共同连接（通过中间节点）
    std::set<std::string> conns_a(a.connections.begin(), a.connections.end());
    int shared = 0;
    for (const auto& c : b.connections) {
        if (conns_a.count(c)) shared++;
    }

    // 特征相似度
    double feature_sim = 0.0;
    int feature_count = 0;
    for (const auto& [key, val_a] : a.features) {
        auto it = b.features.find(key);
        if (it != b.features.end()) {
            double diff = std::abs(val_a - it->second);
            feature_sim += 1.0 - std::min(diff, 1.0);
            feature_count++;
        }
    }
    if (feature_count > 0) feature_sim /= feature_count;

    // 同领域加分
    double domain_bonus = (a.domain == b.domain) ? 0.3 : 0.0;

    return std::min(shared * 0.2 + feature_sim * 0.3 + domain_bonus, 1.0);
}

auto InsightEngine::novelty_score_(const InsightEvent& insight) const
    -> double
{
    // 与已有顿悟比较
    if (insights_.empty()) return 0.8;  // 第一个顿悟天然新颖

    double max_similarity = 0.0;
    for (const auto& past : insights_) {
        // 计算元素重叠
        std::set<std::string> current_set(
            insight.elements_combined.begin(),
            insight.elements_combined.end());
        std::set<std::string> past_set(
            past.elements_combined.begin(),
            past.elements_combined.end());

        int overlap = 0;
        for (const auto& e : current_set) {
            if (past_set.count(e)) overlap++;
        }

        int union_size = static_cast<int>(
            current_set.size() + past_set.size() - overlap);
        double similarity = union_size > 0
            ? static_cast<double>(overlap) / union_size : 0.0;

        max_similarity = std::max(max_similarity, similarity);
    }

    return 1.0 - max_similarity;
}

auto InsightEngine::surprise_score_(const InsightEvent& insight) const
    -> double
{
    // 基于元素之间的距离（越远越惊讶）
    if (insight.elements_combined.size() < 2) return 0.3;

    double total_assoc = 0.0;
    int pairs = 0;
    for (size_t i = 0; i < insight.elements_combined.size(); ++i) {
        for (size_t j = i + 1; j < insight.elements_combined.size(); ++j) {
            total_assoc += association_score_(
                insight.elements_combined[i],
                insight.elements_combined[j]);
            pairs++;
        }
    }

    // 低关联度 = 高惊讶度
    return pairs > 0 ? 1.0 - total_assoc / pairs : 0.5;
}

auto InsightEngine::next_id_() -> std::string
{
    return "insight_" + std::to_string(next_insight_id_++);
}

}  // namespace ai_learning::learning
