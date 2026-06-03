/**
 * @file abstract_concept.cpp
 * @brief 抽象概念形成引擎实现
 */

#include "ai_learning/learning/abstract_concept.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <set>

namespace ai_learning::learning {

// ── 构造 ────────────────────────────────────────────────────────

AbstractConceptEngine::AbstractConceptEngine(
    const AbstractionConfig& config)
    : config_(config) {}

// ── 概念形成 ──────────────────────────────────────────────────

auto AbstractConceptEngine::observe_instance(
    const std::string& instance_id,
    const std::vector<std::string>& attributes,
    const std::map<std::string, double>& features,
    const std::vector<std::string>& relations)
    -> ConceptFormationReport
{
    ConceptFormationReport report;

    // 存储实例
    instances_[instance_id] = {attributes, features, relations};
    report.instances_processed = 1;

    // 尝试将实例归入已有概念
    auto existing_concept = classify_instance(instance_id, attributes);

    if (existing_concept.has_value()) {
        // 加入已有概念
        auto& cpt = concepts_.at(*existing_concept);
        cpt.source_instances.push_back(instance_id);
        instance_to_concepts_[instance_id].insert(*existing_concept);

        // 更新核心属性
        cpt.core_attributes = extract_core_attributes_(
            cpt.source_instances);
        cpt.strength = compute_strength_(cpt);

        // 更新原型特征
        cpt.prototype_features = compute_prototype_features_(
            cpt.source_instances);

        // 检查是否需要分化
        auto diff = check_differentiation(*existing_concept);
        if (diff.has_value()) {
            report.differentiations.push_back(*diff);
        }
    } else {
        // 寻找可以一起形成概念的实例
        std::vector<std::string> candidates;
        for (const auto& [id, data] : instances_) {
            double overlap = attribute_overlap_(attributes, data.attributes);
            if (overlap >= config_.min_core_attribute_ratio) {
                candidates.push_back(id);
            }
        }

        // 如果有足够多的相似实例，形成新概念
        if (static_cast<int>(candidates.size()) >= config_.min_instances_for_concept) {
            auto new_concept = form_concept_from_instances(candidates);
            if (new_concept.has_value()) {
                report.new_concepts.push_back(*new_concept);
            }
        }
    }

    // 尝试发现泛化规则
    for (const auto& [id, cpt] : concepts_) {
        auto rule = discover_generalization(id);
        if (rule.has_value() &&
            rules_.find(rule->id) == rules_.end()) {
            report.new_rules.push_back(*rule);
            rules_[rule->id] = *rule;
        }
    }

    // 计算一致性
    report.coherence_score = coherence();

    return report;
}

auto AbstractConceptEngine::form_concept_from_instances(
    const std::vector<std::string>& instance_ids)
    -> std::optional<ConceptPrototype>
{
    // 验证所有实例都存在
    for (const auto& id : instance_ids) {
        if (instances_.find(id) == instances_.end()) return std::nullopt;
    }

    if (static_cast<int>(instance_ids.size()) < config_.min_instances_for_concept) {
        return std::nullopt;
    }

    ConceptPrototype cpt;
    cpt.id = next_id_();
    cpt.source_instances = instance_ids;

    // 提取核心属性
    cpt.core_attributes = extract_core_attributes_(instance_ids);

    if (cpt.core_attributes.empty()) return std::nullopt;

    // 提取可变属性（非核心属性）
    std::set<std::string> core_set(cpt.core_attributes.begin(),
                                    cpt.core_attributes.end());
    for (const auto& id : instance_ids) {
        for (const auto& attr : instances_.at(id).attributes) {
            if (!core_set.count(attr)) {
                cpt.variable_attributes.push_back(attr);
            }
        }
    }
    // 去重
    std::sort(cpt.variable_attributes.begin(), cpt.variable_attributes.end());
    cpt.variable_attributes.erase(
        std::unique(cpt.variable_attributes.begin(),
                    cpt.variable_attributes.end()),
        cpt.variable_attributes.end());

    // 计算原型特征
    cpt.prototype_features = compute_prototype_features_(instance_ids);

    // 提取关系结构
    std::set<std::string> all_relations;
    for (const auto& id : instance_ids) {
        for (const auto& rel : instances_.at(id).relations) {
            all_relations.insert(rel);
        }
    }
    cpt.relational_structure.assign(all_relations.begin(), all_relations.end());

    // 确定抽象层次
    cpt.level = AbstractionLevel::kBasic;

    // 生成名称
    cpt.name = generate_concept_name_(cpt.core_attributes);

    // 计算强度
    cpt.strength = compute_strength_(cpt);

    if (cpt.strength < config_.min_strength) {
        return std::nullopt;
    }

    // 注册
    concepts_[cpt.id] = cpt;

    // 更新实例→概念映射
    for (const auto& id : instance_ids) {
        instance_to_concepts_[id].insert(cpt.id);
    }

    return cpt;
}

auto AbstractConceptEngine::classify_instance(
    const std::string& instance_id,
    const std::vector<std::string>& attributes)
    -> std::optional<std::string>
{
    auto it = instances_.find(instance_id);
    if (it == instances_.end()) return std::nullopt;

    std::string best_concept;
    double best_overlap = 0.0;

    for (const auto& [cid, cpt] : concepts_) {
        double overlap = attribute_overlap_(
            attributes, cpt.core_attributes);
        if (overlap > config_.min_core_attribute_ratio &&
            overlap > best_overlap) {
            best_overlap = overlap;
            best_concept = cid;
        }
    }

    if (!best_concept.empty()) {
        return best_concept;
    }
    return std::nullopt;
}

// ── 泛化 ──────────────────────────────────────────────────────

auto AbstractConceptEngine::discover_generalization(
    const std::string& concept_id)
    -> std::optional<GeneralizationRule>
{
    auto it = concepts_.find(concept_id);
    if (it == concepts_.end()) return std::nullopt;

    const auto& cpt = it->second;
    if (cpt.source_instances.size() < 2) return std::nullopt;

    GeneralizationRule rule;
    rule.id = "rule_" + concept_id;

    // 构建具体模式（从第一个实例）
    const auto& first = instances_[cpt.source_instances[0]];
    std::ostringstream concrete_oss;
    for (const auto& attr : first.attributes) {
        concrete_oss << attr << " ";
    }
    rule.concrete_pattern = concrete_oss.str();

    // 构建抽象模式（使用核心属性）
    std::ostringstream abstract_oss;
    for (const auto& attr : cpt.core_attributes) {
        abstract_oss << attr << " ";
    }
    rule.abstract_pattern = abstract_oss.str();

    // 变量部分：具体→抽象的差异
    std::set<std::string> core_set(cpt.core_attributes.begin(),
                                    cpt.core_attributes.end());
    std::vector<std::string> variables;
    for (const auto& attr : first.attributes) {
        if (!core_set.count(attr)) {
            variables.push_back(attr);
        }
    }
    std::ostringstream var_oss;
    for (const auto& v : variables) {
        var_oss << v << " ";
    }
    rule.variable_part = var_oss.str();

    rule.support_count = static_cast<int>(cpt.source_instances.size());
    rule.confidence = cpt.strength;

    return rule;
}

auto AbstractConceptEngine::apply_generalization(
    const GeneralizationRule& rule,
    const std::string& new_instance)
    -> std::string
{
    // 简单应用：用抽象模式 + 新实例替换变量
    return rule.abstract_pattern + "+ " + new_instance + " 的特例";
}

auto AbstractConceptEngine::try_promote(
    const std::string& concept_id)
    -> std::optional<ConceptPrototype>
{
    auto it = concepts_.find(concept_id);
    if (it == concepts_.end()) return std::nullopt;

    auto& cpt = it->second;
    int current_level = static_cast<int>(cpt.level);

    if (current_level >= config_.max_abstraction_depth) {
        return std::nullopt;
    }

    // 检查是否有足够的子概念支撑上位概念
    if (cpt.strength > config_.min_strength &&
        cpt.source_instances.size() >=
            static_cast<size_t>(config_.min_instances_for_concept)) {
        cpt.level = static_cast<AbstractionLevel>(current_level + 1);
        return cpt;
    }

    return std::nullopt;
}

// ── 分化 ──────────────────────────────────────────────────────

auto AbstractConceptEngine::check_differentiation(
    const std::string& concept_id)
    -> std::optional<DifferentiationEvent>
{
    auto it = concepts_.find(concept_id);
    if (it == concepts_.end()) return std::nullopt;

    const auto& cpt = it->second;
    if (cpt.source_instances.size() < 4) return std::nullopt;

    // 计算实例间属性不一致性
    const auto& core_attrs = cpt.core_attributes;
    double inconsistency = 0.0;
    std::string diff_attr;

    for (const auto& attr : core_attrs) {
        int has_count = 0;
        for (const auto& inst_id : cpt.source_instances) {
            auto iit = instances_.find(inst_id);
            if (iit != instances_.end()) {
                const auto& inst_attrs = iit->second.attributes;
                if (std::find(inst_attrs.begin(), inst_attrs.end(), attr)
                    != inst_attrs.end()) {
                    has_count++;
                }
            }
        }

        double ratio = static_cast<double>(has_count) /
                        cpt.source_instances.size();
        double dev = std::abs(ratio - 0.5);  // 0.5 表示最分裂
        if (dev > inconsistency && ratio < 0.8 && ratio > 0.2) {
            inconsistency = dev;
            // 找到区分属性：可变属性中在部分实例出现的
        }
    }

    // 寻找真正的区分属性
    for (const auto& inst_id : cpt.source_instances) {
        auto iit = instances_.find(inst_id);
        if (iit == instances_.end()) continue;

        for (const auto& attr : iit->second.attributes) {
            // 检查这个属性是否只在一部分实例中出现
            int count = 0;
            for (const auto& other_id : cpt.source_instances) {
                auto oit = instances_.find(other_id);
                if (oit != instances_.end()) {
                    const auto& other_attrs = oit->second.attributes;
                    if (std::find(other_attrs.begin(), other_attrs.end(), attr)
                        != other_attrs.end()) {
                        count++;
                    }
                }
            }

            double ratio = static_cast<double>(count) /
                            cpt.source_instances.size();
            if (ratio > 0.2 && ratio < 0.8) {
                diff_attr = attr;
                inconsistency = std::max(inconsistency,
                    std::abs(ratio - 0.5));
                break;
            }
        }
        if (!diff_attr.empty()) break;
    }

    if (inconsistency > config_.differentiation_threshold && !diff_attr.empty()) {
        DifferentiationEvent event;
        event.original_concept = concept_id;
        event.differentiating_attribute = diff_attr;
        event.reason = "属性 [" + diff_attr + "] 仅在部分实例中出现";

        // 分成两组
        event.new_subconcepts = {
            concept_id + "_group_a",
            concept_id + "_group_b"
        };

        return event;
    }

    return std::nullopt;
}

auto AbstractConceptEngine::differentiate(
    const DifferentiationEvent& event)
    -> std::vector<ConceptPrototype>
{
    auto it = concepts_.find(event.original_concept);
    if (it == concepts_.end()) return {};

    const auto& original = it->second;

    // 将实例按区分属性分成两组
    std::vector<std::string> group_a, group_b;
    for (const auto& inst_id : original.source_instances) {
        auto iit = instances_.find(inst_id);
        if (iit == instances_.end()) continue;

        const auto& attrs = iit->second.attributes;
        if (std::find(attrs.begin(), attrs.end(),
                      event.differentiating_attribute) != attrs.end()) {
            group_a.push_back(inst_id);
        } else {
            group_b.push_back(inst_id);
        }
    }

    std::vector<ConceptPrototype> new_concepts;

    if (static_cast<int>(group_a.size()) >= config_.min_instances_for_concept) {
        auto concept_a = form_concept_from_instances(group_a);
        if (concept_a.has_value()) {
            concept_a->parent_concept = event.original_concept;
            concepts_[concept_a->id] = *concept_a;
            new_concepts.push_back(*concept_a);
        }
    }

    if (static_cast<int>(group_b.size()) >= config_.min_instances_for_concept) {
        auto concept_b = form_concept_from_instances(group_b);
        if (concept_b.has_value()) {
            concept_b->parent_concept = event.original_concept;
            concepts_[concept_b->id] = *concept_b;
            new_concepts.push_back(*concept_b);
        }
    }

    // 更新原始概念的子概念列表
    if (!new_concepts.empty()) {
        auto& orig = concepts_[event.original_concept];
        for (const auto& nc : new_concepts) {
            orig.child_concepts.push_back(nc.id);
        }
    }

    return new_concepts;
}

// ── 查询 ──────────────────────────────────────────────────────

auto AbstractConceptEngine::get_concept(const std::string& id) const
    -> std::optional<ConceptPrototype>
{
    auto it = concepts_.find(id);
    if (it != concepts_.end()) return it->second;
    return std::nullopt;
}

auto AbstractConceptEngine::concepts_by_level(AbstractionLevel level) const
    -> std::vector<ConceptPrototype>
{
    std::vector<ConceptPrototype> result;
    for (const auto& [id, cpt] : concepts_) {
        if (cpt.level == level) {
            result.push_back(cpt);
        }
    }
    return result;
}

auto AbstractConceptEngine::concept_hierarchy() const
    -> std::map<std::string, std::vector<std::string>>
{
    std::map<std::string, std::vector<std::string>> hierarchy;
    for (const auto& [id, cpt] : concepts_) {
        if (!cpt.child_concepts.empty()) {
            hierarchy[id] = cpt.child_concepts;
        }
    }
    return hierarchy;
}

auto AbstractConceptEngine::instance_concepts(
    const std::string& instance_id) const
    -> std::vector<std::string>
{
    auto it = instance_to_concepts_.find(instance_id);
    if (it != instance_to_concepts_.end()) {
        return {it->second.begin(), it->second.end()};
    }
    return {};
}

auto AbstractConceptEngine::coherence() const -> double
{
    if (concepts_.empty()) return 1.0;

    double total_coherence = 0.0;
    for (const auto& [id, cpt] : concepts_) {
        // 一致性 = 核心属性数 / 总属性数
        int total_attrs = static_cast<int>(
            cpt.core_attributes.size() +
            cpt.variable_attributes.size());
        if (total_attrs > 0) {
            total_coherence += static_cast<double>(
                cpt.core_attributes.size()) / total_attrs;
        } else {
            total_coherence += 1.0;
        }
    }

    return total_coherence / concepts_.size();
}

auto AbstractConceptEngine::stats() const -> std::map<std::string, double>
{
    int total_instances = 0;
    for (const auto& [id, data] : instances_) {
        (void)data;
        total_instances++;
    }

    return {
        {"total_concepts", static_cast<double>(concepts_.size())},
        {"total_instances", static_cast<double>(total_instances)},
        {"total_rules", static_cast<double>(rules_.size())},
        {"coherence", coherence()},
    };
}

// ── 内部方法 ──────────────────────────────────────────────────

auto AbstractConceptEngine::extract_core_attributes_(
    const std::vector<std::string>& instance_ids) const
    -> std::vector<std::string>
{
    if (instance_ids.empty()) return {};

    // 统计每个属性在多少实例中出现
    std::map<std::string, int> attr_counts;
    for (const auto& id : instance_ids) {
        auto it = instances_.find(id);
        if (it == instances_.end()) continue;

        // 去重每个实例的属性
        std::set<std::string> unique_attrs(
            it->second.attributes.begin(),
            it->second.attributes.end());
        for (const auto& attr : unique_attrs) {
            attr_counts[attr]++;
        }
    }

    // 核心属性 = 在 >= min_core_attribute_ratio 的实例中出现
    std::vector<std::string> core;
    double threshold = config_.min_core_attribute_ratio *
                        instance_ids.size();

    for (const auto& [attr, count] : attr_counts) {
        if (count >= threshold) {
            core.push_back(attr);
        }
    }

    return core;
}

auto AbstractConceptEngine::attribute_overlap_(
    const std::vector<std::string>& attrs_a,
    const std::vector<std::string>& attrs_b) const -> double
{
    if (attrs_a.empty() && attrs_b.empty()) return 1.0;
    if (attrs_a.empty() || attrs_b.empty()) return 0.0;

    std::set<std::string> set_a(attrs_a.begin(), attrs_a.end());
    std::set<std::string> set_b(attrs_b.begin(), attrs_b.end());

    int intersection = 0;
    for (const auto& a : set_a) {
        if (set_b.count(a)) intersection++;
    }

    int union_size = static_cast<int>(set_a.size() + set_b.size() - intersection);
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
}

auto AbstractConceptEngine::compute_prototype_features_(
    const std::vector<std::string>& instance_ids) const
    -> std::map<std::string, double>
{
    // 收集所有特征名
    std::set<std::string> all_keys;
    for (const auto& id : instance_ids) {
        auto it = instances_.find(id);
        if (it == instances_.end()) continue;
        for (const auto& [key, val] : it->second.features) {
            all_keys.insert(key);
        }
    }

    // 计算每个特征的平均值
    std::map<std::string, double> prototype;
    for (const auto& key : all_keys) {
        double sum = 0.0;
        int count = 0;
        for (const auto& id : instance_ids) {
            auto it = instances_.find(id);
            if (it == instances_.end()) continue;
            auto fit = it->second.features.find(key);
            if (fit != it->second.features.end()) {
                sum += fit->second;
                count++;
            }
        }
        if (count > 0) {
            prototype[key] = sum / count;
        }
    }

    return prototype;
}

auto AbstractConceptEngine::generate_concept_name_(
    const std::vector<std::string>& core_attrs) const -> std::string
{
    if (core_attrs.empty()) return "empty_concept";

    // 用核心属性组合生成名称
    std::string name;
    for (size_t i = 0; i < std::min(core_attrs.size(), size_t(3)); ++i) {
        if (i > 0) name += "_";
        name += core_attrs[i];
    }
    return name;
}

auto AbstractConceptEngine::compute_strength_(
    const ConceptPrototype& prototype) const -> double
{
    if (prototype.source_instances.empty()) return 0.0;

    // 强度 = 实例数 × 核心属性一致性 × 关系丰富度
    double instance_factor = std::min(
        static_cast<double>(prototype.source_instances.size()) / 10.0, 1.0);

    double attribute_consistency = 0.0;
    if (!prototype.source_instances.empty()) {
        auto core = extract_core_attributes_(prototype.source_instances);
        double total_attrs = 0.0;
        for (const auto& id : prototype.source_instances) {
            auto it = instances_.find(id);
            if (it != instances_.end()) {
                total_attrs += it->second.attributes.size();
            }
        }
        double avg_attrs = total_attrs / prototype.source_instances.size();
        attribute_consistency = avg_attrs > 0
            ? static_cast<double>(core.size()) / avg_attrs
            : 0.0;
    }

    double relation_factor = std::min(
        static_cast<double>(prototype.relational_structure.size()) / 3.0, 1.0);

    return instance_factor * 0.4 +
           std::min(attribute_consistency, 1.0) * 0.4 +
           relation_factor * 0.2;
}

auto AbstractConceptEngine::next_id_() -> std::string
{
    return "concept_" + std::to_string(next_concept_id_++);
}

}  // namespace ai_learning::learning
