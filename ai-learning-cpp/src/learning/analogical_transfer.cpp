/**
 * @file analogical_transfer.cpp
 * @brief 跨领域类比迁移引擎实现
 */

#include "ai_learning/learning/analogical_transfer.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <random>
#include <set>

namespace ai_learning::learning {

// ── 构造 ────────────────────────────────────────────────────────

AnalogicalTransferEngine::AnalogicalTransferEngine(
    const AnalogicalConfig& config)
    : config_(config) {}

// ── 结构映射 ──────────────────────────────────────────────────

auto AnalogicalTransferEngine::describe_concept(
    const std::string& concept_id,
    const std::string& domain,
    const std::vector<std::string>& known_facts,
    const std::map<std::string, double>& features)
    -> ConceptDescriptor
{
    ConceptDescriptor desc;
    desc.id = concept_id;
    desc.domain = domain;
    desc.features = features;

    // 从事实中提取属性和关系
    for (const auto& fact : known_facts) {
        // 关系检测：如果包含因果/条件关键词
        if (fact.find("导致") != std::string::npos ||
            fact.find("引起") != std::string::npos ||
            fact.find("→") != std::string::npos ||
            fact.find("限制") != std::string::npos) {
            desc.relations.push_back(fact);
        }
        // 提取关键词作为属性
        std::istringstream iss(fact);
        std::string word;
        while (iss >> word) {
            if (word.size() >= 2) {
                desc.attributes.push_back(word);
            }
        }
    }

    // 将 feature keys 也作为属性
    for (const auto& [key, val] : features) {
        desc.attributes.push_back(key);
    }

    // 去重
    std::sort(desc.attributes.begin(), desc.attributes.end());
    desc.attributes.erase(
        std::unique(desc.attributes.begin(), desc.attributes.end()),
        desc.attributes.end());

    concept_registry_[concept_id] = desc;
    return desc;
}

auto AnalogicalTransferEngine::find_mapping(
    const ConceptDescriptor& source,
    const ConceptDescriptor& target)
    -> StructureMapping
{
    StructureMapping mapping;
    mapping.source_concept = source.id;
    mapping.target_concept = target.id;

    // 计算表面相似度
    mapping.surface_similarity = surface_similarity_(source, target);

    // 计算关系结构对齐
    mapping.relational_depth = relational_alignment_(source, target);

    // 属性映射
    mapping.attribute_map = match_attributes_(source.attributes, target.attributes);

    // 关系映射
    mapping.relation_map = match_attributes_(source.relations, target.relations);

    // 综合对齐分数
    mapping.alignment_score = mapping.surface_similarity * 0.4 +
                              mapping.relational_depth * 0.6;

    return mapping;
}

auto AnalogicalTransferEngine::find_domain_mapping(
    const std::vector<ConceptDescriptor>& source_domain,
    const std::vector<ConceptDescriptor>& target_domain)
    -> std::vector<StructureMapping>
{
    std::vector<StructureMapping> mappings;

    for (const auto& src : source_domain) {
        for (const auto& tgt : target_domain) {
            auto m = find_mapping(src, tgt);
            if (m.alignment_score >= config_.min_alignment_score) {
                mappings.push_back(std::move(m));
            }
        }
    }

    // 按对齐分数降序排列
    std::sort(mappings.begin(), mappings.end(),
              [](const StructureMapping& a, const StructureMapping& b) {
                  return a.alignment_score > b.alignment_score;
              });

    if (static_cast<int>(mappings.size()) > config_.max_mappings) {
        mappings.resize(config_.max_mappings);
    }

    return mappings;
}

// ── 知识迁移 ──────────────────────────────────────────────────

auto AnalogicalTransferEngine::transfer(
    const std::vector<ConceptDescriptor>& source_concepts,
    const std::vector<ConceptDescriptor>& target_concepts,
    const std::vector<std::string>& source_facts)
    -> TransferResult
{
    TransferResult result;

    if (!source_concepts.empty()) result.source_domain = source_concepts[0].domain;
    if (!target_concepts.empty()) result.target_domain = target_concepts[0].domain;

    // Step 1: 找到映射
    result.mappings = find_domain_mapping(source_concepts, target_concepts);

    // Step 2: 迁移知识
    for (const auto& fact : source_facts) {
        bool transferred = false;
        for (const auto& mapping : result.mappings) {
            auto mapped = map_knowledge(fact, mapping);
            if (mapped.has_value()) {
                if (config_.verify_transfers &&
                    !verify_transfer(*mapped, {})) {
                    result.failed_transfers.push_back(*mapped);
                } else {
                    result.transferred_knowledge.push_back(*mapped);
                }
                transferred = true;
                break;
            }
        }
        if (!transferred) {
            result.failed_transfers.push_back(fact);
        }
    }

    // Step 3: 评估迁移质量
    int total = static_cast<int>(result.transferred_knowledge.size() +
                                  result.failed_transfers.size());
    if (total > 0) {
        result.transfer_quality =
            static_cast<double>(result.transferred_knowledge.size()) / total;
    }

    // Step 4: 生成推理过程
    std::ostringstream oss;
    oss << "源领域 [" << result.source_domain << "] → 目标领域 ["
        << result.target_domain << "]\n";
    oss << "发现 " << result.mappings.size() << " 个结构映射\n";
    oss << "成功迁移 " << result.transferred_knowledge.size() << " 条知识\n";
    oss << "失败 " << result.failed_transfers.size() << " 条\n";
    oss << "迁移质量: " << result.transfer_quality;
    result.reasoning = oss.str();

    // 更新统计
    total_transfers_++;
    if (result.transfer_quality > 0.5) {
        successful_transfers_++;
    }

    return result;
}

auto AnalogicalTransferEngine::map_knowledge(
    const std::string& fact,
    const StructureMapping& mapping)
    -> std::optional<std::string>
{
    if (mapping.attribute_map.empty() && mapping.relation_map.empty()) {
        return std::nullopt;
    }

    std::string result = fact;

    // 按属性映射长度降序替换（避免短词覆盖长词）
    auto sorted_map = mapping.attribute_map;
    std::vector<std::pair<std::string, std::string>> map_vec(
        sorted_map.begin(), sorted_map.end());
    std::sort(map_vec.begin(), map_vec.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });

    for (const auto& [src, tgt] : map_vec) {
        auto pos = result.find(src);
        while (pos != std::string::npos) {
            result.replace(pos, src.size(), tgt);
            pos = result.find(src, pos + tgt.size());
        }
    }

    // 如果结果和原始一样（没有替换发生），返回 nullopt
    if (result == fact) {
        return std::nullopt;
    }

    return result;
}

// ── 验证与学习 ──────────────────────────────────────────────────

auto AnalogicalTransferEngine::verify_transfer(
    const std::string& transferred,
    const std::vector<std::string>& target_facts)
    -> bool
{
    // 简单验证：非空且包含替换内容
    // 在完整实现中，需要检查与目标领域知识的自洽性
    (void)target_facts;
    return !transferred.empty();
}

void AnalogicalTransferEngine::record_experience(
    const AnalogyExperience& experience)
{
    experiences_.push_back(experience);
}

auto AnalogicalTransferEngine::learn_from_failure(
    const TransferResult& failed)
    -> std::string
{
    std::string lesson = "领域 [" + failed.source_domain + "] → [" +
                         failed.target_domain + "] 迁移失败(";

    if (failed.mappings.empty()) {
        lesson += "无结构映射";
    } else {
        lesson += "结构差异过大";
    }

    lesson += ")。原因：";
    if (failed.transfer_quality < 0.2) {
        lesson += "两个领域缺乏共同的深层结构";
    } else {
        lesson += "表面相似但关系结构不匹配";
    }

    record_experience({failed.source_domain, failed.target_domain,
                       false, failed.transfer_quality, lesson});
    return lesson;
}

// ── 查询 ──────────────────────────────────────────────────────

auto AnalogicalTransferEngine::find_analogous_domain(
    const std::string& target_domain,
    const std::vector<std::string>& known_domains,
    const std::map<std::string, std::vector<ConceptDescriptor>>& domain_concepts)
    -> std::optional<std::string>
{
    (void)target_domain;  // 用概念描述来比较

    // 查找目标领域的概念
    std::vector<ConceptDescriptor> target_concepts;
    auto it = domain_concepts.find(target_domain);
    if (it != domain_concepts.end()) {
        target_concepts = it->second;
    }

    if (target_concepts.empty()) return std::nullopt;

    std::string best_domain;
    double best_score = 0.0;

    for (const auto& domain : known_domains) {
        if (domain == target_domain) continue;

        auto dit = domain_concepts.find(domain);
        if (dit == domain_concepts.end()) continue;

        double total_sim = 0.0;
        int count = 0;

        for (const auto& src : dit->second) {
            for (const auto& tgt : target_concepts) {
                total_sim += surface_similarity_(src, tgt);
                count++;
            }
        }

        double avg_sim = count > 0 ? total_sim / count : 0.0;
        if (avg_sim > best_score) {
            best_score = avg_sim;
            best_domain = domain;
        }
    }

    if (best_score > config_.min_surface_similarity) {
        return best_domain;
    }
    return std::nullopt;
}

auto AnalogicalTransferEngine::stats() const
    -> std::map<std::string, double>
{
    return {
        {"total_transfers", static_cast<double>(total_transfers_)},
        {"successful_transfers", static_cast<double>(successful_transfers_)},
        {"success_rate", total_transfers_ > 0
            ? static_cast<double>(successful_transfers_) / total_transfers_
            : 0.0},
        {"concept_registry_size", static_cast<double>(concept_registry_.size())},
        {"experience_count", static_cast<double>(experiences_.size())},
    };
}

// ── 内部算法 ──────────────────────────────────────────────────

auto AnalogicalTransferEngine::surface_similarity_(
    const ConceptDescriptor& a,
    const ConceptDescriptor& b) -> double
{
    if (a.attributes.empty() && b.attributes.empty()) return 0.0;

    // Jaccard 相似度
    std::set<std::string> set_a(a.attributes.begin(), a.attributes.end());
    std::set<std::string> set_b(b.attributes.begin(), b.attributes.end());

    int intersection = 0;
    for (const auto& attr : set_a) {
        if (set_b.count(attr)) intersection++;
    }

    int union_size = static_cast<int>(set_a.size() + set_b.size() - intersection);
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
}

auto AnalogicalTransferEngine::relational_alignment_(
    const ConceptDescriptor& a,
    const ConceptDescriptor& b) -> double
{
    // 如果双方都没有关系信息，不能因此惩罚 — 返回中性分 0.5
    if (a.relations.empty() && b.relations.empty()) return 0.5;

    // 如果只有一方有关系，返回中性分（不影响总分太多）
    if (a.relations.empty() || b.relations.empty()) return 0.3;

    // 关系结构对齐：匹配关系中的模式
    int matched = 0;
    for (const auto& rel_a : a.relations) {
        for (const auto& rel_b : b.relations) {
            // 简单的模式匹配：检查是否有相同的关系动词
            if (rel_a.find("导致") != std::string::npos &&
                rel_b.find("导致") != std::string::npos) {
                matched++;
            } else if (rel_a.find("限制") != std::string::npos &&
                       rel_b.find("限制") != std::string::npos) {
                matched++;
            } else if (rel_a.find("→") != std::string::npos &&
                       rel_b.find("→") != std::string::npos) {
                matched++;
            } else if (rel_a.find("增加") != std::string::npos &&
                       rel_b.find("增加") != std::string::npos) {
                matched++;
            } else if (rel_a.find("减少") != std::string::npos &&
                       rel_b.find("减少") != std::string::npos) {
                matched++;
            }
        }
    }

    int max_relations = std::max(
        static_cast<int>(a.relations.size()),
        static_cast<int>(b.relations.size()));
    return max_relations > 0 ? static_cast<double>(matched) / max_relations : 0.0;
}

auto AnalogicalTransferEngine::match_attributes_(
    const std::vector<std::string>& source,
    const std::vector<std::string>& target)
    -> std::map<std::string, std::string>
{
    std::map<std::string, std::string> mapping;

    // 精确匹配
    std::set<std::string> target_set(target.begin(), target.end());
    for (const auto& s : source) {
        if (target_set.count(s)) {
            mapping[s] = s;  // 自映射
        }
    }

    // 部分匹配：共享子串
    for (const auto& s : source) {
        if (mapping.count(s)) continue;
        for (const auto& t : target) {
            if (mapping.count(s)) break;
            // 检查共享子串（至少2个字符）
            for (size_t len = std::min(s.size(), t.size()); len >= 2; --len) {
                for (size_t i = 0; i + len <= s.size(); ++i) {
                    if (t.find(s.substr(i, len)) != std::string::npos) {
                        mapping[s] = t;
                        break;
                    }
                }
                if (mapping.count(s)) break;
            }
        }
    }

    return mapping;
}

auto AnalogicalTransferEngine::analogy_score_(
    const StructureMapping& mapping) const -> double
{
    return mapping.alignment_score * 0.5 +
           mapping.relational_depth * 0.3 +
           mapping.surface_similarity * 0.2;
}

}  // namespace ai_learning::learning
