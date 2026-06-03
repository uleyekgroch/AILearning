/**
 * @file relation.cpp
 * @brief 知识关系实现
 */

#include "ai_learning/domain/knowledge/relation.hpp"

#include <algorithm>

namespace ai_learning::domain::knowledge {

Relation::Relation(std::string source_id, std::string target_id,
                   std::string type, double confidence,
                   int evidence_count,
                   std::map<std::string, std::string> metadata)
    : source_id_(std::move(source_id)),
      target_id_(std::move(target_id)),
      type_(std::move(type)),
      confidence_(std::max(0.0, std::min(1.0, confidence))),
      evidence_count_(evidence_count),
      metadata_(std::move(metadata)) {}

auto Relation::strengthen(double delta) const -> Relation {
    return Relation(source_id_, target_id_, type_,
                    std::min(1.0, confidence_ + delta),
                    evidence_count_ + 1, metadata_);
}

auto Relation::weaken(double delta) const -> Relation {
    return Relation(source_id_, target_id_, type_,
                    std::max(0.0, confidence_ - delta),
                    evidence_count_, metadata_);
}

auto Relation::to_map() const -> std::map<std::string, std::string> {
    auto result = metadata_;
    result["_source_id"]     = source_id_;
    result["_target_id"]     = target_id_;
    result["_type"]          = type_;
    result["_confidence"]    = std::to_string(confidence_);
    result["_evidence_count"] = std::to_string(evidence_count_);
    return result;
}

auto Relation::from_map(const std::map<std::string, std::string>& m)
    -> Relation {
    return Relation(
        m.count("_source_id") ? m.at("_source_id") : "",
        m.count("_target_id") ? m.at("_target_id") : "",
        m.count("_type") ? m.at("_type") : "",
        m.count("_confidence") ? std::stod(m.at("_confidence")) : 1.0,
        m.count("_evidence_count")
            ? std::stoi(m.at("_evidence_count")) : 0,
        {}  // metadata（简化，不反序列化内部 metadata）
    );
}

}  // namespace ai_learning::domain::knowledge
