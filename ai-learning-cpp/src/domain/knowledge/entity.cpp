/**
 * @file entity.cpp
 * @brief 知识实体实现
 */

#include "ai_learning/domain/knowledge/entity.hpp"

#include <sstream>

namespace ai_learning::domain::knowledge {

Entity::Entity(std::string id, std::string type,
               std::map<std::string, std::string> properties,
               double confidence,
               std::string source,
               std::vector<std::string> tags)
    : id_(std::move(id)),
      type_(std::move(type)),
      properties_(std::move(properties)),
      confidence_(confidence),
      source_(std::move(source)),
      tags_(std::move(tags)) {}

auto Entity::has_property(const std::string& key) const -> bool {
    return properties_.contains(key);
}

auto Entity::get_property(const std::string& key) const -> std::string {
    auto it = properties_.find(key);
    return it != properties_.end() ? it->second : "";
}

auto Entity::matches(
    const std::map<std::string, std::string>& criteria) const -> bool {
    for (const auto& [key, value] : criteria) {
        if (key == "type" && type_ != value) return false;
        if (key == "tag") {
            bool found = false;
            for (const auto& t : tags_) {
                if (t == value) { found = true; break; }
            }
            if (!found) return false;
        }
        auto it = properties_.find(key);
        if (it != properties_.end() && it->second != value) return false;
    }
    return true;
}

auto Entity::has_tag(const std::string& tag) const -> bool {
    for (const auto& t : tags_) {
        if (t == tag) return true;
    }
    return false;
}

auto Entity::with_confidence(double c) const -> Entity {
    return Entity(id_, type_, properties_, c, source_, tags_);
}

auto Entity::with_property(const std::string& key,
                           const std::string& value) const -> Entity {
    auto props = properties_;
    props[key] = value;
    return Entity(id_, type_, std::move(props), confidence_, source_, tags_);
}

auto Entity::with_tag(const std::string& tag) const -> Entity {
    auto new_tags = tags_;
    new_tags.push_back(tag);
    return Entity(id_, type_, properties_, confidence_, source_,
                  std::move(new_tags));
}

auto Entity::to_map() const -> std::map<std::string, std::string> {
    auto result = properties_;
    result["_id"]         = id_;
    result["_type"]       = type_;
    result["_confidence"] = std::to_string(confidence_);
    result["_source"]     = source_;
    std::string tag_str;
    for (size_t i = 0; i < tags_.size(); ++i) {
        if (i > 0) tag_str += ",";
        tag_str += tags_[i];
    }
    result["_tags"] = tag_str;
    return result;
}

auto Entity::from_map(const std::map<std::string, std::string>& m)
    -> Entity {
    std::string id       = m.count("_id") ? m.at("_id") : "";
    std::string type     = m.count("_type") ? m.at("_type") : "";
    double confidence    = m.count("_confidence")
                               ? std::stod(m.at("_confidence")) : 1.0;
    std::string source   = m.count("_source") ? m.at("_source") : "";
    std::string tag_str  = m.count("_tags") ? m.at("_tags") : "";

    std::vector<std::string> tags;
    if (!tag_str.empty()) {
        std::istringstream iss(tag_str);
        std::string tag;
        while (std::getline(iss, tag, ',')) {
            tags.push_back(tag);
        }
    }

    auto props = m;
    props.erase("_id");
    props.erase("_type");
    props.erase("_confidence");
    props.erase("_source");
    props.erase("_tags");

    return Entity(std::move(id), std::move(type), std::move(props),
                  confidence, std::move(source), std::move(tags));
}

}  // namespace ai_learning::domain::knowledge
