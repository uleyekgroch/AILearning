/**
 * @file relation.hpp
 * @brief 知识关系 — 知识图谱中的边（值对象）
 *
 * 连接两个实体，表达它们之间的语义关系。
 * 不可变：strengthen/weaken 返回新实例。
 */
#pragma once

#include <map>
#include <string>

namespace ai_learning::domain::knowledge {

// ── 关系类型常量 ────────────────────────────────────────────────
inline constexpr const char* kRelIsA         = "is_a";
inline constexpr const char* kRelHasProperty = "has_property";
inline constexpr const char* kRelPartOf      = "part_of";
inline constexpr const char* kRelCauses      = "causes";
inline constexpr const char* kRelSimilarTo   = "similar_to";
inline constexpr const char* kRelCollocates  = "collocates_with";
inline constexpr const char* kRelRelatedTo   = "related_to";

class Relation {
public:
    Relation(std::string source_id, std::string target_id,
             std::string type, double confidence = 1.0,
             int evidence_count = 0,
             std::map<std::string, std::string> metadata = {});

    // ── 访问器 ──
    [[nodiscard]] auto source_id() const -> const std::string& {
        return source_id_;
    }
    [[nodiscard]] auto target_id() const -> const std::string& {
        return target_id_;
    }
    [[nodiscard]] auto type() const -> const std::string& { return type_; }
    [[nodiscard]] auto confidence() const -> double { return confidence_; }
    [[nodiscard]] auto evidence_count() const -> int { return evidence_count_; }
    [[nodiscard]] auto metadata() const -> const std::map<std::string, std::string>& {
        return metadata_;
    }

    // ── 修改（返回新实例） ──
    [[nodiscard]] auto strengthen(double delta = 0.1) const -> Relation;
    [[nodiscard]] auto weaken(double delta = 0.1) const -> Relation;

    // ── 序列化 ──
    [[nodiscard]] auto to_map() const
        -> std::map<std::string, std::string>;

    static auto from_map(const std::map<std::string, std::string>& m)
        -> Relation;

    // ── 比较 ──
    bool operator==(const Relation& other) const {
        return source_id_ == other.source_id_ &&
               target_id_ == other.target_id_ &&
               type_ == other.type_;
    }
    bool operator!=(const Relation& other) const {
        return !(*this == other);
    }

private:
    std::string source_id_;
    std::string target_id_;
    std::string type_;
    double      confidence_;
    int         evidence_count_;
    std::map<std::string, std::string> metadata_;
};

}  // namespace ai_learning::domain::knowledge
