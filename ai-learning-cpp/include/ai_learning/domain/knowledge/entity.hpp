/**
 * @file entity.hpp
 * @brief 知识实体 — 知识图谱中的节点（值对象）
 *
 * 代表一个具体的或抽象的概念：物体、属性、动作、事件等。
 * 不可变：所有修改返回新实例。
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain::knowledge {

class Entity {
public:
    Entity(std::string id, std::string type,
           std::map<std::string, std::string> properties = {},
           double confidence = 1.0,
           std::string source = "",
           std::vector<std::string> tags = {});

    // ── 访问器 ──
    [[nodiscard]] auto id() const -> const std::string& { return id_; }
    [[nodiscard]] auto type() const -> const std::string& { return type_; }
    [[nodiscard]] auto properties() const -> const std::map<std::string, std::string>& {
        return properties_;
    }
    [[nodiscard]] auto confidence() const -> double { return confidence_; }
    [[nodiscard]] auto source() const -> const std::string& { return source_; }
    [[nodiscard]] auto tags() const -> const std::vector<std::string>& {
        return tags_;
    }

    // ── 查询 ──
    [[nodiscard]] auto has_property(const std::string& key) const -> bool;
    [[nodiscard]] auto get_property(const std::string& key) const
        -> std::string;

    /// 检查实体是否匹配给定条件
    [[nodiscard]] auto matches(const std::map<std::string, std::string>& criteria) const
        -> bool;

    /// 检查是否包含标签
    [[nodiscard]] auto has_tag(const std::string& tag) const -> bool;

    // ── 修改（返回新实例） ──
    [[nodiscard]] auto with_confidence(double c) const -> Entity;
    [[nodiscard]] auto with_property(const std::string& key,
                                     const std::string& value) const -> Entity;
    [[nodiscard]] auto with_tag(const std::string& tag) const -> Entity;

    // ── 序列化 ──
    [[nodiscard]] auto to_map() const
        -> std::map<std::string, std::string>;

    static auto from_map(const std::map<std::string, std::string>& m)
        -> Entity;

    // ── 比较 ──
    bool operator==(const Entity& other) const { return id_ == other.id_; }
    bool operator!=(const Entity& other) const { return id_ != other.id_; }

private:
    std::string id_;
    std::string type_;
    std::map<std::string, std::string> properties_;
    double confidence_;
    std::string source_;
    std::vector<std::string> tags_;
};

}  // namespace ai_learning::domain::knowledge
