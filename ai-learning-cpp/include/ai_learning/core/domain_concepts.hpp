/**
 * @file domain_concepts.hpp
 * @brief C++20 concepts — 领域接口的类型约束
 *
 * 替代 Python ABC 的纯虚类方案。
 * 使用 concepts 对模板参数进行编译期约束，实现 DIP（依赖倒置）。
 */
#pragma once

#include <concepts>
#include <string>
#include <vector>
#include <optional>
#include <map>

namespace ai_learning::core {

/// 可序列化为字符串字典
template<typename T>
concept Serializable = requires(const T& t) {
    { t.to_map() } -> std::same_as<std::map<std::string, std::string>>;
};

/// 可从字符串字典反序列化
template<typename T>
concept Deserializable = requires(const std::map<std::string, std::string>& m) {
    { T::from_map(m) } -> std::same_as<T>;
};

/// 值对象：不可变，可比较
template<typename T>
concept ValueObject = requires(const T& a, const T& b) {
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
};

/// 实体：有唯一 ID
template<typename T>
concept DomainEntity = requires(const T& e) {
    { e.id() } -> std::convertible_to<std::string>;
};

/// 仓储接口约束
template<typename R, typename E>
concept Repository = requires(R& repo, const E& entity, const std::string& id) {
    { repo.save(entity) } -> std::same_as<void>;
    { repo.find_by_id(id) } -> std::same_as<std::optional<E>>;
    { repo.remove(id) } -> std::same_as<bool>;
};

}  // namespace ai_learning::core
