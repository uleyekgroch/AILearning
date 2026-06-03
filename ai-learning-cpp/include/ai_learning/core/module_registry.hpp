/**
 * @file module_registry.hpp
 * @brief 懒初始化模块注册表 — 替代 Learner 中的 N 个 _module = None
 *
 * DDD 基础设施：提供懒初始化的依赖注入容器。
 * 对应 Python 版 ModuleRegistry。
 */
#pragma once

#include <any>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace ai_learning::core {

/// 懒初始化模块注册表
///
/// 注册工厂函数，首次 get() 时才创建实例。
/// 线程安全性：非线程安全，初始化阶段使用。
class ModuleRegistry {
public:
    /// 注册模块工厂
    void register_factory(const std::string& name,
                          std::function<std::any()> factory) {
        factories_[name] = std::move(factory);
    }

    /// 获取模块（懒初始化，首次访问时创建）
    template<typename T>
    auto get(const std::string& name) -> T& {
        if (!instances_.contains(name)) {
            if (!factories_.contains(name)) {
                throw std::runtime_error(
                    "Module '" + name + "' not registered");
            }
            instances_[name] = factories_[name]();
        }
        return std::any_cast<T&>(instances_[name]);
    }

    /// 获取模块（const 版本）
    template<typename T>
    auto get(const std::string& name) const -> const T& {
        auto it = instances_.find(name);
        if (it == instances_.end()) {
            throw std::runtime_error(
                "Module '" + name + "' not initialized");
        }
        return std::any_cast<const T&>(it->second);
    }

    /// 直接设置模块实例（用于 load 恢复）
    template<typename T>
    void set(const std::string& name, T instance) {
        instances_[name] = std::move(instance);
    }

    /// 模块是否已初始化
    [[nodiscard]] auto has(const std::string& name) const -> bool {
        return instances_.contains(name);
    }

    /// 模块是否已注册
    [[nodiscard]] auto has_factory(const std::string& name) const -> bool {
        return factories_.contains(name);
    }

    /// 获取所有已初始化模块名称
    [[nodiscard]] auto initialized_names() const -> std::vector<std::string> {
        auto names = std::vector<std::string>{};
        names.reserve(instances_.size());
        for (const auto& [name, _] : instances_) {
            names.push_back(name);
        }
        return names;
    }

    /// 清除所有实例（保留工厂）
    void clear_instances() {
        instances_.clear();
    }

private:
    std::map<std::string, std::function<std::any()>> factories_;
    std::map<std::string, std::any> instances_;
};

}  // namespace ai_learning::core
