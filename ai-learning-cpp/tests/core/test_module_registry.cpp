/**
 * @file test_module_registry.cpp
 * @brief ModuleRegistry 单元测试 — TDD Red 阶段
 */

#include <catch2/catch_test_macros.hpp>
#include "ai_learning/core/module_registry.hpp"

using namespace ai_learning::core;

TEST_CASE("ModuleRegistry: 注册并获取模块") {
    ModuleRegistry reg;
    reg.register_factory("counter", []() -> std::any {
        return std::make_shared<int>(42);
    });

    SECTION("懒初始化：首次 get 时创建") {
        REQUIRE_FALSE(reg.has("counter"));
        auto& val = reg.get<std::shared_ptr<int>>("counter");
        REQUIRE(reg.has("counter"));
        REQUIRE(*val == 42);
    }

    SECTION("多次 get 返回同一实例") {
        auto& v1 = reg.get<std::shared_ptr<int>>("counter");
        auto& v2 = reg.get<std::shared_ptr<int>>("counter");
        REQUIRE(v1.get() == v2.get());  // 同一指针
    }
}

TEST_CASE("ModuleRegistry: 未注册模块抛异常") {
    ModuleRegistry reg;
    REQUIRE_THROWS_AS(
        reg.get<int>("nonexistent"),
        std::runtime_error
    );
}

TEST_CASE("ModuleRegistry: 直接设置实例") {
    ModuleRegistry reg;
    reg.set<int>("value", 100);
    REQUIRE(reg.has("value"));
    REQUIRE(reg.get<int>("value") == 100);
}

TEST_CASE("ModuleRegistry: has_factory 和 initialized_names") {
    ModuleRegistry reg;
    reg.register_factory("a", []() -> std::any { return 1; });
    reg.register_factory("b", []() -> std::any { return 2; });

    REQUIRE(reg.has_factory("a"));
    REQUIRE(reg.has_factory("b"));
    REQUIRE_FALSE(reg.has_factory("c"));

    REQUIRE(reg.initialized_names().empty());

    reg.get<int>("a");
    auto names = reg.initialized_names();
    REQUIRE(names.size() == 1);
    REQUIRE(names[0] == "a");
}

TEST_CASE("ModuleRegistry: clear_instances 保留工厂") {
    ModuleRegistry reg;
    reg.register_factory("x", []() -> std::any { return 99; });
    reg.get<int>("x");
    REQUIRE(reg.has("x"));

    reg.clear_instances();
    REQUIRE_FALSE(reg.has("x"));
    REQUIRE(reg.has_factory("x"));

    // 可以重新创建
    REQUIRE(reg.get<int>("x") == 99);
}
