/**
 * @file test_hippocampal_consolidation.cpp
 * @brief 海马记忆 + 睡眠巩固 测试
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "ai_learning/learning/hippocampal.hpp"
#include "ai_learning/learning/consolidation.hpp"

using namespace ai_learning::learning;
using Catch::Approx;

// ── 海马记忆测试 ─────────────────────────────────────────────

TEST_CASE("HippocampalMemory: 单次编码", "[hippocampal]") {
    HippocampalMemory mem(100);
    int id = mem.encode({"猫", "动物"}, {"猫是动物"}, "动物百科");
    REQUIRE(id >= 0);
    REQUIRE(mem.size() == 1);
}

TEST_CASE("HippocampalMemory: 按实体检索", "[hippocampal]") {
    HippocampalMemory mem(100);
    (void)mem.encode({"猫", "动物"}, {"猫是动物"}, "百科");
    (void)mem.encode({"猫", "宠物"}, {"猫是宠物"}, "百科");
    (void)mem.encode({"狗", "动物"}, {"狗是动物"}, "百科");

    auto results = mem.recall("猫", 5);
    REQUIRE(results.size() == 2);
    for (const auto& r : results) {
        bool has_cat = false;
        for (const auto& e : r.entities) {
            if (e == "猫") has_cat = true;
        }
        REQUIRE(has_cat);
    }
}

TEST_CASE("HippocampalMemory: 容量限制", "[hippocampal]") {
    HippocampalMemory mem(3);
    (void)mem.encode({"a"}, {}, "c1");
    (void)mem.encode({"b"}, {}, "c2");
    (void)mem.encode({"c"}, {}, "c3");
    REQUIRE(mem.size() == 3);
    (void)mem.encode({"d"}, {}, "c4");
    REQUIRE(mem.size() == 3);
}

TEST_CASE("HippocampalMemory: 强度衰减", "[hippocampal]") {
    HippocampalMemory mem(100);
    (void)mem.encode({"test"}, {}, "ctx");
    REQUIRE(mem.get_strongest(1)[0].strength == 1.0F);

    mem.decay(0.3F);
    REQUIRE(mem.get_strongest(1)[0].strength == Approx(0.7F));
}

TEST_CASE("HippocampalMemory: 遗忘弱记忆", "[hippocampal]") {
    HippocampalMemory mem(100);
    (void)mem.encode({"a"}, {}, "c1");
    (void)mem.encode({"b"}, {}, "c2");

    mem.decay(0.8F);
    int forgotten = mem.forget_weak(0.3F);
    REQUIRE(forgotten == 2);
    REQUIRE(mem.size() == 0);
}

TEST_CASE("HippocampalMemory: 增强指定条目", "[hippocampal]") {
    HippocampalMemory mem(100);
    int id = mem.encode({"test"}, {}, "ctx");
    mem.decay(0.5F);
    REQUIRE(mem.get_strongest(1)[0].strength == Approx(0.5F));

    mem.strengthen(id, 0.3F);
    REQUIRE(mem.get_strongest(1)[0].strength == Approx(0.8F));
}

TEST_CASE("HippocampalMemory: get_strongest", "[hippocampal]") {
    HippocampalMemory mem(100);
    (void)mem.encode({"a"}, {}, "c1");
    (void)mem.encode({"b"}, {}, "c2");
    (void)mem.encode({"c"}, {}, "c3");

    auto strongest = mem.get_strongest(2);
    REQUIRE(strongest.size() == 2);
    REQUIRE(strongest[0].strength == 1.0F);
}

// ── 皮层记忆测试 ─────────────────────────────────────────────

TEST_CASE("CorticalMemory: 存储和检索", "[consolidation]") {
    CorticalMemory cortical;
    HippocampalEntry entry;
    entry.entities = {"数学"};
    entry.relations = {"数学是学科"};
    entry.context = "教科书";
    entry.strength = 0.8F;
    entry.timestamp = 0;

    cortical.store(entry, 0);
    REQUIRE(cortical.size() == 1);

    auto results = cortical.retrieve(5);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].entities[0] == "数学");
}

TEST_CASE("CorticalMemory: 重复存储会增强", "[consolidation]") {
    CorticalMemory cortical;
    HippocampalEntry entry;
    entry.entities = {"test"};
    entry.relations = {};
    entry.context = "";
    entry.strength = 0.5F;
    entry.timestamp = 42;

    cortical.store(entry, 42);
    REQUIRE(cortical.size() == 1);
    float s1 = cortical.retrieve(1)[0].strength;

    cortical.store(entry, 42);
    REQUIRE(cortical.size() == 1);
    float s2 = cortical.retrieve(1)[0].strength;
    REQUIRE(s2 > s1);
}

TEST_CASE("CorticalMemory: 衰减", "[consolidation]") {
    CorticalMemory cortical;
    HippocampalEntry entry;
    entry.entities = {"x"};
    entry.relations = {};
    entry.context = "";
    entry.strength = 0.5F;
    entry.timestamp = 0;

    cortical.store(entry, 0);
    cortical.decay(0.1F);
    REQUIRE(cortical.retrieve(1)[0].strength == Approx(0.5F * 1.2F - 0.1F));
}

// ── 睡眠巩固测试 ─────────────────────────────────────────────

TEST_CASE("SleepConsolidation: 完整睡眠周期", "[consolidation]") {
    HippocampalMemory hipp(100);
    CorticalMemory cortical;

    (void)hipp.encode({"数学", "学科"}, {"数学是学科"}, "课本");
    (void)hipp.encode({"物理", "学科"}, {"物理是学科"}, "课本");
    (void)hipp.encode({"化学", "学科"}, {"化学是学科"}, "课本");

    SleepConsolidation sleep(hipp, cortical);
    auto report = sleep.sleep();

    REQUIRE(report.memories_consolidated == 3);
    REQUIRE(report.avg_strength > 0.0F);
}

TEST_CASE("SleepConsolidation: 弱记忆被遗忘", "[consolidation]") {
    HippocampalMemory hipp(100);
    CorticalMemory cortical;

    (void)hipp.encode({"强记忆"}, {}, "ctx");
    hipp.decay(0.9F);  // strength: 0.1

    SleepConsolidation sleep(hipp, cortical);
    sleep.set_forget_threshold(0.3F);
    auto report = sleep.sleep();

    REQUIRE(report.memories_forgotten == 1);
}
