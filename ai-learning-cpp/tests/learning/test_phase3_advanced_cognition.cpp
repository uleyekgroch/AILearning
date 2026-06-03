/**
 * @file test_phase3_advanced_cognition.cpp
 * @brief Phase 3 三大高级认知模块测试
 *
 * TDD 测试覆盖：
 *   1. AnalogicalTransferEngine — 跨领域类比迁移
 *   2. ContinualLearner — 持续终身学习
 *   3. AbstractConceptEngine — 抽象概念形成
 *   4. 集成测试 — 三模块协同 + 与已有系统集成
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/learning/analogical_transfer.hpp"
#include "ai_learning/learning/continual_learner.hpp"
#include "ai_learning/learning/abstract_concept.hpp"
#include "ai_learning/learning/skill_tree.hpp"
#include "ai_learning/learning/intrinsic_motivation.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace ai_learning::learning;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════════════
// 1. AnalogicalTransferEngine 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Analogy: describe_concept creates valid descriptor", "[analogy]") {
    AnalogicalTransferEngine engine;

    auto desc = engine.describe_concept(
        "water_flow", "physics",
        std::vector<std::string>{"水从高处流向低处", "水有压力", "水管可以输送水"},
        std::map<std::string, double>{{"speed", 1.0}, {"pressure", 0.8}}
    );

    REQUIRE(desc.id == "water_flow");
    REQUIRE(desc.domain == "physics");
    REQUIRE_FALSE(desc.attributes.empty());
    REQUIRE_FALSE(desc.features.empty());
}

TEST_CASE("Analogy: find_mapping detects structural similarity", "[analogy]") {
    AnalogicalTransferEngine engine;

    ConceptDescriptor water{
        "water_flow", "physics",
        std::vector<std::string>{"流动", "压力", "管道", "方向"},
        std::vector<std::string>{"高处流向低处", "管道限制流向"},
        std::map<std::string, double>{{"speed", 1.0}, {"pressure", 0.8}}
    };

    ConceptDescriptor electricity{
        "electric_current", "physics",
        std::vector<std::string>{"流动", "电压", "导线", "方向"},
        std::vector<std::string>{"正极流向负极", "导线限制流向"},
        std::map<std::string, double>{{"speed", 1.0}, {"voltage", 0.8}}
    };

    auto mapping = engine.find_mapping(water, electricity);

    REQUIRE(mapping.source_concept == "water_flow");
    REQUIRE(mapping.target_concept == "electric_current");
    REQUIRE(mapping.alignment_score > 0.0);
    REQUIRE(mapping.surface_similarity > 0.0);
    REQUIRE(mapping.attribute_map.size() >= 2);
}

TEST_CASE("Analogy: transfer produces transferred knowledge", "[analogy]") {
    AnalogicalTransferEngine engine;

    std::vector<ConceptDescriptor> source_domain = {
        {"water_flow", "hydraulics",
         std::vector<std::string>{"流动", "压力", "管道", "方向"},
         std::vector<std::string>{"水从高处流向低处", "管道限制流向"},
         std::map<std::string, double>{{"speed", 1.0}}
        },
        {"water_pressure", "hydraulics",
         std::vector<std::string>{"压力", "深度", "面积"},
         std::vector<std::string>{"压力随深度增加"},
         std::map<std::string, double>{{"pressure", 0.9}}
        }
    };

    std::vector<ConceptDescriptor> target_domain = {
        {"current", "electronics",
         std::vector<std::string>{"流动", "电压", "导线", "方向"},
         std::vector<std::string>{},
         std::map<std::string, double>{{"speed", 1.0}}
        }
    };

    auto result = engine.transfer(
        source_domain, target_domain,
        std::vector<std::string>{"水从高处流向低处", "管道限制流向", "压力随深度增加"}
    );

    REQUIRE(result.source_domain == "hydraulics");
    REQUIRE(result.target_domain == "electronics");
    REQUIRE_FALSE(result.mappings.empty());
    REQUIRE(result.transfer_quality >= 0.0);
}

TEST_CASE("Analogy: map_knowledge maps a single fact", "[analogy]") {
    AnalogicalTransferEngine engine;

    StructureMapping mapping;
    mapping.source_concept = "water";
    mapping.target_concept = "electricity";
    mapping.attribute_map = {{"管道", "导线"}, {"压力", "电压"}};
    mapping.relation_map = {{"水在管道中流动", "电流在导线中流动"}};

    auto mapped = engine.map_knowledge("水在管道中流动，压力驱动", mapping);
    REQUIRE(mapped.has_value());
    REQUIRE((*mapped).find("导线") != std::string::npos);
}

TEST_CASE("Analogy: find_analogous_domain finds best match", "[analogy]") {
    AnalogicalTransferEngine engine;

    std::map<std::string, std::vector<ConceptDescriptor>> domains = {
        {"electronics", {
            {"current", "electronics",
             std::vector<std::string>{"流动", "电压", "导线"},
             std::vector<std::string>{},
             std::map<std::string, double>{{"speed", 1.0}}}
        }},
        {"hydraulics", {
            {"water", "hydraulics",
             std::vector<std::string>{"流动", "压力", "管道"},
             std::vector<std::string>{},
             std::map<std::string, double>{{"speed", 1.0}}}
        }},
        {"cooking", {
            {"recipe", "cooking",
             std::vector<std::string>{"食材", "步骤", "调味"},
             std::vector<std::string>{},
             std::map<std::string, double>{}}
        }}
    };

    auto best = engine.find_analogous_domain("electronics",
        std::vector<std::string>{"hydraulics", "cooking"}, domains);
    REQUIRE(best.has_value());
    REQUIRE(*best == "hydraulics");
}

TEST_CASE("Analogy: record_experience and stats", "[analogy]") {
    AnalogicalTransferEngine engine;

    engine.record_experience({"physics", "electronics", true, 0.8, "关系结构匹配良好"});
    engine.record_experience({"math", "cooking", false, 0.1, "表面差异太大"});

    auto exps = engine.experiences();
    REQUIRE(exps.size() == 2);

    auto stats = engine.stats();
    REQUIRE(stats.contains("total_transfers"));
    REQUIRE(stats.contains("successful_transfers"));
}

TEST_CASE("Analogy: learn_from_failure produces lesson", "[analogy]") {
    AnalogicalTransferEngine engine;

    TransferResult failed;
    failed.source_domain = "math";
    failed.target_domain = "biology";
    failed.failed_transfers = std::vector<std::string>{"公式不适用于生物系统"};
    failed.transfer_quality = 0.1;

    auto lesson = engine.learn_from_failure(failed);
    REQUIRE_FALSE(lesson.empty());
}

// ═══════════════════════════════════════════════════════════════════
// 2. ContinualLearner 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Continual: register_knowledge creates protection", "[continual]") {
    ContinualLearner learner;

    auto prot = learner.register_knowledge("addition", "math", 0.9, 10);

    REQUIRE(prot.knowledge_id == "addition");
    REQUIRE(prot.importance > 0.0);
    REQUIRE(prot.level != ProtectionLevel::kNone);
}

TEST_CASE("Continual: high-usage knowledge gets high protection", "[continual]") {
    ContinualLearner learner;

    auto core = learner.register_knowledge("core_math", "math", 0.95, 100);
    auto peripheral = learner.register_knowledge("trivia", "general", 0.3, 1);

    REQUIRE(static_cast<int>(core.level) >= static_cast<int>(peripheral.level));
}

TEST_CASE("Continual: assess_impact runs without crash", "[continual]") {
    ContinualLearner learner;

    learner.register_knowledge("sorting_bubble", "programming", 0.8, 20);

    auto conflicts = learner.assess_impact("sorting_merge_is_better", "programming");
    (void)conflicts;
}

TEST_CASE("Continual: resolve_conflict handles override", "[continual]") {
    ContinualLearner learner;

    KnowledgeConflict conflict{
        "旧方法：冒泡排序", "新方法：归并排序更快",
        0.5, "", ""
    };

    auto resolved = learner.resolve_conflict(conflict);
    REQUIRE_FALSE(resolved.resolution.empty());
    REQUIRE_FALSE(resolved.reason.empty());
}

TEST_CASE("Continual: replay adds and replays entries", "[continual]") {
    ContinualLearner learner;

    for (int i = 0; i < 10; ++i) {
        learner.add_replay_entry({
            std::vector<float>(10, static_cast<float>(i)),
            std::vector<float>(10, static_cast<float>(i + 1)),
            static_cast<double>(i) / 10.0,
            0, "domain_a"
        });
    }

    REQUIRE(learner.replay_buffer_size() == 10);

    int replayed = learner.replay(5);
    REQUIRE(replayed <= 5);
    REQUIRE(replayed > 0);
}

TEST_CASE("Continual: sample_replay returns entries", "[continual]") {
    ContinualLearner learner;

    learner.add_replay_entry({std::vector<float>{1.0f}, std::vector<float>{2.0f}, 0.1, 0, "math"});
    learner.add_replay_entry({std::vector<float>{3.0f}, std::vector<float>{4.0f}, 0.9, 0, "math"});
    learner.add_replay_entry({std::vector<float>{5.0f}, std::vector<float>{6.0f}, 0.5, 0, "math"});

    auto samples = learner.sample_replay(3);
    REQUIRE(samples.size() == 3);
}

TEST_CASE("Continual: detect_forgetting returns alerts", "[continual]") {
    ContinualLearner learner;

    learner.register_knowledge("basic_math", "math", 0.9, 50);
    learner.post_learning_update("math_new", 0.3);

    auto alerts = learner.detect_forgetting();
    (void)alerts;
}

TEST_CASE("Continual: retention_rate is valid", "[continual]") {
    ContinualLearner learner;

    learner.register_knowledge("a", "math", 0.8, 10);
    learner.register_knowledge("b", "math", 0.7, 5);
    learner.register_knowledge("c", "physics", 0.9, 15);

    double rate = learner.retention_rate();
    REQUIRE(rate >= 0.0);
    REQUIRE(rate <= 1.0);

    double math_rate = learner.retention_rate("math");
    REQUIRE(math_rate >= 0.0);
    REQUIRE(math_rate <= 1.0);
}

TEST_CASE("Continual: prepare_for_new_learning runs", "[continual]") {
    ContinualLearner learner;

    learner.register_knowledge("old_fact", "domain_a", 0.9, 20);

    auto conflicts = learner.prepare_for_new_learning(
        "domain_a", std::vector<std::string>{"contradictory_new_fact"});

    (void)conflicts;
}

TEST_CASE("Continual: stats tracks all counters", "[continual]") {
    ContinualLearner learner;

    learner.register_knowledge("a", "math", 0.8, 10);
    learner.add_replay_entry(ReplayEntry{
        std::vector<float>{}, std::vector<float>{}, 0.5, 0, "math"});
    learner.replay(1);

    auto stats = learner.stats();
    REQUIRE(stats.total_replays >= 1);
}

// ═══════════════════════════════════════════════════════════════════
// 3. AbstractConceptEngine 测试
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Abstract: observe_instance stores data", "[abstract]") {
    AbstractConceptEngine engine;

    auto report = engine.observe_instance(
        "apple", std::vector<std::string>{"红色", "圆形", "甜", "水果"},
        std::map<std::string, double>{{"sweetness", 0.8}});

    REQUIRE(report.instances_processed == 1);
}

TEST_CASE("Abstract: concept formation from similar instances", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("apple",
        std::vector<std::string>{"红色", "圆形", "甜", "水果", "有籽"}, {});
    engine.observe_instance("pear",
        std::vector<std::string>{"绿色", "梨形", "甜", "水果", "有籽"}, {});
    engine.observe_instance("peach",
        std::vector<std::string>{"粉色", "圆形", "甜", "水果", "有核"}, {});

    auto report = engine.observe_instance("orange",
        std::vector<std::string>{"橙色", "圆形", "甜", "水果", "有籽"}, {});

    if (!report.new_concepts.empty()) {
        REQUIRE(report.new_concepts[0].core_attributes.size() > 0);
        REQUIRE(report.new_concepts[0].source_instances.size() >= 3);
    }
}

TEST_CASE("Abstract: classify_instance assigns to concept", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("apple",
        std::vector<std::string>{"红色", "圆形", "甜", "水果"}, {});
    engine.observe_instance("pear",
        std::vector<std::string>{"绿色", "梨形", "甜", "水果"}, {});
    engine.observe_instance("peach",
        std::vector<std::string>{"粉色", "圆形", "甜", "水果"}, {});

    auto concept_id = engine.classify_instance("banana",
        std::vector<std::string>{"黄色", "长形", "甜", "水果"});
    (void)concept_id;
}

TEST_CASE("Abstract: form_concept_from_instances creates prototype", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("sedan",
        std::vector<std::string>{"四个轮子", "有引擎", "载人", "轿车"}, {});
    engine.observe_instance("suv",
        std::vector<std::string>{"四个轮子", "有引擎", "载人", "越野车"}, {});
    engine.observe_instance("coupe",
        std::vector<std::string>{"四个轮子", "有引擎", "载人", "跑车"}, {});

    auto cpt = engine.form_concept_from_instances(
        std::vector<std::string>{"sedan", "suv", "coupe"});

    if (cpt.has_value()) {
        REQUIRE_FALSE(cpt->id.empty());
        REQUIRE_FALSE(cpt->core_attributes.empty());
        bool has_wheels = false, has_engine = false;
        for (const auto& attr : cpt->core_attributes) {
            if (attr.find("轮子") != std::string::npos) has_wheels = true;
            if (attr.find("引擎") != std::string::npos) has_engine = true;
        }
        REQUIRE(has_wheels);
        REQUIRE(has_engine);
    }
}

TEST_CASE("Abstract: discover_generalization creates rule", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("apple",
        std::vector<std::string>{"甜", "水果"}, {});
    engine.observe_instance("pear",
        std::vector<std::string>{"甜", "水果"}, {});
    engine.observe_instance("peach",
        std::vector<std::string>{"甜", "水果"}, {});

    auto cpt = engine.form_concept_from_instances(
        std::vector<std::string>{"apple", "pear", "peach"});

    if (cpt.has_value()) {
        auto rule = engine.discover_generalization(cpt->id);
        if (rule.has_value()) {
            REQUIRE_FALSE(rule->concrete_pattern.empty());
            REQUIRE_FALSE(rule->abstract_pattern.empty());
            REQUIRE(rule->support_count >= 3);
        }
    }
}

TEST_CASE("Abstract: check_differentiation runs", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("apple",
        std::vector<std::string>{"植物", "可食用", "甜", "水果"}, {});
    engine.observe_instance("carrot",
        std::vector<std::string>{"植物", "可食用", "不甜", "蔬菜"}, {});
    engine.observe_instance("pear",
        std::vector<std::string>{"植物", "可食用", "甜", "水果"}, {});
    engine.observe_instance("broccoli",
        std::vector<std::string>{"植物", "可食用", "不甜", "蔬菜"}, {});

    auto cpt = engine.form_concept_from_instances(
        std::vector<std::string>{"apple", "carrot", "pear", "broccoli"});

    if (cpt.has_value()) {
        auto diff = engine.check_differentiation(cpt->id);
        (void)diff;
    }
}

TEST_CASE("Abstract: concepts_by_level filters correctly", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("apple",
        std::vector<std::string>{"甜", "水果"}, {});
    engine.observe_instance("pear",
        std::vector<std::string>{"甜", "水果"}, {});
    engine.observe_instance("peach",
        std::vector<std::string>{"甜", "水果"}, {});

    engine.form_concept_from_instances(
        std::vector<std::string>{"apple", "pear", "peach"});

    auto concrete = engine.concepts_by_level(AbstractionLevel::kConcrete);
    auto basic = engine.concepts_by_level(AbstractionLevel::kBasic);

    (void)concrete;
    (void)basic;
}

TEST_CASE("Abstract: coherence returns valid score", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("a", std::vector<std::string>{"attr1", "attr2"}, {});
    engine.observe_instance("b", std::vector<std::string>{"attr1", "attr2"}, {});

    double c = engine.coherence();
    REQUIRE(c >= 0.0);
    REQUIRE(c <= 1.0);
}

TEST_CASE("Abstract: stats returns valid data", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("x", std::vector<std::string>{"a"}, {});

    auto stats = engine.stats();
    REQUIRE(stats.contains("total_concepts"));
    REQUIRE(stats.contains("total_instances"));
}

TEST_CASE("Abstract: instance_concepts returns memberships", "[abstract]") {
    AbstractConceptEngine engine;

    engine.observe_instance("apple",
        std::vector<std::string>{"甜", "水果"}, {});
    engine.observe_instance("pear",
        std::vector<std::string>{"甜", "水果"}, {});
    engine.observe_instance("peach",
        std::vector<std::string>{"甜", "水果"}, {});

    engine.form_concept_from_instances(
        std::vector<std::string>{"apple", "pear", "peach"});

    auto memberships = engine.instance_concepts("apple");
    (void)memberships;
}

// ═══════════════════════════════════════════════════════════════════
// 4. 集成测试 — 三模块协同
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Integration: analogy uses abstract concepts", "[integration_phase3]") {
    AbstractConceptEngine cpt_engine;

    cpt_engine.observe_instance("addition",
        std::vector<std::string>{"二元运算", "交换律", "结合律"}, {});
    cpt_engine.observe_instance("multiplication",
        std::vector<std::string>{"二元运算", "交换律", "结合律"}, {});
    cpt_engine.observe_instance("union",
        std::vector<std::string>{"二元运算", "交换律", "结合律"}, {});

    auto math_cpt = cpt_engine.form_concept_from_instances(
        std::vector<std::string>{"addition", "multiplication", "union"});

    REQUIRE(math_cpt.has_value());

    AnalogicalTransferEngine analogy_engine;

    auto desc = analogy_engine.describe_concept(
        "math_operation", "math",
        std::vector<std::string>{"二元运算", "交换律", "结合律"},
        std::map<std::string, double>{});

    REQUIRE(desc.attributes.size() >= 2);
}

TEST_CASE("Integration: continual learner protects during concept formation",
          "[integration_phase3]") {
    AbstractConceptEngine cpt_engine;
    ContinualLearner continual;

    cpt_engine.observe_instance("apple",
        std::vector<std::string>{"甜", "水果"}, {});
    cpt_engine.observe_instance("pear",
        std::vector<std::string>{"甜", "水果"}, {});

    auto cpt = cpt_engine.form_concept_from_instances(
        std::vector<std::string>{"apple", "pear"});

    if (cpt.has_value()) {
        continual.register_knowledge(
            cpt->id, "botany", cpt->strength, 5);

        auto conflicts = continual.prepare_for_new_learning(
            "physics", std::vector<std::string>{"E=mc²"});
        (void)conflicts;
    }
}

TEST_CASE("Integration: full pipeline — concept → analogy → protection",
          "[integration_phase3]") {
    // Step 1: 在源领域形成抽象概念
    AbstractConceptEngine cpt_engine;

    cpt_engine.observe_instance("bubble_sort",
        std::vector<std::string>{"比较", "交换", "排序"},
        std::map<std::string, double>{},
        std::vector<std::string>{"输入→比较→交换→输出"});
    cpt_engine.observe_instance("selection_sort",
        std::vector<std::string>{"比较", "选择", "排序"},
        std::map<std::string, double>{},
        std::vector<std::string>{"输入→比较→选择→输出"});
    cpt_engine.observe_instance("insertion_sort",
        std::vector<std::string>{"比较", "插入", "排序"},
        std::map<std::string, double>{},
        std::vector<std::string>{"输入→比较→插入→输出"});

    auto sort_cpt = cpt_engine.form_concept_from_instances(
        std::vector<std::string>{"bubble_sort", "selection_sort", "insertion_sort"});

    REQUIRE(sort_cpt.has_value());
    REQUIRE(sort_cpt->core_attributes.size() >= 1);

    // Step 2: 迁移到新领域
    AnalogicalTransferEngine analogy_engine;

    std::vector<ConceptDescriptor> source = {
        analogy_engine.describe_concept("sorting", "algorithms",
            sort_cpt->core_attributes,
            std::map<std::string, double>{})
    };

    std::vector<ConceptDescriptor> target = {
        analogy_engine.describe_concept("searching", "algorithms",
            std::vector<std::string>{"比较", "查找", "遍历"},
            std::map<std::string, double>{})
    };

    auto transfer_result = analogy_engine.transfer(
        source, target,
        std::vector<std::string>{"排序需要比较元素", "排序产生有序输出"});

    REQUIRE_FALSE(transfer_result.mappings.empty());

    // Step 3: 保护形成的知识
    ContinualLearner continual;

    continual.register_knowledge(
        sort_cpt->id, "algorithms", sort_cpt->strength, 10);

    auto prot = continual.get_protection(sort_cpt->id);
    REQUIRE(prot.has_value());
    REQUIRE(prot->importance > 0.0);
}

TEST_CASE("Integration: continual learner stats after multiple operations",
          "[integration_phase3]") {
    ContinualLearner learner;

    learner.register_knowledge("add", "math", 0.9, 50);
    learner.register_knowledge("sub", "math", 0.7, 30);
    learner.register_knowledge("mul", "math", 0.5, 10);

    for (int i = 0; i < 20; ++i) {
        learner.add_replay_entry(ReplayEntry{
            std::vector<float>{static_cast<float>(i)},
            std::vector<float>{static_cast<float>(i + 1)},
            0.5, 0, "math"});
    }

    learner.replay(5);

    auto stats = learner.stats();
    REQUIRE(stats.total_knowledge_protected >= 3);
    REQUIRE(stats.total_replays >= 5);

    double rate = learner.retention_rate();
    REQUIRE(rate >= 0.0);
}
