/**
 * @file abstract_concept.hpp
 * @brief 抽象概念形成 — 从具体实例中涌现高层抽象
 *
 * 参考：
 *   - Piaget 发生认识论 (1972)：具体→形式运算
 *   - Rosch 原型理论 (1978)：基本层概念
 *   - Posner & Keele 概念形成实验 (1968)：原型抽象
 *   - Doumas & Hummel LISA (2005)：关系概念学习
 *   - Gärdenfors 概念空间 (2004)：几何化知识表示
 *
 * 核心能力：
 *   1. 原型提取 — 从多个具体实例中抽象出"原型"
 *   2. 层次化抽象 — 具体→基本层→上位层
 *   3. 概念泛化 — 将具体规则泛化为抽象规则
 *   4. 概念分化 — 当抽象概念遇到反例时分裂为更精确的子概念
 *   5. 类比概念 — 通过类比发现跨领域的抽象结构
 *
 * 人类抽象概念形成流程：
 *   苹果、梨、桃子 → [水果]（基本层概念）
 *   水果、蔬菜、谷物 → [食物]（上位层概念）
 *   "把A放入B" + "把C放入D" → [容器-内容关系]（关系概念）
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <optional>
#include <set>

namespace ai_learning::learning {

/// 抽象层次
enum class AbstractionLevel {
    kConcrete = 0,     ///< 具体：特定实例 "苹果是红色的"
    kBasic = 1,        ///< 基本层：类别概括 "水果有维生素"
    kSuperordinate = 2, ///< 上位层：跨类别 "食物提供能量"
    kRelational = 3,   ///< 关系层：模式抽象 "容器-内容关系"
    kMeta = 4,         ///< 元层：关于学习的概念 "分类策略"
};

/// 概念原型 — 一组实例的抽象表示
struct ConceptPrototype {
    std::string id;                         ///< 概念 ID
    std::string name;                       ///< 概念名称
    AbstractionLevel level;                 ///< 抽象层次

    /// 核心属性（所有实例共享）
    std::vector<std::string> core_attributes;

    /// 可变属性（实例间变化的属性）
    std::vector<std::string> variable_attributes;

    /// 原型特征向量（Gärdenfors 概念空间的中心）
    std::map<std::string, double> prototype_features;

    /// 关系结构（LISA 风格）
    std::vector<std::string> relational_structure;

    /// 来源实例
    std::vector<std::string> source_instances;

    /// 概念强度（实例数 × 一致性）
    double strength = 0.0;

    /// 与上位概念的关系
    std::string parent_concept;             ///< 上位概念 ID

    /// 子概念
    std::vector<std::string> child_concepts;
};

/// 泛化规则 — 从具体到抽象的规则
struct GeneralizationRule {
    std::string id;                         ///< 规则 ID
    std::string concrete_pattern;           ///< 具体模式 "苹果是甜的"
    std::string abstract_pattern;           ///< 抽象模式 "水果是甜的"
    std::string variable_part;              ///< 变量部分 "苹果→水果"
    double confidence = 0.0;               ///< 置信度
    int support_count = 0;                  ///< 支持实例数
    std::vector<std::string> counter_examples; ///< 反例
};

/// 概念分化事件 — 概念需要分裂
struct DifferentiationEvent {
    std::string original_concept;           ///< 原概念
    std::vector<std::string> new_subconcepts; ///< 分裂后的子概念
    std::string reason;                     ///< 分化原因（遇到反例）
    std::string differentiating_attribute;   ///< 区分属性
};

/// 抽象概念形成报告
struct ConceptFormationReport {
    std::vector<ConceptPrototype> new_concepts;    ///< 新形成的概念
    std::vector<GeneralizationRule> new_rules;     ///< 新发现的泛化规则
    std::vector<DifferentiationEvent> differentiations; ///< 分化事件
    int instances_processed = 0;                    ///< 处理的实例数
    double coherence_score = 0.0;                  ///< 概念体系一致性
};

/// 抽象概念形成配置
struct AbstractionConfig {
    int min_instances_for_concept = 3;    ///< 形成概念的最少实例数
    double min_core_attribute_ratio = 0.6; ///< 核心属性最小共享率
    double min_strength = 0.3;            ///< 概念强度阈值
    int max_abstraction_depth = 4;        ///< 最大抽象深度
    double differentiation_threshold = 0.3; ///< 分化阈值（不一致性超过此值触发）
};

/// 抽象概念形成引擎
class AbstractConceptEngine {
public:
    explicit AbstractConceptEngine(
        const AbstractionConfig& config = AbstractionConfig{});

    // ── 概念形成 ──────────────────────────────────────

    /// 观察一个实例，可能触发概念形成
    /// @param instance_id 实例标识
    /// @param attributes 实例属性列表
    /// @param features 数值特征
    /// @param relations 关系描述
    auto observe_instance(const std::string& instance_id,
                          const std::vector<std::string>& attributes,
                          const std::map<std::string, double>& features = {},
                          const std::vector<std::string>& relations = {})
        -> ConceptFormationReport;

    /// 从一组实例中形成概念
    auto form_concept_from_instances(
        const std::vector<std::string>& instance_ids)
        -> std::optional<ConceptPrototype>;

    /// 尝试将一个实例归入已有概念
    auto classify_instance(const std::string& instance_id,
                           const std::vector<std::string>& attributes)
        -> std::optional<std::string>;  // 返回概念 ID

    // ── 泛化 ──────────────────────────────────────────

    /// 发现泛化规则（从具体到抽象）
    auto discover_generalization(const std::string& concept_id)
        -> std::optional<GeneralizationRule>;

    /// 应用泛化规则到新实例
    auto apply_generalization(const GeneralizationRule& rule,
                              const std::string& new_instance)
        -> std::string;

    /// 尝试提升概念到更高的抽象层次
    auto try_promote(const std::string& concept_id)
        -> std::optional<ConceptPrototype>;

    // ── 分化 ──────────────────────────────────────────

    /// 检查概念是否需要分化（因为内部不一致）
    auto check_differentiation(const std::string& concept_id)
        -> std::optional<DifferentiationEvent>;

    /// 执行概念分化
    auto differentiate(const DifferentiationEvent& event)
        -> std::vector<ConceptPrototype>;

    // ── 查询 ──────────────────────────────────────────

    /// 获取概念
    [[nodiscard]] auto get_concept(const std::string& id) const
        -> std::optional<ConceptPrototype>;

    /// 获取所有概念
    [[nodiscard]] auto all_concepts() const
        -> const std::map<std::string, ConceptPrototype>& {
        return concepts_;
    }

    /// 按抽象层次获取概念
    [[nodiscard]] auto concepts_by_level(AbstractionLevel level) const
        -> std::vector<ConceptPrototype>;

    /// 获取概念层次结构（树形）
    [[nodiscard]] auto concept_hierarchy() const
        -> std::map<std::string, std::vector<std::string>>;

    /// 获取实例所属的所有概念
    [[nodiscard]] auto instance_concepts(const std::string& instance_id) const
        -> std::vector<std::string>;

    /// 计算概念体系的整体一致性
    [[nodiscard]] auto coherence() const -> double;

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const AbstractionConfig& {
        return config_;
    }

private:
    AbstractionConfig config_;

    /// 概念库
    std::map<std::string, ConceptPrototype> concepts_;

    /// 实例 → 概念的映射
    std::map<std::string, std::set<std::string>> instance_to_concepts_;

    /// 实例属性缓存
    struct InstanceData {
        std::vector<std::string> attributes;
        std::map<std::string, double> features;
        std::vector<std::string> relations;
    };
    std::map<std::string, InstanceData> instances_;

    /// 泛化规则库
    std::map<std::string, GeneralizationRule> rules_;

    // ── 内部方法 ──────────────────────────────────────

    /// 提取一组实例共享的核心属性
    auto extract_core_attributes_(
        const std::vector<std::string>& instance_ids) const
        -> std::vector<std::string>;

    /// 计算实例间的属性重叠率
    auto attribute_overlap_(const std::vector<std::string>& attrs_a,
                            const std::vector<std::string>& attrs_b) const
        -> double;

    /// 计算概念原型特征（平均值）
    auto compute_prototype_features_(
        const std::vector<std::string>& instance_ids) const
        -> std::map<std::string, double>;

    /// 生成概念名称（从核心属性中推导）
    auto generate_concept_name_(
        const std::vector<std::string>& core_attrs) const -> std::string;

    /// 计算概念强度
    auto compute_strength_(const ConceptPrototype& prototype) const -> double;

    /// 下一个概念 ID
    int next_concept_id_ = 0;
    auto next_id_() -> std::string;
};

}  // namespace ai_learning::learning
