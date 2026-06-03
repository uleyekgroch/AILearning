/**
 * @file analogical_transfer.hpp
 * @brief 跨领域类比迁移 — 让系统"举一反三"
 *
 * 参考：
 *   - Gentner 结构映射理论 (SME, 1983)：关系结构对齐
 *   - Holyoak & Thagard ACME (1989)：约束满足类比
 *   - Copycat (Hofstadter & Mitchell, 1994)：流体类比
 *   - 跨领域迁移学习 (Pan & Yang, 2010)
 *
 * 核心能力：
 *   1. 结构映射 — 找到两个领域之间的结构对应关系
 *   2. 知识迁移 — 将源领域的知识映射到目标领域
 *   3. 类比验证 — 检验迁移的知识在目标领域是否成立
 *   4. 失败学习 — 从失败的类比中学习边界条件
 *
 * 人类类比学习流程：
 *   "变量像盒子" → 编程新手理解变量概念
 *   "电流像水流" → 理解电路原理
 *   "原子像太阳系" → 理解原子结构（卢瑟福模型）
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace ai_learning::learning {

/// 概念描述符 — 一个概念的"形状"
struct ConceptDescriptor {
    std::string id;                    ///< 概念 ID
    std::string domain;                ///< 所属领域
    std::vector<std::string> attributes;    ///< 属性列表
    std::vector<std::string> relations;     ///< 关系列表（"A causes B"格式）
    std::map<std::string, double> features; ///< 数值特征
};

/// 结构映射 — 两个概念之间的对应关系
struct StructureMapping {
    std::string source_concept;        ///< 源概念
    std::string target_concept;        ///< 目标概念
    std::map<std::string, std::string> attribute_map;   ///< 属性映射
    std::map<std::string, std::string> relation_map;    ///< 关系映射
    double alignment_score = 0.0;      ///< 结构对齐度 0~1
    double surface_similarity = 0.0;   ///< 表面相似度 0~1
    double relational_depth = 0.0;     ///< 关系深度（深层结构匹配度）
};

/// 迁移结果
struct TransferResult {
    std::string source_domain;         ///< 源领域
    std::string target_domain;         ///< 目标领域
    std::vector<StructureMapping> mappings;  ///< 发现的映射
    std::vector<std::string> transferred_knowledge;  ///< 迁移的知识
    std::vector<std::string> failed_transfers;       ///< 失败的迁移
    double transfer_quality = 0.0;     ///< 迁移质量 0~1
    std::string reasoning;             ///< 迁移推理过程
};

/// 类比经验 — 记录历史类比，用于改进
struct AnalogyExperience {
    std::string source_domain;
    std::string target_domain;
    bool success = false;
    double quality = 0.0;
    std::string lesson;                ///< 学到的教训
};

/// 类比迁移配置
struct AnalogicalConfig {
    double min_alignment_score = 0.3;  ///< 最低对齐分数
    double min_surface_similarity = 0.1; ///< 最低表面相似度
    int max_mappings = 10;             ///< 最大映射数
    bool verify_transfers = true;      ///< 是否验证迁移
};

/// 类比迁移引擎
class AnalogicalTransferEngine {
public:
    explicit AnalogicalTransferEngine(
        const AnalogicalConfig& config = AnalogicalConfig{});

    // ── 结构映射（Gentner SME 核心）─────────────────

    /// 提取概念描述符
    auto describe_concept(const std::string& concept_id,
                          const std::string& domain,
                          const std::vector<std::string>& known_facts,
                          const std::map<std::string, double>& features = {})
        -> ConceptDescriptor;

    /// 在两个概念间寻找结构映射
    auto find_mapping(const ConceptDescriptor& source,
                      const ConceptDescriptor& target)
        -> StructureMapping;

    /// 在两个领域间寻找最佳映射集合
    auto find_domain_mapping(
        const std::vector<ConceptDescriptor>& source_domain,
        const std::vector<ConceptDescriptor>& target_domain)
        -> std::vector<StructureMapping>;

    // ── 知识迁移 ──────────────────────────────────────

    /// 执行跨领域迁移
    /// @param source_concepts 源领域的已知概念
    /// @param target_concepts 目标领域的已知概念
    /// @param source_facts 源领域的事实知识
    auto transfer(const std::vector<ConceptDescriptor>& source_concepts,
                  const std::vector<ConceptDescriptor>& target_concepts,
                  const std::vector<std::string>& source_facts)
        -> TransferResult;

    /// 将源领域的一条知识映射到目标领域
    auto map_knowledge(const std::string& fact,
                       const StructureMapping& mapping)
        -> std::optional<std::string>;

    // ── 验证与学习 ──────────────────────────────────────

    /// 验证迁移的知识是否自洽
    auto verify_transfer(const std::string& transferred,
                         const std::vector<std::string>& target_facts)
        -> bool;

    /// 记录类比经验
    void record_experience(const AnalogyExperience& experience);

    /// 从失败类比中学习教训
    auto learn_from_failure(const TransferResult& failed)
        -> std::string;

    // ── 查询 ──────────────────────────────────────────

    /// 查找与目标领域最相似的源领域
    auto find_analogous_domain(
        const std::string& target_domain,
        const std::vector<std::string>& known_domains,
        const std::map<std::string, std::vector<ConceptDescriptor>>& domain_concepts)
        -> std::optional<std::string>;

    /// 获取类比经验
    [[nodiscard]] auto experiences() const
        -> const std::vector<AnalogyExperience>& {
        return experiences_;
    }

    /// 获取统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

    /// 获取配置
    [[nodiscard]] auto config() const -> const AnalogicalConfig& {
        return config_;
    }

private:
    AnalogicalConfig config_;

    /// 概念注册表
    std::map<std::string, ConceptDescriptor> concept_registry_;

    /// 类比经验库
    std::vector<AnalogyExperience> experiences_;

    // ── 内部算法 ──────────────────────────────────────

    /// 计算表面相似度（属性重叠）
    static auto surface_similarity_(const ConceptDescriptor& a,
                                     const ConceptDescriptor& b) -> double;

    /// 计算关系结构对齐度（Gentner 结构映射核心）
    static auto relational_alignment_(const ConceptDescriptor& a,
                                       const ConceptDescriptor& b) -> double;

    /// 找到最佳属性映射（贪心匹配）
    static auto match_attributes_(const std::vector<std::string>& source,
                                   const std::vector<std::string>& target)
        -> std::map<std::string, std::string>;

    /// 计算综合类比评分
    auto analogy_score_(const StructureMapping& mapping) const -> double;

    /// 统计信息
    int total_transfers_ = 0;
    int successful_transfers_ = 0;
};

}  // namespace ai_learning::learning
