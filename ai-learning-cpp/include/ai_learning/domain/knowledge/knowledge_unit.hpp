/**
 * @file knowledge_unit.hpp
 * @brief 通用知识单元 — 任何领域的概念表示
 *
 * 基于 Bloom's Taxonomy 的掌握度等级。
 * 对应 Python 版 KnowledgeUnit。
 */
#pragma once

#include <string>
#include <vector>
#include <map>

namespace ai_learning::domain::knowledge {

/// 掌握度等级
enum class MasteryLevel : int {
    kUnknown    = 0,  // 完全未知
    kExposed    = 1,  // 接触过
    kRecognized = 2,  // 能识别
    kUnderstood = 3,  // 能理解
    kApplied    = 4,  // 能应用
    kMastered   = 5,  // 精通（能教授）
};

auto mastery_level_to_string(MasteryLevel level) -> std::string;
auto mastery_level_from_string(const std::string& s) -> MasteryLevel;

class KnowledgeUnit {
public:
    KnowledgeUnit(std::string id, std::string name, std::string domain,
                  std::string definition = "",
                  double difficulty = 0.5);

    // ── 访问器 ──
    [[nodiscard]] auto id() const -> const std::string& { return id_; }
    [[nodiscard]] auto name() const -> const std::string& { return name_; }
    [[nodiscard]] auto domain() const -> const std::string& { return domain_; }
    [[nodiscard]] auto definition() const -> const std::string& {
        return definition_;
    }
    [[nodiscard]] auto difficulty() const -> double { return difficulty_; }
    [[nodiscard]] auto mastery() const -> double { return mastery_; }
    [[nodiscard]] auto mastery_level() const -> MasteryLevel {
        return mastery_level_;
    }
    [[nodiscard]] auto exposure_count() const -> int { return exposure_count_; }
    [[nodiscard]] auto practice_count() const -> int { return practice_count_; }
    [[nodiscard]] auto success_count() const -> int { return success_count_; }
    [[nodiscard]] auto prerequisites() const -> const std::vector<std::string>& {
        return prerequisites_;
    }

    // ── 学习操作 ──
    /// 更新掌握度（success + quality → 调整 mastery）
    void update_mastery(bool success, double quality = 0.5);

    /// 增加接触次数
    void add_exposure() { ++exposure_count_; }

    /// 添加前置知识
    void add_prerequisite(const std::string& prereq_id);

    /// 获取成功率
    [[nodiscard]] auto success_rate() const -> double;

    /// 检查前置知识是否满足
    [[nodiscard]] auto prerequisites_met(
        const std::map<std::string, const KnowledgeUnit*>& all_units,
        double threshold = 0.5) const -> bool;

    // ── 序列化 ──
    [[nodiscard]] auto to_map() const
        -> std::map<std::string, std::string>;

private:
    void update_mastery_level_();

    std::string id_;
    std::string name_;
    std::string domain_;
    std::string definition_;
    double      difficulty_;

    double       mastery_        = 0.0;
    MasteryLevel mastery_level_  = MasteryLevel::kUnknown;
    int          exposure_count_ = 0;
    int          practice_count_ = 0;
    int          success_count_  = 0;

    std::vector<std::string> prerequisites_;
};

}  // namespace ai_learning::domain::knowledge
