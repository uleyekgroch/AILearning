/**
 * @file world_model.hpp
 * @brief 世界模型 — 因果 DAG + 反事实推理引擎
 *
 * 参考：
 *   - DreamerV3 (Hafner et al., Nature 2025)：想象中规划
 *   - Pearl 因果阶梯 (Pearl 2009)：观察→干预→反事实
 *   - Active Inference (Friston 2025)：自由能最小化
 *
 * 能力 3：世界模型和因果推理
 * 人类理解"读写文件"不是记住 API，而是构建因果链：
 *   open() → fd → write(fd) → 内核缓冲 → flush → disk
 * 如果 write 返回 -1 → errno=EACCES → 权限问题 → chmod
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <optional>
#include <random>

namespace ai_learning::reasoning {

/// 因果边类型
enum class CausalEdgeType {
    kCauses,        ///< A 导致 B
    kEnables,       ///< A 使 B 成为可能
    kPrevents,      ///< A 阻止 B
    kCorrelates,    ///< A 与 B 相关（无因果方向）
};

/// 因果边
struct CausalEdge {
    std::string from;                 ///< 原因节点
    std::string to;                   ///< 结果节点
    CausalEdgeType type = CausalEdgeType::kCauses;
    double strength = 1.0;            ///< 因果强度 0~1
    std::string condition;            ///< 条件（可选）
};

/// 干预操作
struct Intervention {
    std::string variable;             ///< 被干预的变量
    double value = 0.0;               ///< 设置的值
    std::string description;          ///< 干预描述
};

/// 反事实结果
struct CounterfactualResult {
    std::string original_scenario;    ///< 原始场景
    std::string intervention;         ///< "如果 X 不发生"
    std::string predicted_outcome;    ///< 预测结果
    double confidence = 0.0;          ///< 置信度
    std::vector<std::string> reasoning_chain;  ///< 推理链
};

/// 想象规划结果（参考 DreamerV3）
struct ImaginationPlan {
    std::string goal;                 ///< 目标
    std::vector<std::string> steps;   ///< 规划步骤
    double expected_reward = 0.0;     ///< 预期奖励
    double uncertainty = 0.0;         ///< 不确定性
    std::vector<std::string> alternatives;  ///< 备选方案
};

/// 世界模型统计
struct WorldModelStats {
    int node_count = 0;
    int edge_count = 0;
    int observations_processed = 0;
    int causal_rules_learned = 0;
    double avg_prediction_accuracy = 0.0;
};

/// 世界模型 — 学习因果 DAG，支持三层推理
class WorldModel {
public:
    explicit WorldModel(int seed = 42);

    // ── 观察（Pearl 第一层：Seeing）───────────────────

    /// 观察事件序列，更新因果模型
    /// @param events 按时间顺序的事件列表
    /// @param outcome 最终结果
    void observe_sequence(const std::vector<std::string>& events,
                          const std::string& outcome);

    /// 观察变量相关性
    void observe_correlation(const std::string& var_a,
                              const std::string& var_b,
                              double correlation);

    /// 观察干预效果（Pearl 第二层：Doing）
    void observe_intervention(const std::string& action,
                               const std::map<std::string, double>& before,
                               const std::map<std::string, double>& after);

    // ── 因果推理 ────────────────────────────────────────

    /// 预测：给定原因，预测结果（Pearl 第一层）
    auto predict(const std::vector<std::string>& causes) const
        -> std::vector<std::pair<std::string, double>>;

    /// 干预推理：如果做 X，会怎样？（Pearl 第二层）
    auto intervene(const std::vector<Intervention>& interventions) const
        -> std::map<std::string, double>;

    /// 反事实推理：如果当时不做 X，会怎样？（Pearl 第三层）
    auto counterfactual(const std::string& observed_outcome,
                         const std::vector<Intervention>& was,
                         const std::vector<Intervention>& what_if) const
        -> CounterfactualResult;

    /// 因果链查询：从 A 到 B 的因果路径
    auto find_causal_path(const std::string& from,
                           const std::string& to) const
        -> std::vector<std::vector<std::string>>;

    // ── 想象规划（DreamerV3 风格）────────────────────

    /// 在想象中规划：给定目标，在模型中搜索最佳行动序列
    auto imagine_plan(const std::string& goal,
                       int max_depth = 5) const
        -> ImaginationPlan;

    /// 评估行动序列的预期效果
    auto evaluate_plan(const std::vector<std::string>& actions) const
        -> std::pair<double, double>;  ///< {expected_reward, uncertainty}

    // ── 模型管理 ────────────────────────────────────────

    /// 添加因果规则
    void add_causal_rule(const CausalEdge& edge);

    /// 移除因果规则
    bool remove_causal_rule(const std::string& from, const std::string& to);

    /// 获取所有因果边
    [[nodiscard]] auto causal_edges() const
        -> const std::vector<CausalEdge>& { return edges_; }

    /// 获取所有节点
    [[nodiscard]] auto nodes() const
        -> const std::vector<std::string>& { return nodes_; }

    /// 获取模型统计
    [[nodiscard]] auto stats() const -> WorldModelStats;

    /// 模型验证：检查一致性
    [[nodiscard]] auto validate() const
        -> std::vector<std::string>;  ///< 返回不一致的边列表

private:
    std::vector<std::string> nodes_;
    std::vector<CausalEdge> edges_;
    mutable std::mt19937 rng_;

    // 共现计数（用于因果发现）
    std::map<std::string, std::map<std::string, int>> cooccurrence_;
    std::map<std::string, int> event_counts_;
    int observations_processed_ = 0;

    /// 确保节点存在
    void ensure_node_(const std::string& name);

    /// 从观察中推断因果方向
    auto infer_causal_direction_(const std::string& a,
                                  const std::string& b) const
        -> std::optional<CausalEdge>;

    /// 深度优先搜索因果路径
    auto dfs_paths_(const std::string& current,
                     const std::string& target,
                     std::vector<std::string>& path,
                     std::vector<std::vector<std::string>>& all_paths,
                     int depth) const -> void;

    /// 贝叶斯更新因果强度
    void update_strength_(CausalEdge& edge, double evidence);
};

}  // namespace ai_learning::reasoning
