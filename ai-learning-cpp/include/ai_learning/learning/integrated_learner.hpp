/**
 * @file integrated_learner.hpp
 * @brief Phase 6 深度整合 — 跨模块协同编排器
 *
 * 五大深度整合：
 *   1. 全流水线闭环 — observe→learn→analogize→experiment→reflect→insight
 *   2. 元学习驱动策略选择 — MetaLearner 为 IntrinsicMotivation 推荐策略
 *   3. 情感调制全系统 — EmotionEngine 影响所有模块的学习率
 *   4. 实验驱动探索 — ActiveExperimenter 假设 → IntrinsicMotivation 探索目标
 *   5. 社会学习加速类比 — SocialLearning 观察 → AnalogicalTransfer 源知识
 */
#pragma once

#include "ai_learning/learning/meta_learner.hpp"
#include "ai_learning/learning/active_experimenter.hpp"
#include "ai_learning/learning/emotion_engine.hpp"
#include "ai_learning/learning/social_learning.hpp"
#include "ai_learning/learning/analogical_transfer.hpp"
#include "ai_learning/learning/insight_engine.hpp"
#include "ai_learning/learning/intrinsic_motivation.hpp"
#include "ai_learning/learning/continual_learner.hpp"
#include "ai_learning/learning/abstract_concept.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ai_learning::learning {

// ── 整合 1：全流水线报告 ─────────────────────────────────────

/// 单步学习报告
struct PipelineStepReport {
    std::string step_name;             ///< 步骤名
    bool success = false;              ///< 是否成功
    double confidence_delta = 0.0;     ///< 置信度变化
    std::string summary;              ///< 摘要
};

/// 全流水线报告
struct IntegratedPipelineReport {
    // 六步结果
    PipelineStepReport observe_step;
    PipelineStepReport learn_step;
    PipelineStepReport analogize_step;
    PipelineStepReport experiment_step;
    PipelineStepReport reflect_step;
    PipelineStepReport insight_step;

    // 综合指标
    double overall_progress = 0.0;     ///< 总体进步 0~1
    double emotion_valence = 0.0;      ///< 最终情绪效价
    std::string dominant_emotion;      ///< 主导情绪
    std::vector<std::string> insights; ///< 获得的洞察
    std::vector<std::string> theories; ///< 构建的理论
    int steps_completed = 0;           ///< 完成的步骤数
};

// ── 整合 2：元学习驱动 ─────────────────────────────────────

/// 元学习驱动的学习会话报告
struct MetaGuidedSessionReport {
    LearningGoal chosen_goal;                     ///< 选择的动机目标
    MetaLearningRecommendation strategy_rec;      ///< 策略推荐
    double learning_rate = 0.0;                   ///< 使用的学习率
    LearningExperience experience;                ///< 记录的经验
    MemoryModulation emotion_modulation;          ///< 情感调制参数
    std::vector<std::string> reflections;         ///< 反思结果
};

// ── 整合 3：情感调制 ───────────────────────────────────────

/// 情感调制的系统级参数
struct EmotionModulatedParams {
    double learning_rate = 0.1;       ///< 调制后的学习率
    double encoding_boost = 1.0;      ///< 记忆编码增强
    double exploration_tendency = 0.5; ///< 探索倾向
    double risk_tolerance = 0.5;      ///< 风险容忍度
    std::string emotion_label;        ///< 当前情绪标签
};

// ── 整合 4：实验驱动探索 ───────────────────────────────────

/// 实验驱动的探索报告
struct ExperimentDrivenExplorationReport {
    std::string domain;               ///< 探索的领域
    std::vector<Hypothesis> hypotheses; ///< 生成的假设
    std::optional<ExperimentDesign> design; ///< 设计的实验
    double information_value = 0.0;    ///< 总信息价值
    double domain_uncertainty = 0.0;   ///< 领域不确定性
    LearningGoal suggested_goal;       ///< 建议的学习目标
};

// ── 整合 5：社会加速类比 ───────────────────────────────────

/// 社会学习加速类比的报告
struct SocialAnalogicalReport {
    std::string source_domain;         ///< 源领域（来自社会观察）
    std::string target_domain;         ///< 目标领域
    TransferResult transfer_result;    ///< 类比迁移结果
    int social_observations_used = 0;  ///< 使用的社会观察数
    double transfer_quality = 0.0;     ///< 迁移质量
};

// ── 整合编排器 ──────────────────────────────────────────────

/// 深度整合编排器 — 串联所有 Phase 3-5 模块
class IntegratedLearner {
public:
    IntegratedLearner(
        MetaLearner& meta,
        ActiveExperimenter& experimenter,
        EmotionEngine& emotion,
        SocialLearningEngine& social,
        AnalogicalTransferEngine& analogy,
        InsightEngine& insight,
        IntrinsicMotivationEngine& motivation,
        ContinualLearner& continual,
        AbstractConceptEngine& concept_eng);

    // ── 整合 1：全流水线闭环 ────────────────────────────────

    /// 执行完整的六步学习闭环
    auto run_full_pipeline(
        const std::string& observation_text,
        const std::string& domain)
        -> IntegratedPipelineReport;

    // ── 整合 2：元学习驱动策略选择 ──────────────────────────

    /// 元学习驱动的学习会话：动机选择→策略推荐→情感调制→执行→反思
    auto meta_guided_session(
        const std::vector<std::string>& known_topics,
        const std::map<std::string, double>& mastery_map)
        -> MetaGuidedSessionReport;

    // ── 整合 3：情感调制全系统 ──────────────────────────────

    /// 获取当前情感调制的系统级参数
    auto compute_system_params() const -> EmotionModulatedParams;

    /// 处理情感事件并更新系统状态
    auto process_emotion_and_modulate(const EmotionEvent& event)
        -> EmotionModulatedParams;

    // ── 整合 4：实验驱动探索 ────────────────────────────────

    /// 实验驱动的自主探索：假设→实验→目标
    auto experiment_driven_exploration(const std::string& domain)
        -> ExperimentDrivenExplorationReport;

    // ── 整合 5：社会学习加速类比 ────────────────────────────

    /// 将社会观察获得的知识通过类比迁移到新领域
    auto social_analogical_transfer(
        const std::string& source_domain,
        const std::string& target_domain,
        const std::vector<ConceptDescriptor>& target_concepts)
        -> SocialAnalogicalReport;

    // ── 辅助 ──────────────────────────────────────────────

    /// 获取整合统计
    [[nodiscard]] auto stats() const -> std::map<std::string, double>;

private:
    // 子系统引用
    MetaLearner& meta_;
    ActiveExperimenter& experimenter_;
    EmotionEngine& emotion_;
    SocialLearningEngine& social_;
    AnalogicalTransferEngine& analogy_;
    InsightEngine& insight_;
    IntrinsicMotivationEngine& motivation_;
    ContinualLearner& continual_;
    AbstractConceptEngine& concept_eng_;

    // 统计
    int pipeline_runs_ = 0;
    int meta_sessions_ = 0;
    int emotion_modulations_ = 0;
    int experiment_explorations_ = 0;
    int social_transfers_ = 0;
};

}  // namespace ai_learning::learning
