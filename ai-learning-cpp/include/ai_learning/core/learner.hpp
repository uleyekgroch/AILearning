/**
 * @file learner.hpp
 * @brief 统一学习体 — 瘦编排器
 *
 * C++20 重构的核心改变：从 5140 行 God Class → ~200 行编排器。
 *
 * 设计原则：
 * - 组合优于继承：持有各子系统引用，不继承
 * - DDD 聚合根：Learner 是事务边界
 * - 依赖倒置：通过接口引用子系统
 *
 * 核心闭环：
 *   observe → perceive → predict → choose_action → learn → remember
 */
#pragma once

#include "ai_learning/core/config.hpp"
#include "ai_learning/core/module_registry.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/learning/predictive_coding_engine.hpp"
#include "ai_learning/learning/text_learner.hpp"
#include "ai_learning/learning/stdp_learning.hpp"
#include "ai_learning/learning/verification.hpp"
#include "ai_learning/memory/episodic_memory.hpp"
#include "ai_learning/reasoning/activation_spread.hpp"
#include "ai_learning/reasoning/simulation.hpp"
#include "ai_learning/reasoning/unified_engine.hpp"
#include "ai_learning/learning/statistical_learner.hpp"
#include "ai_learning/learning/distributional_semantics.hpp"
#include "ai_learning/learning/embedding_trainer.hpp"
#include "ai_learning/language/grounding.hpp"
#include "ai_learning/language/development.hpp"
#include "ai_learning/learning/hippocampal.hpp"
#include "ai_learning/learning/consolidation.hpp"
#include "ai_learning/domain/perception/multimodal_encoder.hpp"
#include "ai_learning/domain/environment/i_environment.hpp"
#include "ai_learning/learning/execution_sandbox.hpp"
#include "ai_learning/learning/self_modifier.hpp"
#include "ai_learning/reasoning/world_model.hpp"
#include "ai_learning/learning/metacognition.hpp"
#include "ai_learning/learning/intrinsic_motivation.hpp"
#include "ai_learning/learning/skill_tree.hpp"
#include "ai_learning/learning/autonomous_learning_loop.hpp"
#include "ai_learning/learning/problem_solver.hpp"
#include "ai_learning/learning/development_milestones.hpp"
#include "ai_learning/learning/analogical_transfer.hpp"
#include "ai_learning/learning/continual_learner.hpp"
#include "ai_learning/learning/abstract_concept.hpp"
#include "ai_learning/learning/social_learning.hpp"
#include "ai_learning/learning/emotion_engine.hpp"
#include "ai_learning/learning/insight_engine.hpp"
#include "ai_learning/learning/meta_learner.hpp"
#include "ai_learning/learning/active_experimenter.hpp"
#include "ai_learning/learning/integrated_learner.hpp"

#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ai_learning::core {

/// 发展阶段序列
inline const std::vector<std::string> kStageOrder = {
    "sensorimotor", "single_word", "two_word", "complex", "literacy"
};

/// 自主学习循环结果
struct AutonomousLearnResult {
    int total_steps = 0;
    int episodes = 0;
    double total_reward = 0.0;
    double avg_error = 0.0;
    int consolidations = 0;
    double elapsed_ms = 0.0;
};

/// 统一学习体
class Learner {
public:
    explicit Learner(const LearnerConfig& config);

    // ── 文本学习 ─────────────────────────────────────────────────

    /// 从文本学习
    auto learn_from_text(const std::string& text,
                         const std::string& source = "text")
        -> learning::TextLearnResult;

    /// 使用统计学习器观察文本（概念涌现）
    auto observe_text(const std::string& text)
        -> std::map<std::string, std::vector<std::string>>;

    /// 统一推理（多模式）
    auto reason(const std::string& question) const
        -> std::vector<reasoning::ReasoningResult>;

    /// 思考/回答问题
    auto think(const std::string& question) const
        -> std::string;

    // ── 感知循环 ─────────────────────────────────────────────────

    /// 感知原始输入（使用多模态编码器）
    auto perceive(const std::map<std::string, std::vector<float>>& raw_input)
        -> std::vector<float>;

    /// 选择动作（好奇心驱动）
    auto choose_action(const std::vector<float>& obs) -> int;

    /// 从经验学习
    auto learn_from_experience(const std::vector<float>& obs,
                                int action,
                                const std::vector<float>& next_obs,
                                float reward) -> double;

    // ── 记忆 ─────────────────────────────────────────────────────

    /// 存储经验到情景记忆
    void remember(const std::vector<float>& obs, int action,
                  const std::vector<float>& next_obs, float reward, float error);

    /// 按线索检索相关经验
    auto recall(const std::vector<float>& cue, int k = 5) const
        -> std::vector<domain::MemoryItem>;

    /// 巩固记忆（模拟睡眠）
    auto consolidate() -> std::map<std::string, double>;

    // ── 自主学习循环 ─────────────────────────────────────────────

    /// 自主学习：在环境中运行完整闭环
    /// @param env 环境接口
    /// @param max_steps 最大总步数
    /// @param max_episodes 最大回合数（0=无限制）
    /// @param consolidation_interval 每隔多少步巩固一次
    auto autonomous_learn(domain::IEnvironment& env,
                          int max_steps = 1000,
                          int max_episodes = 0,
                          int consolidation_interval = 100)
        -> AutonomousLearnResult;

    // ── 发展阶段 ─────────────────────────────────────────────────

    /// 获取当前发展阶段
    [[nodiscard]] auto stage() const -> const std::string& {
        return stage_;
    }

    /// 尝试晋升阶段
    auto try_advance(const std::map<std::string, double>& evaluation) -> bool;

    // ── 自主进化 ─────────────────────────────────────────────────

    /// 评估能力
    auto evaluate_capabilities() -> std::map<std::string, std::map<std::string, double>>;

    /// 执行进化
    auto evolve(int iterations = 1)
        -> std::map<std::string, double>;

    // ── 统计与持久化 ─────────────────────────────────────────────

    /// 获取统计
    [[nodiscard]] auto get_stats() const
        -> std::map<std::string, double>;

    /// 保存状态到文件
    void save(const std::string& path) const;

    /// 从文件加载状态
    void load(const std::string& path);

    // ── 子系统访问 ───────────────────────────────────────────────

    [[nodiscard]] auto knowledge_graph()
        -> domain::knowledge::KnowledgeGraph& { return kg_; }
    [[nodiscard]] auto knowledge_graph() const
        -> const domain::knowledge::KnowledgeGraph& { return kg_; }

    [[nodiscard]] auto engine()
        -> learning::PredictiveCodingEngine& { return engine_; }
    [[nodiscard]] auto engine() const
        -> const learning::PredictiveCodingEngine& { return engine_; }

    [[nodiscard]] auto text_learner()
        -> learning::TextLearner& { return text_learner_; }
    [[nodiscard]] auto text_learner() const
        -> const learning::TextLearner& { return text_learner_; }

    [[nodiscard]] auto encoder()
        -> perception::MultiModalEncoder& { return encoder_; }
    [[nodiscard]] auto encoder() const
        -> const perception::MultiModalEncoder& { return encoder_; }

    [[nodiscard]] auto distributional_semantics()
        -> learning::DistributionalSemantics& { return ds_; }
    [[nodiscard]] auto distributional_semantics() const
        -> const learning::DistributionalSemantics& { return ds_; }
    [[nodiscard]] auto embedding_trainer()
        -> learning::EmbeddingTrainer& { return embedding_trainer_; }
    [[nodiscard]] auto embedding_trainer() const
        -> const learning::EmbeddingTrainer& { return embedding_trainer_; }

    [[nodiscard]] auto config() const -> const LearnerConfig& {
        return config_;
    }

    // ── 四大人类核心能力 ─────────────────────────────────────────

    /// 能力 1：动手做 — 在沙箱中执行代码，获取真实反馈
    auto learn_by_doing(const std::string& code,
                         const std::string& language = "cpp")
        -> learning::ExecutionFeedback;

    /// 能力 2：自我修改 — 执行一轮自我进化
    auto self_evolve()
        -> MutationResult;

    /// 能力 3：因果推理 — 在世界模型中学习因果并推理
    void learn_causal(const std::vector<std::string>& events,
                       const std::string& outcome);

    auto reason_causal(const std::string& question) const
        -> reasoning::CounterfactualResult;

    auto plan_with_world_model(const std::string& goal) const
        -> reasoning::ImaginationPlan;

    /// 能力 4：元认知 — "知道自己不知道什么"
    auto metacognitive_report()
        -> MetacognitiveReport;

    auto knows_about(const std::string& topic) const
        -> bool;

    auto what_should_i_learn() const
        -> std::vector<KnowledgeGap>;

    // ── 自主学习系统 ─────────────────────────────────────

    /// 生成自主学习目标
    auto generate_learning_goal()
        -> learning::LearningGoal;

    /// 自主学习循环（内置策略）
    auto autonomous_learning_run(int iterations = 10)
        -> learning::AutonomousLoopReport;

    /// 求解问题
    auto solve_problem(const std::string& problem_description)
        -> learning::Solution;

    /// 检查发展里程碑
    auto check_milestones()
        -> std::vector<learning::MilestoneEvent>;

    /// 获取学习进展快照
    [[nodiscard]] auto learning_progress() const
        -> learning::ProgressSnapshot;

    /// 访问内在动机引擎
    [[nodiscard]] auto motivation_engine()
        -> learning::IntrinsicMotivationEngine& { return motivation_; }
    [[nodiscard]] auto motivation_engine() const
        -> const learning::IntrinsicMotivationEngine& { return motivation_; }

    /// 访问技能树
    [[nodiscard]] auto skill_tree()
        -> learning::SkillTree& { return skill_tree_; }
    [[nodiscard]] auto skill_tree() const
        -> const learning::SkillTree& { return skill_tree_; }

    /// 访问里程碑系统
    [[nodiscard]] auto milestones()
        -> learning::DevelopmentMilestones& { return milestones_; }
    [[nodiscard]] auto milestones() const
        -> const learning::DevelopmentMilestones& { return milestones_; }

    /// 访问问题求解器
    [[nodiscard]] auto problem_solver()
        -> learning::ProblemSolver& { return problem_solver_; }

    // ── Phase 3：高级认知能力 ──────────────────────────────────

    /// 跨领域类比迁移：将源领域知识迁移到目标领域
    auto analogical_transfer(
        const std::vector<learning::ConceptDescriptor>& source_concepts,
        const std::vector<learning::ConceptDescriptor>& target_concepts,
        const std::vector<std::string>& source_facts)
        -> learning::TransferResult;

    /// 持续学习保护：注册知识防止遗忘
    void protect_knowledge(const std::string& knowledge_id,
                           const std::string& domain,
                           double confidence,
                           int usage_count);

    /// 持续学习保护：检测遗忘
    auto detect_forgetting() const
        -> std::vector<learning::ForgettingAlert>;

    /// 抽象概念形成：观察实例，触发概念涌现
    auto form_abstractions(const std::string& instance_id,
                           const std::vector<std::string>& attributes,
                           const std::map<std::string, double>& features = {},
                           const std::vector<std::string>& relations = {})
        -> learning::ConceptFormationReport;

    /// 访问类比迁移引擎
    [[nodiscard]] auto analogy_engine()
        -> learning::AnalogicalTransferEngine& { return analogy_engine_; }

    /// 访问持续学习引擎
    [[nodiscard]] auto continual_learner()
        -> learning::ContinualLearner& { return continual_; }

    /// 访问抽象概念引擎
    [[nodiscard]] auto concept_engine()
        -> learning::AbstractConceptEngine& { return concept_engine_; }

    // ── Phase 4：增强智能 ────────────────────────────────────

    /// 社会观察学习：观察一次行为
    auto observe_behavior(const learning::BehaviorObservation& observation)
        -> learning::SocialLearningReport;

    /// 情感处理：处理一次情绪事件
    auto process_emotion(const learning::EmotionEvent& event)
        -> learning::EmotionState;

    /// 尝试产生顿悟
    auto try_insight(const std::string& problem_context)
        -> std::optional<learning::InsightEvent>;

    /// 访问社会学习引擎
    [[nodiscard]] auto social_engine()
        -> learning::SocialLearningEngine& { return social_engine_; }

    /// 访问情感引擎
    [[nodiscard]] auto emotion_engine()
        -> learning::EmotionEngine& { return emotion_engine_; }

    /// 访问顿悟引擎
    [[nodiscard]] auto insight_engine_ref()
        -> learning::InsightEngine& { return insight_engine_; }

    // ── Phase 5：高级元认知 ────────────────────────────────────

    /// 元学习：推荐最佳学习策略
    auto meta_recommend(const learning::TaskDescriptor& task) const
        -> learning::MetaLearningRecommendation;

    /// 元学习：记录学习经验
    void meta_record(const learning::LearningExperience& experience);

    /// 元学习：自我反思
    auto meta_reflect() const -> std::vector<std::string>;

    /// 主动实验：从观察生成假设
    auto generate_hypothesis(const std::string& observation,
                              const std::string& domain)
        -> learning::Hypothesis;

    /// 主动实验：自动设计实验
    auto design_experiment()
        -> std::optional<learning::ExperimentDesign>;

    /// 主动实验：记录实验结果
    auto record_experiment(const learning::ExperimentResult& result)
        -> std::string;

    /// 主动实验：构建理论
    auto build_theory(const std::string& domain)
        -> std::optional<learning::Theory>;

    /// 访问元学习引擎
    [[nodiscard]] auto meta_learner()
        -> learning::MetaLearner& { return meta_learner_; }

    /// 访问实验引擎
    [[nodiscard]] auto experimenter()
        -> learning::ActiveExperimenter& { return experimenter_; }

    // ── Phase 6：深度整合 ────────────────────────────────────

    /// 全流水线学习闭环（观察→学习→类比→实验→反思→顿悟）
    auto integrated_pipeline(const std::string& observation,
                              const std::string& domain)
        -> learning::IntegratedPipelineReport;

    /// 元学习驱动的学习会话
    auto meta_guided_learn(
        const std::vector<std::string>& known_topics,
        const std::map<std::string, double>& mastery_map)
        -> learning::MetaGuidedSessionReport;

    /// 情感调制后的系统参数
    auto emotion_modulated_params() const
        -> learning::EmotionModulatedParams;

    /// 实验驱动的自主探索
    auto experiment_driven_explore(const std::string& domain)
        -> learning::ExperimentDrivenExplorationReport;

    /// 社会学习加速类比迁移
    auto social_accelerated_transfer(
        const std::string& source_domain,
        const std::string& target_domain,
        const std::vector<learning::ConceptDescriptor>& target_concepts)
        -> learning::SocialAnalogicalReport;

    /// 访问整合编排器
    [[nodiscard]] auto integrated()
        -> learning::IntegratedLearner& { return integrated_; }

private:
    /// 检查晋升条件
    [[nodiscard]] auto check_promotion_(
        const std::map<std::string, double>& evaluation) const -> bool;

    // 配置
    LearnerConfig config_;

    // 核心子系统（值语义，不使用指针）
    domain::knowledge::KnowledgeGraph kg_;
    learning::PredictiveCodingEngine engine_;
    learning::TextLearner text_learner_;
    memory::EpisodicMemory           episodic_memory_;
    learning::STDP                   stdp_;
    reasoning::ActivationSpread      activation_spread_;
    learning::KnowledgeVerifier      verifier_;
    reasoning::SimulationReasoning   simulation_reasoning_;
    learning::StatisticalLearner     stat_learner_;
    // 嵌入学习子系统
    learning::DistributionalSemantics ds_;
    learning::EmbeddingTrainer        embedding_trainer_;
    int                               consolidation_count_ = 0;
    reasoning::UnifiedReasoningEngine unified_engine_;
    language::GroundingModule         grounding_;
    language::DevelopmentTracker      dev_tracker_;
    learning::HippocampalMemory       hippocampal_;
    learning::CorticalMemory          cortical_;
    learning::SleepConsolidation      sleep_consolidat_;
    perception::MultiModalEncoder     encoder_;

    // 四大人类核心能力子系统
    learning::LocalProcessSandbox     sandbox_;
    SelfModifier                      self_modifier_;
    reasoning::WorldModel             world_model_;
    MetacognitionEngine               metacognition_;

    // 自主学习系统
    learning::IntrinsicMotivationEngine motivation_;
    learning::SkillTree                 skill_tree_;
    learning::DevelopmentMilestones     milestones_;
    learning::ProblemSolver             problem_solver_;

    // Phase 3：高级认知能力
    learning::AnalogicalTransferEngine  analogy_engine_;
    learning::ContinualLearner          continual_;
    learning::AbstractConceptEngine     concept_engine_;

    // Phase 4：增强智能
    learning::SocialLearningEngine      social_engine_;
    learning::EmotionEngine             emotion_engine_;
    learning::InsightEngine             insight_engine_;

    // Phase 5：高级元认知
    learning::MetaLearner               meta_learner_;
    learning::ActiveExperimenter        experimenter_;

    // Phase 6：深度整合
    learning::IntegratedLearner         integrated_;

    // 发展状态
    std::string stage_;
    int stage_index_ = 0;

    // 统计
    int total_steps_ = 0;
    std::deque<float> error_history_;

    // 能力评估
    std::map<std::string, std::map<std::string, double>> capabilities_;
};

}  // namespace ai_learning::core
