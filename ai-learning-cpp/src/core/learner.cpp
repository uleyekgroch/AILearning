/**
 * @file learner.cpp
 * @brief 统一学习体实现 — 瘦编排器
 */

#include "ai_learning/core/learner.hpp"
#include "ai_learning/core/tensor_ops.hpp"
#include "ai_learning/learning/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace ai_learning::core {

using namespace learning;
using namespace domain::knowledge;

// ── 构造 ────────────────────────────────────────────────────────

Learner::Learner(const LearnerConfig& config)
    : config_(config),
      kg_(),
      engine_(PredictiveCodingConfig{
          config.obs_dim,
          config.action_dim,
          config.hidden_dims.size() > 0 ? config.hidden_dims[0] : 64,
          config.hidden_dims.size() > 1 ? config.hidden_dims[1] : 32,
          config.learning_rate,
          config.inference_lr,
          config.max_inference_steps,
          config.convergence_threshold,
      }),
      text_learner_(kg_),
      episodic_memory_(config.episodic_memory_capacity),
      stdp_(),
      activation_spread_(kg_),
      verifier_(),
      simulation_reasoning_(kg_),
      stat_learner_(),
      ds_(learning::DistributionalSemanticsConfig{
          5,                          // window_size
          config.ds_min_freq,         // min_cpt_freq
          0.5,                        // ppmi_threshold
          500,                        // max_dimensions
          10000,                      // max_cpts
          0.3,                        // similarity_threshold
          false                       // use_causal_prior: 关闭因果先验（构造时未连接 WorldModel）
      }),
      embedding_trainer_(learning::EmbeddingTrainerConfig{
          config.embedding_dim,       // embedding_dim
          config.embedding_window_size, // window_size
          config.embedding_neg_samples, // neg_samples
          config.embedding_learning_rate, // learning_rate
          0.001,                      // min_learning_rate
          config.embedding_epochs,    // epochs
          config.embedding_min_count, // min_count
          512,                        // batch_size
          50000,                      // max_vocab
          42                          // seed
      }),
      consolidation_count_(0),
      unified_engine_(kg_),
      hippocampal_(config.episodic_memory_capacity),
      cortical_(),
      sleep_consolidat_(hippocampal_, cortical_),
      encoder_(config.obs_dim),
      sandbox_(SandboxConfig{}),
      self_modifier_(),
      world_model_(42),
      metacognition_(0.5),
      motivation_(),
      skill_tree_(),
      milestones_(),
      problem_solver_(),
       analogy_engine_(),
       continual_(),
       concept_engine_(),
       social_engine_(),
       emotion_engine_(),
       insight_engine_(),
       meta_learner_(),
       experimenter_(),
       integrated_(meta_learner_, experimenter_, emotion_engine_,
                   social_engine_, analogy_engine_, insight_engine_,
                   motivation_, continual_, concept_engine_),
       goal_manager_(kg_, &metacognition_),
       stage_(config.initial_stage) {

    // 设置阶段索引
    for (int i = 0; i < static_cast<int>(kStageOrder.size()); ++i) {
        if (kStageOrder[i] == stage_) {
            stage_index_ = i;
            break;
        }
    }
}

// ── 文本学习 ─────────────────────────────────────────────────────

auto Learner::learn_from_text(const std::string& text,
                               const std::string& source)
    -> TextLearnResult {
    auto result = text_learner_.learn_from_text(text, source);

    // 通过统一分词器处理
    auto tokens = learning::tokenize(text, config_.language);
    stat_learner_.observe_tokens(tokens);

    // 嵌入管线（config 开关控制）
    if (config_.embedding_learning_enabled) {
        ds_.learn_from_tokens(tokens);
        embedding_trainer_.add_tokens(tokens);
    }

    // PC 嵌入预测学习
    if (config_.embedding_predictive_learning) {
        learn_predictive_(tokens);
    }

    // 海马快速记忆：存储提取的实体和关系
    std::vector<std::string> rel_strs;
    for (const auto& t : result.triples) {
        rel_strs.push_back(t.subject + "->" + t.relation + "->" + t.object);
    }
    (void)hippocampal_.encode(result.entities, rel_strs, text);

    return result;
}

void Learner::learn_predictive_(const std::vector<std::string>& tokens) {
    // 从 DS 获取有 PPMI 向量的概念
    std::vector<std::vector<float>> embeddings;
    for (const auto& t : tokens) {
        auto vec = ds_.get_dense_vector(t, config_.obs_dim);
        if (vec) {
            embeddings.push_back(std::move(*vec));
        }
    }

    // 连续概念嵌入对送入 PC engine
    int steps = 0;
    for (size_t i = 0; i + 1 < embeddings.size() && steps < config_.pc_max_steps_per_text; ++i) {
        auto action = std::vector<float>{1.0f};  // 固定 "预测下一个"
        engine_.learn(embeddings[i], action, embeddings[i + 1]);
        ++steps;
    }
    pc_steps_ += steps;
}

auto Learner::observe_text(const std::string& text)
    -> std::map<std::string, std::vector<std::string>> {
    return stat_learner_.observe(text);
}

auto Learner::reason(const std::string& question) const
    -> std::vector<reasoning::ReasoningResult> {
    return unified_engine_.reason(question);
}

auto Learner::think(const std::string& question) const
    -> std::string {
    // 路径 0-3: TextLearner 已有的推理管线
    auto answer = text_learner_.think(question);
    if (answer != "抱歉，我暂时不知道答案") return answer;

    // 路径 4: 模拟推理
    auto entities = learning::KnowledgeExtractor::extract_entities(question);
    if (!entities.empty()) {
        auto activated = activation_spread_.spread(entities, 2);
        if (!activated.empty()) {
            std::vector<std::string> concepts;
            for (const auto& node : activated) {
                concepts.push_back(node.entity_id);
            }

            auto result = simulation_reasoning_.reason(question, concepts);
            if (result.confidence >= 0.2) {
                auto sim_answer = simulation_reasoning_.express(result, question);
                if (!sim_answer.empty() && sim_answer.size() > 5) {
                    return sim_answer;
                }
            }
        }
    }

    return answer;
}

// ── 感知循环 ─────────────────────────────────────────────────────

auto Learner::perceive(
    const std::map<std::string, std::vector<float>>& raw_input)
    -> std::vector<float> {
    return encoder_.encode(raw_input);
}

auto Learner::choose_action(const std::vector<float>& obs) -> int {
    auto curiosity = engine_.get_curiosity();

    // ε-贪心：高好奇心时更多探索
    auto epsilon = static_cast<float>(config_.motivation_epsilon);
    if (curiosity > 0.5) epsilon = std::min(1.0f, epsilon * 2.0f);

    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng) < epsilon) {
        // 随机探索
        std::uniform_int_distribution<int> action_dist(0, config_.action_dim - 1);
        return action_dist(rng);
    }

    // 选择预测误差最大的动作（好奇心驱动）
    float max_error = -1.0f;
    int best_action = 0;

    for (int a = 0; a < config_.action_dim; ++a) {
        auto action_vec = std::vector<float>(config_.action_dim, 0.0f);
        action_vec[a] = 1.0f;
        auto pred = engine_.predict(obs, action_vec);

        float uncertainty = 0.0f;
        for (auto v : pred) uncertainty += v * v;
        uncertainty = std::sqrt(uncertainty / static_cast<float>(pred.size()));

        if (uncertainty > max_error) {
            max_error = uncertainty;
            best_action = a;
        }
    }

    return best_action;
}

auto Learner::learn_from_experience(const std::vector<float>& obs,
                                      int action,
                                      const std::vector<float>& next_obs,
                                      float /*reward*/) -> double {
    auto action_vec = std::vector<float>(config_.action_dim, 0.0f);
    if (action >= 0 && action < config_.action_dim) {
        action_vec[action] = 1.0f;
    }

    auto error = engine_.learn(obs, action_vec, next_obs);
    error_history_.push_back(static_cast<float>(error));
    ++total_steps_;

    // STDP 赫布学习：强化 obs→next_obs 关联
    auto obs_id = "s" + std::to_string(total_steps_ % 100);
    auto next_id = "s" + std::to_string((total_steps_ + 1) % 100);
    stdp_.strengthen(obs_id, next_id, 1.0);

    return error;
}

// ── 记忆 ─────────────────────────────────────────────────────────

void Learner::remember(const std::vector<float>& obs, int action,
                       const std::vector<float>& next_obs,
                       float reward, float error) {
    std::map<std::string, std::string> meta;
    meta["action"] = std::to_string(action);
    meta["reward"] = std::to_string(reward);
    meta["error"]  = std::to_string(error);
    meta["step"]   = std::to_string(total_steps_);

    std::vector<float> repr;
    repr.reserve(obs.size() + 1 + next_obs.size());
    repr.insert(repr.end(), obs.begin(), obs.end());
    repr.push_back(static_cast<float>(action));
    repr.insert(repr.end(), next_obs.begin(), next_obs.end());

    episodic_memory_.store(repr, meta);
}

auto Learner::recall(const std::vector<float>& cue, int k) const
    -> std::vector<domain::MemoryItem> {
    return episodic_memory_.retrieve(cue, k);
}

auto Learner::consolidate() -> std::map<std::string, double> {
    auto report = episodic_memory_.consolidate();
    stdp_.decay_all();

    // 睡眠巩固：海马→皮层转移
    auto sleep_report = sleep_consolidat_.sleep();

    report["hippocampal_episodes"] = static_cast<double>(hippocampal_.size());
    report["cortical_facts"]       = static_cast<double>(cortical_.size());
    report["consolidated"]         = static_cast<double>(sleep_report.memories_consolidated);
    report["forgotten"]            = static_cast<double>(sleep_report.memories_forgotten);

    // 嵌入训练（周期性）
    if (config_.embedding_learning_enabled) {
        ++consolidation_count_;
        if (consolidation_count_ % config_.embedding_train_interval == 0) {
            auto ds_concepts = ds_.all_cpts();
            embedding_trainer_.import_vocabulary(ds_concepts);
            auto emb_result = embedding_trainer_.train();
            report["embedding_vocab_size"] = static_cast<double>(emb_result.vocab_size);
            report["embedding_loss"]       = emb_result.final_loss;
            report["embedding_trained"]    = 1.0;
        } else {
            report["embedding_trained"] = 0.0;
        }
        auto ds_stats = ds_.stats();
        report["ds_concepts"]   = static_cast<double>(ds_stats.cpts_represented);
        report["ds_dimensions"] = static_cast<double>(ds_stats.total_dimensions);
    }

    report["pc_steps"] = static_cast<double>(pc_steps_);

    return report;
}

// ── 自主学习循环 ─────────────────────────────────────────────────

auto Learner::autonomous_learn(domain::IEnvironment& env,
                                int max_steps,
                                int max_episodes,
                                int consolidation_interval)
    -> AutonomousLearnResult {
    AutonomousLearnResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    int step = 0;
    int episode = 0;
    double total_reward = 0.0;
    double total_error = 0.0;
    int consolidations = 0;

    while (step < max_steps) {
        // 重置环境开始新一轮
        auto raw_obs = env.reset();
        auto obs = perceive(raw_obs);
        ++episode;

        bool done = false;
        double episode_reward = 0.0;

        while (!done && step < max_steps) {
            // 选择动作
            int action = choose_action(obs);

            // 环境执行
            auto [reward, env_done] = env.step(action);
            done = env_done;
            episode_reward += reward;

            // 获取新观测
            auto next_raw = env.observe();
            auto next_obs = perceive(next_raw);

            // 从经验学习
            auto error = learn_from_experience(obs, action, next_obs,
                                                static_cast<float>(reward));
            total_error += error;

            // 记忆存储
            remember(obs, action, next_obs,
                     static_cast<float>(reward), static_cast<float>(error));

            obs = next_obs;
            ++step;

            // 周期性巩固
            if (step % consolidation_interval == 0) {
                (void)consolidate();
                ++consolidations;
            }
        }

        total_reward += episode_reward;

        // 检查回合限制
        if (max_episodes > 0 && episode >= max_episodes) break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.total_steps = step;
    result.episodes = episode;
    result.total_reward = total_reward;
    result.avg_error = (step > 0) ? total_error / step : 0.0;
    result.consolidations = consolidations;
    result.elapsed_ms = elapsed;

    return result;
}

// ── 发展阶段 ─────────────────────────────────────────────────────

auto Learner::try_advance(
    const std::map<std::string, double>& evaluation) -> bool {
    if (!check_promotion_(evaluation)) return false;

    if (stage_index_ + 1 < static_cast<int>(kStageOrder.size())) {
        ++stage_index_;
        stage_ = kStageOrder[stage_index_];
        return true;
    }
    return false;
}

auto Learner::check_promotion_(
    const std::map<std::string, double>& evaluation) const -> bool {
    if (evaluation.empty()) return false;

    double avg = 0.0;
    for (const auto& [_, v] : evaluation) avg += v;
    avg /= static_cast<double>(evaluation.size());

    return avg >= 0.7;
}

// ── 自主进化 ─────────────────────────────────────────────────────

auto Learner::evaluate_capabilities()
    -> std::map<std::string, std::map<std::string, double>> {
    std::map<std::string, std::map<std::string, double>> caps;

    auto text_result = learn_from_text("人工智能是计算机科学的一个分支");
    caps["text_learning"] = {
        {"score", text_result.verification_passed ? 1.0 : 0.5},
        {"entities", static_cast<double>(text_result.entities.size())},
    };

    auto answer = think("什么是人工智能");
    caps["knowledge_retrieval"] = {
        {"score", answer.find("人工智能") != std::string::npos ? 1.0 : 0.0},
    };

    capabilities_ = caps;
    return caps;
}

auto Learner::evolve(int iterations)
    -> std::map<std::string, double> {
    auto caps_before = evaluate_capabilities();

    double score_before = 0.0;
    for (const auto& [_, m] : caps_before) {
        if (m.contains("score")) score_before += m.at("score");
    }
    score_before /= static_cast<double>(caps_before.size());

    for (int i = 0; i < iterations; ++i) {
        learn_from_text("机器学习是人工智能的子领域");
        learn_from_text("深度学习使用神经网络");
    }

    auto caps_after = evaluate_capabilities();

    double score_after = 0.0;
    for (const auto& [_, m] : caps_after) {
        if (m.contains("score")) score_after += m.at("score");
    }
    score_after /= static_cast<double>(caps_after.size());

    return {
        {"iterations", static_cast<double>(iterations)},
        {"score_before", score_before},
        {"score_after", score_after},
    };
}

// ── 统计 ─────────────────────────────────────────────────────────

auto Learner::get_stats() const -> std::map<std::string, double> {
    std::map<std::string, double> stats{
        {"total_steps", static_cast<double>(total_steps_)},
        {"entity_count", static_cast<double>(kg_.entity_count())},
        {"relation_count", static_cast<double>(kg_.relation_count())},
        {"stage_index", static_cast<double>(stage_index_)},
        {"learning_progress", engine_.get_learning_progress()},
        {"curiosity", engine_.get_curiosity()},
        {"hippocampal_episodes", static_cast<double>(hippocampal_.size())},
        {"cortical_facts", static_cast<double>(cortical_.size())},
    };

    if (config_.embedding_learning_enabled) {
        auto ds_stats = ds_.stats();
        stats["ds_concepts"] = static_cast<double>(ds_stats.cpts_represented);
        stats["embedding_vocab"] = static_cast<double>(embedding_trainer_.vocab_size());
        stats["embedding_trained"] = embedding_trainer_.is_trained() ? 1.0 : 0.0;
    }

    if (config_.embedding_predictive_learning) {
        stats["pc_steps"] = static_cast<double>(pc_steps_);
    }

    return stats;
}

// ── Phase 3：高级认知能力 ────────────────────────────────────────

auto Learner::analogical_transfer(
    const std::vector<learning::ConceptDescriptor>& source_concepts,
    const std::vector<learning::ConceptDescriptor>& target_concepts,
    const std::vector<std::string>& source_facts)
    -> learning::TransferResult
{
    auto result = analogy_engine_.transfer(
        source_concepts, target_concepts, source_facts);

    // 将迁移成功的知识注册到持续学习保护
    for (const auto& knowledge : result.transferred_knowledge) {
        continual_.register_knowledge(
            knowledge, result.target_domain, result.transfer_quality, 1);
    }

    return result;
}

void Learner::protect_knowledge(const std::string& knowledge_id,
                                 const std::string& domain,
                                 double confidence,
                                 int usage_count) {
    continual_.register_knowledge(knowledge_id, domain, confidence, usage_count);
}

auto Learner::detect_forgetting() const
    -> std::vector<learning::ForgettingAlert> {
    return continual_.detect_forgetting();
}

auto Learner::form_abstractions(
    const std::string& instance_id,
    const std::vector<std::string>& attributes,
    const std::map<std::string, double>& features,
    const std::vector<std::string>& relations)
    -> learning::ConceptFormationReport
{
    return concept_engine_.observe_instance(
        instance_id, attributes, features, relations);
}

// ── Phase 4：增强智能 ────────────────────────────────────────

auto Learner::observe_behavior(
    const learning::BehaviorObservation& observation)
    -> learning::SocialLearningReport
{
    // 情感调制：高唤醒状态增强社会学习效果
    auto report = social_engine_.observe(observation);

    // 将学到的策略注册到持续学习保护
    for (const auto& strategy : report.learned_strategies) {
        continual_.register_knowledge(
            strategy.id, strategy.domain,
            strategy.observed_success_rate, strategy.observation_count);
    }

    return report;
}

auto Learner::process_emotion(const learning::EmotionEvent& event)
    -> learning::EmotionState
{
    return emotion_engine_.process_event(event);
}

auto Learner::try_insight(const std::string& problem_context)
    -> std::optional<learning::InsightEvent>
{
    auto insight = insight_engine_.try_insight(problem_context);

    // 如果产生顿悟，触发情感事件（兴奋）
    if (insight.has_value()) {
        learning::EmotionEvent event;
        event.event_type = "surprise";
        event.domain = "insight";
        event.description = insight->new_perspective;
        event.magnitude = insight->surprise_level;
        event.actual_outcome = insight->confidence;
        event.expected_outcome = 0.3;
        emotion_engine_.process_event(event);

        // 保护顿悟知识
        continual_.register_knowledge(
            insight->id, "insight",
            insight->confidence, 1);
    }

    return insight;
}

// ── Phase 5：高级元认知 ────────────────────────────────────

auto Learner::meta_recommend(
    const learning::TaskDescriptor& task) const
    -> learning::MetaLearningRecommendation
{
    return meta_learner_.recommend_strategy(task);
}

void Learner::meta_record(
    const learning::LearningExperience& experience)
{
    meta_learner_.record_experience(experience);
}

auto Learner::meta_reflect() const -> std::vector<std::string>
{
    return meta_learner_.reflect();
}

auto Learner::generate_hypothesis(
    const std::string& observation,
    const std::string& domain)
    -> learning::Hypothesis
{
    return experimenter_.generate_hypothesis(observation, domain);
}

auto Learner::design_experiment()
    -> std::optional<learning::ExperimentDesign>
{
    return experimenter_.auto_design_experiment();
}

auto Learner::record_experiment(
    const learning::ExperimentResult& result)
    -> std::string
{
    return experimenter_.record_result(result);
}

auto Learner::build_theory(const std::string& domain)
    -> std::optional<learning::Theory>
{
    return experimenter_.build_theory(domain);
}

// ── Phase 6：深度整合 ────────────────────────────────────

auto Learner::integrated_pipeline(const std::string& observation,
                                   const std::string& domain)
    -> learning::IntegratedPipelineReport
{
    return integrated_.run_full_pipeline(observation, domain);
}

auto Learner::meta_guided_learn(
    const std::vector<std::string>& known_topics,
    const std::map<std::string, double>& mastery_map)
    -> learning::MetaGuidedSessionReport
{
    return integrated_.meta_guided_session(known_topics, mastery_map);
}

auto Learner::emotion_modulated_params() const
    -> learning::EmotionModulatedParams
{
    return integrated_.compute_system_params();
}

auto Learner::experiment_driven_explore(const std::string& domain)
    -> learning::ExperimentDrivenExplorationReport
{
    return integrated_.experiment_driven_exploration(domain);
}

auto Learner::social_accelerated_transfer(
    const std::string& source_domain,
    const std::string& target_domain,
    const std::vector<learning::ConceptDescriptor>& target_concepts)
    -> learning::SocialAnalogicalReport
{
    return integrated_.social_analogical_transfer(
        source_domain, target_domain, target_concepts);
}

}  // namespace ai_learning::core
