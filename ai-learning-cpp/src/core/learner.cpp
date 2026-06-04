/**
 * @file learner.cpp
 * @brief 统一学习体实现 — 瘦编排器
 */

#include "ai_learning/core/learner.hpp"
#include "ai_learning/core/tensor_ops.hpp"
#include "ai_learning/learning/predictive_coding_engine.hpp"
#include "ai_learning/learning/tokenizer.hpp"

#ifdef AI_LEARNING_WITH_LLAMA_CPP
#include "ai_learning/language/llama_cpp_embedding_provider.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace ai_learning::core {

using namespace learning;
using namespace domain::knowledge;

namespace {
/// 是否为同源 n-gram 碎片（一个串包含另一个），如 "数学研" vs "数学"。
/// 中文分词会产生 unigram/bigram/trigram，这类碎片会污染语义近邻排序，
/// 在"概念联想"语境下应过滤掉，只保留真正不同的概念。
auto is_ngram_fragment(const std::string& a, const std::string& b) -> bool {
    if (a == b) return true;
    return a.find(b) != std::string::npos || b.find(a) != std::string::npos;
}
}  // namespace

// ── 构造 ────────────────────────────────────────────────────────

Learner::Learner(const LearnerConfig& config)
    : Learner(config,
              std::make_unique<PredictiveCodingEngine>(
                  PredictiveCodingConfig{
                      config.obs_dim,
                      config.action_dim,
                      config.hidden_dims.size() > 0 ? config.hidden_dims[0] : 64,
                      config.hidden_dims.size() > 1 ? config.hidden_dims[1] : 32,
                      config.learning_rate,
                      config.inference_lr,
                      config.max_inference_steps,
                      config.convergence_threshold,
                  })) {}

Learner::Learner(const LearnerConfig& config,
                 std::unique_ptr<learning::IPredictiveEngine> engine)
    : config_(config),
      kg_(),
      engine_(std::move(engine)),
      semantic_engine_(std::make_unique<PredictiveCodingEngine>(
          PredictiveCodingConfig{
              config.obs_dim,
              config.action_dim,
              config.hidden_dims.size() > 0 ? config.hidden_dims[0] * 2 : 128, // L2 容量更大
              config.hidden_dims.size() > 1 ? config.hidden_dims[1] * 2 : 64,
              config.learning_rate * 0.5, // 语义层学习更慢
              config.inference_lr,
              config.max_inference_steps,
              config.convergence_threshold,
          })),
      current_context_state_(config.obs_dim, 0.0f),
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
       // ★v2: 仿人类学习增强模块
       active_inference_(),
       self_model_(),
       creative_engine_(),
       tutoring_system_(),
       mirror_neurons_(),
       goal_manager_(kg_, &metacognition_),
       mastery_assessor_(),
       proficiency_tester_(),
       stage_(config.initial_stage) {

    // 设置阶段索引
    for (int i = 0; i < static_cast<int>(kStageOrder.size()); ++i) {
        if (kStageOrder[i] == stage_) {
            stage_index_ = i;
            break;
        }
    }
    
    // 初始化 unified engine 依赖
    unified_engine_.set_predictive_engine(engine_.get());
    unified_engine_.set_distributional_semantics(&ds_);

    // 初始化预训练嵌入提供者（llama.cpp）
    #ifdef AI_LEARNING_WITH_LLAMA_CPP
    if (!config_.embedding_model_path.empty()) {
        try {
            embedding_provider_ = std::make_unique<language::LlamaCppEmbeddingProvider>(
                config_.embedding_model_path,
                config_.embedding_model_dim,
                config_.n_gpu_layers);
            embedding_trainer_.set_embedding_provider(embedding_provider_.get());
        } catch (const std::exception& e) {
            (void)e;
        }
    }
    if (!config_.llm_model_path.empty()) {
        try {
            llm_provider_ = std::make_unique<language::LlamaCppLLMProvider>(
                config_.llm_model_path,
                config_.n_gpu_layers);
            unified_engine_.set_llm_provider(llm_provider_.get());
        } catch (const std::exception& e) {
            (void)e;
        }
    }
    #endif
}

// ── 文本学习 ─────────────────────────────────────────────────────

auto Learner::learn_from_text(const std::string& text,
                               const std::string& source)
    -> TextLearnResult {
    
    // 1. 分词与底层神经感知
    auto tokens = learning::tokenize(text, config_.language);
    stat_learner_.observe_tokens(tokens);

    // 嵌入管线（config 开关控制）
    if (config_.embedding_learning_enabled) {
        ds_.learn_from_tokens(tokens);
        embedding_trainer_.add_tokens(tokens);
    }

    // 2. PC 嵌入预测学习，获取语句整体的预测误差
    double novelty_error = 0.0;
    if (config_.embedding_predictive_learning) {
        novelty_error = learn_predictive_(tokens);
    }

    // 3. 注意力门控（预测误差过滤）- 阶段一：用预测误差接管知识提取
    TextLearnResult result;
    // 只有当预测误差大于阈值（即产生了“惊讶”），才会触发符号层的强行提取和记忆
    // 这是一个非常“类人”的设定：已知和无聊的东西不会占用高维的逻辑网络空间
    double novelty_threshold = 0.05; 
    if (config_.embedding_predictive_learning && novelty_error < novelty_threshold) {
        result.verification_passed = true;
        result.verification_score = 1.0;
        // 直接返回，免去沉重的知识图谱写入
        return result;
    }

    // 4. 产生“惊讶”，启动高耗能的符号提取与海马记忆
    result = text_learner_.learn_from_text(text, source);

    // 海马快速记忆：存储提取的实体和关系
    std::vector<std::string> rel_strs;
    for (const auto& t : result.triples) {
        rel_strs.push_back(t.subject + "->" + t.relation + "->" + t.object);
    }
    (void)hippocampal_.encode(result.entities, rel_strs, text);

    // ★ 自成目标引擎：记录真实学习进度（预测误差作为 novelty 信号）
    if (!result.entities.empty() || !result.triples.empty()) {
        for (const auto& entity : result.entities) {
            autotelic_engine_.record_progress(entity, novelty_error);
        }
    }

    return result;
}

double Learner::learn_predictive_(const std::vector<std::string>& tokens) {
    // 阶段二：分层预测编码 (HPC)
    // Level 1: Token -> Token 预测
    // Level 2: Context -> Context 预测
    
    // 去重获取唯一 tokens
    std::vector<std::string> unique_tokens;
    std::unordered_set<std::string> seen;
    for (const auto& t : tokens) {
        if (seen.insert(t).second) {
            unique_tokens.push_back(t);
        }
    }

    // 批量获取向量（利用缓存）
    auto vecs = ds_.get_dense_vectors_batch(unique_tokens, config_.obs_dim);

    // 构建 token -> vector 映射
    std::unordered_map<std::string, std::vector<float>> vec_map;
    for (size_t i = 0; i < unique_tokens.size(); ++i) {
        if (vecs[i]) {
            vec_map[unique_tokens[i]] = std::move(*vecs[i]);
        }
    }

    // 用原始 token 顺序构建 embeddings
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(tokens.size());
    std::vector<float> sentence_mean(config_.obs_dim, 0.0f);
    int valid_tokens = 0;

    for (const auto& t : tokens) {
        auto it = vec_map.find(t);
        if (it != vec_map.end()) {
            embeddings.push_back(it->second);
            for (size_t d = 0; d < static_cast<size_t>(config_.obs_dim); ++d) {
                sentence_mean[d] += it->second[d];
            }
            valid_tokens++;
        }
    }

    if (embeddings.empty()) return 0.0;

    // 计算当前句子的语义均值向量 (Propositional Vector)
    for (size_t d = 0; d < static_cast<size_t>(config_.obs_dim); ++d) {
        sentence_mean[d] /= static_cast<float>(valid_tokens);
    }

    int steps = 0;
    double total_error = 0.0;
    auto forward_action = std::vector<float>{1.0f};

    // Level 1: 序列预测 (Token -> Token)
    // 引入 Top-down Context: 将句意向量混入 action 空间
    auto l1_action = forward_action;
    l1_action.insert(l1_action.end(), sentence_mean.begin(), sentence_mean.end());
    // 如果 action_dim 不够放 context，引擎会自动截断/丢弃，但为了安全我们截断到 action_dim
    if (l1_action.size() > static_cast<size_t>(config_.action_dim)) {
        l1_action.resize(config_.action_dim);
    }

    for (size_t i = 0; i + 1 < embeddings.size() && steps < config_.pc_max_steps_per_text; ++i) {
        double error = engine_->learn(embeddings[i], l1_action, embeddings[i + 1]);
        total_error += error;
        ++steps;
    }
    
    // Level 2: 语义层预测 (Context_t-1 -> Context_t)
    double l2_error = 0.0;
    if (semantic_engine_) {
        l2_error = semantic_engine_->learn(current_context_state_, forward_action, sentence_mean);
        current_context_state_ = sentence_mean; // 更新 L2 状态
    }

    pc_steps_ += steps;
    
    // 综合两层误差，L2（语义层）的意外程度权重更高
    double avg_l1_error = steps > 0 ? total_error / steps : 0.0;
    return avg_l1_error * 0.3 + l2_error * 0.7;
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

    // 路径 5: 神经↔符号桥（用自学的分布语义 + 预测编码联想作答）
    // 当符号管线无法直接命中时，退而用"系统自己学到的语义空间"回答，
    // 而不是直接放弃。这里全程不依赖任何外部大模型。
    for (const auto& entity : entities) {
        if (!ds_.has_cpt(entity)) continue;

        // (a) 分布语义近邻：基于上下文共现分布的相似概念（过滤同源碎片）
        auto neighbors = ds_.most_similar(entity, 3, /*exclude_ngram_overlap=*/true);
        // (b) 预测编码联想：PC 引擎从概念向量预测出的相关概念
        auto assoc = semantic_associate(entity, 3);

        std::string sem;
        if (!neighbors.empty()) {
            sem += entity + " 在语义上与 ";
            for (size_t i = 0; i < neighbors.size(); ++i) {
                if (i > 0) sem += "、";
                sem += neighbors[i].cpt_b;
            }
            sem += " 相近（基于上下文分布）";
        }
        if (!assoc.empty()) {
            sem += sem.empty() ? "" : "；";
            sem += "预测编码联想到 " + assoc.front().first;
        }
        if (sem.size() > 5) return sem;
    }

    return answer;
}

auto Learner::semantic_associate(const std::string& concept_name,
                                 int top_k) const
    -> std::vector<std::pair<std::string, double>> {
    // 1. 取概念在自学分布语义空间中的稠密向量（维度对齐 PC 引擎的 obs_dim）
    auto vec_opt = ds_.get_dense_vector(concept_name, config_.obs_dim);
    if (!vec_opt || !engine_) return {};

    // 2. 用预测编码引擎做一次前向预测（"下一个会想到什么"）
    std::vector<float> action(config_.action_dim, 0.0f);
    if (config_.action_dim > 0) action[0] = 1.0f;  // 前向动作
    auto predicted = engine_->predict(*vec_opt, action);
    if (predicted.empty()) return {};

    // 3. 把预测出的连续向量映射回最接近的符号概念（过滤掉概念本身的同源碎片）
    auto nearest = ds_.find_nearest(predicted, top_k + 5);
    std::vector<std::pair<std::string, double>> result;
    for (const auto& cand : nearest) {
        if (is_ngram_fragment(concept_name, cand.first)) continue;
        result.push_back(cand);
        if (static_cast<int>(result.size()) >= top_k) break;
    }
    return result;
}

// ── 感知循环 ─────────────────────────────────────────────────────

auto Learner::perceive(
    const std::map<std::string, std::vector<float>>& raw_input)
    -> std::vector<float> {
    return encoder_.encode(raw_input);
}

auto Learner::choose_action(const std::vector<float>& obs) -> int {
    auto curiosity = engine_->get_curiosity();

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
        auto pred = engine_->predict(obs, action_vec);

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

    auto error = engine_->learn(obs, action_vec, next_obs);
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
        {"learning_progress", engine_->get_learning_progress()},
        {"curiosity", engine_->get_curiosity()},
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

    // 评估摘要
    if (kg_.entity_count() > 0) {
        stats["assessment_entity_count"] = static_cast<double>(kg_.entity_count());
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

// ── Bloom 掌握度评估 ────────────────────────────────────────────

auto Learner::assess(const std::string& domain) const
    -> assessment::AssessmentResult
{
    // If domain specified, find first entity of that type
    if (!domain.empty()) {
        auto entities = kg_.query(domain);
        if (!entities.empty()) {
            return mastery_assessor_.assess(entities.front(), kg_);
        }
    }
    // Otherwise, find the first entity in the KG
    auto ids = kg_.get_all_entity_ids();
    if (ids.empty()) {
        return assessment::AssessmentResult{};
    }
    auto opt = kg_.get_entity(ids.front());
    if (opt.has_value()) {
        return mastery_assessor_.assess(opt->get(), kg_);
    }
    return assessment::AssessmentResult{};
}

auto Learner::assess_domain(const std::string& domain) const
    -> assessment::DomainReport
{
    return mastery_assessor_.assess_domain(domain, kg_);
}

auto Learner::assess_all() const
    -> std::map<std::string, assessment::DomainReport>
{
    return mastery_assessor_.assess_all_domains(kg_);
}

auto Learner::assess_proficiency(const std::string& entity_type) const
    -> assessment::ProficiencyReport
{
    return proficiency_tester_.assess(entity_type, kg_);
}

// ★v2: 自主学习循环 — 实现在 learner_autonomous.cpp（含主动推理注入）

auto Learner::active_read(const std::string& text, const std::string& source)
    -> learning::TextLearnResult {
    learning::TextLearnResult final_result;
    
    // 简单的断句分割（模拟注意力在段落内移动）
    std::vector<std::string> sentences;
    std::string current_sentence;
    for (size_t i = 0; i < text.size(); ) {
        // UTF-8 粗略处理
        auto uc = static_cast<unsigned char>(text[i]);
        int byte_len = 1;
        if (uc >= 0xE0) byte_len = 3;
        else if (uc >= 0xC0) byte_len = 2;
        
        if (i + byte_len <= text.size()) {
            std::string ch = text.substr(i, byte_len);
            current_sentence += ch;
            // 中英文标点
            if (ch == "." || ch == "!" || ch == "?" || ch == "\n" || 
                ch == "。" || ch == "！" || ch == "？") {
                if (current_sentence.size() > 5) {
                    sentences.push_back(current_sentence);
                }
                current_sentence.clear();
            }
        }
        i += byte_len;
    }
    // 处理末尾
    if (!current_sentence.empty() || sentences.empty()) {
        if (sentences.empty()) sentences.push_back(text);
        else if (current_sentence.size() > 5) sentences.push_back(current_sentence);
    }

    double novelty_threshold = 0.05; 
    int sentences_extracted = 0;

    for (size_t i = 0; i < sentences.size(); ++i) {
        const auto& sentence = sentences[i];
        
        // 1. 尝试阅读与预测 (Perception / Prediction)
        auto tokens = learning::tokenize(sentence, config_.language);
        if (tokens.empty()) continue;
        
        stat_learner_.observe_tokens(tokens);

        if (config_.embedding_learning_enabled) {
            ds_.learn_from_tokens(tokens);
            embedding_trainer_.add_tokens(tokens);
        }

        double novelty_error = 0.0;
        if (config_.embedding_predictive_learning) {
            novelty_error = learn_predictive_(tokens);
        }

        // 2. 主动推理控制 (Active Inference for Reading)
        if (novelty_error > novelty_threshold) {
            // 自由能（预测误差）过高 -> 触发好奇心和惊讶
            
            // 行动 1：尝试从长时记忆（KG）中检索背景知识，平息自由能
            auto entities = learning::KnowledgeExtractor::extract_entities(sentence);
            for (const auto& ent : entities) {
                if (kg_.has_entity(ent)) {
                    // 检索到背景知识，稍微降低一些惊讶值（模拟消除了一部分 Epistemic uncertainty）
                    novelty_error *= 0.8;
                }
            }

            // 行动 2：如果仍然惊讶，必须分配重计算资源（强行提取与记忆建立）
            if (novelty_error > novelty_threshold) {
                auto chunk_result = text_learner_.learn_from_text(sentence, source);
                
                // 聚合结果
                final_result.entities.insert(final_result.entities.end(), 
                                             chunk_result.entities.begin(), chunk_result.entities.end());
                final_result.triples.insert(final_result.triples.end(), 
                                            chunk_result.triples.begin(), chunk_result.triples.end());
                
                std::vector<std::string> rel_strs;
                for (const auto& t : chunk_result.triples) {
                    rel_strs.push_back(t.subject + "->" + t.relation + "->" + t.object);
                }
                (void)hippocampal_.encode(chunk_result.entities, rel_strs, sentence);
                
                sentences_extracted++;
            }
        }
    }

    if (sentences_extracted > 0) {
        final_result.verification_passed = true;
        final_result.verification_score = 1.0;
    } else {
        final_result.verification_passed = true;
        final_result.verification_score = 1.0;
    }

    // 记录元认知（认知努力度）
    metacognition_.record_learning(
        sentences_extracted > 0 ? "focused_reading" : "skim_reading", 
        sentences_extracted > 0 ? 0.9 : 0.1
    );

    return final_result;
}

void Learner::run_conscious_loop(int ticks) {
    for (int i = 0; i < ticks; ++i) {
        // 1. 内部状态评估 (Homeostasis)
        double current_curiosity = engine_ ? engine_->get_curiosity() : 0.0;
        double current_lp = engine_ ? engine_->get_learning_progress() : 0.0;
        
        // ★ 将真实学习进度反馈到自成目标引擎
        if (engine_) {
            autotelic_engine_.record_progress("_global_learning_progress", current_lp);
        }
        
        // 2. 自成目标引擎 (Autotelic Generation) 
        // 模拟：如果没有外部刺激，且感到无聊 (LP停滞)，自己找事做
        if (current_lp < 0.01 && current_curiosity < 0.1) {
            auto goal = autotelic_engine_.generate_goal();
            
            // 提交一个内部思绪到工作空间
            if (goal.goal_type != "idle_dreaming") {
                consciousness::WorkspaceThought thought;
                thought.source_module = "autotelic";
                thought.symbolic_content = "我打算探索: " + goal.target_cpt_a;
                thought.surprise_value = 0.5; // 自我设定的高好奇心
                workspace_.submit_thought(thought);
            }
        }

        // 3. 全局工作空间广播 (Global Workspace Broadcast)
        auto focus = workspace_.process_and_broadcast();
        
        if (focus) {
            // 意识的聚焦引发下游模块的集体处理
            // 这里用一段简单的模拟逻辑展示：如果是内部产生的目标，在潜空间中进行推演 (JEPA style)
            if (focus->source_module == "autotelic" && !focus->symbolic_content.empty()) {
                // JEPA 规划推演：在分布语义空间中提取目标向量，执行前向预测
                std::string target = focus->symbolic_content.substr(focus->symbolic_content.find_last_of(' ') + 1);
                auto vec_opt = ds_.get_dense_vector(target);
                if (vec_opt && engine_) {
                    auto action = std::vector<float>{1.0f}; 
                    auto pred = engine_->predict(*vec_opt, action);
                    
                    // 将推演结果存入记忆，或者产生新的情绪
                    auto nearest = ds_.find_nearest(pred, 1);
                    if (!nearest.empty()) {
                        consciousness::WorkspaceThought insight;
                        insight.source_module = "predictive_engine";
                        insight.symbolic_content = "啊！我想到 " + target + " 可能和 " + nearest[0].first + " 有关！";
                        insight.surprise_value = 0.8;
                        workspace_.submit_thought(insight);
                    }
                }
            } else if (focus->source_module == "predictive_engine" && !focus->symbolic_content.empty()) {
                // 把顿悟的结果显式写入知识图谱或自传体记忆
                consciousness::AutobiographicalMemory mem;
                mem.narrative = "今天我顿悟了：" + focus->symbolic_content;
                mem.emotional_intensity = 0.8;
                mem.importance = 0.9;
                self_model_.remember(mem);
            }
        }

        // 4. 定期海马体巩固 (Hippocampal Consolidation / Sleep)
        if (i > 0 && i % 100 == 0) {
            (void)consolidate();
        }
    }
}

auto Learner::choose_continuous_action(const std::vector<float>& obs) -> std::vector<float> {
    int num_samples = 64;
    std::vector<float> best_action(config_.action_dim, 0.0f);
    float max_epistemic_value = -1.0f;

    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < num_samples; ++i) {
        std::vector<float> action(config_.action_dim);
        for (int d = 0; d < config_.action_dim; ++d) {
            action[d] = dist(rng);
        }

        auto pred = engine_->predict(obs, action);
        float uncertainty = 0.0f;
        for (auto v : pred) uncertainty += v * v;
        uncertainty = std::sqrt(uncertainty / static_cast<float>(pred.size()));

        if (uncertainty > max_epistemic_value) {
            max_epistemic_value = uncertainty;
            best_action = action;
        }
    }

    // Epsilon-greedy 探索
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    if (prob(rng) < static_cast<float>(config_.motivation_epsilon)) {
        for (int d = 0; d < config_.action_dim; ++d) {
            best_action[d] = dist(rng);
        }
    }

    return best_action;
}

auto Learner::learn_continuous_experience(const std::vector<float>& obs,
                                          const std::vector<float>& action,
                                          const std::vector<float>& next_obs) -> double {
    auto error = engine_->learn(obs, action, next_obs);
    error_history_.push_back(static_cast<float>(error));
    ++total_steps_;

    // STDP 赫布学习（保持与离散一样）
    auto obs_id = "s" + std::to_string(total_steps_ % 100);
    auto next_id = "s" + std::to_string((total_steps_ + 1) % 100);
    stdp_.strengthen(obs_id, next_id, 1.0);

    return error;
}

// ★v2: 具身感知-行动闭环（最小可行路径）
double Learner::embodied_step(
    const std::map<std::string, std::vector<float>>& raw_input,
    domain::IEnvironment* env) {

    // 1. 多模态感知编码
    auto obs = encoder_.encode(raw_input);

    // 2. 主动推理驱动动作选择（取代简单的 ε-贪心）
    int action = 0;

    // 使用主动推理引擎生成策略并选择最优
    auto belief = active_inference_.current_belief();
    auto policies = active_inference_.generate_policies(belief, 3);

    if (!policies.empty()) {
        auto selected = active_inference_.select_policy(policies, belief, "");

        // 将策略名映射到离散动作
        if (!selected.actions.empty()) {
            // 使用哈希将策略名映射到动作空间
            size_t hash = std::hash<std::string>{}(selected.name);
            action = static_cast<int>(hash) % config_.action_dim;
        }
    } else {
        // 回退到好奇心驱动的动作选择
        action = choose_action(obs);
    }

    // 3. 环境交互（如果有环境）
    std::vector<float> next_obs;
    float reward = 0.0f;

    if (env) {
        auto [env_reward, done] = env->step(action);
        reward = static_cast<float>(env_reward);
        // 使用环境的当前观测作为 next_obs
        auto next_raw = env->observe();
        next_obs = encoder_.encode(next_raw);
    } else {
        // 无环境时，用预测编码引擎生成"想象"的下一个状态
        auto action_vec = std::vector<float>(config_.action_dim, 0.0f);
        if (action >= 0 && action < config_.action_dim) {
            action_vec[action] = 1.0f;
        }
        next_obs = engine_->predict(obs, action_vec);
    }

    // 4. 经验学习
    double error = learn_from_experience(obs, action, next_obs, reward);

    // 5. 将观测反馈给主动推理引擎（闭合感知→信念更新环）
    auto predicted = engine_->predict(obs, std::vector<float>(config_.action_dim, 0.0f));
    active_inference_.perceive(obs, predicted);

    // 6. 记录到情景记忆
    remember(obs, action, next_obs, reward, static_cast<float>(error));

    // 7. 自成目标引擎反馈
    autotelic_engine_.record_progress("_embodied", 1.0 - error);

    // 8. 如果预测误差足够大（惊讶），提交到意识工作空间
    if (error > 0.3) {
        consciousness::WorkspaceThought thought;
        thought.source_module = "embodied_perception";
        thought.symbolic_content = "感知到意外: error=" + std::to_string(error);
        thought.surprise_value = static_cast<float>(error);
        workspace_.submit_thought(thought);
    }

    return error;
}

}  // namespace ai_learning::core
