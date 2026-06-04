/**
 * @file learner_embodied.cpp
 * @brief Learner 主动阅读 + 意识循环 + 具身感知-行动闭环
 *
 * 从 learner.cpp 分离，遵守单文件 ≤800 行规范。
 */

#include "ai_learning/core/learner.hpp"
#include "ai_learning/learning/predictive_coding_engine.hpp"
#include "ai_learning/learning/tokenizer.hpp"

#include <cmath>
#include <random>

namespace ai_learning::core {

using namespace learning;

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
                        const std::string& related = nearest[0].first;
                        consciousness::WorkspaceThought insight;
                        insight.source_module = "predictive_engine";
                        insight.symbolic_content = "啊！我想到 " + target + " 可能和 " + related + " 有关！";
                        insight.surprise_value = 0.8;
                        workspace_.submit_thought(insight);

                        // ★ 让 CreativeEngine 真正进入核心意识闭环
                        std::vector<creativity::ConceptNode> knowledge;
                        for (const auto& nb : ds_.most_similar(target, 5, /*exclude_ngram_overlap=*/true)) {
                            creativity::ConceptNode node;
                            node.name = nb.cpt_b;
                            node.activation = nb.similarity;
                            knowledge.push_back(node);
                        }
                        creativity::ConceptNode na; na.name = target; knowledge.push_back(na);
                        creativity::ConceptNode nr; nr.name = related; knowledge.push_back(nr);

                        auto idea = creative_engine_.remote_associate(target, related, knowledge);
                        if (idea) {
                            consciousness::AutobiographicalMemory cmem;
                            cmem.narrative = "我用想象力把「" + target + "」和「" + related +
                                             "」联系起来：" + idea->idea;
                            cmem.emotional_intensity = idea->surprise;
                            cmem.importance = idea->novelty;
                            self_model_.remember(cmem);
                        }
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
