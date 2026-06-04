/**
 * @file learner_io.cpp
 * @brief Learner 持久化 — 生产级 JSON Checkpoint (序列化/反序列化)
 */

#include "ai_learning/core/learner.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace ai_learning::core {

void Learner::save(const std::string& path) const {
    json snap;

    // 1. Meta (元数据)
    snap["meta"] = {
        {"version", "2.0"},
        {"stage", stage_},
        {"stage_index", stage_index_},
        {"total_steps", total_steps_},
        {"pc_steps", pc_steps_}
    };

    // 2. Config (重建大脑所需的关键配置)
    snap["config"] = {
        {"obs_dim", config_.obs_dim},
        {"action_dim", config_.action_dim},
        {"hidden_dims", config_.hidden_dims},
        {"learning_rate", config_.learning_rate},
        {"initial_stage", config_.initial_stage}
    };

    // 3. Predictive Engine Tensors (潜意识神经突触权重)
    if (engine_) {
        auto p_state = engine_->save_state();
        snap["predictive_engine"]["weights"] = p_state.weights;
        snap["predictive_engine"]["shape"] = p_state.shape;
    }
    if (semantic_engine_) {
        auto s_state = semantic_engine_->save_state();
        snap["semantic_engine"]["weights"] = s_state.weights;
        snap["semantic_engine"]["shape"] = s_state.shape;
    }

    // 4. Knowledge Graph (大脑皮层语义图谱)
    json kg_json = json::array();
    auto all_ids = kg_.get_all_entity_ids();
    for (const auto& id : all_ids) {
        auto ent_opt = kg_.get_entity(id);
        if (!ent_opt) continue;
        const auto& ent = ent_opt->get();
        json e_json;
        e_json["id"] = ent.id();
        e_json["type"] = ent.type();
        e_json["confidence"] = ent.confidence();
        
        json rel_json = json::array();
        auto rels = kg_.get_relations_of(id, "out");
        for (const auto& r_ref : rels) {
            const auto& r = r_ref.get();
            rel_json.push_back({
                {"target", r.target_id()},
                {"type", r.type()},
                {"confidence", r.confidence()}
            });
        }
        e_json["relations"] = rel_json;
        kg_json.push_back(e_json);
    }
    snap["knowledge_graph"] = kg_json;

    // 保存到磁盘
    std::ofstream out(path);
    if (out.is_open()) {
        out << snap.dump(); // 紧凑格式，节省空间
        std::cout << "[IO] 成功保存脑快照至: " << path << " (包含 " << all_ids.size() << " 个概念节点)\n";
    } else {
        std::cerr << "[IO] 保存失败: 无法打开文件 " << path << "\n";
    }
}

void Learner::load(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "[IO] 无法加载脑快照: " << path << " (作为全新大脑启动)\n";
        return;
    }

    try {
        json snap = json::parse(in);

        // 1. Meta
        if (snap.contains("meta")) {
            stage_ = snap["meta"].value("stage", "sensorimotor");
            total_steps_ = snap["meta"].value("total_steps", 0);
        }

        // 2. Predictive Engine
        if (engine_ && snap.contains("predictive_engine")) {
            learning::PredictiveEngineState p_state;
            p_state.weights = snap["predictive_engine"]["weights"].get<std::vector<float>>();
            p_state.shape = snap["predictive_engine"]["shape"].get<std::vector<int>>();
            engine_->load_state(p_state);
        }
        if (semantic_engine_ && snap.contains("semantic_engine")) {
            learning::PredictiveEngineState s_state;
            s_state.weights = snap["semantic_engine"]["weights"].get<std::vector<float>>();
            s_state.shape = snap["semantic_engine"]["shape"].get<std::vector<int>>();
            semantic_engine_->load_state(s_state);
        }

        // 3. Knowledge Graph
        if (snap.contains("knowledge_graph")) {
            for (const auto& item : snap["knowledge_graph"]) {
                domain::knowledge::Entity ent(
                    item["id"].get<std::string>(),
                    item["type"].get<std::string>(),
                    {}, // 略去属性恢复
                    item["confidence"].get<double>(),
                    "loaded_memory"
                );
                kg_.add_entity(ent);
                
                for (const auto& rel : item["relations"]) {
                    kg_.add_relation(domain::knowledge::Relation(
                        item["id"].get<std::string>(),
                        rel["target"].get<std::string>(),
                        rel["type"].get<std::string>(),
                        rel["confidence"].get<double>()
                    ));
                }
            }
        }
        std::cout << "[IO] 成功唤醒加载脑快照: " << path << " (恢复了 " << kg_.entity_count() << " 个概念)\n";
    } catch (const std::exception& e) {
        std::cerr << "[IO] 脑快照加载失败 (文件损坏?): " << e.what() << "\n";
    }
}

}  // namespace ai_learning::core