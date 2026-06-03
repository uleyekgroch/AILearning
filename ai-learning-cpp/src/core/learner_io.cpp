/**
 * @file learner_io.cpp
 * @brief Learner 持久化 — save / load 序列化实现
 */

#include "ai_learning/core/learner.hpp"

#include <fstream>
#include <string>

namespace ai_learning::core {

void Learner::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) return;

    // -- 元数据 --
    out << "[meta]\n";
    out << "version=1\n";
    out << "stage=" << stage_ << "\n";
    out << "stage_index=" << stage_index_ << "\n";
    out << "total_steps=" << total_steps_ << "\n";

    // -- 配置 --
    out << "[config]\n";
    out << "obs_dim=" << config_.obs_dim << "\n";
    out << "action_dim=" << config_.action_dim << "\n";
    out << "learning_rate=" << config_.learning_rate << "\n";

    // -- 知识图谱 --
    out << "[knowledge_graph]\n";
    out << "entities=" << kg_.entity_count() << "\n";
    out << "relations=" << kg_.relation_count() << "\n";

    // -- 统计学习 --
    out << "[statistics]\n";
    out << "hippocampal_episodes=" << hippocampal_.size() << "\n";
    out << "cortical_facts=" << cortical_.size() << "\n";
    out << "learning_progress=" << engine_->get_learning_progress() << "\n";
    out << "curiosity=" << engine_->get_curiosity() << "\n";

    // -- 误差历史 --
    out << "[error_history]\n";
    out << "count=" << error_history_.size() << "\n";
    int cnt = 0;
    for (auto e : error_history_) {
        out << "e" << cnt << "=" << e << "\n";
        ++cnt;
    }

    // -- 模态权重 --
    out << "[modality_weights]\n";
    for (const auto& [mod, w] : encoder_.get_modality_weights()) {
        out << mod << "=" << w << "\n";
    }

    // -- 嵌入学习 --
    if (config_.embedding_learning_enabled) {
        out << "[embeddings]\n";
        auto ds_stats = ds_.stats();
        out << "ds_concepts=" << ds_stats.cpts_represented << "\n";
        out << "ds_dimensions=" << ds_stats.total_dimensions << "\n";
        out << "ds_texts_processed=" << ds_stats.texts_processed << "\n";
        out << "embedding_vocab_size=" << embedding_trainer_.vocab_size() << "\n";
        out << "embedding_trained=" << (embedding_trainer_.is_trained() ? 1 : 0) << "\n";
        out << "consolidation_count=" << consolidation_count_ << "\n";
    }
}

void Learner::load(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return;

    std::string section;
    std::string line;
    while (std::getline(in, line)) {
        // 空行跳过
        if (line.empty()) continue;

        // 检测节
        if (line[0] == '[') {
            section = line.substr(1, line.size() - 2);
            continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto key = line.substr(0, eq);
        auto val = line.substr(eq + 1);

        if (section == "meta") {
            if (key == "stage") {
                stage_ = val;
                for (int i = 0; i < static_cast<int>(kStageOrder.size()); ++i) {
                    if (kStageOrder[i] == stage_) {
                        stage_index_ = i;
                        break;
                    }
                }
            } else if (key == "total_steps") {
                total_steps_ = std::stoi(val);
            }
        } else if (section == "error_history") {
            if (key[0] == 'e') {
                error_history_.push_back(std::stof(val));
                // 限制历史长度
                if (error_history_.size() > 1000) {
                    error_history_.pop_front();
                }
            }
        } else if (section == "modality_weights") {
            encoder_.update_weight(key, std::stod(val) -
                encoder_.get_modality_weights().count(key)
                ? (encoder_.get_modality_weights().at(key))
                : 0.0);
        } else if (section == "embeddings") {
            if (key == "consolidation_count") {
                consolidation_count_ = std::stoi(val);
            }
        }
    }
}

}  // namespace ai_learning::core
