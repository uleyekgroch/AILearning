/**
 * @file simulation.cpp
 * @brief 模拟推理实现
 */

#include "ai_learning/reasoning/simulation.hpp"
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

#include <algorithm>
#include <cmath>

namespace ai_learning::reasoning {

using domain::knowledge::KnowledgeGraph;

SimulationReasoning::SimulationReasoning(const KnowledgeGraph& kg)
    : kg_(kg),
      stats_({
          {"total_reasoning", 0},
          {"causal_chains_found", 0},
          {"counterfactuals_run", 0},
          {"analogies_found", 0},
      }) {}

auto SimulationReasoning::reason(
    const std::string& question,
    const std::vector<std::string>& activated_concepts) const -> SimulationResult {

    stats_["total_reasoning"]++;

    // 1. 场景构建
    auto scene = build_scene_(activated_concepts);

    // 2. 因果追踪
    auto chains = trace_causal_chains_(scene, question);
    stats_["causal_chains_found"] += static_cast<double>(chains.size());

    // 3. 反事实
    auto counterfactuals = std::vector<std::map<std::string, std::string>>{};
    if (is_counterfactual_(question)) {
        counterfactuals = simulate_counterfactuals_(scene, question);
        stats_["counterfactuals_run"] += static_cast<double>(counterfactuals.size());
    }

    // 4. 类比
    auto analogies = find_analogies_(scene, activated_concepts);
    stats_["analogies_found"] += static_cast<double>(analogies.size());

    // 5. 置信度
    auto confidence = assess_confidence_(scene, chains, analogies);

    return {
        .scene            = std::move(scene),
        .causal_chains    = std::move(chains),
        .counterfactuals  = std::move(counterfactuals),
        .analogies        = std::move(analogies),
        .confidence       = confidence,
        .reasoning_type   = classify_question_(question),
    };
}

auto SimulationReasoning::express(
    const SimulationResult& result,
    const std::string& /*question*/) const -> std::string {

    if (result.confidence < 0.1) return "";

    // 因果链表达
    if (!result.causal_chains.empty()) {
        const auto& best = result.causal_chains[0];
        if (best.steps.size() >= 2) {
            std::string answer = "根据因果分析：";
            for (size_t i = 0; i < best.steps.size(); ++i) {
                if (i > 0) answer += " → ";
                answer += best.steps[i];
            }
            return answer;
        }
    }

    // 反事实表达
    if (!result.counterfactuals.empty()) {
        const auto& cf = result.counterfactuals[0];
        auto it_condition = cf.find("condition");
        auto it_result   = cf.find("result");
        if (it_condition != cf.end() && it_result != cf.end()) {
            return "如果" + it_condition->second + "，那么"
                 + it_result->second;
        }
    }

    // 类比表达
    if (!result.analogies.empty()) {
        const auto& analogy = result.analogies[0];
        auto it_src = analogy.find("source");
        auto it_tgt = analogy.find("target");
        auto it_sim = analogy.find("similarity");
        if (it_src != analogy.end() && it_tgt != analogy.end()) {
            std::string sim_str = "0.5";
            if (it_sim != analogy.end()) sim_str = it_sim->second;
            return it_src->second + "与" + it_tgt->second
                 + "有类似的关系（相似度" + sim_str + "）";
        }
    }

    // 场景表达
    if (!result.scene.concepts.empty()) {
        std::string answer;
        for (size_t i = 0; i < std::min(result.scene.concepts.size(), size_t(3)); ++i) {
            if (i > 0) answer += "、";
            answer += result.scene.concepts[i];
        }
        return "相关概念：" + answer;
    }

    return "";
}

auto SimulationReasoning::get_stats() const -> std::map<std::string, double> {
    return stats_;
}

// ── 私有方法 ─────────────────────────────────────────────────────────

auto SimulationReasoning::build_scene_(
    const std::vector<std::string>& concepts) const -> SimulationScene {

    SimulationScene scene;
    scene.concepts = concepts;

    for (const auto& c : concepts) {
        // 从知识图谱获取实体的属性
        auto props = kg_.get_entity_properties(c);
        std::map<std::string, std::string> feat;
        for (const auto& [k, v] : props) {
            feat[k] = v;
        }
        if (!feat.empty()) {
            scene.features.push_back(std::move(feat));
        }

        // 获取关系
        auto rels = kg_.get_relations_of(c);
        for (const auto& r : rels) {
            std::map<std::string, std::string> rel_map;
            rel_map["source"] = c;
            rel_map["target"] = r.get().target_id();
            rel_map["type"]   = r.get().type();
            scene.relations.push_back(std::move(rel_map));
        }
    }

    // 场景置信度基于概念的覆盖度
    scene.confidence = concepts.empty() ? 0.0
                     : std::min(1.0, static_cast<double>(
                         scene.features.size() + scene.relations.size())
                         / static_cast<double>(concepts.size() * 2 + 1));

    return scene;
}

auto SimulationReasoning::trace_causal_chains_(
    const SimulationScene& scene,
    const std::string& /*question*/) const -> std::vector<CausalChain> {

    std::vector<CausalChain> chains;

    // 从场景关系构建因果链
    // 查找 "导致"、"引起" 等因果类型的关系
    static const std::vector<std::string> causal_types = {
        "导致", "引起", "使", "造成", "产生"
    };

    for (const auto& rel : scene.relations) {
        auto it = rel.find("type");
        if (it == rel.end()) continue;

        bool is_causal = false;
        for (const auto& ct : causal_types) {
            if (it->second.find(ct) != std::string::npos) {
                is_causal = true;
                break;
            }
        }

        if (is_causal) {
            CausalChain chain;
            chain.steps.push_back(rel.at("source"));
            chain.steps.push_back(it->second);
            chain.steps.push_back(rel.at("target"));
            chain.confidence = 0.7;
            chain.evidence.push_back(rel.at("source") + " " + it->second + " " + rel.at("target"));
            chains.push_back(std::move(chain));
        }
    }

    // 也尝试从连续概念构建传递链（A→B→C）
    if (chains.empty() && scene.concepts.size() >= 2) {
        // 通过知识图谱查找两个概念之间的路径
        for (size_t i = 0; i + 1 < scene.concepts.size(); ++i) {
            auto rels = kg_.get_relations_of(scene.concepts[i]);
            for (const auto& r : rels) {
                // 检查关系目标是否在场景中
                for (size_t j = i + 1; j < scene.concepts.size(); ++j) {
                    if (r.get().target_id() == scene.concepts[j]) {
                        CausalChain chain;
                        chain.steps.push_back(scene.concepts[i]);
                        chain.steps.push_back(r.get().type());
                        chain.steps.push_back(scene.concepts[j]);
                        chain.confidence = 0.5;
                        chains.push_back(std::move(chain));
                        break;
                    }
                }
            }
        }
    }

    return chains;
}

auto SimulationReasoning::is_counterfactual_(const std::string& question) -> bool {
    static const std::vector<std::string> markers = {
        "如果", "假如", "假设", "要是", "若"
    };
    for (const auto& m : markers) {
        if (question.find(m) != std::string::npos) return true;
    }
    return false;
}

auto SimulationReasoning::simulate_counterfactuals_(
    const SimulationScene& scene,
    const std::string& question) const
    -> std::vector<std::map<std::string, std::string>> {

    std::vector<std::map<std::string, std::string>> cfs;

    // 提取 "如果 X" 部分
    auto if_pos = question.find("如果");
    if (if_pos == std::string::npos) {
        if_pos = question.find("假如");
    }
    if (if_pos == std::string::npos) return cfs;

    auto condition = question.substr(if_pos);
    // 取到逗号或问号
    auto comma = condition.find_first_of("，,？?");
    if (comma != std::string::npos) {
        condition = condition.substr(0, comma);
    }

    // 用场景中的因果链生成反事实结果
    for (const auto& rel : scene.relations) {
        auto it_type = rel.find("type");
        if (it_type == rel.end()) continue;

        // 对于因果型关系，翻转条件
        static const std::vector<std::string> causal = {"导致", "引起"};
        for (const auto& ct : causal) {
            if (it_type->second.find(ct) != std::string::npos) {
                std::map<std::string, std::string> cf;
                cf["condition"] = condition;
                cf["result"] = rel.at("source") + "不会" + ct + rel.at("target");
                cfs.push_back(std::move(cf));
                break;
            }
        }
        if (cfs.size() >= 2) break;
    }

    // 如果没有因果链，用场景概念生成简单的反事实
    if (cfs.empty() && scene.concepts.size() >= 2) {
        std::map<std::string, std::string> cf;
        cf["condition"] = condition;
        cf["result"] = scene.concepts[0] + "和" + scene.concepts[1] + "的关系会改变";
        cfs.push_back(std::move(cf));
    }

    return cfs;
}

auto SimulationReasoning::find_analogies_(
    const SimulationScene& scene,
    const std::vector<std::string>& /*concepts*/) const
    -> std::vector<std::map<std::string, std::string>> {

    std::vector<std::map<std::string, std::string>> analogies;

    // 遍历场景中的关系，在知识图谱中找结构相似的对
    for (const auto& rel : scene.relations) {
        auto it_src = rel.find("source");
        auto it_tgt = rel.find("target");
        auto it_type = rel.find("type");
        if (it_src == rel.end() || it_tgt == rel.end() || it_type == rel.end())
            continue;

        const auto& src = it_src->second;
        const auto& tgt = it_tgt->second;
        const auto& type = it_type->second;

        // 查找具有相同关系类型的其他概念对
        auto all_srcs = kg_.get_all_entity_ids();
        int count = 0;
        for (const auto& other_src : all_srcs) {
            if (other_src == src) continue;
            auto other_rels = kg_.get_relations_of(other_src);
            for (const auto& other_r : other_rels) {
                if (other_r.get().type() == type && other_r.get().target_id() != tgt) {
                    std::map<std::string, std::string> analogy;
                    analogy["source"]     = src + "→" + tgt;
                    analogy["target"]     = other_src + "→" + other_r.get().target_id();
                    analogy["similarity"] = "0.6";
                    analogies.push_back(std::move(analogy));
                    if (++count >= 3) break;
                }
            }
            if (count >= 3) break;
        }
    }

    return analogies;
}

auto SimulationReasoning::assess_confidence_(
    const SimulationScene& scene,
    const std::vector<CausalChain>& chains,
    const std::vector<std::map<std::string, std::string>>& analogies)
    -> double {

    double confidence = 0.0;

    // 场景置信度
    confidence += scene.confidence * 0.3;

    // 因果链置信度
    if (!chains.empty()) {
        double best_conf = 0.0;
        for (const auto& c : chains) {
            best_conf = std::max(best_conf, c.confidence);
        }
        confidence += best_conf * 0.4;
    }

    // 类比支持度
    if (!analogies.empty()) {
        confidence += 0.6 * 0.3;
    }

    return std::min(1.0, confidence);
}

auto SimulationReasoning::classify_question_(const std::string& question)
    -> std::string {

    if (is_counterfactual_(question)) return "counterfactual";

    static const std::vector<std::string> causal_verbs = {
        "导致", "引起", "使", "让", "造成", "产生", "带来"
    };
    for (const auto& v : causal_verbs) {
        if (question.find(v) != std::string::npos) return "causal";
    }

    if (question.find("关系") != std::string::npos ||
        question.find("联系") != std::string::npos ||
        question.find("和")   != std::string::npos) {
        return "analogical";
    }

    return "factual";
}

}  // namespace ai_learning::reasoning
