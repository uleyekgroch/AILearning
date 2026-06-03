/**
 * @file graph_exporter.hpp
 * @brief 知识图谱导出工具 — GraphML, Cytoscape JSON, D3 JSON
 *
 * 零外部依赖，纯 C++20 + nlohmann/json 实现。
 */

#pragma once

#include "ai_learning/domain/knowledge/knowledge_graph.hpp"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace ai_learning::server {

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════
// GraphML 导出 — Cytoscape / Gephi 兼容
// ═══════════════════════════════════════════════════════════════════

/// 导出知识图谱为 GraphML XML
inline auto export_graphml(const domain::knowledge::KnowledgeGraph& kg)
    -> std::string {
    std::ostringstream out;
    out << R"(<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="type" for="node" attr.name="type" attr.type="string"/>
  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>
  <key id="rel_type" for="edge" attr.name="rel_type" attr.type="string"/>
  <key id="rel_confidence" for="edge" attr.name="rel_confidence" attr.type="double"/>
  <graph id="knowledge_graph" edgedefault="directed">
)";

    auto ids = kg.get_all_entity_ids();
    for (const auto& id : ids) {
        auto ent = kg.get_entity(id);
        if (!ent) continue;
        out << "    <node id=\"" << id << "\">\n";
        out << "      <data key=\"type\">" << ent->get().type() << "</data>\n";
        out << "      <data key=\"confidence\">" << ent->get().confidence() << "</data>\n";
        for (const auto& [k, v] : ent->get().properties()) {
            out << "      <data key=\"" << k << "\">" << v << "</data>\n";
        }
        out << "    </node>\n";
    }

    auto rels = kg.get_relations_of("", "both");
    int edge_id = 0;
    for (const auto& rel_ref : rels) {
        const auto& rel = rel_ref.get();
        out << "    <edge id=\"e" << edge_id++ << "\" source=\""
            << rel.source_id() << "\" target=\"" << rel.target_id() << "\">\n";
        out << "      <data key=\"rel_type\">" << rel.type() << "</data>\n";
        out << "      <data key=\"rel_confidence\">" << rel.confidence() << "</data>\n";
        out << "    </edge>\n";
    }

    out << "  </graph>\n</graphml>";
    return out.str();
}

// ═══════════════════════════════════════════════════════════════════
// Cytoscape JSON — cytoscape.js 直接消费
// ═══════════════════════════════════════════════════════════════════

/// 导出知识图谱为 Cytoscape.js 元素数组
inline auto export_cytoscape(const domain::knowledge::KnowledgeGraph& kg)
    -> json {
    json elements = json::array();

    auto ids = kg.get_all_entity_ids();
    for (const auto& id : ids) {
        auto ent = kg.get_entity(id);
        if (!ent) continue;
        json node;
        node["data"]["id"] = id;
        node["data"]["label"] = ent->get().type() + ":" + id;
        node["data"]["type"] = ent->get().type();
        node["data"]["confidence"] = ent->get().confidence();
        for (const auto& [k, v] : ent->get().properties()) {
            node["data"][k] = v;
        }
        node["group"] = "nodes";
        elements.push_back(node);
    }

    auto rels = kg.get_relations_of("", "both");
    int edge_id = 0;
    for (const auto& rel_ref : rels) {
        const auto& rel = rel_ref.get();
        json edge;
        edge["data"]["id"] = "e" + std::to_string(edge_id++);
        edge["data"]["source"] = rel.source_id();
        edge["data"]["target"] = rel.target_id();
        edge["data"]["label"] = rel.type();
        edge["data"]["confidence"] = rel.confidence();
        edge["group"] = "edges";
        elements.push_back(edge);
    }

    return elements;
}

// ═══════════════════════════════════════════════════════════════════
// D3.js 力导向图 JSON
// ═══════════════════════════════════════════════════════════════════

/// 导出知识图谱为 D3.js 力导向图格式 {nodes, links}
inline auto export_d3(const domain::knowledge::KnowledgeGraph& kg)
    -> json {
    json result;
    json nodes = json::array();
    json links = json::array();

    auto ids = kg.get_all_entity_ids();
    for (const auto& id : ids) {
        auto ent = kg.get_entity(id);
        if (!ent) continue;
        json n;
        n["id"] = id;
        n["label"] = id;
        n["type"] = ent->get().type();
        n["group"] = ent->get().type();  // 用于 D3 颜色分组
        n["confidence"] = ent->get().confidence();
        for (const auto& [k, v] : ent->get().properties()) {
            n[k] = v;
        }
        nodes.push_back(n);
    }

    auto rels = kg.get_relations_of("", "both");
    for (const auto& rel_ref : rels) {
        const auto& rel = rel_ref.get();
        json l;
        l["source"] = rel.source_id();
        l["target"] = rel.target_id();
        l["type"] = rel.type();
        l["value"] = rel.confidence();
        links.push_back(l);
    }

    result["nodes"] = nodes;
    result["links"] = links;
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// 统一导出 — 根据 format 参数分发
// ═══════════════════════════════════════════════════════════════════

inline auto export_graph(const domain::knowledge::KnowledgeGraph& kg,
                         const std::string& format) -> std::string {
    if (format == "graphml") return export_graphml(kg);
    if (format == "cytoscape") return export_cytoscape(kg).dump(2);
    if (format == "d3") return export_d3(kg).dump(2);
    // 默认：内部 JSON 格式
    return export_d3(kg).dump(2);
}

}  // namespace ai_learning::server
