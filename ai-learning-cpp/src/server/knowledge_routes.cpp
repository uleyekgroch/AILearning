/**
 * @file knowledge_routes.cpp
 * @brief 知识图谱可视化路由（K.2）
 *
 * 端点:
 *   GET  /api/knowledge/graph?format=cytoscape|d3|graphml
 *   GET  /api/knowledge/stats
 *   GET  /api/knowledge/search?q=<query>&type=<type>
 *   POST /api/knowledge/query — BFS 路径查找
 */

#include "route_groups.hpp"
#include "ai_learning/server/graph_exporter.hpp"
#include "ai_learning/server/logger.hpp"
#include "ai_learning/server/metrics_collector.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>

using json = nlohmann::json;

static auto ok_resp(const json& body) -> crow::response {
    crow::response res{200, body.dump()};
    res.set_header("Content-Type", "application/json");
    return res;
}
static auto json_resp(int code, const json& body) -> crow::response {
    crow::response res{code, body.dump()};
    res.set_header("Content-Type", "application/json");
    return res;
}
static auto err_resp(int code, const std::string& msg) -> crow::response {
    return json_resp(code, json{{"error", msg}});
}

void ai_learning::server::register_knowledge_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    const ServerConfig& /*config*/,
    SharedState& state) {

    // GET /api/knowledge/graph — 导出图谱（支持多种格式）
    CROW_ROUTE(app, "/api/knowledge/graph").methods("GET"_method)
    ([&](const crow::request& req) -> crow::response {
        ScopedTimer timer(state.metrics, "knowledge_graph_export");
        auto format = req.url_params.get("format");
        std::string fmt = format ? format : "d3";

        const auto& kg = learner.knowledge_graph();

        if (fmt == "graphml") {
            auto xml = export_graphml(kg);
            crow::response res{200, xml};
            res.set_header("Content-Type", "application/xml");
            return res;
        }
        if (fmt == "cytoscape") {
            auto cy = export_cytoscape(kg);
            crow::response res{200, cy.dump(2)};
            res.set_header("Content-Type", "application/json");
            return res;
        }
        // 默认 d3 / json
        auto d3 = export_d3(kg);
        crow::response res{200, d3.dump(2)};
        res.set_header("Content-Type", "application/json");
        return res;
    });

    // GET /api/knowledge/stats — 图谱统计
    CROW_ROUTE(app, "/api/knowledge/stats").methods("GET"_method)
    ([&](const crow::request& /*req*/) -> crow::response {
        ScopedTimer timer(state.metrics, "knowledge_stats");
        const auto& kg = learner.knowledge_graph();

        json body;
        body["entity_count"] = kg.entity_count();
        body["relation_count"] = kg.relation_count();
        body["type_distribution"] = kg.type_distribution();

        // 统计实体最多的前 5 个类型
        auto dist = kg.type_distribution();
        std::vector<std::pair<std::string, int>> sorted(dist.begin(), dist.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        json top_types = json::array();
        for (size_t i = 0; i < std::min<size_t>(5, sorted.size()); ++i) {
            top_types.push_back({{"type", sorted[i].first}, {"count", sorted[i].second}});
        }
        body["top_types"] = top_types;

        return ok_resp(body);
    });

    // GET /api/knowledge/search — 语义搜索实体
    CROW_ROUTE(app, "/api/knowledge/search").methods("GET"_method)
    ([&](const crow::request& req) -> crow::response {
        ScopedTimer timer(state.metrics, "knowledge_search");
        auto q = req.url_params.get("q");
        auto type_filter = req.url_params.get("type");
        if (!q) return err_resp(400, "missing 'q' parameter");

        std::string query = q;
        std::string q_lower = query;
        std::transform(q_lower.begin(), q_lower.end(), q_lower.begin(), ::tolower);

        const auto& kg = learner.knowledge_graph();
        json results = json::array();

        for (const auto& id : kg.get_all_entity_ids()) {
            auto ent = kg.get_entity(id);
            if (!ent) continue;

            // 类型过滤
            if (type_filter && ent->get().type() != std::string(type_filter)) continue;

            // 简单子串匹配（ID、类型、属性值）
            bool match = false;
            std::string id_lower = id;
            std::transform(id_lower.begin(), id_lower.end(), id_lower.begin(), ::tolower);
            if (id_lower.find(q_lower) != std::string::npos) match = true;

            std::string type_lower = ent->get().type();
            std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), ::tolower);
            if (type_lower.find(q_lower) != std::string::npos) match = true;

            for (const auto& [k, v] : ent->get().properties()) {
                std::string v_lower = v;
                std::transform(v_lower.begin(), v_lower.end(), v_lower.begin(), ::tolower);
                if (v_lower.find(q_lower) != std::string::npos) {
                    match = true;
                    break;
                }
            }

            if (match) {
                json item;
                item["id"] = id;
                item["type"] = ent->get().type();
                item["confidence"] = ent->get().confidence();
                item["properties"] = ent->get().properties();
                results.push_back(item);
            }
        }

        json body;
        body["query"] = query;
        body["count"] = results.size();
        body["results"] = results;
        return ok_resp(body);
    });

    // POST /api/knowledge/path — BFS 查找两个实体间的最短路径
    CROW_ROUTE(app, "/api/knowledge/path").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        ScopedTimer timer(state.metrics, "knowledge_path");
        try {
            auto body = json::parse(req.body);
            if (!body.contains("source") || !body.contains("target")) {
                return err_resp(400, "missing 'source' or 'target' field");
            }
            auto source = body["source"].get<std::string>();
            auto target = body["target"].get<std::string>();
            int max_depth = body.value("max_depth", 3);

            const auto& kg = learner.knowledge_graph();
            auto path = kg.find_path(source, target, max_depth);

            json resp;
            resp["found"] = path.found;
            resp["path"] = path.node_ids;
            resp["depth"] = static_cast<int>(path.node_ids.size()) - 1;
            return ok_resp(resp);
        } catch (const std::exception& e) {
            return err_resp(400, e.what());
        }
    });

    log_info("knowledge", "registered /api/knowledge/* routes");
}
