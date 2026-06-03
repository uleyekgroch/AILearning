/**
 * @file society_routes.cpp
 * @brief 社会路由（Phase 9 多 Agent 社会学习）
 *
 * 包含端点：
 * - POST   /society/create          创建新 Agent
 * - DELETE /society/agents/<id>      移除 Agent
 * - GET    /society/agents           列出所有 Agent
 * - POST   /society/observe          触发社会观察
 * - POST   /society/broadcast        广播知识
 * - GET    /society/metrics           社会指标
 */

#include "route_groups.hpp"
#include "dto.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace dto = ai_learning::server::dto;

void ai_learning::server::register_society_routes(
    crow::SimpleApp& app,
    core::Learner& /*learner*/,
    SharedState& state) {

    // POST /society/create — 创建新 Agent
    CROW_ROUTE(app, "/society/create").methods("POST"_method)
    ([&state](const crow::request& req) -> crow::response {
        try {
            if (!state.society) {
                state.society = std::make_unique<society::Society>(society::SocietyConfig{});
            }
            int port = 0;
            if (!req.body.empty()) {
                auto body = json::parse(req.body);
                port = body.value("port", 0);
            }
            std::string agent_id = state.society->create_agent(port);

            json resp;
            resp["status"] = "ok";
            resp["agent_id"] = agent_id;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // DELETE /society/agents/<id> — 移除 Agent
    CROW_ROUTE(app, "/society/agents/<string>").methods("DELETE"_method)
    ([&state](const std::string& agent_id) -> crow::response {
        try {
            if (!state.society) {
                crow::response res{400, dto::make_error_response("society not initialized").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            bool removed = state.society->remove_agent(agent_id);

            json resp;
            resp["status"] = removed ? "ok" : "not_found";
            resp["agent_id"] = agent_id;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // GET /society/agents — 列出所有 Agent
    CROW_ROUTE(app, "/society/agents").methods("GET"_method)
    ([&state]() -> crow::response {
        try {
            if (!state.society) {
                state.society = std::make_unique<society::Society>(society::SocietyConfig{});
            }
            auto agents = state.society->list_agents();
            json arr = json::array();
            for (const auto& a : agents) {
                json item;
                item["agent_id"] = a.agent_id;
                item["port"] = a.port;
                item["pid"] = a.pid;
                item["status"] = society::to_string(a.status);
                item["knowledge_count"] = a.knowledge_count;
                arr.push_back(item);
            }

            json resp;
            resp["agents"] = arr;
            resp["count"] = static_cast<int>(agents.size());

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /society/observe — 触发社会观察
    CROW_ROUTE(app, "/society/observe").methods("POST"_method)
    ([&state](const crow::request& req) -> crow::response {
        try {
            if (!state.society) {
                state.society = std::make_unique<society::Society>(society::SocietyConfig{});
            }
            auto body = json::parse(req.body);
            if (!body.contains("observer_id") || !body.contains("model_id")
                || !body.contains("domain")) {
                crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: observer_id, model_id, domain").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto result = state.society->trigger_observation(
                body["observer_id"].get<std::string>(),
                body["model_id"].get<std::string>(),
                body["domain"].get<std::string>());

            crow::response res{result.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /society/broadcast — 广播知识
    CROW_ROUTE(app, "/society/broadcast").methods("POST"_method)
    ([&state](const crow::request& req) -> crow::response {
        try {
            if (!state.society) {
                state.society = std::make_unique<society::Society>(society::SocietyConfig{});
            }
            auto body = json::parse(req.body);
            if (!body.contains("from_id") || !body.contains("domain")) {
                crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: from_id, domain").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto result = state.society->broadcast_knowledge(
                body["from_id"].get<std::string>(),
                body["domain"].get<std::string>());

            crow::response res{result.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // GET /society/metrics — 社会指标
    CROW_ROUTE(app, "/society/metrics").methods("GET"_method)
    ([&state]() -> crow::response {
        try {
            if (!state.society) {
                state.society = std::make_unique<society::Society>(society::SocietyConfig{});
            }
            auto metrics = state.society->social_metrics();
            json resp;
            resp["total_agents"] = metrics.total_agents;
            resp["active_agents"] = metrics.active_agents;
            resp["avg_knowledge_per_agent"] = metrics.avg_knowledge_per_agent;
            resp["knowledge_diversity"] = metrics.knowledge_diversity;
            resp["cultural_transmission_count"] = metrics.cultural_transmission_count;
            resp["collective_learning_speed"] = metrics.collective_learning_speed;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });
}
