/**
 * @file runtime_routes.cpp
 * @brief 运行时路由（Phase 9 持续在线学习）
 *
 * 包含端点：
 * - GET  /api/runtime/status         获取运行时状态
 * - POST /api/runtime/start          启动持续学习循环
 * - POST /api/runtime/stop           停止持续学习循环
 * - POST /api/runtime/checkpoint     强制检查点
 * - POST /api/runtime/feed           喂数据给学习循环
 * - POST /api/runtime/recover        从检查点恢复
 */

#include "route_groups.hpp"
#include "dto.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace dto = ai_learning::server::dto;

void ai_learning::server::register_runtime_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    SharedState& state) {

    // GET /api/runtime/status — 获取运行时状态
    CROW_ROUTE(app, "/api/runtime/status").methods("GET"_method)
    ([&state]() -> crow::response {
        try {
            if (!state.continuous_loop) {
                json resp;
                resp["running"] = false;
                resp["iterations"] = 0;
                resp["data_processed"] = 0;
                resp["pending_data_count"] = 0;
                resp["message"] = "continuous loop not started";

                crow::response res{resp.dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto s = state.continuous_loop->status();
            json resp;
            resp["running"] = s.running;
            resp["iterations"] = s.iterations;
            resp["data_processed"] = s.data_processed;
            resp["last_checkpoint_time"] = s.last_checkpoint_time;
            resp["last_consolidation_time"] = s.last_consolidation_time;
            resp["knowledge_retention"] = s.knowledge_retention;
            resp["uptime_seconds"] = s.uptime_seconds;
            resp["pending_data_count"] = s.pending_data_count;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/runtime/start — 启动持续学习循环
    CROW_ROUTE(app, "/api/runtime/start").methods("POST"_method)
    ([&state, &learner](const crow::request& req) -> crow::response {
        try {
            if (!state.continuous_loop) {
                state.continuous_loop = std::make_unique<learning::ContinuousLearningLoop>(learner);
            }
            learning::ContinuousLoopConfig cfg;
            if (!req.body.empty()) {
                auto body = json::parse(req.body);
                cfg.checkpoint_interval_seconds = body.value("checkpoint_interval_seconds", 300);
                cfg.consolidation_interval_seconds = body.value("consolidation_interval_seconds", 60);
                cfg.max_iterations = body.value("max_iterations", 0);
                cfg.data_buffer_size = body.value("data_buffer_size", 100);
                cfg.checkpoint_dir = body.value("checkpoint_dir", "./checkpoints");
                cfg.auto_recover = body.value("auto_recover", true);
            }
            state.continuous_loop->start(cfg);

            json resp;
            resp["status"] = "ok";
            resp["message"] = "continuous loop started";

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/runtime/stop — 停止持续学习循环
    CROW_ROUTE(app, "/api/runtime/stop").methods("POST"_method)
    ([&state](const crow::request& /*req*/) -> crow::response {
        try {
            if (state.continuous_loop) {
                state.continuous_loop->stop();
            }
            json resp;
            resp["status"] = "ok";
            resp["message"] = "continuous loop stopped";

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/runtime/checkpoint — 强制检查点
    CROW_ROUTE(app, "/api/runtime/checkpoint").methods("POST"_method)
    ([&state](const crow::request& /*req*/) -> crow::response {
        try {
            if (!state.continuous_loop) {
                crow::response res{400,
                    dto::make_error_response("continuous loop not initialized").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            state.continuous_loop->checkpoint_now();

            json resp;
            resp["status"] = "ok";
            resp["message"] = "checkpoint triggered";

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/runtime/feed — 喂数据给学习循环
    CROW_ROUTE(app, "/api/runtime/feed").methods("POST"_method)
    ([&state, &learner](const crow::request& req) -> crow::response {
        try {
            if (!state.continuous_loop) {
                state.continuous_loop = std::make_unique<learning::ContinuousLearningLoop>(learner);
            }
            auto body = json::parse(req.body);
            if (!body.contains("data")) {
                crow::response res{400,
                    dto::make_error_response("missing 'data' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string data = body["data"].get<std::string>();
            std::string source = body.value("source", "api");
            state.continuous_loop->feed_data(data, source);

            json resp;
            resp["status"] = "ok";
            resp["message"] = "data enqueued";

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/runtime/recover — 从检查点恢复
    CROW_ROUTE(app, "/api/runtime/recover").methods("POST"_method)
    ([&state, &learner](const crow::request& req) -> crow::response {
        try {
            if (!state.continuous_loop) {
                state.continuous_loop = std::make_unique<learning::ContinuousLearningLoop>(learner);
            }
            std::string dir = "./checkpoints";
            if (!req.body.empty()) {
                auto body = json::parse(req.body);
                dir = body.value("dir", "./checkpoints");
            }
            bool ok = state.continuous_loop->recover(dir);

            json resp;
            resp["status"] = ok ? "ok" : "failed";
            resp["message"] = ok ? "recovered from checkpoint" : "no checkpoint found";
            resp["dir"] = dir;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });
}
