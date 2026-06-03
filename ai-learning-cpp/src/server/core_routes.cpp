/**
 * @file core_routes.cpp
 * @brief 系统 + 核心 API 路由（Phase 7.1-7.2）
 *
 * 端点：health, stats, stage, tasks, learn/text, observe, reason, think,
 *       perceive, remember, recall, consolidate, autonomous, save, load
 */

#include "route_groups.hpp"
#include "dto.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <thread>

using json = nlohmann::json;
namespace dto = ai_learning::server::dto;

// ── 辅助：构造 JSON 响应 ──────────────────────────────────────────
static auto json_resp(int code, const json& body) -> crow::response {
    crow::response res{code, body.dump()};
    res.set_header("Content-Type", "application/json");
    return res;
}
static auto ok_resp(const json& body) -> crow::response { return json_resp(200, body); }
static auto err_resp(int code, const std::string& msg) -> crow::response {
    return json_resp(code, dto::make_error_response(msg));
}

auto ai_learning::server::generate_task_id() -> std::string {
    static std::atomic<int> counter{0};
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "task_" + std::to_string(now) + "_" + std::to_string(counter++);
}

void ai_learning::server::register_core_routes(
    crow::SimpleApp& app, core::Learner& learner,
    const ServerConfig& config, SharedState& state) {

    // GET /api/health
    CROW_ROUTE(app, "/api/health").methods("GET"_method)
    ([&]() -> crow::response {
        json body;
        body["status"] = "ok";
        body["version"] = config.version;
        body["stage"] = learner.stage();
        auto stats = learner.get_stats();
        body["total_steps"] = stats.count("total_steps")
            ? static_cast<int>(stats.at("total_steps")) : 0;
        return ok_resp(body);
    });

    // GET /api/stats
    CROW_ROUTE(app, "/api/stats").methods("GET"_method)
    ([&]() -> crow::response { return ok_resp(learner.get_stats()); });

    // GET /api/stage
    CROW_ROUTE(app, "/api/stage").methods("GET"_method)
    ([&]() -> crow::response { return ok_resp(json{{"stage", learner.stage()}}); });

    // GET /api/tasks/<string>
    CROW_ROUTE(app, "/api/tasks/<string>").methods("GET"_method)
    ([&state](const std::string& task_id) -> crow::response {
        std::lock_guard<std::mutex> lock(state.tasks_mutex);
        auto it = state.tasks.find(task_id);
        if (it == state.tasks.end()) return err_resp(404, "task not found");
        json body;
        body["task_id"] = it->second.task_id;
        body["status"] = it->second.status;
        if (it->second.status == "completed") body["result"] = it->second.result;
        else if (it->second.status == "failed") body["error"] = it->second.error;
        return ok_resp(body);
    });

    // POST /api/learn/text
    CROW_ROUTE(app, "/api/learn/text").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("text")) return err_resp(400, "missing 'text' field");
            auto result = learner.learn_from_text(
                body["text"].get<std::string>(), body.value("source", "text"));
            return ok_resp(dto::to_json_text_learn_result(result));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/observe
    CROW_ROUTE(app, "/api/observe").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("text")) return err_resp(400, "missing 'text' field");
            auto result = learner.observe_text(body["text"].get<std::string>());
            json resp;
            for (const auto& [k, v] : result) resp[k] = v;
            return ok_resp(resp);
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/reason
    CROW_ROUTE(app, "/api/reason").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("question")) return err_resp(400, "missing 'question' field");
            auto results = learner.reason(body["question"].get<std::string>());
            json arr = json::array();
            for (const auto& r : results) arr.push_back(dto::to_json_reasoning_result(r));
            return ok_resp(arr);
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/think
    CROW_ROUTE(app, "/api/think").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("question")) return err_resp(400, "missing 'question' field");
            return ok_resp(json{{"answer", learner.think(body["question"].get<std::string>())}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/perceive
    CROW_ROUTE(app, "/api/perceive").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("raw_input")) return err_resp(400, "missing 'raw_input' field");
            return ok_resp(json{{"perception", learner.perceive(dto::parse_raw_input(body["raw_input"]))}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/remember
    CROW_ROUTE(app, "/api/remember").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("obs") || !body.contains("action")
                || !body.contains("next_obs") || !body.contains("reward")
                || !body.contains("error"))
                return err_resp(400, "missing required fields: obs, action, next_obs, reward, error");
            learner.remember(
                dto::parse_float_vector(body["obs"]), body["action"].get<int>(),
                dto::parse_float_vector(body["next_obs"]),
                body["reward"].get<float>(), body["error"].get<float>());
            return ok_resp(json{{"status", "ok"}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/recall
    CROW_ROUTE(app, "/api/recall").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("cue")) return err_resp(400, "missing 'cue' field");
            auto items = learner.recall(dto::parse_float_vector(body["cue"]), body.value("k", 5));
            json arr = json::array();
            for (const auto& m : items) arr.push_back(dto::to_json_memory_item(m));
            return ok_resp(arr);
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/consolidate
    CROW_ROUTE(app, "/api/consolidate").methods("POST"_method)
    ([&](const crow::request&) -> crow::response {
        try { return ok_resp(learner.consolidate()); }
        catch (const std::exception& e) { return err_resp(500, e.what()); }
    });

    // POST /api/autonomous（异步）
    CROW_ROUTE(app, "/api/autonomous").methods("POST"_method)
    ([&state, &learner](const crow::request& req) -> crow::response {
        try {
            int iterations = 10;
            if (!req.body.empty()) iterations = json::parse(req.body).value("iterations", 10);
            std::string task_id = generate_task_id();
            {
                std::lock_guard<std::mutex> lock(state.tasks_mutex);
                state.tasks[task_id] = AsyncTask{task_id, "pending", json{}, ""};
            }
            std::thread([&learner, &state, task_id, iterations]() {
                { std::lock_guard<std::mutex> lk(state.tasks_mutex);
                  state.tasks[task_id].status = "running"; }
                try {
                    auto report = learner.autonomous_learning_run(iterations);
                    std::lock_guard<std::mutex> lk(state.tasks_mutex);
                    state.tasks[task_id].status = "completed";
                    state.tasks[task_id].result = dto::to_json_loop_report(report);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(state.tasks_mutex);
                    state.tasks[task_id].status = "failed";
                    state.tasks[task_id].error = e.what();
                }
            }).detach();
            return json_resp(202, json{{"task_id", task_id}, {"status", "pending"}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/save
    CROW_ROUTE(app, "/api/save").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("path")) return err_resp(400, "missing 'path' field");
            auto path = body["path"].get<std::string>();
            if (path.find("..") != std::string::npos) return err_resp(400, "path traversal not allowed");
            learner.save(path);
            return ok_resp(json{{"status", "ok"}, {"path", path}});
        } catch (const std::exception& e) { return err_resp(500, e.what()); }
    });

    // POST /api/load
    CROW_ROUTE(app, "/api/load").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("path")) return err_resp(400, "missing 'path' field");
            auto path = body["path"].get<std::string>();
            if (path.find("..") != std::string::npos) return err_resp(400, "path traversal not allowed");
            learner.load(path);
            return ok_resp(json{{"status", "ok"}, {"path", path}});
        } catch (const std::exception& e) { return err_resp(500, e.what()); }
    });
}
