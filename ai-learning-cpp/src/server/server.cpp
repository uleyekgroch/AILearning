/**
 * @file server.cpp
 * @brief REST HTTP + WebSocket 服务实现 — 路由注册与请求处理
 *
 * Phase 7.1: 基础骨架 + 健康检查端点
 * Phase 7.2: 核心 API 端点（13 个）
 * Phase 7.3: 高级认知 API 端点（Phase 3-6，14 个）
 * Phase 7.4: WebSocket 实时事件推送
 * Phase 7.5: Web Console 静态文件服务
 */

#include "server.hpp"
#include "dto.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#endif

// 文件级 JSON 类型别名，避免每个 lambda 内重复声明
using json = nlohmann::json;

// ── 全局信号处理 ──────────────────────────────────────────────────

static std::atomic<bool> g_shutdown_requested{false};

#ifdef _WIN32
static BOOL WINAPI windows_signal_handler_(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT) {
        g_shutdown_requested = true;
        return TRUE;
    }
    return FALSE;
}
#else
static void posix_signal_handler_(int /*signal*/) {
    g_shutdown_requested = true;
}
#endif

// ── 构造 ────────────────────────────────────────────────────────

ai_learning::server::LearningServer::LearningServer(
    ai_learning::core::Learner& learner,
    ai_learning::server::ServerConfig config)
    : learner_(learner), config_(std::move(config)) {}

// ── 路由注册 ────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::register_routes_(::crow::SimpleApp& app) -> void {
    register_system_routes_(app);
    register_core_routes_(app);
    register_advanced_routes_(app);
    register_static_routes_(app);
    register_ws_routes_(app);
}

// ── 系统路由 ────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::register_system_routes_(::crow::SimpleApp& app) -> void {

    // GET /api/health — 健康检查
    CROW_ROUTE(app, "/api/health").methods("GET"_method)
    ([this]() -> ::crow::response {
        json body;
        body["status"] = "ok";
        body["version"] = config_.version;
        body["stage"] = learner_.stage();

        auto stats = learner_.get_stats();
        body["total_steps"] = stats.count("total_steps")
            ? static_cast<int>(stats.at("total_steps")) : 0;

        ::crow::response res{body.dump()};
        res.set_header("Content-Type", "application/json");
        return res;
    });

    // GET /api/stats — 学习统计
    CROW_ROUTE(app, "/api/stats").methods("GET"_method)
    ([this]() -> ::crow::response {
        auto stats = learner_.get_stats();
        json body = stats;

        ::crow::response res{body.dump()};
        res.set_header("Content-Type", "application/json");
        return res;
    });

    // GET /api/stage — 发展阶段
    CROW_ROUTE(app, "/api/stage").methods("GET"_method)
    ([this]() -> ::crow::response {
        json body;
        body["stage"] = learner_.stage();

        ::crow::response res{body.dump()};
        res.set_header("Content-Type", "application/json");
        return res;
    });

    // GET /api/tasks/<string> — 查询异步任务状态
    CROW_ROUTE(app, "/api/tasks/<string>").methods("GET"_method)
    ([this](const std::string& task_id) -> ::crow::response {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        auto it = tasks_.find(task_id);
        if (it == tasks_.end()) {
            ::crow::response res{404, ai_learning::server::dto::make_error_response("task not found").dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
        json body;
        body["task_id"] = it->second.task_id;
        body["status"] = it->second.status;
        if (it->second.status == "completed") {
            body["result"] = it->second.result;
        } else if (it->second.status == "failed") {
            body["error"] = it->second.error;
        }
        ::crow::response res{body.dump()};
        res.set_header("Content-Type", "application/json");
        return res;
    });
}

// ── 核心 API 路由 ────────────────────────────────────────────────

auto ai_learning::server::LearningServer::register_core_routes_(::crow::SimpleApp& app) -> void {

    // POST /api/learn/text — 学习文本
    CROW_ROUTE(app, "/api/learn/text").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("text")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'text' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string text = body["text"].get<std::string>();
            std::string source = body.value("source", "text");

            auto result = learner_.learn_from_text(text, source);
            json resp = ai_learning::server::dto::to_json_text_learn_result(result);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/observe — 观察输入
    CROW_ROUTE(app, "/api/observe").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("text")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'text' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string text = body["text"].get<std::string>();
            auto result = learner_.observe_text(text);

            json resp;
            for (const auto& [key, vals] : result) {
                resp[key] = vals;
            }

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/reason — 推理问答
    CROW_ROUTE(app, "/api/reason").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("question")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'question' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string question = body["question"].get<std::string>();
            auto results = learner_.reason(question);

            json arr = json::array();
            for (const auto& r : results) {
                arr.push_back(ai_learning::server::dto::to_json_reasoning_result(r));
            }

            ::crow::response res{arr.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/think — 深度思考
    CROW_ROUTE(app, "/api/think").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("question")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'question' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string question = body["question"].get<std::string>();
            auto answer = learner_.think(question);

            json resp;
            resp["answer"] = answer;

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/perceive — 感知处理
    CROW_ROUTE(app, "/api/perceive").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("raw_input")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'raw_input' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto raw_input = ai_learning::server::dto::parse_raw_input(body["raw_input"]);
            auto perception = learner_.perceive(raw_input);

            json resp;
            resp["perception"] = perception;

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/remember — 记忆存储
    CROW_ROUTE(app, "/api/remember").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("obs") || !body.contains("action")
                || !body.contains("next_obs") || !body.contains("reward")
                || !body.contains("error")) {
                ::crow::response res{400,
                    ai_learning::server::dto::make_error_response(
                        "missing required fields: obs, action, next_obs, reward, error").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }

            auto obs = ai_learning::server::dto::parse_float_vector(body["obs"]);
            int action = body["action"].get<int>();
            auto next_obs = ai_learning::server::dto::parse_float_vector(body["next_obs"]);
            float reward = body["reward"].get<float>();
            float error = body["error"].get<float>();

            learner_.remember(obs, action, next_obs, reward, error);

            json resp;
            resp["status"] = "ok";

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/recall — 记忆检索
    CROW_ROUTE(app, "/api/recall").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("cue")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'cue' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto cue = ai_learning::server::dto::parse_float_vector(body["cue"]);
            int k = body.value("k", 5);

            auto items = learner_.recall(cue, k);

            json arr = json::array();
            for (const auto& m : items) {
                arr.push_back(ai_learning::server::dto::to_json_memory_item(m));
            }

            ::crow::response res{arr.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/consolidate — 记忆巩固
    CROW_ROUTE(app, "/api/consolidate").methods("POST"_method)
    ([this](const ::crow::request& /*req*/) -> ::crow::response {
        try {
            auto report = learner_.consolidate();

            json resp = report;

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{500, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/autonomous — 自主学习循环（异步）
    CROW_ROUTE(app, "/api/autonomous").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            int iterations = 10;
            if (!req.body.empty()) {
                auto body = json::parse(req.body);
                iterations = body.value("iterations", 10);
            }

            std::string task_id = generate_task_id_();

            // 注册任务
            {
                std::lock_guard<std::mutex> lock(tasks_mutex_);
                tasks_[task_id] = AsyncTask{
                    task_id, "pending", json{}, ""
                };
            }

            // 异步执行自主学习
            std::thread([this, task_id, iterations]() {
                {
                    std::lock_guard<std::mutex> lock(tasks_mutex_);
                    tasks_[task_id].status = "running";
                }
                try {
                    auto report = learner_.autonomous_learning_run(iterations);
                    json result = ai_learning::server::dto::to_json_loop_report(report);
                    std::lock_guard<std::mutex> lock(tasks_mutex_);
                    tasks_[task_id].status = "completed";
                    tasks_[task_id].result = result;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(tasks_mutex_);
                    tasks_[task_id].status = "failed";
                    tasks_[task_id].error = e.what();
                }
            }).detach();

            json resp;
            resp["task_id"] = task_id;
            resp["status"] = "pending";

            ::crow::response res{202, resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/save — 持久化
    CROW_ROUTE(app, "/api/save").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("path")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'path' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string path = body["path"].get<std::string>();

            // 安全检查：不允许路径遍历
            if (path.find("..") != std::string::npos) {
                ::crow::response res{400,
                    ai_learning::server::dto::make_error_response("path traversal not allowed").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }

            learner_.save(path);

            json resp;
            resp["status"] = "ok";
            resp["path"] = path;

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{500, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/load — 加载状态
    CROW_ROUTE(app, "/api/load").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("path")) {
                ::crow::response res{400, ai_learning::server::dto::make_error_response("missing 'path' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string path = body["path"].get<std::string>();

            // 安全检查：不允许路径遍历
            if (path.find("..") != std::string::npos) {
                ::crow::response res{400,
                    ai_learning::server::dto::make_error_response("path traversal not allowed").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }

            learner_.load(path);

            json resp;
            resp["status"] = "ok";
            resp["path"] = path;

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{500, ai_learning::server::dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });
}

// ── 高级 API 路由 ────────────────────────────────────────────────

auto ai_learning::server::LearningServer::register_advanced_routes_(::crow::SimpleApp& app) -> void {
    using json = nlohmann::json;
    namespace dto = ai_learning::server::dto;

    // ── Phase 3：高级认知能力 ──────────────────────────────────────

    // POST /api/analogize — 跨领域类比迁移
    CROW_ROUTE(app, "/api/analogize").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("source_concepts") || !body.contains("target_concepts")) {
                ::crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: source_concepts, target_concepts").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::vector<ai_learning::learning::ConceptDescriptor> source;
            for (const auto& sc : body["source_concepts"]) {
                source.push_back(dto::parse_concept_descriptor(sc));
            }
            std::vector<ai_learning::learning::ConceptDescriptor> target;
            for (const auto& tc : body["target_concepts"]) {
                target.push_back(dto::parse_concept_descriptor(tc));
            }
            std::vector<std::string> source_facts;
            if (body.contains("source_facts") && body["source_facts"].is_array()) {
                source_facts = body["source_facts"].get<std::vector<std::string>>();
            }

            auto result = learner_.analogical_transfer(source, target, source_facts);
            json resp = dto::to_json_transfer_result(result);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/protect — 注册知识保护（防遗忘）
    CROW_ROUTE(app, "/api/protect").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("knowledge_id") || !body.contains("domain")) {
                ::crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: knowledge_id, domain").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string knowledge_id = body["knowledge_id"].get<std::string>();
            std::string domain = body["domain"].get<std::string>();
            double confidence = body.value("confidence", 0.5);
            int usage_count = body.value("usage_count", 0);

            learner_.protect_knowledge(knowledge_id, domain, confidence, usage_count);

            json resp;
            resp["status"] = "ok";
            resp["knowledge_id"] = knowledge_id;

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // GET /api/forgetting — 检测遗忘
    CROW_ROUTE(app, "/api/forgetting").methods("GET"_method)
    ([this]() -> ::crow::response {
        try {
            auto alerts = learner_.detect_forgetting();
            json arr = json::array();
            for (const auto& a : alerts) {
                arr.push_back(dto::to_json_forgetting_alert(a));
            }
            json resp;
            resp["alerts"] = arr;
            resp["count"] = static_cast<int>(alerts.size());

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/abstract — 抽象概念形成
    CROW_ROUTE(app, "/api/abstract").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("instance_id") || !body.contains("attributes")) {
                ::crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: instance_id, attributes").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string instance_id = body["instance_id"].get<std::string>();
            auto attributes = body["attributes"].get<std::vector<std::string>>();
            std::map<std::string, double> features;
            if (body.contains("features") && body["features"].is_object()) {
                features = body["features"].get<std::map<std::string, double>>();
            }
            std::vector<std::string> relations;
            if (body.contains("relations") && body["relations"].is_array()) {
                relations = body["relations"].get<std::vector<std::string>>();
            }

            auto report = learner_.form_abstractions(instance_id, attributes, features, relations);
            json resp = dto::to_json_concept_formation_report(report);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // ── Phase 4：增强智能 ──────────────────────────────────────────

    // POST /api/observe-behavior — 社会观察学习
    CROW_ROUTE(app, "/api/observe-behavior").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("agent_id") || !body.contains("action")) {
                ::crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: agent_id, action").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto observation = dto::parse_behavior_observation(body);
            auto report = learner_.observe_behavior(observation);
            json resp = dto::to_json_social_learning_report(report);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/emotion — 情感处理
    CROW_ROUTE(app, "/api/emotion").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("event_type")) {
                ::crow::response res{400,
                    dto::make_error_response("missing 'event_type' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto event = dto::parse_emotion_event(body);
            auto state = learner_.process_emotion(event);
            json resp = dto::to_json_emotion_state(state);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/insight — 尝试顿悟
    CROW_ROUTE(app, "/api/insight").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("problem_context")) {
                ::crow::response res{400,
                    dto::make_error_response("missing 'problem_context' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string problem_context = body["problem_context"].get<std::string>();
            auto insight_opt = learner_.try_insight(problem_context);

            json resp;
            if (insight_opt.has_value()) {
                resp["insight"] = dto::to_json_insight_event(insight_opt.value());
                resp["found"] = true;
            } else {
                resp["found"] = false;
                resp["message"] = "no insight emerged";
            }

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // ── Phase 5：高级元认知 ────────────────────────────────────────

    // POST /api/meta/recommend — 元学习策略推荐
    CROW_ROUTE(app, "/api/meta/recommend").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("domain") || !body.contains("task_type")) {
                ::crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: domain, task_type").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto task = dto::parse_task_descriptor(body);
            auto rec = learner_.meta_recommend(task);
            json resp = dto::to_json_meta_recommendation(rec);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/meta/reflect — 元学习自我反思
    CROW_ROUTE(app, "/api/meta/reflect").methods("POST"_method)
    ([this](const ::crow::request& /*req*/) -> ::crow::response {
        try {
            auto reflections = learner_.meta_reflect();
            json resp;
            resp["reflections"] = reflections;
            resp["count"] = static_cast<int>(reflections.size());

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/experiment/design — 自动设计实验
    CROW_ROUTE(app, "/api/experiment/design").methods("POST"_method)
    ([this](const ::crow::request& /*req*/) -> ::crow::response {
        try {
            auto design_opt = learner_.design_experiment();
            json resp;
            if (design_opt.has_value()) {
                resp["experiment"] = dto::to_json_experiment_design(design_opt.value());
                resp["found"] = true;
            } else {
                resp["found"] = false;
                resp["message"] = "no hypothesis suitable for experiment";
            }

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/experiment/record — 记录实验结果
    CROW_ROUTE(app, "/api/experiment/record").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("experiment_id") || !body.contains("hypothesis_id")) {
                ::crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: experiment_id, hypothesis_id").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            auto result = dto::parse_experiment_result(body);
            auto result_id = learner_.record_experiment(result);

            json resp;
            resp["status"] = "ok";
            resp["result_id"] = result_id;

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // ── Phase 6：深度整合 ──────────────────────────────────────────

    // POST /api/integrated/pipeline — 全流水线闭环学习
    CROW_ROUTE(app, "/api/integrated/pipeline").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("observation") || !body.contains("domain")) {
                ::crow::response res{400,
                    dto::make_error_response(
                        "missing required fields: observation, domain").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string observation = body["observation"].get<std::string>();
            std::string domain = body["domain"].get<std::string>();

            auto report = learner_.integrated_pipeline(observation, domain);
            json resp = dto::to_json_integrated_pipeline(report);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // POST /api/integrated/meta-guided — 元学习驱动学习会话
    CROW_ROUTE(app, "/api/integrated/meta-guided").methods("POST"_method)
    ([this](const ::crow::request& req) -> ::crow::response {
        try {
            auto body = json::parse(req.body);
            std::vector<std::string> known_topics;
            if (body.contains("known_topics") && body["known_topics"].is_array()) {
                known_topics = body["known_topics"].get<std::vector<std::string>>();
            }
            std::map<std::string, double> mastery_map;
            if (body.contains("mastery_map") && body["mastery_map"].is_object()) {
                mastery_map = body["mastery_map"].get<std::map<std::string, double>>();
            }

            auto report = learner_.meta_guided_learn(known_topics, mastery_map);
            json resp = dto::to_json_meta_guided_session(report);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // GET /api/integrated/emotion-params — 情感调制参数
    CROW_ROUTE(app, "/api/integrated/emotion-params").methods("GET"_method)
    ([this]() -> ::crow::response {
        try {
            auto params = learner_.emotion_modulated_params();
            json resp = dto::to_json_emotion_modulated_params(params);

            ::crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            ::crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });
}


// ── 静态文件路由（Phase 7.5 — Web Console）─────────────────────────

auto ai_learning::server::LearningServer::register_static_routes_(::crow::SimpleApp& app) -> void {
    // 确定静态文件目录
    const std::string web_dir = config_.static_dir.empty()
        ? std::string("web/")
        : config_.static_dir;

    // ── 辅助 lambda：读取文件内容 ──
    static const auto read_file = [](const std::string& filepath) -> std::string {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) return {};
        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        return content;
    };

    // ── 辅助 lambda：根据扩展名返回 MIME 类型 ──
    static const auto get_mime = [](const std::string& path) -> std::string {
        auto pos = path.rfind('.');
        if (pos == std::string::npos) return "text/plain";
        std::string ext = path.substr(pos + 1);
        if (ext == "html" || ext == "htm") return "text/html; charset=utf-8";
        if (ext == "css") return "text/css; charset=utf-8";
        if (ext == "js")  return "application/javascript; charset=utf-8";
        if (ext == "json") return "application/json; charset=utf-8";
        if (ext == "png") return "image/png";
        if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
        if (ext == "svg") return "image/svg+xml";
        if (ext == "ico") return "image/x-icon";
        return "application/octet-stream";
    };

    // GET / — 重定向到 /console
    CROW_ROUTE(app, "/").methods("GET"_method)
    ([&web_dir](const ::crow::request&, ::crow::response& res) {
        std::string filepath = web_dir + "index.html";
        // 路径安全检查
        if (filepath.find("..") != std::string::npos) {
            res.code = 400;
            res.write("Bad Request");
            res.end();
            return;
        }
        std::string body = read_file(filepath);
        if (body.empty()) {
            res.code = 404;
            res.write("AILearning server running. Web console not found at " + filepath);
            res.end();
            return;
        }
        res.set_header("Content-Type", "text/html; charset=utf-8");
        res.write(body);
        res.end();
    });

    // GET /console — serve index.html explicitly
    CROW_ROUTE(app, "/console").methods("GET"_method)
    ([&web_dir](const ::crow::request&, ::crow::response& res) {
        std::string filepath = web_dir + "index.html";
        if (filepath.find("..") != std::string::npos) {
            res.code = 400;
            res.write("Bad Request");
            res.end();
            return;
        }
        std::string body = read_file(filepath);
        if (body.empty()) {
            res.code = 404;
            res.write("File not found");
            res.end();
            return;
        }
        res.set_header("Content-Type", "text/html; charset=utf-8");
        res.write(body);
        res.end();
    });

    // GET /web/<path> — serve static files from web/ directory
    // This handles /web/style.css, /web/app.js, etc.
    CROW_ROUTE(app, "/web/<path>").methods("GET"_method)
    ([&web_dir](const ::crow::request&, ::crow::response& res, std::string file_path) {
        // 路径安全检查
        if (file_path.find("..") != std::string::npos) {
            res.code = 400;
            res.write("Bad Request");
            res.end();
            return;
        }
        std::string filepath = web_dir + file_path;
        std::string body = read_file(filepath);
        if (body.empty()) {
            res.code = 404;
            res.write("File not found: " + file_path);
            res.end();
            return;
        }
        res.set_header("Content-Type", get_mime(file_path));
        res.write(body);
        res.end();
    });

    std::cout << "[Server] Static files serving from: " << web_dir << "\n";
}

// ── WebSocket 路由 ──────────────────────────────────────────────────

auto ai_learning::server::LearningServer::register_ws_routes_(::crow::SimpleApp& app) -> void {

    // ── /ws/events — 所有学习事件实时推送 ──────────────────────────
    CROW_WEBSOCKET_ROUTE(app, "/ws/events")
    .onopen([this](::crow::websocket::connection& conn) {
        event_adapter_.add_connection(&conn);
        heartbeat_.add_connection(&conn);
        // 发送历史缓冲区事件给新连接
        auto history = event_adapter_.get_history_json();
        if (!history.empty() && history != "[]") {
            try {
                conn.send_text(history);
            } catch (const std::exception& e) {
                std::cerr << "[WS/events] send history failed: " << e.what() << "\n";
            }
        }
    })
    .onclose([this](::crow::websocket::connection& conn,
                     const std::string& /*reason*/) {
        event_adapter_.remove_connection(&conn);
        heartbeat_.remove_connection(&conn);
    })
    .onmessage([this](::crow::websocket::connection& conn,
                       const std::string& data, bool is_binary) {
        // 客户端发来消息，更新心跳活动时间
        heartbeat_.touch(&conn);

        // 处理客户端请求
        if (!is_binary) {
            try {
                auto msg = json::parse(data);
                if (msg.contains("type") && msg["type"] == "ping") {
                    json pong;
                    pong["type"] = "pong";
                    pong["timestamp"] = std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count();
                    conn.send_text(pong.dump());
                }
            } catch (const std::exception&) {
                // 忽略无法解析的消息
            }
        }
    });

    // ── /ws/stats — 统计摘要推送 ──────────────────────────────────
    CROW_WEBSOCKET_ROUTE(app, "/ws/stats")
    .onopen([this](::crow::websocket::connection& conn) {
        stats_pusher_.add_connection(&conn);
        heartbeat_.add_connection(&conn);
        // 立即发送一次当前统计
        try {
            auto stats = learner_.get_stats();
            json msg;
            msg["type"] = "stats";
            msg["data"] = stats;
            conn.send_text(msg.dump());
        } catch (const std::exception& e) {
            std::cerr << "[WS/stats] initial send failed: " << e.what() << "\n";
        }
    })
    .onclose([this](::crow::websocket::connection& conn,
                     const std::string& /*reason*/) {
        stats_pusher_.remove_connection(&conn);
        heartbeat_.remove_connection(&conn);
    })
    .onmessage([this](::crow::websocket::connection& conn,
                       const std::string& /*data*/, bool /*is_binary*/) {
        heartbeat_.touch(&conn);
    });
}

// ── WebSocket 后台任务 ──────────────────────────────────────────────

auto ai_learning::server::LearningServer::start_ws_background_tasks_() -> void {
    ws_bg_running_ = true;

    // 心跳 + 超时检查线程（每 30 秒 ping 一次）
    std::jthread heartbeat_thread([this]() {
        while (ws_bg_running_) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            if (!ws_bg_running_) break;

            // 发送 ping
            heartbeat_.send_pings();

            // 检查超时（60 秒无活动）
            auto expired = heartbeat_.check_timeouts(std::chrono::seconds{60});
            for (auto* conn : expired) {
                std::cout << "[WS/heartbeat] Closing timed-out connection\n";
                try {
                    conn->close("timeout");
                } catch (const std::exception& e) {
                    std::cerr << "[WS/heartbeat] close failed: " << e.what() << "\n";
                }
                // 清理连接（onclose 回调会处理，但以防万一）
                event_adapter_.remove_connection(conn);
                stats_pusher_.remove_connection(conn);
                heartbeat_.remove_connection(conn);
            }
        }
    });

    // 统计推送线程（每 1 秒推送一次）
    std::jthread stats_thread([this]() {
        while (ws_bg_running_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (!ws_bg_running_) break;

            try {
                auto stats = learner_.get_stats();
                json msg;
                msg["type"] = "stats";
                msg["timestamp"] = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count();
                msg["data"] = stats;
                stats_pusher_.push_stats(msg);
            } catch (const std::exception& e) {
                std::cerr << "[WS/stats] push failed: " << e.what() << "\n";
            }
        }
    });

    // 分离线程，让它们在后台运行直到 ws_bg_running_ 为 false
    heartbeat_thread.detach();
    stats_thread.detach();
}

auto ai_learning::server::LearningServer::stop_ws_background_tasks_() -> void {
    ws_bg_running_ = false;
}

// ── 启动 ────────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::run() -> void {
    ::crow::SimpleApp app;

    // 注册所有路由
    register_routes_(app);

    // 设置日志级别
    app.loglevel(::crow::LogLevel::Info);

    // 安装信号处理
#ifdef _WIN32
    SetConsoleCtrlHandler(windows_signal_handler_, TRUE);
#else
    signal(SIGINT, posix_signal_handler_);
    signal(SIGTERM, posix_signal_handler_);
#endif

    running_ = true;
    std::cout << "[Server] AILearning REST API starting on port "
              << config_.port << " (" << config_.threads << " threads)\n";

    // 启动 WebSocket 后台任务（心跳 + 统计推送）
    start_ws_background_tasks_();

    // 在独立线程中监控关闭信号
    std::jthread monitor([this, &app]() {
        while (running_ && !g_shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (g_shutdown_requested) {
            std::cout << "\n[Server] Shutdown signal received, stopping...\n";
            app.stop();
            running_ = false;
        }
    });

    // 启动 HTTP 服务（阻塞，直到 app.stop() 被调用）
    app.port(config_.port).concurrency(
        static_cast<std::uint16_t>(config_.threads)).run();

    // 等待监控线程结束
    monitor.request_stop();
    if (monitor.joinable()) {
        monitor.join();
    }

    running_ = false;

    // 停止 WebSocket 后台任务
    stop_ws_background_tasks_();

    std::cout << "[Server] Stopped gracefully.\n";
}

// ── 关闭 ────────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::shutdown() -> void {
    g_shutdown_requested = true;
}

auto ai_learning::server::LearningServer::is_running() const -> bool {
    return running_.load();
}

// ── 工具方法 ────────────────────────────────────────────────────

auto ai_learning::server::LearningServer::generate_task_id_() -> std::string {
    static std::atomic<int> counter{0};
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "task_" + std::to_string(now) + "_" + std::to_string(counter++);
}
