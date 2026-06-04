/**
 * @file test_server_routes.cpp
 * @brief REST API 集成测试 — 通过 Crow 的 handle_full() 测试路由处理器
 *
 * 测试策略：
 *   - 创建 crow::SimpleApp，注册路由，调用 app.handle_full(req, res)
 *   - 无需启动 HTTP 服务器，直接调用路由处理逻辑
 *   - 覆盖所有 5 个路由组文件的端点
 *
 * 测试分组：
 *   [server][core]      — 健康检查、统计、阶段、学习、推理、记忆、持久化
 *   [server][advanced]  — Phase 3-6 高级认知端点
 *   [server][society]   — Phase 9 多 Agent 社会
 *   [server][chat]      — Phase 9 对话
 *   [server][runtime]   — Phase 9 持续在线学习
 */

#include <catch2/catch_test_macros.hpp>

#include "route_groups.hpp"
#include "dto.hpp"
#include "server_config.hpp"
#include "event_adapter.hpp"

#include "ai_learning/core/learner.hpp"
#include "ai_learning/core/learner_factory.hpp"

#include <crow.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <thread>

using json = nlohmann::json;
using namespace ai_learning;
using namespace ai_learning::server;

// ═══════════════════════════════════════════════════════════
// 辅助工具
// ═══════════════════════════════════════════════════════════

/// 创建一个 literacy 阶段的 Learner
static auto make_test_learner() -> core::Learner {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    config.embedding_learning_enabled = false;
    return core::Learner(config);
}

/// 构造 GET 请求
static auto make_get(const std::string& url) -> crow::request {
    crow::request req;
    req.method = crow::HTTPMethod::Get;
    req.raw_url = url;
    req.url = url;
    return req;
}

/// 构造 POST 请求（JSON body）
static auto make_post(const std::string& url, const json& body) -> crow::request {
    crow::request req;
    req.method = crow::HTTPMethod::Post;
    req.raw_url = url;
    req.url = url;
    req.body = body.dump();
    return req;
}

/// 构造 DELETE 请求
static auto make_delete(const std::string& url) -> crow::request {
    crow::request req;
    req.method = crow::HTTPMethod::Delete;
    req.raw_url = url;
    req.url = url;
    return req;
}

/// 注册所有路由到 app 并执行 validate
static void register_all_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    const ServerConfig& config,
    SharedState& state)
{
    register_core_routes(app, learner, config, state);
    register_advanced_routes(app, learner, state);
    register_society_routes(app, learner, state);
    register_chat_routes(app, learner, state);
    register_runtime_routes(app, learner, state);
    app.validate();
}

/// 调用 handle_full 并返回解析后的 JSON 响应体
static auto dispatch(crow::SimpleApp& app, crow::request& req) -> std::pair<int, json> {
    crow::response res;
    app.handle_full(req, res);
    json body;
    try {
        body = json::parse(res.body);
    } catch (...) {
        // 某些端点返回非 JSON（如错误文本），跳过解析
    }
    return {res.code, body};
}

// ═══════════════════════════════════════════════════════════
// 核心路由测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server Core: 健康检查返回 200 + 正确结构",
          "[server][core][health]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    config.version = "0.2.0";
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/health");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["status"] == "ok");
    CHECK(body["version"] == "0.2.0");
    CHECK(body.contains("stage"));
    CHECK(body.contains("total_steps"));
    CHECK(body["engine_type"] == "pc");
    CHECK(body.contains("avg_inference_steps"));
}

TEST_CASE("Server Core: 统计端点返回 learner 统计",
          "[server][core][stats]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/stats");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_object());
}

TEST_CASE("Server Core: 阶段端点返回当前阶段",
          "[server][core][stage]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/stage");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("stage"));
    CHECK(body["stage"].is_string());
}

TEST_CASE("Server Core: 学习文本端点",
          "[server][core][learn]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/learn/text", {
        {"text", "光合作用是植物利用阳光合成有机物的过程"},
        {"source", "test"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("entities"));
    CHECK(body.contains("triples"));
    CHECK(body.contains("verification_passed"));
}

TEST_CASE("Server Core: 学习文本缺少 text 字段返回 400",
          "[server][core][learn]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/learn/text", {{"source", "test"}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Server Core: 观察文本端点",
          "[server][core][observe]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/observe", {
        {"text", "太阳系有八大行星"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_object());
}

TEST_CASE("Server Core: 推理端点",
          "[server][core][reason]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 先学习一些知识
    learner.learn_from_text("金属在加热后膨胀", "test");

    auto req = make_post("/api/reason", {
        {"question", "金属受热会怎样"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_array());
}

TEST_CASE("Server Core: 思考端点",
          "[server][core][think]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/think", {
        {"question", "什么是光合作用"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("answer"));
    CHECK(body["answer"].is_string());
}

TEST_CASE("Server Core: 记忆巩固端点",
          "[server][core][consolidate]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/consolidate", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_object());
}

TEST_CASE("Server Core: 记忆存储端点",
          "[server][core][remember]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/remember", {
        {"obs", {0.1f, 0.2f, 0.3f}},
        {"action", 1},
        {"next_obs", {0.4f, 0.5f, 0.6f}},
        {"reward", 1.0f},
        {"error", 0.1f}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["status"] == "ok");
}

TEST_CASE("Server Core: 记忆检索端点",
          "[server][core][recall]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/recall", {
        {"cue", {0.1f, 0.2f, 0.3f}},
        {"k", 3}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_array());
}

TEST_CASE("Server Core: 异步自主学习端点返回 task_id",
          "[server][core][autonomous]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/autonomous", {{"iterations", 2}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 202);
    CHECK(body.contains("task_id"));
    CHECK(body["status"] == "pending");

    // 等待异步任务完成
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 查询任务状态
    auto task_id = body["task_id"].get<std::string>();
    auto status_req = make_get("/api/tasks/" + task_id);
    auto [status_code, status_body] = dispatch(app, status_req);

    CHECK(status_code == 200);
    CHECK(status_body.contains("status"));
    // 任务应该是 completed 或 running
    CHECK((status_body["status"] == "completed" ||
           status_body["status"] == "running"));
}

TEST_CASE("Server Core: 查询不存在的任务返回 404",
          "[server][core][tasks]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/tasks/nonexistent_task");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 404);
    CHECK(body.contains("error"));
}

// ═══════════════════════════════════════════════════════════
// 高级路由测试 (Phase 3-6)
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server Advanced: 遗忘检测端点",
          "[server][advanced][forgetting]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/forgetting");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("alerts"));
    CHECK(body.contains("count"));
}

TEST_CASE("Server Advanced: 知识保护端点",
          "[server][advanced][protect]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/protect", {
        {"knowledge_id", "test_knowledge"},
        {"domain", "physics"},
        {"confidence", 0.9},
        {"usage_count", 5}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["status"] == "ok");
}

TEST_CASE("Server Advanced: 情感处理端点",
          "[server][advanced][emotion]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/emotion", {
        {"event_type", "discovery"},
        {"domain", "physics"},
        {"description", "发现了新知识"},
        {"magnitude", 0.8}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("valence"));
    CHECK(body.contains("arousal"));
    CHECK(body.contains("dominance"));
    CHECK(body.contains("label"));
}

TEST_CASE("Server Advanced: 顿悟触发端点",
          "[server][advanced][insight]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/insight", {
        {"problem_context", "如何理解量子纠缠"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("found"));
}

TEST_CASE("Server Advanced: 元认知推荐端点",
          "[server][advanced][meta]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/meta/recommend", {
        {"domain", "math"},
        {"task_type", "learning"},
        {"difficulty", 0.5}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("recommended_strategy"));
    CHECK(body.contains("confidence"));
    CHECK(body.contains("suggested_learning_rate"));
}

TEST_CASE("Server Advanced: 元认知反思端点",
          "[server][advanced][meta]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/meta/reflect", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("reflections"));
    CHECK(body.contains("count"));
}

TEST_CASE("Server Advanced: 社会观察学习端点",
          "[server][advanced][social]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/observe-behavior", {
        {"agent_id", "agent_1"},
        {"action", "experiment"},
        {"context", "lab"},
        {"domain", "chemistry"},
        {"outcome_quality", 0.8}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("learned_strategies"));
    CHECK(body.contains("imitation_success_rate"));
}

TEST_CASE("Server Advanced: 实验设计端点",
          "[server][advanced][experiment]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/experiment/design", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("found"));
}

TEST_CASE("Server Advanced: 实验记录端点",
          "[server][advanced][experiment]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/experiment/record", {
        {"experiment_id", "exp_001"},
        {"hypothesis_id", "hyp_001"},
        {"supports_hypothesis", true},
        {"confidence_delta", 0.2},
        {"observation", "test observation"},
        {"analysis", "test analysis"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["status"] == "ok");
    CHECK(body.contains("result_id"));
}

TEST_CASE("Server Advanced: 整合流水线端点",
          "[server][advanced][integrated]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/integrated/pipeline", {
        {"observation", "金属在加热后膨胀"},
        {"domain", "physics"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("observe_step"));
    CHECK(body.contains("learn_step"));
    CHECK(body.contains("overall_progress"));
    CHECK(body.contains("steps_completed"));
}

TEST_CASE("Server Advanced: 元学习驱动会话端点",
          "[server][advanced][integrated]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/integrated/meta-guided", {
        {"known_topics", {"physics", "chemistry"}},
        {"mastery_map", {{"physics", 0.5}, {"chemistry", 0.3}}}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("chosen_goal"));
    CHECK(body.contains("learning_rate"));
}

TEST_CASE("Server Advanced: 情感调制参数端点",
          "[server][advanced][integrated]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/integrated/emotion-params");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("learning_rate"));
    CHECK(body.contains("encoding_boost"));
    CHECK(body.contains("exploration_tendency"));
    CHECK(body.contains("emotion_label"));
}

TEST_CASE("Server Advanced: 抽象概念端点",
          "[server][advanced][abstract]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/abstract", {
        {"instance_id", "apple_1"},
        {"attributes", {"red", "round", "sweet"}},
        {"features", {{"weight", 0.2}, {"size", 0.5}}}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("new_concepts"));
    CHECK(body.contains("instances_processed"));
    CHECK(body.contains("coherence_score"));
}

TEST_CASE("Server Advanced: 缺少必填字段返回 400",
          "[server][advanced][error]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 缺少 event_type
    auto req = make_post("/api/emotion", {{"domain", "test"}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

