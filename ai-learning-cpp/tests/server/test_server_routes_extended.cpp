/**
 * @file test_server_routes_extended.cpp
 * @brief REST API 集成测试 — 社会/对话/运行时路由
 *
 * 从 test_server_routes.cpp 分离，遵守单文件 ≤800 行规范。
 *
 * 测试分组：
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
// 辅助工具（与 test_server_routes.cpp 相同）
// ═══════════════════════════════════════════════════════════

static auto make_test_learner() -> core::Learner {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    config.embedding_learning_enabled = false;
    return core::Learner(config);
}

static auto make_get(const std::string& url) -> crow::request {
    crow::request req;
    req.method = crow::HTTPMethod::Get;
    req.raw_url = url;
    req.url = url;
    return req;
}

static auto make_post(const std::string& url, const json& body) -> crow::request {
    crow::request req;
    req.method = crow::HTTPMethod::Post;
    req.raw_url = url;
    req.url = url;
    req.body = body.dump();
    return req;
}

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
    register_goals_routes(app, learner, state);
    app.validate();
}

static auto dispatch(crow::SimpleApp& app, crow::request& req) -> std::pair<int, json> {
    crow::response res;
    app.handle_full(req, res);
    json body;
    try {
        body = json::parse(res.body);
    } catch (...) {}
    return {res.code, body};
}

// ═══════════════════════════════════════════════════════════
// 社会路由测试 (Phase 9)
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server Society: 列出空 Agent 列表",
          "[server][society]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/society/agents");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("agents"));
    CHECK(body["agents"].is_array());
    CHECK(body["count"].get<int>() == 0);
}

TEST_CASE("Server Society: 社会指标端点",
          "[server][society]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/society/metrics");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("total_agents"));
    CHECK(body.contains("active_agents"));
    CHECK(body.contains("avg_knowledge_per_agent"));
    CHECK(body.contains("knowledge_diversity"));
}

TEST_CASE("Server Society: 创建 Agent 端点",
          "[server][society]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/society/create", {{"port", 0}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["status"] == "ok");
    CHECK(body.contains("agent_id"));
}

TEST_CASE("Server Society: 删除不存在的 Agent",
          "[server][society]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_delete("/society/agents/nonexistent_agent");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
}

// ═══════════════════════════════════════════════════════════
// 对话路由测试 (Phase 9)
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server Chat: 空历史对话端点",
          "[server][chat]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/chat/history");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("history"));
    CHECK(body["history"].is_array());
    CHECK(body["count"].get<int>() == 0);
}

TEST_CASE("Server Chat: 对话端点（Stub LLM）",
          "[server][chat]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/chat", {
        {"message", "什么是光合作用"},
        {"session_id", "test_session"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("assistant_message"));
    CHECK(body.contains("intent"));
}

TEST_CASE("Server Chat: 缺少 message 字段返回 400",
          "[server][chat]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/chat", {{"session_id", "test"}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Server Chat: 清除会话端点",
          "[server][chat]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_delete("/api/chat/session/test_session");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["status"] == "ok");
}

// ═══════════════════════════════════════════════════════════
// 运行时路由测试 (Phase 9)
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server Runtime: 初始状态端点",
          "[server][runtime]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/runtime/status");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("running"));
    CHECK(body["running"].get<bool>() == false);
}

TEST_CASE("Server Runtime: 喂数据端点",
          "[server][runtime]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/runtime/feed", {
        {"data", "金属是良好的导体"},
        {"source", "test"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["status"] == "ok");
}

TEST_CASE("Server Runtime: 缺少 data 字段返回 400",
          "[server][runtime]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/runtime/feed", {{"source", "test"}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Server Runtime: 检查点端点（未启动循环）",
          "[server][runtime]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/runtime/checkpoint", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Server Runtime: 恢复端点",
          "[server][runtime]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/runtime/recover", {
        {"dir", "./nonexistent_checkpoints"}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("status"));
    // 恢复可能成功或失败，取决于检查点是否存在
    CHECK((body["status"] == "ok" || body["status"] == "failed"));
}

// ═══════════════════════════════════════════════════════════
// DTO 序列化测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server DTO: make_error_response 格式正确",
          "[server][dto]") {
    auto result = dto::make_error_response("test error");
    CHECK(result["error"] == "test error");
}

TEST_CASE("Server DTO: parse_float_vector 正确解析",
          "[server][dto]") {
    json arr = {1.0f, 2.5f, 3.7f};
    auto result = dto::parse_float_vector(arr);
    CHECK(result.size() == 3);
    CHECK(result[0] == 1.0f);
    CHECK(result[1] == 2.5f);
    CHECK(result[2] == 3.7f);
}

TEST_CASE("Server DTO: parse_float_vector 非数组返回空",
          "[server][dto]") {
    json obj = {{"key", "value"}};
    auto result = dto::parse_float_vector(obj);
    CHECK(result.empty());
}

TEST_CASE("Server DTO: parse_raw_input 正确解析",
          "[server][dto]") {
    json obj = {
        {"sensor1", {0.1f, 0.2f}},
        {"sensor2", {0.3f, 0.4f, 0.5f}}
    };
    auto result = dto::parse_raw_input(obj);
    CHECK(result.size() == 2);
    CHECK(result["sensor1"].size() == 2);
    CHECK(result["sensor2"].size() == 3);
}

TEST_CASE("Server DTO: parse_concept_descriptor 正确解析",
          "[server][dto]") {
    json j = {
        {"id", "concept_1"},
        {"domain", "physics"},
        {"attributes", {"hot", "conductive"}},
        {"relations", {"heated_by", "conducts_to"}},
        {"features", {{"temperature", 0.8}}}
    };
    auto cd = dto::parse_concept_descriptor(j);
    CHECK(cd.id == "concept_1");
    CHECK(cd.domain == "physics");
    CHECK(cd.attributes.size() == 2);
    CHECK(cd.relations.size() == 2);
    CHECK(cd.features.size() == 1);
}

TEST_CASE("Server DTO: parse_task_descriptor 正确解析",
          "[server][dto]") {
    json j = {
        {"domain", "math"},
        {"task_type", "problem_solving"},
        {"difficulty", 0.7},
        {"novelty", 0.3},
        {"urgency", 0.5}
    };
    auto td = dto::parse_task_descriptor(j);
    CHECK(td.domain == "math");
    CHECK(td.task_type == "problem_solving");
}

TEST_CASE("Server DTO: strategy_type_to_string 全覆盖",
          "[server][dto]") {
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kRoteMemorization) == "rote_memorization");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kSpacedRepetition) == "spaced_repetition");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kActiveRecall) == "active_recall");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kTrialAndError) == "trial_and_error");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kAnalogicalTransfer) == "analogical_transfer");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kDecomposition) == "decomposition");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kExplanationBased) == "explanation_based");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kExploratory) == "exploratory");
    CHECK(dto::strategy_type_to_string(
        learning::LearningStrategyType::kStructuredPractice) == "structured_practice");
}

// ═══════════════════════════════════════════════════════════
// SharedState 管理测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server: SharedState 初始状态正确",
          "[server][shared_state]") {
    SharedState state;
    CHECK(state.tasks.empty());
    CHECK_FALSE(state.society);
    CHECK_FALSE(state.dialog);
    CHECK_FALSE(state.continuous_loop);
    CHECK_FALSE(state.llm_provider);
}

TEST_CASE("Server: generate_task_id 生成唯一 ID",
          "[server][shared_state]") {
    auto id1 = generate_task_id();
    auto id2 = generate_task_id();
    CHECK_FALSE(id1.empty());
    CHECK_FALSE(id2.empty());
    CHECK(id1 != id2);
}

// ═══════════════════════════════════════════════════════════
// EventAdapter 单元测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server: EventAdapter 初始状态",
          "[server][event_adapter]") {
    server::EventAdapter adapter;
    CHECK(adapter.connection_count() == 0);
    // 空历史
    auto history = adapter.get_history_json();
    CHECK(history == "[]");
}

TEST_CASE("Server: EventAdapter 接收领域事件并缓存",
          "[server][event_adapter]") {
    server::EventAdapter adapter;
    domain::EntityCreated evt{
        "evt_1",
        std::chrono::steady_clock::now(),
        "entity_1",
        "concept"
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "learn");
    CHECK(arr[0]["data"]["entity_id"] == "entity_1");
}

TEST_CASE("Server: EventAdapter 环形缓冲区溢出",
          "[server][event_adapter]") {
    server::EventAdapter adapter;

    // 插入超过缓冲区大小的事件
    for (int i = 0; i < 150; ++i) {
        domain::EntityCreated evt{
            "evt_" + std::to_string(i),
            std::chrono::steady_clock::now(),
            "entity_" + std::to_string(i),
            "concept"
        };
        adapter.on_event(domain::DomainEvent{evt});
    }

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    // 缓冲区最大容量为 kEventBufferSize = 100
    CHECK(arr.size() == 100);
}

TEST_CASE("Server: EventAdapter publish_custom 正确推送",
          "[server][event_adapter]") {
    server::EventAdapter adapter;
    adapter.publish_custom("test_type", R"({"key":"value"})");

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "test_type");
    CHECK(arr[0]["data"]["key"] == "value");
}

TEST_CASE("Server: StatsPusher 初始无连接",
          "[server][event_adapter]") {
    server::StatsPusher pusher;
    CHECK(pusher.connection_count() == 0);
}

TEST_CASE("Server: HeartbeatManager 初始无连接",
          "[server][event_adapter]") {
    server::HeartbeatManager hb;
    // 初始无连接，超时检查应返回空列表
    auto expired = hb.check_timeouts(std::chrono::seconds{60});
    CHECK(expired.empty());
}

// ═══════════════════════════════════════════════════════════
// 引擎选择测试 (Phase F)
// ═══════════════════════════════════════════════════════════

TEST_CASE("Server Engine: MLP 引擎健康检查",
          "[server][engine][mlp]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    auto learner = core::LearnerFactory::create_with_engine(
        config, core::LearnerFactory::make_engine("mlp", config));

    crow::SimpleApp app;
    ServerConfig srv_config;
    SharedState state;
    register_all_routes(app, *learner, srv_config, state);

    auto req = make_get("/api/health");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["engine_type"] == "mlp");
    CHECK(body["avg_inference_steps"] == 1.0);
}

TEST_CASE("Server Engine: Light 引擎健康检查",
          "[server][engine][light]") {
    core::LearnerConfig config;
    config.initial_stage = "literacy";
    auto learner = core::LearnerFactory::create_with_engine(
        config, core::LearnerFactory::make_engine("light", config));

    crow::SimpleApp app;
    ServerConfig srv_config;
    SharedState state;
    register_all_routes(app, *learner, srv_config, state);

    auto req = make_get("/api/health");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["engine_type"] == "light");
    CHECK(body["avg_inference_steps"] == 1.0);
}
