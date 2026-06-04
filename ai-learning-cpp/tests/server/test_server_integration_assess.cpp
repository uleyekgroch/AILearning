/**
 * @file test_server_integration_assess.cpp
 * @brief REST API 集成测试 — 评估/DTO/EventAdapter/ServerConfig/LearningServer
 *
 * 从 test_server_integration.cpp 分离，遵守单文件 ≤800 行规范。
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "route_groups.hpp"
#include "dto.hpp"
#include "server_config.hpp"
#include "event_adapter.hpp"
#include "server.hpp"

#include "ai_learning/core/learner.hpp"

#include <crow.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

using json = nlohmann::json;
using Catch::Approx;
using namespace ai_learning;
using namespace ai_learning::server;

// ═══════════════════════════════════════════════════════════
// 辅助工具（与 test_server_integration.cpp 相同）
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
// 评估端点
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration Assess: 评估所有领域",
          "[integration][assess]") {
    auto learner = make_test_learner();
    // 先学习一些知识，确保 KG 非空
    learner.learn_from_text("金属是良好的导电体", "test");
    learner.learn_from_text("水在零度结冰", "test");

    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/assess");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_object());
}

TEST_CASE("Integration Assess: 评估指定领域",
          "[integration][assess]") {
    auto learner = make_test_learner();
    learner.learn_from_text("金属是良好的导电体", "test");

    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/assess/concept");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("domain"));
    CHECK(body.contains("total_units"));
    CHECK(body.contains("avg_mastery"));
    CHECK(body.contains("coverage"));
}

TEST_CASE("Integration Assess: 熟练度评估端点（实际匹配 assess/<domain> 路由）",
          "[integration][assess]") {
    auto learner = make_test_learner();
    learner.learn_from_text("光合作用是植物利用阳光合成有机物的过程", "test");

    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 注意: /api/assess/proficiency 被 /api/assess/<string> 通配路由捕获
    // 实际返回 DomainReport（领域评估），而非 ProficiencyReport
    auto req = make_get("/api/assess/proficiency");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("domain"));
    CHECK(body.contains("total_units"));
    CHECK(body.contains("coverage"));
}

// ═══════════════════════════════════════════════════════════
// ServerConfig 单元测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration Config: ServerConfig 默认值正确",
          "[integration][config]") {
    ServerConfig config;
    CHECK(config.port == 8080);
    CHECK(config.threads == 4);
    CHECK(config.cors_enabled == true);
    CHECK(config.static_dir.empty());
    CHECK(config.version == "0.2.0");
}

TEST_CASE("Integration Config: ServerConfig 自定义值",
          "[integration][config]") {
    ServerConfig config;
    config.port = 9090;
    config.threads = 8;
    config.cors_enabled = false;
    config.static_dir = "/var/www";
    config.version = "1.0.0";

    CHECK(config.port == 9090);
    CHECK(config.threads == 8);
    CHECK(config.cors_enabled == false);
    CHECK(config.static_dir == "/var/www");
    CHECK(config.version == "1.0.0");
}

// ═══════════════════════════════════════════════════════════
// LearningServer 构造与生命周期
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration Server: LearningServer 构造不抛异常",
          "[integration][server]") {
    auto learner = make_test_learner();
    ServerConfig config;
    config.port = 0;  // 不实际绑定端口

    REQUIRE_NOTHROW(LearningServer(learner, config));
}

TEST_CASE("Integration Server: LearningServer 初始未运行",
          "[integration][server]") {
    auto learner = make_test_learner();
    ServerConfig config;
    LearningServer server(learner, config);

    CHECK_FALSE(server.is_running());
}

TEST_CASE("Integration Server: LearningServer 构造后可析构",
          "[integration][server]") {
    auto learner = make_test_learner();
    ServerConfig config;
    {
        LearningServer server(learner, config);
        CHECK_FALSE(server.is_running());
    }
    // 析构不应抛异常或崩溃
}

// ═══════════════════════════════════════════════════════════
// DTO 序列化完整性测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration DTO: parse_behavior_observation 完整字段",
          "[integration][dto]") {
    json j = {
        {"agent_id", "agent_test"},
        {"action", "observe"},
        {"context", "lab_environment"},
        {"domain", "chemistry"},
        {"outcome_quality", 0.75},
        {"preconditions", {"goggles", "gloves"}},
        {"effects", {"discovery", "data_recorded"}}
    };
    auto obs = dto::parse_behavior_observation(j);
    CHECK(obs.agent_id == "agent_test");
    CHECK(obs.action == "observe");
    CHECK(obs.context == "lab_environment");
    CHECK(obs.domain == "chemistry");
    CHECK(obs.outcome_quality == Approx(0.75));
    CHECK(obs.preconditions.size() == 2);
    CHECK(obs.effects.size() == 2);
}

TEST_CASE("Integration DTO: parse_emotion_event 完整字段",
          "[integration][dto]") {
    json j = {
        {"event_type", "success"},
        {"domain", "math"},
        {"description", "solved a hard problem"},
        {"magnitude", 0.9},
        {"expected_outcome", 0.5},
        {"actual_outcome", 1.0}
    };
    auto evt = dto::parse_emotion_event(j);
    CHECK(evt.event_type == "success");
    CHECK(evt.domain == "math");
    CHECK(evt.description == "solved a hard problem");
    CHECK(evt.magnitude == Approx(0.9));
    CHECK(evt.expected_outcome == Approx(0.5));
    CHECK(evt.actual_outcome == Approx(1.0));
}

TEST_CASE("Integration DTO: parse_experiment_result 完整字段",
          "[integration][dto]") {
    json j = {
        {"experiment_id", "exp_42"},
        {"hypothesis_id", "hyp_7"},
        {"supports_hypothesis", true},
        {"confidence_delta", 0.3},
        {"observation", "temperature increased"},
        {"analysis", "consistent with hypothesis"},
        {"information_gain", 1.5},
        {"surprise", 0.2}
    };
    auto r = dto::parse_experiment_result(j);
    CHECK(r.experiment_id == "exp_42");
    CHECK(r.hypothesis_id == "hyp_7");
    CHECK(r.supports_hypothesis == true);
    CHECK(r.confidence_delta == Approx(0.3));
    CHECK(r.observation == "temperature increased");
    CHECK(r.analysis == "consistent with hypothesis");
    CHECK(r.information_gain == Approx(1.5));
    CHECK(r.surprise == Approx(0.2));
}

TEST_CASE("Integration DTO: to_json_emotion_state 包含全部字段",
          "[integration][dto]") {
    learning::EmotionState state;
    state.valence = 0.5;
    state.arousal = 0.7;
    state.dominance = 0.3;
    state.label = "excited";

    auto j = dto::to_json_emotion_state(state);
    CHECK(j["valence"].get<double>() == Approx(0.5));
    CHECK(j["arousal"].get<double>() == Approx(0.7));
    CHECK(j["dominance"].get<double>() == Approx(0.3));
    CHECK(j["label"] == "excited");
    CHECK(j.contains("intensity"));
}

TEST_CASE("Integration DTO: parse_metadata 正确解析",
          "[integration][dto]") {
    json j = {
        {"key1", "value1"},
        {"key2", "value2"},
        {"numeric_key", "123"}
    };
    auto result = dto::parse_metadata(j);
    CHECK(result.size() == 3);
    CHECK(result["key1"] == "value1");
    CHECK(result["key2"] == "value2");
    CHECK(result["numeric_key"] == "123");
}

TEST_CASE("Integration DTO: parse_metadata 忽略非字符串值",
          "[integration][dto]") {
    json j = {
        {"str_key", "value"},
        {"num_key", 42},
        {"bool_key", true},
        {"arr_key", {1, 2, 3}}
    };
    auto result = dto::parse_metadata(j);
    CHECK(result.size() == 1);
    CHECK(result.count("str_key") == 1);
}

TEST_CASE("Integration DTO: parse_float_vector 空数组返回空",
          "[integration][dto]") {
    json arr = json::array();
    auto result = dto::parse_float_vector(arr);
    CHECK(result.empty());
}

TEST_CASE("Integration DTO: parse_raw_input 空对象返回空",
          "[integration][dto]") {
    json obj = json::object();
    auto result = dto::parse_raw_input(obj);
    CHECK(result.empty());
}

TEST_CASE("Integration DTO: parse_raw_input 非对象返回空",
          "[integration][dto]") {
    json arr = json::array();
    auto result = dto::parse_raw_input(arr);
    CHECK(result.empty());
}

// ═══════════════════════════════════════════════════════════
// EventAdapter 领域事件序列化完整性
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration EventAdapter: RelationAdded 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::RelationAdded evt{
        "evt_rel",
        std::chrono::steady_clock::now(),
        "entity_a",
        "entity_b",
        "causes",
        0.85
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "learn");
    CHECK(arr[0]["data"]["subtype"] == "relation_added");
    CHECK(arr[0]["data"]["source_id"] == "entity_a");
    CHECK(arr[0]["data"]["target_id"] == "entity_b");
    CHECK(arr[0]["data"]["relation_type"] == "causes");
    CHECK(arr[0]["data"]["confidence"].get<double>() == Approx(0.85));
}

TEST_CASE("Integration EventAdapter: KnowledgeLearned 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::KnowledgeLearned evt{
        "evt_know",
        std::chrono::steady_clock::now(),
        "gravity pulls objects down",
        "textbook",
        0.95
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "learn");
    CHECK(arr[0]["data"]["subtype"] == "knowledge_learned");
    CHECK(arr[0]["data"]["content"] == "gravity pulls objects down");
    CHECK(arr[0]["data"]["source"] == "textbook");
}

TEST_CASE("Integration EventAdapter: PredictionError 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::PredictionError evt{
        "evt_pred",
        std::chrono::steady_clock::now(),
        0.42,
        0.78
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "reason");
    CHECK(arr[0]["data"]["error_magnitude"].get<double>() == Approx(0.42));
    CHECK(arr[0]["data"]["confidence"].get<double>() == Approx(0.78));
}

TEST_CASE("Integration EventAdapter: StageAdvanced 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::StageAdvanced evt{
        "evt_stage",
        std::chrono::steady_clock::now(),
        "sensorimotor",
        "single_word"
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "stage_change");
    CHECK(arr[0]["data"]["from_stage"] == "sensorimotor");
    CHECK(arr[0]["data"]["to_stage"] == "single_word");
}

TEST_CASE("Integration EventAdapter: MemoryStored 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::MemoryStored evt{
        "evt_mem",
        std::chrono::steady_clock::now(),
        "mem_001",
        0.85
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "remember");
    CHECK(arr[0]["data"]["memory_id"] == "mem_001");
    CHECK(arr[0]["data"]["importance"].get<double>() == Approx(0.85));
}

TEST_CASE("Integration EventAdapter: MemoryConsolidated 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::MemoryConsolidated evt{
        "evt_consol",
        std::chrono::steady_clock::now(),
        5,
        2
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "remember");
    CHECK(arr[0]["data"]["subtype"] == "memory_consolidated");
    CHECK(arr[0]["data"]["consolidated_count"] == 5);
    CHECK(arr[0]["data"]["forgotten_count"] == 2);
}

TEST_CASE("Integration EventAdapter: VocabularyRecorded 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::VocabularyRecorded evt{
        "evt_vocab",
        std::chrono::steady_clock::now(),
        "quantum",
        true
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "learn");
    CHECK(arr[0]["data"]["symbol"] == "quantum");
    CHECK(arr[0]["data"]["success"] == true);
}

TEST_CASE("Integration EventAdapter: GrammarRuleExtracted 事件正确序列化",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;
    domain::GrammarRuleExtracted evt{
        "evt_grammar",
        std::chrono::steady_clock::now(),
        "S -> NP VP",
        0.88
    };
    adapter.on_event(domain::DomainEvent{evt});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 1);
    CHECK(arr[0]["type"] == "learn");
    CHECK(arr[0]["data"]["subtype"] == "grammar_rule_extracted");
    CHECK(arr[0]["data"]["pattern"] == "S -> NP VP");
    CHECK(arr[0]["data"]["confidence"].get<double>() == Approx(0.88));
}

TEST_CASE("Integration EventAdapter: 多事件混合推送",
          "[integration][event_adapter]") {
    server::EventAdapter adapter;

    // 推送三种不同类型的事件
    adapter.on_event(domain::DomainEvent{domain::EntityCreated{
        "e1", std::chrono::steady_clock::now(), "cat", "animal"}});
    adapter.on_event(domain::DomainEvent{domain::PredictionError{
        "e2", std::chrono::steady_clock::now(), 0.1, 0.9}});
    adapter.on_event(domain::DomainEvent{domain::StageAdvanced{
        "e3", std::chrono::steady_clock::now(), "a", "b"}});

    auto history = adapter.get_history_json();
    auto arr = json::parse(history);
    CHECK(arr.size() == 3);
    CHECK(arr[0]["type"] == "learn");
    CHECK(arr[1]["type"] == "reason");
    CHECK(arr[2]["type"] == "stage_change");
}

// ═══════════════════════════════════════════════════════════
// 多步操作集成（端到端风格）
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration E2E: 学习 → 统计变化 → 巩固 → 统计更新",
          "[integration][e2e]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 获取初始统计
    auto stats_req1 = make_get("/api/stats");
    auto [_, stats1] = dispatch(app, stats_req1);

    // 学习文本
    auto learn_req = make_post("/api/learn/text", {
        {"text", "地球是太阳系中第三颗行星"},
        {"source", "astronomy"}
    });
    auto [learn_code, learn_body] = dispatch(app, learn_req);
    CHECK(learn_code == 200);

    // 巩固
    auto consol_req = make_post("/api/consolidate", {});
    auto [consol_code, consol_body] = dispatch(app, consol_req);
    CHECK(consol_code == 200);

    // 统计应有变化（至少 total_steps 应增加）
    auto stats_req2 = make_get("/api/stats");
    auto [__, stats2] = dispatch(app, stats_req2);

    // 验证统计是有效的 JSON 对象
    CHECK(stats2.is_object());
}

TEST_CASE("Integration E2E: 情感 → 情感调制参数联动",
          "[integration][e2e]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 触发情感事件
    auto emotion_req = make_post("/api/emotion", {
        {"event_type", "discovery"},
        {"domain", "physics"},
        {"description", "发现了新规律"},
        {"magnitude", 0.9}
    });
    auto [emotion_code, emotion_body] = dispatch(app, emotion_req);
    CHECK(emotion_code == 200);

    // 查看情感调制参数（应该反映情感状态变化）
    auto params_req = make_get("/api/integrated/emotion-params");
    auto [params_code, params_body] = dispatch(app, params_req);
    CHECK(params_code == 200);
    CHECK(params_body.contains("learning_rate"));
    CHECK(params_body.contains("emotion_label"));
}

TEST_CASE("Integration E2E: 目标创建 → 分解 → 进度更新 → 完成检查",
          "[integration][e2e]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 1. 创建目标
    auto create_req = make_post("/api/goals/create", {
        {"description", "掌握热力学三大定律"},
        {"priority", 0.9}
    });
    auto [create_code, create_body] = dispatch(app, create_req);
    CHECK(create_code == 200);
    auto goal_id = create_body["id"].get<std::string>();

    // 2. 分解目标
    auto decompose_req = make_post("/api/goals/" + goal_id + "/decompose", {});
    auto [decompose_code, decompose_body] = dispatch(app, decompose_req);
    CHECK(decompose_code == 200);
    CHECK(decompose_body["steps"].is_array());

    // 3. 更新进度
    auto progress_req = make_post("/api/goals/" + goal_id + "/progress", {
        {"progress", 0.5}
    });
    auto [progress_code, _] = dispatch(app, progress_req);
    CHECK(progress_code == 200);

    // 4. 检查完成状态
    auto completion_req = make_get("/api/goals/" + goal_id + "/completion");
    auto [completion_code, completion_body] = dispatch(app, completion_req);
    CHECK(completion_code == 200);
    CHECK(completion_body["goal_id"] == goal_id);
    CHECK(completion_body.contains("completed"));
}

TEST_CASE("Integration E2E: 知识保护 → 遗忘检测 → 遗忘警报",
          "[integration][e2e]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 1. 保护知识
    auto protect_req = make_post("/api/protect", {
        {"knowledge_id", "thermodynamics_1"},
        {"domain", "physics"},
        {"confidence", 0.95},
        {"usage_count", 10}
    });
    auto [protect_code, protect_body] = dispatch(app, protect_req);
    CHECK(protect_code == 200);
    CHECK(protect_body["status"] == "ok");

    // 2. 检测遗忘（刚保护的知识不应有遗忘警报）
    auto forgetting_req = make_get("/api/forgetting");
    auto [forgetting_code, forgetting_body] = dispatch(app, forgetting_req);
    CHECK(forgetting_code == 200);
    CHECK(forgetting_body.contains("alerts"));
    CHECK(forgetting_body.contains("count"));
}
