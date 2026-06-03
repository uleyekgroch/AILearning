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
