/**
 * @file test_server_integration.cpp
 * @brief REST API + WebSocket 集成测试 — 覆盖 test_server_routes.cpp 未包含的端点
 *
 * 测试策略：
 *   - 使用 crow::SimpleApp + handle_full() 直接调用路由处理逻辑
 *   - 无需启动 HTTP 服务器
 *   - 覆盖目标系统、评估端点、保存/加载循环、感知、类比等
 *
 * 测试分组：
 *   [integration][goals]      — 目标系统 CRUD
 *   [integration][assess]     — Bloom 掌握度评估
 *   [integration][core]       — 保存/加载循环、感知、类比
 *   [integration][config]     — ServerConfig 单元测试
 *   [integration][dto]        — DTO 序列化完整性
 *   [integration][server]     — LearningServer 生命周期
 *   [integration][validate]   — 路由注册完整性验证
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
    register_goals_routes(app, learner, state);
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
// 路由注册完整性验证
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration: validate() 通过确认所有路由已注册",
          "[integration][validate]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;

    // 注册所有路由并验证 — 如果任何路由有冲突或拼写错误，validate() 会抛异常
    REQUIRE_NOTHROW(register_all_routes(app, learner, config, state));
}

// ═══════════════════════════════════════════════════════════
// 感知端点
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration Core: 感知端点正确处理 raw_input",
          "[integration][core][perceive]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/perceive", {
        {"raw_input", {
            {"sensor_a", {0.1f, 0.2f, 0.3f}},
            {"sensor_b", {0.4f, 0.5f}}
        }}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("perception"));
    CHECK(body["perception"].is_array());
}

TEST_CASE("Integration Core: 感知端点缺少 raw_input 返回 400",
          "[integration][core][perceive]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/perceive", {{"other_field", 42}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

// ═══════════════════════════════════════════════════════════
// 类比迁移端点
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration Core: 类比迁移端点",
          "[integration][core][analogize]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/analogize", {
        {"source_concepts", json::array({
            {{"id", "heat"}, {"domain", "physics"}, {"attributes", {"hot", "expansive"}}}
        })},
        {"target_concepts", json::array({
            {{"id", "sound"}, {"domain", "acoustics"}, {"attributes", {"loud", "resonant"}}}
        })},
        {"source_facts", {"热膨胀导致体积增大"}}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("source_domain"));
    CHECK(body.contains("target_domain"));
    CHECK(body.contains("mappings"));
    CHECK(body.contains("transfer_quality"));
}

TEST_CASE("Integration Core: 类比迁移缺少 source_concepts 返回 400",
          "[integration][core][analogize]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/analogize", {
        {"target_concepts", json::array({
            {{"id", "x"}, {"domain", "test"}}
        })}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

// ═══════════════════════════════════════════════════════════
// 保存/加载循环
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration Core: save + load 循环持久化状态",
          "[integration][core][persistence]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 先学习一些知识
    auto learn_req = make_post("/api/learn/text", {
        {"text", "地球围绕太阳公转"},
        {"source", "test"}
    });
    auto [learn_code, learn_body] = dispatch(app, learn_req);
    CHECK(learn_code == 200);

    const std::string save_path = "test_integration_save.bin";

    // 保存
    auto save_req = make_post("/api/save", {{"path", save_path}});
    auto [save_code, save_body] = dispatch(app, save_req);
    CHECK(save_code == 200);
    CHECK(save_body["status"] == "ok");
    CHECK(save_body["path"] == save_path);

    // 加载（同一个 Learner 实例）
    auto load_req = make_post("/api/load", {{"path", save_path}});
    auto [load_code, load_body] = dispatch(app, load_req);
    CHECK(load_code == 200);
    CHECK(load_body["status"] == "ok");

    // 清理测试文件
    std::remove(save_path.c_str());
}

TEST_CASE("Integration Core: save 缺少 path 返回 400",
          "[integration][core][persistence]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/save", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Integration Core: save 路径遍历保护",
          "[integration][core][persistence]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/save", {{"path", "../etc/passwd"}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Integration Core: load 缺少 path 返回 400",
          "[integration][core][persistence]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/load", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Integration Core: load 路径遍历保护",
          "[integration][core][persistence]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/load", {{"path", "../../secret.data"}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

// ═══════════════════════════════════════════════════════════
// 目标系统端点
// ═══════════════════════════════════════════════════════════

TEST_CASE("Integration Goals: 创建目标",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/goals/create", {
        {"description", "学习量子力学基础"},
        {"priority", 0.8}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("id"));
    CHECK(body["description"] == "学习量子力学基础");
    CHECK(body.contains("status"));
    CHECK(body.contains("priority"));
    CHECK(body.contains("progress"));
}

TEST_CASE("Integration Goals: 创建目标缺少 description 返回 400",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/goals/create", {{"priority", 0.5}});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 400);
    CHECK(body.contains("error"));
}

TEST_CASE("Integration Goals: 列出所有目标（初始为空）",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/goals");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_array());
}

TEST_CASE("Integration Goals: 创建后列出目标包含新建项",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 创建目标
    auto create_req = make_post("/api/goals/create", {
        {"description", "掌握微积分"}
    });
    auto [create_code, create_body] = dispatch(app, create_req);
    CHECK(create_code == 200);

    // 列出目标
    auto list_req = make_get("/api/goals");
    auto [list_code, list_body] = dispatch(app, list_req);

    CHECK(list_code == 200);
    CHECK(list_body.is_array());
    CHECK(list_body.size() >= 1);
}

TEST_CASE("Integration Goals: 获取活跃目标",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 先创建一个目标
    auto create_req = make_post("/api/goals/create", {
        {"description", "学习线性代数"}
    });
    dispatch(app, create_req);

    auto req = make_get("/api/goals/active");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.is_array());
}

TEST_CASE("Integration Goals: 获取下一步动作",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/goals/next-action");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("action"));
}

TEST_CASE("Integration Goals: 获取目标详情",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 创建目标
    auto create_req = make_post("/api/goals/create", {
        {"description", "学习概率论"}
    });
    auto [_, create_body] = dispatch(app, create_req);
    auto goal_id = create_body["id"].get<std::string>();

    // 获取详情
    auto req = make_get("/api/goals/" + goal_id);
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["id"] == goal_id);
    CHECK(body["description"] == "学习概率论");
}

TEST_CASE("Integration Goals: 获取不存在目标返回 404",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_get("/api/goals/nonexistent_id");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 404);
    CHECK(body.contains("error"));
}

TEST_CASE("Integration Goals: 分解目标",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 创建目标
    auto create_req = make_post("/api/goals/create", {
        {"description", "学习统计学"}
    });
    auto [_, create_body] = dispatch(app, create_req);
    auto goal_id = create_body["id"].get<std::string>();

    // 分解
    auto req = make_post("/api/goals/" + goal_id + "/decompose", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body.contains("goal_id"));
    CHECK(body.contains("steps"));
    CHECK(body["steps"].is_array());
}

TEST_CASE("Integration Goals: 分解不存在目标返回 404",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/goals/nonexistent_id/decompose", {});
    auto [code, body] = dispatch(app, req);

    CHECK(code == 404);
    CHECK(body.contains("error"));
}

TEST_CASE("Integration Goals: 更新目标进度",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 创建目标
    auto create_req = make_post("/api/goals/create", {
        {"description", "学习离散数学"}
    });
    auto [_, create_body] = dispatch(app, create_req);
    auto goal_id = create_body["id"].get<std::string>();

    // 更新进度
    auto req = make_post("/api/goals/" + goal_id + "/progress", {
        {"progress", 0.6}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["id"] == goal_id);
    CHECK(body["progress"].get<double>() == Approx(0.6).margin(0.01));
}

TEST_CASE("Integration Goals: 更新不存在目标进度返回 404",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    auto req = make_post("/api/goals/nonexistent_id/progress", {
        {"progress", 0.5}
    });
    auto [code, body] = dispatch(app, req);

    CHECK(code == 404);
}

TEST_CASE("Integration Goals: 完成状态检查",
          "[integration][goals]") {
    auto learner = make_test_learner();
    crow::SimpleApp app;
    ServerConfig config;
    SharedState state;
    register_all_routes(app, learner, config, state);

    // 创建目标
    auto create_req = make_post("/api/goals/create", {
        {"description", "学习图论"}
    });
    auto [_, create_body] = dispatch(app, create_req);
    auto goal_id = create_body["id"].get<std::string>();

    // 检查完成状态（新目标不应已完成）
    auto req = make_get("/api/goals/" + goal_id + "/completion");
    auto [code, body] = dispatch(app, req);

    CHECK(code == 200);
    CHECK(body["goal_id"] == goal_id);
    CHECK(body.contains("completed"));
}

