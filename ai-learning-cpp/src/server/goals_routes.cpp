/**
 * @file goals_routes.cpp
 * @brief 目标系统 REST 端点 — 目标创建、分解、进度、查询
 *
 * 端点：
 *   POST /api/goals/create          — 创建目标
 *   POST /api/goals/<id>/decompose  — 分解并规划
 *   POST /api/goals/<id>/progress   — 更新进度
 *   GET  /api/goals                 — 列出所有目标
 *   GET  /api/goals/active          — 活跃目标
 *   GET  /api/goals/next-action     — 下一步动作
 *   GET  /api/goals/<id>            — 目标详情
 *   GET  /api/goals/<id>/completion — 完成状态
 */

#include "route_groups.hpp"
#include "dto.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── 辅助：构造 JSON 响应 ──────────────────────────────────────────
static auto json_resp(int code, const json& body) -> crow::response {
    crow::response res{code, body.dump()};
    res.set_header("Content-Type", "application/json");
    return res;
}
static auto ok_resp(const json& body) -> crow::response { return json_resp(200, body); }
static auto err_resp(int code, const std::string& msg) -> crow::response {
    return json_resp(code, json{{"error", msg}});
}

// ── 辅助：Goal → JSON ──────────────────────────────────────────────
static auto goal_to_json(const ai_learning::goals::Goal& g) -> json {
    return json{
        {"id",                 g.id},
        {"description",        g.description},
        {"status",             ai_learning::goals::goal_status_to_string(g.status)},
        {"sub_goals",          g.sub_goals},
        {"parent_goal",        g.parent_goal},
        {"required_knowledge", g.required_knowledge},
        {"priority",           g.priority},
        {"progress",           g.progress},
        {"created_step",       g.created_step}
    };
}

static auto step_to_json(const ai_learning::goals::LearningStep& s) -> json {
    return json{
        {"action",       ai_learning::goals::action_to_string(s.action)},
        {"target",       s.target},
        {"strategy",     ai_learning::goals::strategy_to_string(s.strategy)},
        {"difficulty",   s.difficulty},
        {"analogy_from", s.analogy_from},
        {"reason",       s.reason},
        {"done",         s.done}
    };
}

static auto plan_to_json(const ai_learning::goals::LearningPlan& p) -> json {
    json steps_arr = json::array();
    for (const auto& s : p.steps) steps_arr.push_back(step_to_json(s));
    return json{
        {"goal_id",          p.goal_id},
        {"steps",            steps_arr},
        {"estimated_effort", p.estimated_effort}
    };
}

void ai_learning::server::register_goals_routes(
    crow::SimpleApp& app, core::Learner& learner, SharedState& /*state*/) {

    auto& gm = learner.goal_manager();

    // POST /api/goals/create
    CROW_ROUTE(app, "/api/goals/create").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("description"))
                return err_resp(400, "missing 'description' field");
            auto goal = gm.create_goal(
                body["description"].get<std::string>(),
                body.value("priority", 0.5));
            return ok_resp(goal_to_json(goal));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/goals/<id>/decompose
    CROW_ROUTE(app, "/api/goals/<string>/decompose").methods("POST"_method)
    ([&](const crow::request&, const std::string& goal_id) -> crow::response {
        try {
            auto plan = gm.decompose_and_plan(goal_id);
            if (!plan.has_value())
                return err_resp(404, "goal not found: " + goal_id);
            return ok_resp(plan_to_json(*plan));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/goals/<id>/progress
    CROW_ROUTE(app, "/api/goals/<string>/progress").methods("POST"_method)
    ([&](const crow::request& req, const std::string& goal_id) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("progress"))
                return err_resp(400, "missing 'progress' field");
            gm.update_progress(goal_id, body["progress"].get<double>());
            auto* g = gm.get_goal(goal_id);
            if (!g) return err_resp(404, "goal not found: " + goal_id);
            return ok_resp(goal_to_json(*g));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // GET /api/goals
    CROW_ROUTE(app, "/api/goals").methods("GET"_method)
    ([&]() -> crow::response {
        auto state = gm.save_state();
        json arr = json::array();
        for (const auto& [gid, goal] : state.goals) {
            arr.push_back(goal_to_json(goal));
        }
        return ok_resp(arr);
    });

    // GET /api/goals/active
    CROW_ROUTE(app, "/api/goals/active").methods("GET"_method)
    ([&]() -> crow::response {
        auto active = gm.get_active_goals();
        json arr = json::array();
        for (const auto* g : active) {
            arr.push_back(goal_to_json(*g));
        }
        return ok_resp(arr);
    });

    // GET /api/goals/next-action
    CROW_ROUTE(app, "/api/goals/next-action").methods("GET"_method)
    ([&]() -> crow::response {
        auto step = gm.get_next_action();
        if (!step.has_value())
            return ok_resp(json{{"action", "none"}, {"message", "no pending actions"}});
        return ok_resp(step_to_json(*step));
    });

    // GET /api/goals/<id>
    CROW_ROUTE(app, "/api/goals/<string>").methods("GET"_method)
    ([&](const std::string& goal_id) -> crow::response {
        auto* g = gm.get_goal(goal_id);
        if (!g) return err_resp(404, "goal not found: " + goal_id);
        return ok_resp(goal_to_json(*g));
    });

    // GET /api/goals/<id>/completion
    CROW_ROUTE(app, "/api/goals/<string>/completion").methods("GET"_method)
    ([&](const std::string& goal_id) -> crow::response {
        bool completed = gm.check_completion(goal_id);
        return ok_resp(json{{"goal_id", goal_id}, {"completed", completed}});
    });
}
