/**
 * @file advanced_routes.cpp
 * @brief 高级认知 API 路由（Phase 3-6）
 *
 * Phase 3: analogize, protect, forgetting, abstract
 * Phase 4: observe-behavior, emotion, insight
 * Phase 5: meta/recommend, meta/reflect, experiment/design, experiment/record
 * Phase 6: integrated/pipeline, integrated/meta-guided, integrated/emotion-params
 */

#include "route_groups.hpp"
#include "dto.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using json = nlohmann::json;
namespace dto = ai_learning::server::dto;

static auto ok_resp(const json& body) -> crow::response {
    crow::response res{body.dump()};
    res.set_header("Content-Type", "application/json");
    return res;
}
static auto err_resp(int code, const std::string& msg) -> crow::response {
    crow::response res{code, dto::make_error_response(msg).dump()};
    res.set_header("Content-Type", "application/json");
    return res;
}

void ai_learning::server::register_advanced_routes(
    crow::SimpleApp& app, core::Learner& learner, SharedState&) {

    // ── Phase 3 ──────────────────────────────────────────────────────

    // POST /api/analogize
    CROW_ROUTE(app, "/api/analogize").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("source_concepts") || !body.contains("target_concepts"))
                return err_resp(400, "missing required fields: source_concepts, target_concepts");
            std::vector<ai_learning::learning::ConceptDescriptor> source, target;
            for (const auto& sc : body["source_concepts"]) source.push_back(dto::parse_concept_descriptor(sc));
            for (const auto& tc : body["target_concepts"]) target.push_back(dto::parse_concept_descriptor(tc));
            std::vector<std::string> facts;
            if (body.contains("source_facts") && body["source_facts"].is_array())
                facts = body["source_facts"].get<std::vector<std::string>>();
            return ok_resp(dto::to_json_transfer_result(learner.analogical_transfer(source, target, facts)));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/protect
    CROW_ROUTE(app, "/api/protect").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("knowledge_id") || !body.contains("domain"))
                return err_resp(400, "missing required fields: knowledge_id, domain");
            auto kid = body["knowledge_id"].get<std::string>();
            learner.protect_knowledge(kid, body["domain"].get<std::string>(),
                                      body.value("confidence", 0.5), body.value("usage_count", 0));
            return ok_resp(json{{"status", "ok"}, {"knowledge_id", kid}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // GET /api/forgetting
    CROW_ROUTE(app, "/api/forgetting").methods("GET"_method)
    ([&]() -> crow::response {
        try {
            auto alerts = learner.detect_forgetting();
            json arr = json::array();
            for (const auto& a : alerts) arr.push_back(dto::to_json_forgetting_alert(a));
            return ok_resp(json{{"alerts", arr}, {"count", static_cast<int>(alerts.size())}});
        } catch (const std::exception& e) { return err_resp(500, e.what()); }
    });

    // POST /api/abstract
    CROW_ROUTE(app, "/api/abstract").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("instance_id") || !body.contains("attributes"))
                return err_resp(400, "missing required fields: instance_id, attributes");
            std::map<std::string, double> features;
            if (body.contains("features") && body["features"].is_object())
                features = body["features"].get<std::map<std::string, double>>();
            std::vector<std::string> relations;
            if (body.contains("relations") && body["relations"].is_array())
                relations = body["relations"].get<std::vector<std::string>>();
            return ok_resp(dto::to_json_concept_formation_report(
                learner.form_abstractions(
                    body["instance_id"].get<std::string>(),
                    body["attributes"].get<std::vector<std::string>>(), features, relations)));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // ── Phase 4 ──────────────────────────────────────────────────────

    // POST /api/observe-behavior
    CROW_ROUTE(app, "/api/observe-behavior").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("agent_id") || !body.contains("action"))
                return err_resp(400, "missing required fields: agent_id, action");
            return ok_resp(dto::to_json_social_learning_report(
                learner.observe_behavior(dto::parse_behavior_observation(body))));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/emotion
    CROW_ROUTE(app, "/api/emotion").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("event_type")) return err_resp(400, "missing 'event_type' field");
            return ok_resp(dto::to_json_emotion_state(
                learner.process_emotion(dto::parse_emotion_event(body))));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/insight
    CROW_ROUTE(app, "/api/insight").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("problem_context")) return err_resp(400, "missing 'problem_context' field");
            auto opt = learner.try_insight(body["problem_context"].get<std::string>());
            json resp;
            if (opt.has_value()) {
                resp["insight"] = dto::to_json_insight_event(opt.value());
                resp["found"] = true;
            } else { resp["found"] = false; resp["message"] = "no insight emerged"; }
            return ok_resp(resp);
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // ── Phase 5 ──────────────────────────────────────────────────────

    // POST /api/meta/recommend
    CROW_ROUTE(app, "/api/meta/recommend").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("domain") || !body.contains("task_type"))
                return err_resp(400, "missing required fields: domain, task_type");
            return ok_resp(dto::to_json_meta_recommendation(
                learner.meta_recommend(dto::parse_task_descriptor(body))));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/meta/reflect
    CROW_ROUTE(app, "/api/meta/reflect").methods("POST"_method)
    ([&](const crow::request&) -> crow::response {
        try {
            auto reflections = learner.meta_reflect();
            return ok_resp(json{{"reflections", reflections}, {"count", static_cast<int>(reflections.size())}});
        } catch (const std::exception& e) { return err_resp(500, e.what()); }
    });

    // POST /api/experiment/design
    CROW_ROUTE(app, "/api/experiment/design").methods("POST"_method)
    ([&](const crow::request&) -> crow::response {
        try {
            auto opt = learner.design_experiment();
            json resp;
            if (opt.has_value()) {
                resp["experiment"] = dto::to_json_experiment_design(opt.value());
                resp["found"] = true;
            } else { resp["found"] = false; resp["message"] = "no hypothesis suitable for experiment"; }
            return ok_resp(resp);
        } catch (const std::exception& e) { return err_resp(500, e.what()); }
    });

    // POST /api/experiment/record
    CROW_ROUTE(app, "/api/experiment/record").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("experiment_id") || !body.contains("hypothesis_id"))
                return err_resp(400, "missing required fields: experiment_id, hypothesis_id");
            return ok_resp(json{{"status", "ok"}, {"result_id", learner.record_experiment(dto::parse_experiment_result(body))}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // ── Phase 6 ──────────────────────────────────────────────────────

    // POST /api/integrated/pipeline
    CROW_ROUTE(app, "/api/integrated/pipeline").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("observation") || !body.contains("domain"))
                return err_resp(400, "missing required fields: observation, domain");
            return ok_resp(dto::to_json_integrated_pipeline(
                learner.integrated_pipeline(body["observation"].get<std::string>(), body["domain"].get<std::string>())));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/integrated/meta-guided
    CROW_ROUTE(app, "/api/integrated/meta-guided").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            std::vector<std::string> topics;
            if (body.contains("known_topics") && body["known_topics"].is_array())
                topics = body["known_topics"].get<std::vector<std::string>>();
            std::map<std::string, double> mastery;
            if (body.contains("mastery_map") && body["mastery_map"].is_object())
                mastery = body["mastery_map"].get<std::map<std::string, double>>();
            return ok_resp(dto::to_json_meta_guided_session(learner.meta_guided_learn(topics, mastery)));
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // GET /api/integrated/emotion-params
    CROW_ROUTE(app, "/api/integrated/emotion-params").methods("GET"_method)
    ([&]() -> crow::response {
        try { return ok_resp(dto::to_json_emotion_modulated_params(learner.emotion_modulated_params())); }
        catch (const std::exception& e) { return err_resp(500, e.what()); }
    });
}
