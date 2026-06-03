/**
 * @file core_routes.cpp
 * @brief 系统 + 核心 API 路由（Phase 7.1-7.2）
 *
 * 端点：health, stats, stage, tasks, learn/text, observe, reason, think,
 *       perceive, remember, recall, consolidate, autonomous, save, load
 */

#include "route_groups.hpp"
#include "dto.hpp"
#include "ai_learning/server/metrics_collector.hpp"
#include "ai_learning/perception/image_encoder.hpp"
#include "ai_learning/perception/imodal_encoder.hpp"
#include "ai_learning/perception/onnx_clip_encoder.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread>

// ── 工具函数：base64 解码 ──────────────────────────────────────────
static auto decode_base64_(const std::string& b64_input) -> std::vector<uint8_t> {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> out;
    std::string b64 = b64_input;
    auto pos = b64.find(',');
    if (pos != std::string::npos) b64 = b64.substr(pos + 1);

    int val = 0, valb = -8;
    for (uint8_t c : b64) {
        if (c == '=') break;
        auto p = base64_chars.find(c);
        if (p == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(p);
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

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

    // GET /api/health — Kubernetes-style 综合健康检查
    CROW_ROUTE(app, "/api/health").methods("GET"_method)
    ([&]() -> crow::response {
        json body;
        body["status"] = "ok";
        body["version"] = config.version;
        body["stage"] = learner.stage();
        body["engine_type"] = learner.engine_type();
        body["avg_inference_steps"] = learner.engine().get_avg_inference_steps();
        auto stats = learner.get_stats();
        body["total_steps"] = stats.count("total_steps")
            ? static_cast<int>(stats.at("total_steps")) : 0;
        // J.2: 详细系统状态
        body["uptime_seconds"] = static_cast<int>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() % 86400);
        body["api_version"] = "v1";
        return ok_resp(body);
    });

    // GET /api/health/live — 存活探针（进程是否运行）
    CROW_ROUTE(app, "/api/health/live").methods("GET"_method)
    ([]() -> crow::response {
        return ok_resp(json{{"status", "alive"}, {"checks", json::array()}});
    });

    // GET /api/health/ready — 就绪探针（是否可以接受请求）
    CROW_ROUTE(app, "/api/health/ready").methods("GET"_method)
    ([&]() -> crow::response {
        json checks = json::array();
        bool ready = true;

        // 检查引擎是否就绪
        try {
            (void)learner.engine_type();  // 轻量检查
            json engine_check;
            engine_check["name"] = "engine";
            engine_check["status"] = "pass";
            checks.push_back(engine_check);
        } catch (...) {
            json engine_check;
            engine_check["name"] = "engine";
            engine_check["status"] = "fail";
            checks.push_back(engine_check);
            ready = false;
        }

        // 检查知识图谱（非空）
        try {
            auto stats = learner.get_stats();
            auto it = stats.find("entities");
            bool has_knowledge = (it != stats.end() && it->second > 0.0);
            json kg_check;
            kg_check["name"] = "knowledge_graph";
            kg_check["status"] = has_knowledge ? "pass" : "warn";
            kg_check["entities"] = (it != stats.end()) ? it->second : 0.0;
            checks.push_back(kg_check);
        } catch (...) {
            json kg_check;
            kg_check["name"] = "knowledge_graph";
            kg_check["status"] = "warn";
            checks.push_back(kg_check);
        }

        json body;
        body["status"] = ready ? "ready" : "not_ready";
        body["ready"] = ready;
        body["checks"] = checks;
        return json_resp(ready ? 200 : 503, body);
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
        ScopedTimer timer(state.metrics, "learn_text");
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
        ScopedTimer timer(state.metrics, "reason");
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
        ScopedTimer timer(state.metrics, "think");
        try {
            auto body = json::parse(req.body);
            if (!body.contains("question")) return err_resp(400, "missing 'question' field");
            return ok_resp(json{{"answer", learner.think(body["question"].get<std::string>())}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/perceive
    CROW_ROUTE(app, "/api/perceive").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        ScopedTimer timer(state.metrics, "perceive");
        try {
            auto body = json::parse(req.body);
            if (!body.contains("raw_input")) return err_resp(400, "missing 'raw_input' field");
            return ok_resp(json{{"perception", learner.perceive(dto::parse_raw_input(body["raw_input"]))}});
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/remember
    CROW_ROUTE(app, "/api/remember").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        ScopedTimer timer(state.metrics, "remember");
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
        ScopedTimer timer(state.metrics, "recall");
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
        ScopedTimer timer(state.metrics, "consolidate");
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

    // ── K.1+K.1++: 多模态感知（图像 + 音频）──────────────────────────

    // 辅助：构建多模态编码器（图像优先 ONNX CLIP，否则 Stub）
    auto build_multimodal_encoder = []() -> perception::MultiModalEncoder {
        perception::MultiModalEncoder mme;

        // 注册图像编码器
#ifdef AI_LEARNING_WITH_ONNX
        std::string clip_model = "models/clip-vit-base-patch32.onnx";
        if (std::filesystem::exists(clip_model)) {
            mme.register_image(std::make_shared<perception::OnnxClipImageEncoder>(
                clip_model, 512));
        } else
#endif
        {
            mme.register_image(std::make_shared<perception::StubImageEncoder>(128));
        }

        // 注册音频编码器（当前仅 Stub，未来可接入 Whisper / Wav2Vec2 ONNX）
        mme.register_audio(std::make_shared<perception::StubAudioEncoder>(128));

        return mme;
    };

    // POST /api/perceive/image — 接收 base64 图像，编码为嵌入，注入知识图谱
    CROW_ROUTE(app, "/api/perceive/image").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        ScopedTimer timer(state.metrics, "perceive_image");
        try {
            auto body = json::parse(req.body);
            if (!body.contains("image_base64")) {
                return err_resp(400, "missing 'image_base64' field");
            }

            auto image_data = decode_base64_(body["image_base64"].get<std::string>());
            if (image_data.empty()) {
                return err_resp(400, "invalid base64 image data");
            }

            auto mme = build_multimodal_encoder();
            std::map<std::string, int> params;
            params["width"] = body.value("width", 0);
            params["height"] = body.value("height", 0);
            auto result = mme.encode(image_data, perception::ModalityType::Image, params);

            if (!result.success) {
                return err_resp(500, result.error);
            }

            // 将图像嵌入注入感知系统（视为 "visual" 模态）
            std::map<std::string, std::vector<float>> raw_input;
            raw_input["visual"] = result.embedding;
            auto perception = learner.perceive(raw_input);

            json resp;
            resp["status"] = "ok";
            resp["encoder"] = mme.name();
            resp["modality"] = perception::modality_name(result.modality);
            resp["embedding_dim"] = result.embedding.size();
            resp["image_size"] = image_data.size();
            resp["perception_dim"] = perception.size();
            return ok_resp(resp);
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // POST /api/perceive/audio — 接收 base64 音频，编码为嵌入，注入知识图谱
    CROW_ROUTE(app, "/api/perceive/audio").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        ScopedTimer timer(state.metrics, "perceive_audio");
        try {
            auto body = json::parse(req.body);
            if (!body.contains("audio_base64")) {
                return err_resp(400, "missing 'audio_base64' field");
            }

            auto audio_data = decode_base64_(body["audio_base64"].get<std::string>());
            if (audio_data.empty()) {
                return err_resp(400, "invalid base64 audio data");
            }

            auto mme = build_multimodal_encoder();
            std::map<std::string, int> params;
            params["sample_rate"] = body.value("sample_rate", 16000);
            params["channels"] = body.value("channels", 1);
            auto result = mme.encode(audio_data, perception::ModalityType::Audio, params);

            if (!result.success) {
                return err_resp(500, result.error);
            }

            // 将音频嵌入注入感知系统（视为 "auditory" 模态）
            std::map<std::string, std::vector<float>> raw_input;
            raw_input["auditory"] = result.embedding;
            auto perception = learner.perceive(raw_input);

            json resp;
            resp["status"] = "ok";
            resp["encoder"] = mme.name();
            resp["modality"] = perception::modality_name(result.modality);
            resp["embedding_dim"] = result.embedding.size();
            resp["audio_size"] = audio_data.size();
            resp["sample_rate"] = result.meta.count("sample_rate") ?
                std::stoi(result.meta.at("sample_rate")) : 0;
            resp["channels"] = result.meta.count("channels") ?
                std::stoi(result.meta.at("channels")) : 0;
            resp["perception_dim"] = perception.size();
            return ok_resp(resp);
        } catch (const std::exception& e) { return err_resp(400, e.what()); }
    });

    // ═══════════════════════════════════════════════════════════
    // ★v2: 仿人类学习增强端点
    // ═══════════════════════════════════════════════════════════

    // POST /api/self/reflect — 自我反思
    CROW_ROUTE(app, "/api/self/reflect").methods("POST"_method)
    ([&]() -> crow::response {
        auto reflection = learner.self_model().reflect_on_self();
        json resp;
        resp["status"] = "ok";
        resp["reflection"] = reflection;
        resp["self_efficacy"] = learner.self_model().self_efficacy();
        resp["self_continuity"] = learner.self_model().self_continuity();
        return ok_resp(resp);
    });

    // GET /api/self/whoami — 自我认知
    CROW_ROUTE(app, "/api/self/whoami").methods("GET"_method)
    ([&]() -> crow::response {
        json resp;
        resp["status"] = "ok";
        resp["identity"] = learner.self_model().who_am_i_now();
        resp["narrative"] = learner.self_model().life_narrative();
        resp["self_beliefs"] = learner.self_model().self_concept().size();
        resp["memories"] = learner.self_model().stats().autobiographical_memories;
        return ok_resp(resp);
    });

    // POST /api/creativity/diverge — 发散思维
    CROW_ROUTE(app, "/api/creativity/diverge").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        auto body = json::parse(req.body);
        std::string problem = body.value("problem", "如何学习");
        auto result = learner.creative_engine().brainstorm(problem, 5);
        json resp;
        resp["status"] = "ok";
        resp["problem"] = problem;
        resp["ideas"] = result;
        return ok_resp(resp);
    });

    // POST /api/tutor/teach — 教学相长
    CROW_ROUTE(app, "/api/tutor/teach").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        auto body = json::parse(req.body);
        social::KnowledgeUnit topic;
        topic.topic = body.value("topic", "math");
        topic.explanation = body.value("explanation", "basic concepts");
        topic.mastery = body.value("mastery", 0.5);

        social::LearnerModel student;
        student.learner_id = body.value("student_id", "peer");
        student.knowledge[topic.topic] = body.value("student_mastery", 0.3);

        auto result = learner.tutoring_system().teach(topic, student, "self");
        json resp;
        resp["status"] = "ok";
        resp["tutor_gain"] = result.tutor_gain;
        resp["student_gain"] = result.knowledge_gain;
        resp["tutor_reflection"] = result.tutor_reflection;
        return ok_resp(resp);
    });

    // POST /api/mirror/observe — 镜像神经元观察
    CROW_ROUTE(app, "/api/mirror/observe").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        auto body = json::parse(req.body);
        learning::ObservedAction action;
        action.action_name = body.value("action", "observe");
        action.actor_id = body.value("actor", "other");
        action.confidence = body.value("confidence", 0.8);

        auto motor = learner.mirror_neurons().observe_action(action);
        json resp;
        resp["status"] = "ok";
        resp["motor_activation"] = motor.activation;
        resp["mastered"] = motor.mastered;
        resp["total_observations"] = learner.mirror_neurons().total_observations();
        return ok_resp(resp);
    });

    // POST /api/autonomous/run — 运行自主学习循环（v2: 集成主动推理）
    CROW_ROUTE(app, "/api/autonomous/run").methods("POST"_method)
    ([&](const crow::request& req) -> crow::response {
        auto body = json::parse(req.body);
        int iterations = body.value("iterations", 10);
        auto report = learner.autonomous_learning_run(iterations);
        json resp;
        resp["status"] = "ok";
        resp["iterations"] = report.total_iterations;
        resp["goals_attempted"] = report.goals_attempted;
        resp["goals_completed"] = report.goals_completed;
        resp["avg_motivation"] = report.avg_motivation;
        return ok_resp(resp);
    });
}
