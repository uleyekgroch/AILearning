/**
 * @file chat_routes.cpp
 * @brief 对话路由（Phase 9 语言接口）
 *
 * 包含端点：
 * - POST   /api/chat                  多轮对话
 * - GET    /api/chat/history           获取对话历史
 * - DELETE /api/chat/session/<id>      清除会话
 */

#include "route_groups.hpp"
#include "dto.hpp"
#include "ai_learning/distributed/inference_cluster.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <string>

using json = nlohmann::json;
namespace dto = ai_learning::server::dto;

namespace {
auto ok_resp(const json& data) -> crow::response {
    crow::response r;
    r.code = 200;
    r.set_header("Content-Type", "application/json");
    r.body = data.dump();
    return r;
}

auto err_resp(int code, const std::string& msg) -> crow::response {
    crow::response r;
    r.code = code;
    r.set_header("Content-Type", "application/json");
    json err;
    err["error"] = msg;
    r.body = err.dump();
    return r;
}
}

void ai_learning::server::register_chat_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    SharedState& state) {

    // POST /api/chat — 多轮对话
    CROW_ROUTE(app, "/api/chat").methods("POST"_method)
    ([&state, &learner](const crow::request& req) -> crow::response {
        try {
            // 懒初始化 LLM 提供者和对话管理器
            // 优先级：1) llama.cpp 本地模型 2) OpenAI 兼容 API 3) Stub
            if (!state.llm_provider) {
                const std::string& local_path = learner.config().llm_model_path;
                if (!local_path.empty()) {
#ifdef AI_LEARNING_WITH_LLAMA_CPP
                    try {
                        state.llm_provider = std::make_unique<language::LlamaCppLLMProvider>(
                            local_path,
                            learner.config().n_gpu_layers);
                    } catch (const std::exception& e) {
                        std::cerr << "[Warning] Failed to load local LLM: "
                                  << e.what() << ", falling back to API/Stub\n";
                    }
#endif
                }
                if (!state.llm_provider) {
                    const char* api_key = std::getenv("DASHSCOPE_API_KEY");
                    if (api_key && api_key[0] != '\0') {
                        state.llm_provider = std::make_unique<language::OpenAICompatibleProvider>(
                            "dashscope.aliyuncs.com", api_key, "qwen-plus-latest");
                    } else {
                        state.llm_provider = std::make_unique<language::StubLLMProvider>();
                    }
                }
            }
            if (!state.dialog) {
                state.dialog = std::make_unique<language::DialogManager>(
                    *state.llm_provider, learner);
            }

            auto body = json::parse(req.body);
            if (!body.contains("message")) {
                crow::response res{400,
                    dto::make_error_response("missing 'message' field").dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            std::string message = body["message"].get<std::string>();
            std::string session_id = body.value("session_id", "default");

            auto response = state.dialog->chat(message, session_id);
            json resp;
            resp["assistant_message"] = response.assistant_message;
            resp["knowledge_learned"] = response.knowledge_learned;
            resp["learner_action"] = response.learner_action;
            resp["intent"] = response.intent;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // GET /api/chat/history — 获取对话历史
    CROW_ROUTE(app, "/api/chat/history").methods("GET"_method)
    ([&state](const crow::request& req) -> crow::response {
        try {
            if (!state.dialog) {
                json resp;
                resp["history"] = json::array();
                resp["count"] = 0;

                crow::response res{resp.dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
            // 从 query string 获取 session 参数
            std::string session_id = "default";
            std::string url = req.raw_url;
            auto pos = url.find("?session=");
            if (pos != std::string::npos) {
                session_id = url.substr(pos + 9);
                auto amp = session_id.find('&');
                if (amp != std::string::npos) session_id = session_id.substr(0, amp);
            }

            int last_n = 20;
            auto history = state.dialog->get_history(session_id, last_n);
            json arr = json::array();
            for (const auto& turn : history) {
                json item;
                item["role"] = turn.role;
                item["content"] = turn.content;
                item["timestamp"] = turn.timestamp;
                arr.push_back(item);
            }

            json resp;
            resp["history"] = arr;
            resp["count"] = static_cast<int>(history.size());
            resp["session_id"] = session_id;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{500, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // DELETE /api/chat/session/<id> — 清除会话
    CROW_ROUTE(app, "/api/chat/session/<string>").methods("DELETE"_method)
    ([&state](const std::string& session_id) -> crow::response {
        try {
            if (state.dialog) {
                state.dialog->clear_session(session_id);
            }
            json resp;
            resp["status"] = "ok";
            resp["session_id"] = session_id;

            crow::response res{resp.dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        } catch (const std::exception& e) {
            crow::response res{400, dto::make_error_response(e.what()).dump()};
            res.set_header("Content-Type", "application/json");
            return res;
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // J.1: 推理性能优化端点
    // ═══════════════════════════════════════════════════════════════

    // POST /api/inference/kv-cache/clear — 清除 KV cache
    CROW_ROUTE(app, "/api/inference/kv-cache/clear").methods("POST"_method)
    ([&state]() -> crow::response {
        try {
#ifdef AI_LEARNING_WITH_LLAMA_CPP
            auto* llama = dynamic_cast<language::LlamaCppLLMProvider*>(
                state.llm_provider.get());
            if (llama) {
                llama->clear_kv_cache();
                json resp;
                resp["status"] = "ok";
                resp["action"] = "kv_cache_cleared";
                crow::response res{resp.dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
#endif
            return err_resp(400, "llama.cpp provider not active");
        } catch (const std::exception& e) {
            return err_resp(500, e.what());
        }
    });

    // GET /api/inference/kv-cache/stats — KV cache 统计
    CROW_ROUTE(app, "/api/inference/kv-cache/stats").methods("GET"_method)
    ([&state]() -> crow::response {
        try {
#ifdef AI_LEARNING_WITH_LLAMA_CPP
            auto* llama = dynamic_cast<language::LlamaCppLLMProvider*>(
                state.llm_provider.get());
            if (llama) {
                json resp;
                resp["status"] = "ok";
                resp["kv_cache_tokens"] = llama->kv_cache_token_count();
                resp["speculative_enabled"] = llama->speculative_decoding_enabled();
                crow::response res{resp.dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
#endif
            return err_resp(400, "llama.cpp provider not active");
        } catch (const std::exception& e) {
            return err_resp(500, e.what());
        }
    });

    // POST /api/inference/batch — 批量推理
    CROW_ROUTE(app, "/api/inference/batch").methods("POST"_method)
    ([&state](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("prompts") || !body["prompts"].is_array()) {
                return err_resp(400, "missing 'prompts' array");
            }
            std::vector<std::string> prompts;
            for (const auto& p : body["prompts"]) {
                prompts.push_back(p.get<std::string>());
            }

            std::vector<std::string> systems;
            if (body.contains("system_prompts") && body["system_prompts"].is_array()) {
                for (const auto& s : body["system_prompts"]) {
                    systems.push_back(s.get<std::string>());
                }
            }

#ifdef AI_LEARNING_WITH_LLAMA_CPP
            auto* llama = dynamic_cast<language::LlamaCppLLMProvider*>(
                state.llm_provider.get());
            if (llama) {
                auto results = llama->complete_batch(prompts, systems);
                json resp;
                resp["status"] = "ok";
                resp["count"] = results.size();
                json arr = json::array();
                for (const auto& r : results) {
                    json item;
                    item["text"] = r;
                    arr.push_back(item);
                }
                resp["results"] = arr;
                crow::response res{resp.dump()};
                res.set_header("Content-Type", "application/json");
                return res;
            }
#endif
            return err_resp(400, "llama.cpp provider not active");
        } catch (const std::exception& e) {
            return err_resp(500, e.what());
        }
    });

    // ═══════════════════════════════════════════════════════════════
    // K.3: 模型量化信息
    // ═══════════════════════════════════════════════════════════════

    // GET /api/model/quantize/info — 量化类型参考
    CROW_ROUTE(app, "/api/model/quantize/info").methods("GET"_method)
    ([]() -> crow::response {
        json resp;
        resp["status"] = "ok";
        json types = json::array();
        struct QType { std::string type, desc, ratio, scene; };
        std::vector<QType> qtypes = {
            {"q8_0", "INT8 量化", "~50%", "精度优先"},
            {"q6_k", "Q6_K 混合量化", "~38%", "高精度 6-bit"},
            {"q5_k_m", "Q5_K_M 混合量化", "~31%", "平衡精度/速度"},
            {"q4_k_m", "Q4_K_M 混合量化 (推荐)", "~25%", "最佳平衡点"},
            {"q4_k_s", "Q4_K_S 更小更快", "~25%", "边缘设备"},
            {"iq4_nl", "IQ4_NL 高质量 4-bit", "~25%", "新方案质量高"},
            {"q3_k_m", "Q3_K_M 混合量化", "~19%", "高压缩可接受"},
            {"q2_k", "Q2_K 极限压缩", "~13%", "仅测试/边缘"},
        };
        for (const auto& q : qtypes) {
            json item;
            item["type"] = q.type;
            item["description"] = q.desc;
            item["ratio"] = q.ratio;
            item["scene"] = q.scene;
            types.push_back(item);
        }
        resp["quantization_types"] = types;
        resp["note"] = "Use scripts/quantize_model.sh to quantize models";
        crow::response res{resp.dump()};
        res.set_header("Content-Type", "application/json");
        return res;
    });

    // ═══════════════════════════════════════════════════════════════
    // L.1: 分布式推理集群
    // ═══════════════════════════════════════════════════════════════

    static distributed::InferenceCluster cluster;

    // POST /api/cluster/nodes — 注册节点
    CROW_ROUTE(app, "/api/cluster/nodes").methods("POST"_method)
    ([&state](const crow::request& req) -> crow::response {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("id") || !body.contains("endpoint")) {
                return err_resp(400, "missing 'id' or 'endpoint'");
            }
            cluster.register_node(
                body["id"].get<std::string>(),
                body["endpoint"].get<std::string>(),
                body.value("gpu_info", ""),
                body.value("max_concurrent", 1));
            json resp;
            resp["status"] = "ok";
            resp["node_count"] = cluster.node_count();
            return ok_resp(resp);
        } catch (const std::exception& e) {
            return err_resp(500, e.what());
        }
    });

    // GET /api/cluster/nodes — 列出节点
    CROW_ROUTE(app, "/api/cluster/nodes").methods("GET"_method)
    ([&state]() -> crow::response {
        json resp;
        resp["status"] = "ok";
        json arr = json::array();
        for (const auto& n : cluster.nodes()) {
            json item;
            item["id"] = n.id;
            item["endpoint"] = n.endpoint;
            item["gpu_info"] = n.gpu_info;
            item["healthy"] = n.healthy.load();
            item["active_requests"] = n.active_requests.load();
            arr.push_back(item);
        }
        resp["nodes"] = arr;
        resp["total"] = cluster.node_count();
        resp["healthy"] = cluster.healthy_node_count();
        return ok_resp(resp);
    });

    // POST /api/cluster/nodes/<id>/heartbeat — 节点心跳
    CROW_ROUTE(app, "/api/cluster/nodes/<string>/heartbeat").methods("POST"_method)
    ([&state](const std::string& id) -> crow::response {
        cluster.heartbeat(id);
        json resp;
        resp["status"] = "ok";
        return ok_resp(resp);
    });

    // DELETE /api/cluster/nodes/<id> — 注销节点
    CROW_ROUTE(app, "/api/cluster/nodes/<string>").methods("DELETE"_method)
    ([&state](const std::string& id) -> crow::response {
        cluster.unregister_node(id);
        json resp;
        resp["status"] = "ok";
        return ok_resp(resp);
    });

    // GET /api/cluster/stats — 集群统计
    CROW_ROUTE(app, "/api/cluster/stats").methods("GET"_method)
    ([&state]() -> crow::response {
        auto stats = cluster.cluster_stats();
        json resp;
        resp["status"] = "ok";
        for (const auto& [k, v] : stats) {
            resp[k] = v;
        }
        return ok_resp(resp);
    });
}
