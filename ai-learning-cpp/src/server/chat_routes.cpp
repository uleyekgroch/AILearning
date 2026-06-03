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

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <string>

using json = nlohmann::json;
namespace dto = ai_learning::server::dto;

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
                            local_path);
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
}
