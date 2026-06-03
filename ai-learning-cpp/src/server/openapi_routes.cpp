/**
 * @file openapi_routes.cpp
 * @brief OpenAPI 3.0 JSON 端点 — 自动生成 API 文档
 *
 * GET /api/openapi.json — 返回完整的 OpenAPI 3.0.3 规范
 */

#include "route_groups.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static auto make_openapi_spec_(const ai_learning::server::ServerConfig& config) -> json {
    json spec;
    spec["openapi"] = "3.0.3";

    // ── Info ──────────────────────────────────────────────────────
    spec["info"] = {
        {"title", "AILearning REST API"},
        {"version", config.version},
        {"description",
         "从学习本源出发的人工智能系统 REST API。\n"
         "涵盖学习、推理、记忆、对话、目标管理、社会学习等 47+ 端点。"},
        {"contact", {{"name", "AI Learning Project"}}}
    };

    // ── Servers ─────────────────────────────────────────────────
    spec["servers"] = json::array({
        {{"url", "/"}, {"description", "当前服务器"}}
    });

    // ── Tags ────────────────────────────────────────────────────
    spec["tags"] = json::array({
        {{"name", "Health"}, {"description", "健康检查"}},
        {{"name", "System"}, {"description", "系统统计与状态"}},
        {{"name", "Learning"}, {"description", "学习与观察"}},
        {{"name", "Reasoning"}, {"description", "推理与思考"}},
        {{"name", "Memory"}, {"description", "记忆系统"}},
        {{"name", "Dialog"}, {"description", "对话与聊天"}},
        {{"name", "Goals"}, {"description", "目标管理"}},
        {{"name", "Society"}, {"description", "多 Agent 社会"}},
        {{"name", "Runtime"}, {"description", "运行时控制"}},
        {{"name", "Advanced"}, {"description", "高级认知功能"}}
    });

    // ── Paths ───────────────────────────────────────────────────
    json paths;

    // Health
    paths["/api/health"] = {
        {"get", {
            {"tags", json::array({"Health"})},
            {"summary", "综合健康检查"},
            {"responses", {
                {"200", {
                    {"description", "服务健康"},
                    {"content", {{"application/json", {
                        {"schema", {{"$ref", "#/components/schemas/HealthResponse"}}}
                    }}}}
                }}
            }}
        }}
    };
    paths["/api/health/live"] = {
        {"get", {
            {"tags", json::array({"Health"})},
            {"summary", "存活探针"},
            {"description", "Kubernetes liveness probe"},
            {"responses", {{"200", {{"description", "进程存活"}}}}}
        }}
    };
    paths["/api/health/ready"] = {
        {"get", {
            {"tags", json::array({"Health"})},
            {"summary", "就绪探针"},
            {"description", "Kubernetes readiness probe"},
            {"responses", {
                {"200", {{"description", "服务就绪"}}},
                {"503", {{"description", "服务未就绪"}}}
            }}
        }}
    };

    // System
    paths["/api/stats"] = {
        {"get", {
            {"tags", json::array({"System"})},
            {"summary", "获取系统统计"},
            {"responses", {{"200", {{"description", "统计信息"}}}}}
        }}
    };
    paths["/api/stage"] = {
        {"get", {
            {"tags", json::array({"System"})},
            {"summary", "获取当前发展阶段"},
            {"responses", {{"200", {{"description", "阶段信息"}}}}}
        }}
    };

    // Learning
    paths["/api/learn/text"] = {
        {"post", {
            {"tags", json::array({"Learning"})},
            {"summary", "从文本学习"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"$ref", "#/components/schemas/LearnTextRequest"}}}
                }}}}
            }},
            {"responses", {
                {"200", {{"description", "学习结果"}}},
                {"400", {{"description", "请求格式错误"}}}
            }}
        }}
    };
    paths["/api/observe"] = {
        {"post", {
            {"tags", json::array({"Learning"})},
            {"summary", "观察文本"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"text", {{"type", "string"}}}
                    }}, {"required", json::array({"text"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "观察结果"}}}}}
        }}
    };

    // Reasoning
    paths["/api/reason"] = {
        {"post", {
            {"tags", json::array({"Reasoning"})},
            {"summary", "推理问答"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"query", {{"type", "string"}}},
                        {"mode", {{"type", "string"}, {"enum", json::array({
                            "direct", "causal", "inductive", "analogical",
                            "counterfactual", "probabilistic"
                        })}}}
                    }}, {"required", json::array({"query"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "推理结果"}}}}}
        }}
    };
    paths["/api/think"] = {
        {"post", {
            {"tags", json::array({"Reasoning"})},
            {"summary", "思考（元认知）"},
            {"responses", {{"200", {{"description", "思考结果"}}}}}
        }}
    };

    // Memory
    paths["/api/remember"] = {
        {"post", {
            {"tags", json::array({"Memory"})},
            {"summary", "记忆"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"key", {{"type", "string"}}},
                        {"value", {{"type", "string"}}}
                    }}, {"required", json::array({"key", "value"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "记忆结果"}}}}}
        }}
    };
    paths["/api/recall"] = {
        {"post", {
            {"tags", json::array({"Memory"})},
            {"summary", "回忆"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"query", {{"type", "string"}}}
                    }}, {"required", json::array({"query"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "回忆结果"}}}}}
        }}
    };
    paths["/api/consolidate"] = {
        {"post", {
            {"tags", json::array({"Memory"})},
            {"summary", "睡眠记忆巩固"},
            {"responses", {{"200", {{"description", "巩固结果"}}}}}
        }}
    };

    // Dialog
    paths["/api/chat"] = {
        {"post", {
            {"tags", json::array({"Dialog"})},
            {"summary", "对话"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"message", {{"type", "string"}}},
                        {"session_id", {{"type", "string"}}}
                    }}, {"required", json::array({"message"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "回复"}}}}}
        }}
    };

    // Goals
    paths["/api/goals"] = {
        {"get", {
            {"tags", json::array({"Goals"})},
            {"summary", "列出所有目标"},
            {"responses", {{"200", {{"description", "目标列表"}}}}}
        }},
        {"post", {
            {"tags", json::array({"Goals"})},
            {"summary", "创建目标"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"description", {{"type", "string"}}},
                        {"priority", {{"type", "string"}, {"enum", json::array({"low", "medium", "high", "critical"})}}}
                    }}, {"required", json::array({"description"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "创建结果"}}}}}
        }}
    };

    // Persistence
    paths["/api/save"] = {
        {"post", {
            {"tags", json::array({"System"})},
            {"summary", "保存状态到文件"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"path", {{"type", "string"}}}
                    }}, {"required", json::array({"path"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "保存成功"}}}}}
        }}
    };
    paths["/api/load"] = {
        {"post", {
            {"tags", json::array({"System"})},
            {"summary", "从文件加载状态"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"path", {{"type", "string"}}}
                    }}, {"required", json::array({"path"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "加载成功"}}}}}
        }}
    };

    // Runtime
    paths["/api/runtime/status"] = {
        {"get", {
            {"tags", json::array({"Runtime"})},
            {"summary", "获取运行时状态"},
            {"responses", {{"200", {{"description", "状态信息"}}}}}
        }}
    };
    paths["/api/runtime/start"] = {
        {"post", {
            {"tags", json::array({"Runtime"})},
            {"summary", "启动持续学习循环"},
            {"responses", {{"200", {{"description", "启动结果"}}}}}
        }}
    };
    paths["/api/runtime/stop"] = {
        {"post", {
            {"tags", json::array({"Runtime"})},
            {"summary", "停止持续学习循环"},
            {"responses", {{"200", {{"description", "停止结果"}}}}}
        }}
    };

    // Society
    paths["/api/society/agents"] = {
        {"get", {
            {"tags", json::array({"Society"})},
            {"summary", "列出所有 Agent"},
            {"responses", {{"200", {{"description", "Agent 列表"}}}}}
        }},
        {"post", {
            {"tags", json::array({"Society"})},
            {"summary", "创建 Agent"},
            {"requestBody", {
                {"required", true},
                {"content", {{"application/json", {
                    {"schema", {{"type", "object"}, {"properties", {
                        {"name", {{"type", "string"}}},
                        {"role", {{"type", "string"}}}
                    }}, {"required", json::array({"name"})}}}
                }}}}
            }},
            {"responses", {{"200", {{"description", "创建结果"}}}}}
        }}
    };

    // WebSocket
    paths["/ws/events"] = {
        {"get", {
            {"tags", json::array({"System"})},
            {"summary", "WebSocket — 实时事件流"},
            {"description", "连接后接收所有学习事件的实时推送。支持 ping/pong 心跳。"},
            {"responses", {{"101", {{"description", "Switching Protocols"}}}}}
        }}
    };
    paths["/ws/stats"] = {
        {"get", {
            {"tags", json::array({"System"})},
            {"summary", "WebSocket — 统计推送"},
            {"description", "定期推送系统统计摘要。"},
            {"responses", {{"101", {{"description", "Switching Protocols"}}}}}
        }}
    };

    spec["paths"] = paths;

    // ── Components ──────────────────────────────────────────────
    spec["components"] = {
        {"schemas", {
            {"HealthResponse", {
                {"type", "object"},
                {"properties", {
                    {"status", {{"type", "string"}, {"example", "ok"}}},
                    {"version", {{"type", "string"}}},
                    {"stage", {{"type", "string"}}},
                    {"engine_type", {{"type", "string"}}},
                    {"total_steps", {{"type", "integer"}}},
                    {"uptime_seconds", {{"type", "integer"}}},
                    {"api_version", {{"type", "string"}, {"example", "v1"}}}
                }}
            }},
            {"LearnTextRequest", {
                {"type", "object"},
                {"properties", {
                    {"text", {{"type", "string"}, {"description", "要学习的文本"}}},
                    {"source", {{"type", "string"}, {"description", "来源标识"}, {"default", "text"}}}
                }},
                {"required", json::array({"text"})}
            }},
            {"ErrorResponse", {
                {"type", "object"},
                {"properties", {
                    {"error", {{"type", "string"}}}
                }}
            }}
        }}
    };

    return spec;
}

void ai_learning::server::register_openapi_routes(
    crow::SimpleApp& app,
    const ServerConfig& config) {

    CROW_ROUTE(app, "/api/openapi.json").methods("GET"_method)
    ([&config]() -> crow::response {
        auto spec = make_openapi_spec_(config);
        crow::response res{200, spec.dump(2)};
        res.set_header("Content-Type", "application/json");
        res.set_header("Access-Control-Allow-Origin", "*");
        return res;
    });

    // GET /api/docs — 重定向到 Swagger UI（如果有）
    CROW_ROUTE(app, "/api/docs").methods("GET"_method)
    ([]() -> crow::response {
        crow::response res{302};
        res.set_header("Location",
            "https://petstore.swagger.io/?url=/api/openapi.json");
        return res;
    });
}
