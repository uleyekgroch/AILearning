# Research: C++ REST API Frameworks for ML/Serving

- **Query**: Compare C++ REST frameworks for serving the AILearning cognitive system
- **Scope**: External (framework comparison and evaluation)
- **Date**: 2026-06-02

## Findings

### Project Context

The AILearning C++ system needs an application-layer REST API to expose its cognitive learning capabilities. Key constraints from the codebase:

- **C++20** with MSVC + GCC/Clang support
- **Windows 11** as primary development platform (also WSL and CUDA builds)
- **CMake 3.22+** build system
- Static library output (`ai_learning`) with optional CUDA
- The system is compute-intensive (predictive coding, knowledge graph operations, autonomous learning loops)
- API needs to expose ~40+ Learner methods plus subsystem access
- Async request handling is important for long-running operations (autonomous_learn, integrated_pipeline)

### Framework Comparison

| Feature | crow | cpp-httplib | Pistache | oat++ | userver |
|---------|------|-------------|----------|-------|---------|
| **License** | BSD-3 | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **C++ Standard** | C++14+ | C++14+ | C++17+ | C++17+ | C++17+ (C++20 preferred) |
| **Header-only** | No (CMake) | Yes (single header) | No (CMake) | No (CMake) | No (CMake) |
| **Design** | Flask-like (Sinatra) | Minimal blocking | Async futures | Full MVC framework | Full async framework |
| **HTTP/2** | No | No | No | No | Yes |
| **WebSocket** | No (separate lib) | No | No | Yes | Yes |
| **Async model** | Thread-per-connection (with thread pool) | Blocking with threads | True async (futures) | Async with coroutine support | Coroutine-based (Boost.Asio) |
| **JSON built-in** | No (recommends nlohmann/json) | No | No | Yes (oatpp::String) | Yes (userver::formats::json) |
| **SSE/WebSocket** | No built-in | No | No | WebSocket support | WebSocket + SSE |
| **Windows support** | Partial (MSVC issues reported) | Yes | Poor (Linux-focused) | Yes (experimental) | Yes |
| **Performance** | Good (~100K req/s) | Moderate (~50K req/s) | Good (~80K req/s) | Good (~100K req/s) | Excellent (~200K+ req/s) |
| **Binary size** | Small | Tiny (~400KB header) | Medium | Medium | Large (Boost dependency) |
| **Production readiness** | Mature | Mature (used in production) | Mature | Mature | Mature (Yandex production) |
| **C++20 compat** | Yes | Yes | Yes | Yes | Yes (designed for C++20) |
| **Last updated** | Active (2025) | Active (2025) | Low activity | Active (2025) | Active (2025) |
| **GitHub stars** | ~18K | ~13K | ~3K | ~8K | ~3K |

### Detailed Analysis

#### 1. crow (recommended for this project)

**Why it fits**: Flask-like API design makes it the most approachable for exposing the Learner's methods. Widely used, active development.

```cpp
#include <crow.h>
#include "ai_learning/core/learner.hpp"

int main() {
    crow::SimpleApp app;

    CROW_ROUTE(app, "/api/v1/learn/text").methods("POST"_method)
    ([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        LearnerConfig config;
        // ... configure from body
        Learner learner(config);
        auto result = learner.learn_from_text(body["text"].s(), body["source"].s());
        crow::json::wvalue response;
        response["success"] = result.success;
        response["confidence"] = result.confidence;
        return crow::response(response);
    });

    app.port(8080).multithreaded(4).run();
}
```

**Pros**:
- Flask-like routing — intuitive for Python developers transitioning
- Built-in JSON support (crow::json) but also works with nlohmann/json
- Middleware support (CORS, logging, authentication)
- Static file serving (useful for dashboard)
- Good documentation and examples
- Compile-time route registration

**Cons**:
- No native WebSocket (would need separate library for real-time metrics)
- No native async/coroutine support — uses thread pool
- MSVC compatibility has historical issues (mostly resolved in recent versions)
- Template-heavy compilation can be slow

**Windows note**: crow works with MSVC but requires careful CMake setup. The `ASIO_STANDALONE` define must be set.

#### 2. cpp-httplib

**Why it fits**: Single-header, zero-dependency, easiest to integrate into existing CMake project.

```cpp
#include <httplib.h>
#include "ai_learning/core/learner.hpp"

int main() {
    httplib::Server svr;
    Learner learner(LearnerConfig{});

    svr.Post("/api/v1/learn/text", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse JSON body
        auto body = nlohmann::json::parse(req.body);
        auto result = learner.learn_from_text(body["text"], body["source"]);
        nlohmann::json response;
        response["success"] = result.success;
        res.set_content(response.dump(), "application/json");
    });

    svr.listen("0.0.0.0", 8080);
}
```

**Pros**:
- Single file inclusion — trivial to add to existing project
- Excellent Windows support
- SSL/TLS support via OpenSSL
- Built-in file upload, download, compression
- Very well tested, production-proven
- No external dependencies

**Cons**:
- Blocking I/O model — not ideal for long-running ML operations
- No async support
- No built-in WebSocket
- Limited middleware system
- Lower performance under high concurrency

#### 3. oat++ (alternative if full framework is desired)

**Why it fits**: Full MVC framework with built-in WebSocket and async support.

```cpp
#include OATPP_CODEGEN_BEGIN(DTO)  // oat++ DTO macros

class LearnRequestDto : public oatpp::DTO {
    DTO_INIT(LearnRequestDto, DTO)
    DTO_FIELD(String, text);
    DTO_FIELD(String, source) = "text";
};

class LearnResponseDto : public oatpp::DTO {
    DTO_INIT(LearnResponseDto, DTO)
    DTO_FIELD(Boolean, success);
    DTO_FIELD(Float64, confidence);
};

#include OATPP_CODEGEN_END(DTO)
```

**Pros**:
- Full MVC framework with DTO, Controller, Service layers
- Built-in WebSocket support
- Built-in ORM (not needed for this project but indicates completeness)
- Async support with coroutines
- Swagger UI auto-generation
- Clean API design

**Cons**:
- Heavier framework — more boilerplate
- Heavily macro-based DTO system
- Windows support is "experimental"
- More opinionated — less flexibility
- Build system integration more complex

#### 4. userver (for maximum performance)

**Why it fits**: Coroutine-based async framework designed for high-load services. Best fit if the system needs to handle many concurrent learning sessions.

**Pros**:
- Best performance (200K+ req/s)
- Coroutine-based (C++20 coroutines)
- Built-in metrics, tracing, logging
- WebSocket + HTTP/2
- Production-proven at Yandex scale
- Excellent async handling for long-running operations

**Cons**:
- **Large dependency footprint** (Boost.Asio, many Boost libs)
- Complex build setup
- Steep learning curve
- Overkill for a research/prototype system
- Documentation primarily in Russian (improving)

#### 5. Pistache

**Not recommended**: Linux-focused, poor Windows support, low recent activity.

### JSON Serialization

#### nlohmann/json (recommended)

The de facto standard for C++ JSON. Header-only or CMake-integrated.

```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Automatic serialization for structs via NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE
namespace ai_learning::core {
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LearnerConfig,
        obs_dim, action_dim, learning_rate, curiosity_alpha,
        // ... all fields
    )
}

// Or manual to_json/from_json for complex types
void to_json(json& j, const EmotionState& e) {
    j = json{{"valence", e.valence}, {"arousal", e.arousal},
             {"dominance", e.dominance}, {"label", e.label}};
}
void from_json(const json& j, EmotionState& e) {
    j.at("valence").get_to(e.valence);
    j.at("arousal").get_to(e.arousal);
    // ...
}
```

**Integration with frameworks**:
- crow: crow::json is separate; convert via string dump/parse, or use nlohmann directly
- cpp-httplib: Use nlohmann/json directly (set_content with dump())
- oat++: Has its own DTO system, can convert to/from nlohmann if needed

**For this project's 45+ result structs**, nlohmann/json with `to_json`/`from_json` free functions is the most maintainable approach. Each struct gets a pair of serialization functions.

### Async Request Handling Patterns

For long-running operations like `autonomous_learn()` (which can take thousands of steps), the API should:

1. **Start async, return task ID**:
```cpp
// POST /api/v1/learn/autonomous → returns task_id
// GET /api/v1/learn/autonomous/{task_id}/status → poll for progress
// GET /api/v1/learn/autonomous/{task_id}/result → get final result
```

2. **Thread pool pattern** (works with all frameworks):
```cpp
// Background task manager
std::unordered_map<std::string, std::future<AutonomousLearnResult>> tasks;

svr.Post("/api/v1/learn/autonomous", [&](const auto& req, auto& res) {
    std::string task_id = generate_uuid();
    tasks[task_id] = std::async(std::launch::async, [&learner]() {
        return learner.autonomous_learn(env, 10000);
    });
    res.set_content(json{{"task_id", task_id}}.dump(), "application/json");
});
```

3. **WebSocket for real-time progress** (requires crow + separate WebSocket lib OR oat++/userver):
```
Client: WS connect to /ws/learning
Server: push progress events during autonomous_learn
```

### CMake Integration Pattern

```cmake
# Option 1: crow (recommended for this project)
FetchContent_Declare(
    crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG        v1.2.0
)
FetchContent_MakeAvailable(crow)

# nlohmann/json (needed regardless of framework choice)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# REST server executable
add_executable(ai_learning_server
    server/main.cpp
    server/routes/learning_routes.cpp
    server/routes/knowledge_routes.cpp
    server/routes/system_routes.cpp
    server/serializers.cpp
)
target_link_libraries(ai_learning_server
    PRIVATE ai_learning Crow::Crow nlohmann_json::nlohmann_json
)
```

### Recommended Architecture for This Project

Given the project constraints (Windows, C++20, CMake, research prototype):

**Primary recommendation: crow + nlohmann/json**

Rationale:
1. crow's Flask-like API is approachable and maps well to Learner's method surface
2. nlohmann/json handles the 45+ struct types cleanly with `to_json`/`from_json`
3. Both have good MSVC/Windows support
4. Moderate dependency footprint (no Boost)
5. Thread pool model is sufficient for the expected concurrency level
6. For real-time visualization, add a separate WebSocket library (e.g., `websocketpp` or `libwebsockets`)

**If async/coroutines are critical**: Consider upgrading to userver or using crow with a custom async layer using C++20 `std::jthread` and `std::execution` (when available).

### API Route Design (Illustrative)

```
POST   /api/v1/learner/create              → Create learner with config
POST   /api/v1/learner/learn/text          → learn_from_text()
POST   /api/v1/learner/observe             → observe_text()
POST   /api/v1/learner/reason              → reason()
POST   /api/v1/learner/think               → think()
POST   /api/v1/learner/perceive            → perceive()
POST   /api/v1/learner/remember            → remember()
POST   /api/v1/learner/consolidate         → consolidate()
POST   /api/v1/learner/autonomous/start    → Start autonomous_learn (async)
GET    /api/v1/learner/autonomous/{id}     → Get status/result
GET    /api/v1/learner/stats               → get_stats()
GET    /api/v1/learner/stage               → stage()
POST   /api/v1/learner/save                → save()
POST   /api/v1/learner/load                → load()
POST   /api/v1/learner/evolve              → evolve()
GET    /api/v1/learner/capabilities         → evaluate_capabilities()
POST   /api/v1/learner/integrated-pipeline → integrated_pipeline()
POST   /api/v1/learner/emotion             → process_emotion()
POST   /api/v1/learner/insight             → try_insight()
GET    /api/v1/knowledge/entities           → KG query
GET    /api/v1/knowledge/entity/{id}        → get_entity
GET    /api/v1/knowledge/relations          → get_relations_of
GET    /api/v1/knowledge/path               → find_path
GET    /api/v1/health                       → Health check
WS     /ws/v1/metrics                       → Real-time learning metrics
```

### External References

- [Crow Documentation](https://crowcpp.org/master/) — v1.2.0, Flask-like C++ web framework
- [cpp-httplib GitHub](https://github.com/yhirose/cpp-httplib) — v0.18.x, single-header HTTP server
- [oat++ Documentation](https://oatpp.io/) — v1.3.0, full MVC framework
- [userver Documentation](https://userver.tech/) — coroutine-based async framework by Yandex
- [nlohmann/json GitHub](https://github.com/nlohmann/json) — v3.11.3, JSON for Modern C++
- [Pistache GitHub](https://github.com/oktal/pistache) — v0.4.x, async HTTP framework (Linux-focused)

### Related Specs

- `ai-learning-cpp/CMakeLists.txt` — existing build configuration to extend
- `ai-learning-cpp/CPP_REFACTORING_PLAN.md` — architecture overview

## Caveats / Not Found

- **No existing REST or HTTP code**: The project has zero server code today.
- **crow + Windows MSVC**: While generally working, some template-heavy compilation can be slow on MSVC. Test with the actual project.
- **WebSocket**: None of the lightweight frameworks (crow, cpp-httplib) have built-in WebSocket. If real-time metrics are needed, a separate library must be added. See `visualization-options.md`.
- **Production deployment**: For a research prototype, all frameworks are adequate. For production ML serving, consider a Python wrapper with FastAPI (using the pybind11 bindings) for better ecosystem support.
