# Error Handling

> Exceptions for fatal errors, `std::optional` for absent results, catch-at-boundary.

---

## Overview

No custom exception hierarchy. Use `std::runtime_error` for unrecoverable errors and `std::optional<T>` for operations that may not produce a result.

---

## Error Types

- **`std::runtime_error`** — fatal/config errors (model loading, invalid state)
- **`std::optional<T>`** — operations that may return nothing (insight, experiment design, theory)
- **`std::variant<...>`** — domain events (9 types via `std::visit`)

---

## Error Handling Patterns

### 1. Throw for fatal errors (constructor, I/O)

```cpp
// OK: model file missing = unrecoverable
LlamaCppEmbeddingProvider::LlamaCppEmbeddingProvider(const std::string& path) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("model not found: " + path);
}
```

### 2. Return optional for absent results

```cpp
// OK: insight may not happen
auto try_insight(const std::string& context) -> std::optional<InsightEvent>;
auto design_experiment() -> std::optional<ExperimentDesign>;
```

### 3. Catch at boundary (server routes)

```cpp
// Server route: catch-all, convert to HTTP error
CROW_ROUTE(app, "/api/learn/text").methods("POST"_method)
([&](const crow::request& req) -> crow::response {
    try {
        auto body = json::parse(req.body);
        // ... call learner ...
        return ok_resp(result);
    } catch (const std::exception& e) { return err_resp(400, e.what()); }
});
```

### 4. Suppress in non-critical paths

```cpp
// OK: don't crash if stats push fails
try { conn->send_text(msg); }
catch (...) { /* log and continue */ }
```

---

## API Error Responses

All errors follow `{ "error": "message" }` format:

```cpp
static auto err_resp(int code, const std::string& msg) -> crow::response {
    return json_resp(code, {{"error", msg}});
}
```

HTTP status codes: `400` (bad request), `404` (not found), `500` (internal), `202` (async accepted).

---

## Common Mistakes

- **Don't throw from learning algorithms** — return `optional` or default values instead
- **Don't let exceptions escape route handlers** — always wrap in try/catch
- **Don't catch and swallow in core logic** — let it propagate to the boundary
- **Don't use error codes** — prefer `optional<T>` or exceptions, not `int` return codes
