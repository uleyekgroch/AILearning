# Logging Guidelines

> No logging framework. Use `std::cout` / `std::cerr` with `[Tag]` prefix.

---

## Overview

This is a research prototype. Logging is ad-hoc via `std::cout` (info) and `std::cerr` (warnings/errors). No structured logging library.

---

## Log Levels (by convention)

| Stream | Level | When |
|--------|-------|------|
| `std::cout` | Info | Startup, shutdown, client connect/disconnect, CUDA ready |
| `std::cerr` | Warn/Error | GPU alloc failure, WS send failure, model load failure |

---

## Format

Tagged prefix pattern: `[Component] Message`

```cpp
// Server lifecycle
std::cout << "[Server] AILearning REST API starting on port " << port << "\n";
std::cout << "[Server] Stopped gracefully.\n";

// WebSocket events
std::cout << "[WS/events] Client connected. Total: " << count << "\n";
std::cerr << "[WS/events] send_text failed: " << e.what() << "\n";

// CUDA operations
std::cout << "[Flash Attention] CUDA ready on " << prop.name << "\n";
std::cerr << "[KnowledgeGraph CUDA] GPU buffer alloc failed\n";
```

---

## What to Log

- Server start/stop, port binding
- WebSocket client connect/disconnect count
- CUDA initialization success (GPU name)
- Async task failures
- Model loading errors

---

## What NOT to Log

- Per-step learning data (too frequent, use WebSocket push instead)
- Internal tensor values
- Full JSON request/response bodies
- Knowledge graph entity contents on every operation
