# Backend Development Guidelines

> C++20 cognitive learning system — DDD layered architecture.

---

## Overview

Backend is a C++20 static library (`ai_learning`) with a Crow-based REST server (`ai_learning_server`). Follows DDD with `Learner` as aggregate root.

---

## Guidelines Index

| Guide | Description |
|-------|-------------|
| [Directory Structure](./directory-structure.md) | DDD layer organization, include/src mirror |
| [Error Handling](./error-handling.md) | Exceptions + optional, catch-at-boundary pattern |
| [Quality Guidelines](./quality-guidelines.md) | SOLID, naming, 800-line limit, testing |
| [Logging Guidelines](./logging-guidelines.md) | `std::cout`/`std::cerr` with `[Tag]` prefix |
| [Database Guidelines](./database-guidelines.md) | INI-style file persistence, no ORM |

---

## Key Conventions

- **Language**: C++20, `auto` + trailing return types everywhere
- **Namespace**: `ai_learning::{core,domain,learning,reasoning,memory,language,server}`
- **Build**: CMake 3.22+, FetchContent for deps, MinGW on Windows
- **Testing**: Catch2 v3, tag-based (`[phase3]`, `[phase4]`, etc.)
