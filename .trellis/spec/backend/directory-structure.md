# Directory Structure

> DDD layered layout — include/ and src/ mirror each other.

---

## Directory Layout

```
ai-learning-cpp/
├── include/ai_learning/
│   ├── core/            # Aggregates, config, types, tensor ops
│   │   ├── learner.hpp       # Aggregate root (~440 lines)
│   │   ├── config.hpp        # LearnerConfig, TrainerConfig value objects
│   │   ├── types.hpp         # Tensor, Matrix, Properties, Metadata aliases
│   │   ├── evolver.hpp       # Autonomous evolution
│   │   └── tensor_ops.hpp    # CPU/CUDA unified tensor operations
│   ├── domain/          # Domain layer: entities, interfaces, events
│   │   ├── domain_events.hpp # 9 event types, IEventPublisher/Subscriber
│   │   ├── knowledge/        # KnowledgeGraph aggregate, Entity, Relation
│   │   ├── environment/      # IEnvironment interface, SimpleEnvironment
│   │   ├── memory/           # IMemory interface, MemoryItem
│   │   ├── perception/       # IPerception, MultiModalEncoder
│   │   └── social/           # ISocialAgent interface
│   ├── learning/        # All learning algorithms (~45 files)
│   ├── reasoning/       # Activation spread, simulation, unified engine
│   ├── memory/          # EpisodicMemory
│   ├── language/        # Grounding, development, dialog, LLM providers
│   ├── society/         # Society controller, AgentHandle
│   ├── goals/           # GoalManager, Goal value objects
│   ├── assessment/      # MasteryAssessor, ProficiencyTester
│   └── utils/           # UTF-8 utilities
├── src/                 # Mirrors include/ structure exactly
│   ├── core/            # learner.cpp, learner_io.cpp, tensor_ops...
│   ├── domain/          # knowledge/, environment/...
│   ├── learning/        # One .cpp per subsystem
│   ├── server/          # REST server (Crow-based)
│   │   ├── server.hpp/cpp        # Server class
│   │   ├── server_config.hpp     # ServerConfig value object
│   │   ├── route_groups.hpp      # SharedState + route registration decls
│   │   ├── dto.hpp               # JSON serialization helpers
│   │   ├── event_adapter.hpp/cpp # WebSocket event bridge
│   │   ├── core_routes.cpp       # Health, stats, learn, reason, memory
│   │   ├── advanced_routes.cpp   # Phase 3-6 endpoints
│   │   ├── society_routes.cpp    # Multi-agent endpoints
│   │   ├── chat_routes.cpp       # Dialog/chat endpoints
│   │   ├── runtime_routes.cpp    # Continuous loop control
│   │   └── goals_routes.cpp      # Goal management
│   └── society/         # Society controller implementation
├── tests/               # Mirrors include/ structure
│   ├── server/          # Server integration tests
│   ├── core/ learning/ reasoning/ ...
│   └── test_*.cpp       # Phase-specific test files
├── web/                 # Vue 3 SPA (no build step)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── CMakeLists.txt
├── server_main.cpp      # Server entry point
├── main.cpp             # CLI entry point
└── benchmark.cpp
```

---

## Module Organization

Each subsystem follows: `include/ai_learning/<layer>/<module>.hpp` + `src/<layer>/<module>.cpp`

- **Headers** declare class + all inline doc. No `.cpp` for header-only modules.
- **One class per file**. Struct/helper types can share a header.
- **Server routes** split by domain: `core_routes`, `advanced_routes`, etc.
- **CUDA files**: `.cu`/`.cuh` alongside corresponding `.cpp`/`.hpp`.

---

## Naming Conventions

- **Files**: `snake_case.cpp` / `snake_case.hpp`
- **Classes**: `PascalCase` (e.g., `PredictiveCodingEngine`)
- **Interfaces**: `IPascalCase` (e.g., `IEnvironment`, `IEventPublisher`)
- **Enums**: `kCamelCase` values (e.g., `Device::kCpu`)
- **Private members**: trailing underscore (e.g., `learner_`, `mutex_`)
- **Namespaces**: `ai_learning::core`, `ai_learning::domain`, etc.
