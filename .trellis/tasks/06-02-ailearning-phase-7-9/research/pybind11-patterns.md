# Research: pybind11 Patterns for AILearning C++ Codebase

- **Query**: How to expose a large C++ class (26-subsystem Learner) to Python via pybind11
- **Scope**: Mixed (internal codebase analysis + external pybind11 best practices)
- **Date**: 2026-06-02

## Findings

### Project Context

The AILearning C++ codebase (`ai-learning-cpp/`) is a C++20 cognitive learning system with the following structure:

**Core class**: `ai_learning::core::Learner` (file: `include/ai_learning/core/learner.hpp`)
- A thin orchestrator (~475 lines) that composes ~26 subsystems via value semantics (not pointers)
- Public API: ~40+ methods spanning text learning, perception loop, memory, autonomous learning, development stages, evolution, Phase 3-6 cognitive abilities
- All subsystems are held as member variables directly (no `unique_ptr`/`shared_ptr`)

**Key CMake facts** (from `CMakeLists.txt`):
- CMake 3.22+, C++20, MSVC + GCC/Clang support
- Builds as a static library: `add_library(ai_learning STATIC ...)`
- Optional CUDA backend via `ai_learning_cuda` static library
- Uses FetchContent for Catch2
- Current build targets: `ai_learning_tests`, `ai_learning_main`, `ai_learning_benchmark`, `debug_main`

**No existing pybind11 code found** (grep for `pybind11|PYBIND11_MODULE` returned zero results).

### Files Found

| File Path | Description |
|---|---|
| `ai-learning-cpp/include/ai_learning/core/learner.hpp` | Main Learner orchestrator class, ~475 lines |
| `ai-learning-cpp/include/ai_learning/core/types.hpp` | Core type aliases: Tensor=vector<float>, Properties, Metadata, TagSet, EntityId, Stats |
| `ai-learning-cpp/include/ai_learning/core/config.hpp` | LearnerConfig (25+ fields), TrainerConfig, StageDefinition, enums Device/PlasticitySchedule/ConsolidationStrategy |
| `ai-learning-cpp/include/ai_learning/core/module_registry.hpp` | Lazy-init module registry using std::any |
| `ai-learning-cpp/include/ai_learning/domain/domain_events.hpp` | 9 event types via std::variant, IEventPublisher/IEventSubscriber interfaces |
| `ai-learning-cpp/include/ai_learning/domain/environment/i_environment.hpp` | IEnvironment virtual interface (observe/step/reset) |
| `ai-learning-cpp/include/ai_learning/domain/memory/i_memory.hpp` | IMemory virtual interface, MemoryItem struct |
| `ai-learning-cpp/include/ai_learning/domain/knowledge/knowledge_graph.hpp` | KnowledgeGraph aggregate root, uses std::reference_wrapper |
| `ai-learning-cpp/include/ai_learning/learning/predictive_coding_engine.hpp` | PredictiveCodingEngine, InferenceResult, State struct |
| `ai-learning-cpp/include/ai_learning/learning/text_learner.hpp` | TextLearner, TextLearnResult |
| `ai-learning-cpp/include/ai_learning/learning/emotion_engine.hpp` | EmotionEngine, EmotionState, EmotionEvent, MemoryModulation, DopamineSignal |
| `ai-learning-cpp/include/ai_learning/learning/meta_learner.hpp` | MetaLearner, LearningStrategyType enum, TaskDescriptor, LearningExperience, MetaLearningRecommendation |
| `ai-learning-cpp/include/ai_learning/learning/integrated_learner.hpp` | IntegratedLearner, 5 integration report types |
| `ai-learning-cpp/include/ai_learning/core/evolver.hpp` | Evolver, CapabilityResult, EvolutionResult |
| `ai-learning-cpp/CMakeLists.txt` | Build configuration, static library, CUDA optional |

### Type Surface Analysis (What pybind11 Must Map)

**STL containers used throughout the codebase**:
- `std::vector<float>` — Tensor type, used in 415+ locations across 52 headers
- `std::vector<std::string>` — entity lists, step lists, observation results
- `std::map<std::string, std::string>` — properties, metadata (178+ occurrences across 44 files)
- `std::map<std::string, double>` — stats, feature maps, mastery maps
- `std::map<std::string, std::vector<float>>` — environment observations
- `std::map<std::string, std::vector<std::string>>` — observe_text return type
- `std::map<std::string, std::map<std::string, double>>` — nested capability maps
- `std::unordered_map<std::string, std::string>` — Metadata type
- `std::unordered_map<std::string, std::unordered_set<std::string>>` — KG indices
- `std::optional<T>` — used 44 times across 21 files (optional returns)
- `std::variant<...9 types...>` — DomainEvent type (domain_events.hpp)
- `std::deque<float>` — error history, emotion history, dopamine history
- `std::pair<std::string, float>` — STDP query results
- `std::reference_wrapper<const Entity>` — KG return types (5 occurrences)
- `std::chrono::steady_clock::time_point` — EventTime in domain events

**Enums to expose**:
- `Device` (kCpu, kCuda, kAuto)
- `PlasticitySchedule` (kExponential, kSigmoid, kLinear, kStep, kNone)
- `ConsolidationStrategy` (kRandom, kSuccess, kRecent, kSurprising)
- `LearningStrategyType` (9 strategies: kRoteMemorization through kStructuredPractice)

**Structs to expose** (45+ result/report/config types):
- `LearnerConfig` (25+ fields), `TrainerConfig`, `StageDefinition`
- `AutonomousLearnResult`, `TextLearnResult`, `InferenceResult`
- `EmotionState`, `EmotionEvent`, `MemoryModulation`, `DopamineSignal`, `EmotionConfig`
- `TaskDescriptor`, `LearningExperience`, `MetaLearningRecommendation`, `LearningRateSchedule`, `StrategyStats`
- `IntegratedPipelineReport`, `MetaGuidedSessionReport`, `EmotionModulatedParams`, `ExperimentDrivenExplorationReport`, `SocialAnalogicalReport`
- `CapabilityResult`, `EvolutionResult`, `Improvement`
- `MemoryItem`, `AggregateResult`, `PathResult`
- All 9 domain event structs in `domain_events.hpp`

**Interfaces requiring Python trampolines**:
- `IEnvironment` (observe/step/reset/configure_for_stage)
- `IMemory` (store/retrieve/consolidate)
- `IEventPublisher` (publish)
- `IEventSubscriber` (on_event)

### pybind11 Best Practices for Large Codebases

#### 1. Modular Binding Organization

For a 26-subsystem codebase, split bindings into multiple compilation units to avoid single-file compilation bloat:

```
bindings/
  CMakeLists.txt
  module.cpp           # PYBIND11_MODULE declaration
  core_types.cpp       # Tensor, Properties, Stats, enums
  core_config.cpp      # LearnerConfig, TrainerConfig
  core_learner.cpp     # Learner class binding
  domain_kg.cpp        # KnowledgeGraph, Entity, Relation
  domain_events.cpp    # DomainEvent variant, event structs
  domain_memory.cpp    # MemoryItem, IMemory trampoline
  domain_env.cpp       # IEnvironment trampoline
  learning_engine.cpp  # PredictiveCodingEngine
  learning_text.cpp    # TextLearner
  learning_emotion.cpp # EmotionEngine
  learning_meta.cpp    # MetaLearner
  learning_social.cpp  # SocialLearningEngine
  learning_insight.cpp # InsightEngine
  learning_integrated.cpp # IntegratedLearner
  reasoning.cpp        # UnifiedReasoningEngine, WorldModel
```

Each file uses `py::module_::def()` / `py::class_<>` with a shared module object passed in:

```cpp
// module.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(ai_learning, m) {
    m.doc() = "AILearning C++20 cognitive learning system";
    init_core_types(m);
    init_core_config(m);
    init_core_learner(m);
    // ... etc
}

// core_types.cpp
void init_core_types(py::module_& m) {
    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer(...);
}
```

#### 2. Memory Management Patterns

**The Learner class uses value semantics** — all 26 subsystems are direct member variables, not pointers. This is favorable for pybind11:

```cpp
// Value semantics: pybind11 can hold Learner directly
py::class_<ai_learning::core::Learner>(m, "Learner")
    .def(py::init<const LearnerConfig&>())
    .def("learn_from_text", &Learner::learn_from_text)
    // ...
```

**Subsystem access returns references** — the `knowledge_graph()`, `engine()`, etc. methods return `T&`. Use `py::return_value_policy::reference_internal`:

```cpp
py::class_<Learner>(m, "Learner")
    .def("knowledge_graph",
         py::overload_cast<>(&Learner::knowledge_graph),
         py::return_value_policy::reference_internal)
```

This ties the subsystem's Python lifetime to the parent Learner.

**For IEnvironment (virtual interface)**, use PYBIND11_OVERRIDE macros for Python-side subclassing:

```cpp
class PyEnvironment : public ai_learning::domain::IEnvironment {
public:
    using IEnvironment::IEnvironment;
    auto observe() const -> std::map<std::string, std::vector<float>> override {
        PYBIND11_OVERRIDE_PURE(std::map<std::string, std::vector<float>>, IEnvironment, observe);
    }
    auto step(int action) -> std::pair<double, bool> override {
        PYBIND11_OVERRIDE_PURE(std::pair<double, bool>, IEnvironment, step, action);
    }
    auto reset() -> std::map<std::string, std::vector<float>> override {
        PYBIND11_OVERRIDE_PURE(std::map<std::string, std::vector<float>>, IEnvironment, reset);
    }
};
```

#### 3. STL Container Mappings

pybind11 provides automatic conversion for standard containers. Include `<pybind11/stl.h>` for:
- `std::vector<T>` <-> Python `list`
- `std::map<K,V>` <-> Python `dict`
- `std::optional<T>` <-> Python `T | None`
- `std::pair<A,B>` <-> Python `tuple`
- `std::deque<T>` <-> Python `list`

For `std::unordered_map` and `std::unordered_set`, also covered by `<pybind11/stl.h>`.

**std::variant mapping**: Include `<pybind11/stl.h>` handles `std::variant` automatically. For the `DomainEvent` variant with 9 types, pybind11 will convert to the matching Python type. Consider using `py::class_<DomainEvent>` with visitor-based access instead for cleaner Python API.

**std::reference_wrapper** (used in KnowledgeGraph): This is NOT automatically handled. Options:
1. Return copies instead of references (modify binding layer, not C++ code)
2. Use custom type caster for `std::reference_wrapper`
3. Expose wrapper methods that return by value

**std::chrono::steady_clock::time_point** (EventTime): Not auto-converted. Options:
1. Expose as opaque or convert to Python `datetime`
2. Use `int64_t` milliseconds instead in the Python API

#### 4. Performance Considerations

**GIL Release**: Release the GIL for all computation-heavy methods:

```cpp
py::class_<Learner>(m, "Learner")
    .def("learn_from_text", [](Learner& self, const std::string& text, const std::string& source) {
        py::gil_scoped_release release;
        return self.learn_from_text(text, source);
    }, py::arg("text"), py::arg("source") = "text")
    .def("autonomous_learn", [](Learner& self, IEnvironment& env, int max_steps, int max_episodes, int interval) {
        py::gil_scoped_release release;
        return self.autonomous_learn(env, max_steps, max_episodes, interval);
    }, py::arg("env"), py::arg("max_steps")=1000, py::arg("max_episodes")=0, py::arg("consolidation_interval")=100);
```

**Buffer protocol for Tensor**: Since `Tensor = std::vector<float>`, expose it with buffer protocol for zero-copy numpy interop:

```cpp
// Option A: Register Tensor as a buffer-compatible type
py::class_<std::vector<float>>(m, "Tensor", py::buffer_protocol())
    .def_buffer([](std::vector<float>& v) -> py::buffer_info {
        return py::buffer_info(
            v.data(), sizeof(float), py::format_descriptor<float>::format(),
            1, {v.size()}, {sizeof(float)}
        );
    });
```

**Avoid copies for large vectors**: Use `py::array_t<float>` for function arguments that accept numpy arrays:

```cpp
.def("perceive", [](Learner& self, const std::map<std::string, py::array_t<float>>& raw_input) {
    // Convert py::array_t to std::vector<float>
    std::map<std::string, std::vector<float>> cpp_input;
    for (auto& [key, arr] : raw_input) {
        auto buf = arr.request();
        float* ptr = static_cast<float*>(buf.ptr);
        cpp_input[key] = std::vector<float>(ptr, ptr + buf.size);
    }
    py::gil_scoped_release release;
    return self.perceive(cpp_input);
})
```

#### 5. CMake Integration

Add to existing `CMakeLists.txt` or create a separate `bindings/CMakeLists.txt`:

```cmake
# Option A: FetchContent pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6  # Latest stable as of 2025
)
FetchContent_MakeAvailable(pybind11)

# Option B: find_package (if pybind11 installed system-wide)
# find_package(pybind11 REQUIRED)

# The binding library
pybind11_add_module(ai_learning_py
    bindings/module.cpp
    bindings/core_types.cpp
    bindings/core_config.cpp
    bindings/core_learner.cpp
    bindings/domain_kg.cpp
    # ... other binding files
)
target_link_libraries(ai_learning_py PRIVATE ai_learning)
target_include_directories(ai_learning_py PRIVATE include)

# CUDA variant
if(CUDA_FOUND)
    target_link_libraries(ai_learning_py PRIVATE ai_learning_cuda)
endif()
```

**Important**: `pybind11_add_module` creates a shared library (`.pyd` on Windows / `.so` on Linux). The static `ai_learning` library must be compiled with `POSITION_INDEPENDENT_CODE ON`:

```cmake
set_target_properties(ai_learning PROPERTIES POSITION_INDEPENDENT_CODE ON)
```

**Windows-specific notes**:
- MSVC requires `/permissive-` and `/Zc:__cplusplus` (already set in project)
- Python development headers must be available (Python installed with "Download debug binaries")
- Debug builds require `python3xx_d.lib` — often simpler to build Release only for bindings
- Consider `CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded"` for consistent runtime linking with Python

#### 6. Handling std::reference_wrapper Returns

KnowledgeGraph methods return `std::vector<std::reference_wrapper<const Entity>>`. pybind11 cannot auto-convert these. Strategies:

**Strategy A — Lambda wrapper returning copies** (simplest):
```cpp
.def("get_related", [](const KnowledgeGraph& kg, const std::string& id, const std::string& rel_type) {
    auto refs = kg.get_related(id, rel_type);
    std::vector<Entity> copies;
    copies.reserve(refs.size());
    for (auto& ref : refs) copies.push_back(ref.get());
    return copies;
}, py::arg("entity_id"), py::arg("relation_type") = "")
```

**Strategy B — Custom type caster** (zero-copy, more complex):
```cpp
namespace pybind11 { namespace detail {
    template<> struct type_caster<std::reference_wrapper<const Entity>> {
        // custom implementation
    };
}}
```

#### 7. Enum Binding Pattern

```cpp
py::enum_<Device>(m, "Device")
    .value("CPU", Device::kCpu)
    .value("CUDA", Device::kCuda)
    .value("AUTO", Device::kAuto)
    .export_values();

py::enum_<LearningStrategyType>(m, "LearningStrategyType")
    .value("ROTE_MEMORIZATION", LearningStrategyType::kRoteMemorization)
    .value("SPACED_REPETITION", LearningStrategyType::kSpacedRepetition)
    // ... all 9 strategies
    .export_values();
```

#### 8. Struct Binding Pattern (for Result Types)

```cpp
py::class_<EmotionState>(m, "EmotionState")
    .def(py::init<>())
    .def_readonly("valence", &EmotionState::valence)
    .def_readonly("arousal", &EmotionState::arousal)
    .def_readonly("dominance", &EmotionState::dominance)
    .def_readonly("label", &EmotionState::label)
    .def("intensity", &EmotionState::intensity);
```

For structs with many fields (like `LearnerConfig` with 25+ fields), use `.def_readwrite()` for mutable config:

```cpp
py::class_<LearnerConfig>(m, "LearnerConfig")
    .def(py::init<>())
    .def_readwrite("obs_dim", &LearnerConfig::obs_dim)
    .def_readwrite("action_dim", &LearnerConfig::action_dim)
    .def_readwrite("learning_rate", &LearnerConfig::learning_rate)
    // ... all 25+ fields
    ;
```

### External References

- [pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/) — official docs, v2.13.x
- [pybind11 GitHub](https://github.com/pybind/pybind11) — v2.13.6 latest stable
- pybind11 supports C++20 features including concepts, ranges, span (v2.12+)
- `<pybind11/stl.h>` covers `std::optional`, `std::variant`, `std::vector`, `std::map`, `std::unordered_map`, `std::pair`, `std::deque`
- `<pybind11/functional.h>` needed for `std::function` parameters (used in ModuleRegistry)

### Related Specs

- `.trellis/spec/backend/directory-structure.md` — backend directory conventions
- `ai-learning-cpp/CPP_REFACTORING_PLAN.md` — existing C++ refactoring roadmap (Phases 1-6 complete)

## Caveats / Not Found

- **No existing pybind11 code**: The project has zero binding code today. Everything must be built from scratch.
- **std::reference_wrapper**: KnowledgeGraph uses `std::reference_wrapper<const Entity/Relation>` in 5 return positions. These require custom handling (lambda wrappers or type casters).
- **std::chrono::steady_clock::time_point**: Used in all 9 domain event structs. No automatic pybind11 conversion. Needs custom caster or conversion to int64_t.
- **Windows MSVC debug builds**: pybind11 debug builds need Python debug libs (`python3xx_d.lib`). Consider Release-only binding builds.
- **ModuleRegistry uses std::any**: This is not directly exposable to Python. If Python-side module registration is needed, a thin wrapper would be required.
- **CUDA linkage**: The CUDA static library must also have PIC enabled for the Python binding shared module.
- **45+ struct types**: Full binding of all result/report types is substantial. Consider a phased approach: expose core Learner API first, subsystems later.
