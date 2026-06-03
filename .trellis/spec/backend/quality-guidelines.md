# Quality Guidelines

> Code standards derived from project CLAUDE.md and actual codebase patterns.

---

## Forbidden Patterns

| Pattern | Why |
|---------|-----|
| `new`/`delete` for subsystems | Use value members or `unique_ptr` |
| Raw pointers for ownership | `unique_ptr` only for injected/optional deps |
| `auto` without trailing return | Always write `-> ReturnType` |
| `concept` as identifier | C++20 reserved keyword; use `cpt` or `prototype` |
| Member/variable name clash | Counter `id_counter_`, not `next_id_()` |
| God class (>800 lines) | Split by responsibility |
| Method >50 lines | Extract helper methods |
| Nested >3 levels | Guard clauses or extract function |

---

## Required Patterns

```cpp
// 1. Trailing return type on ALL functions
auto learn_from_text(const std::string& text) -> TextLearnResult;

// 2. Value semantics for subsystems
class Learner {
    KnowledgeGraph kg_;                        // not unique_ptr<KnowledgeGraph>
    std::unique_ptr<IPredictiveEngine> engine_; // OK: injected dependency
};

// 3. Const-correct accessors
[[nodiscard]] auto knowledge_graph() -> KnowledgeGraph& { return kg_; }
[[nodiscard]] auto knowledge_graph() const -> const KnowledgeGraph& { return kg_; }

// 4. Private method suffix underscore
auto check_promotion_(const std::map<std::string, double>& eval) const -> bool;

// 5. DDD value objects for config
struct LearnerConfig { int obs_dim = 128; /* ... */ };
```

---

## Testing Requirements

- **Framework**: Catch2 v3 with `TEST_CASE` + `SECTION`
- **Tags**: `[phase3]`, `[phase4]`, `[phase5]`, `[phase6]`, `[integration]`
- **Float assertions**: `WithinAbs(value, epsilon)` or `WithinRel(value, tolerance)`, never `==`
- **Coverage target**: 80%+ unit, 60%+ integration
- **One test file per phase**: `tests/learning/test_phase5_meta_cognition.cpp`
- **Server tests**: `tests/server/test_server_routes.cpp`

```cpp
TEST_CASE("Emotion state intensity", "[phase4]") {
    EmotionState s{0.5, 0.7, 0.3, "excited"};
    CHECK(s.intensity() == Approx(0.5).epsilon(0.01));
}
```

---

## Code Review Checklist

- [ ] File under 800 lines? Methods under 50 lines?
- [ ] Trailing return types on all functions?
- [ ] Value semantics (not unnecessary pointers)?
- [ ] DDD layers respected (domain has no learning deps)?
- [ ] No C++20 keyword conflicts (`concept`, `requires`)?
- [ ] Float comparisons use Catch2 matchers?
