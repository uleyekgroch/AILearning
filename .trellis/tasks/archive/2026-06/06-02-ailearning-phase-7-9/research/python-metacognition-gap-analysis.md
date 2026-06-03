# Research: Python Metacognition Module vs C++ Metacognition (Gap Analysis)

- **Query**: Compare Python src/metacognition/ with C++ metacognition.hpp/cpp to find gaps
- **Scope**: internal
- **Date**: 2026-06-03

## Findings

### Python Module Overview

`src/metacognition/` has 3 files, **377 lines total** (excluding `__init__.py`).

| File | Lines | Description |
|---|---|---|
| `monitor.py` | 163 | Confidence tracking, gap detection, difficulty estimation |
| `strategy.py` | 111 | Strategy selection with EMA effectiveness tracking |
| `assessor.py` | 103 | High-level facade combining monitor + strategy |
| `__init__.py` | 3 | Re-exports all public classes |

### C++ Module Overview

`ai-learning-cpp/include/ai_learning/learning/metacognition.hpp` (169 lines) + `src/learning/metacognition.cpp` (373 lines) = **542 lines total**.

### Detailed Comparison

#### 1. Knowledge Confidence Assessment

| Aspect | Python (monitor.py) | C++ (metacognition.hpp/cpp) |
|---|---|---|
| Method name | `knowledge_confidence(topic)` | `assess_confidence(topic)` |
| Approach | KG-based: entity_conf(0.4) + neighbor_avg(0.3) + relation_avg(0.3) | Usage-based: times_correct / times_used, calibrated by usage count |
| Data source | Queries `KnowledgeGraph` directly | Internal `confidence_registry_` map |
| Result type | `float` [0,1] | `KnowledgeConfidence` struct with topic, confidence, uncertainty, times_used, times_correct, last_source |

**Key difference**: Python's monitor computes confidence dynamically from the knowledge graph topology (entity + neighbor + relation confidences). C++ uses a simpler approach: tracking usage outcomes (correct/incorrect) in an internal registry. The Python approach is more "online" (always reflects current KG state), while C++ is more "cumulative" (builds up from recorded outcomes).

#### 2. Knowledge Gap Detection

| Aspect | Python | C++ |
|---|---|---|
| Method name | `identify_knowledge_gaps()` | `detect_gaps(learner)` |
| Gap criteria | 1. confidence < 0.3, 2. no relations (isolated), 3. no properties | 1. entity_count < 10, 2. relation_count < 5, 3. learning_progress < 0.3, 4. confidence < threshold on core topics |
| Returns | `List[str]` (entity IDs) | `List<KnowledgeGap>` with urgency, reason, related_known, suggested_action |

**Key difference**: Python detects gaps purely from knowledge graph properties. C++ additionally considers learner-level statistics (progress, step count) and has a hardcoded list of "core topics". C++ gaps carry richer metadata (urgency, reasons, suggested actions).

#### 3. Difficulty Estimation

| Aspect | Python | C++ |
|---|---|---|
| Method name | `estimate_difficulty(task_description)` | No direct equivalent |
| Approach | Extracts entity IDs from text, computes unknown_ratio(0.6) + scale_factor(0.4) | N/A |

**GAP**: C++ has no `estimate_difficulty` method. The closest is `evaluate_strategy(recent_performance, performance_trend)` which assesses strategy effectiveness, not task difficulty.

#### 4. Strategy Selection

| Aspect | Python (strategy.py) | C++ |
|---|---|---|
| Strategy enum | `explore`, `practice`, `analogize`, `decompose`, `seek_help`, `hypothesize` | No explicit strategy list. `evaluate_strategy()` returns recommendations like "interval repetition + active recall", "decompose + trial-error", "analogical transfer" |
| Selection logic | Rule-based on difficulty + gap_count | Rule-based on performance + trend |
| Effectiveness tracking | EMA (alpha=0.3) per strategy, usage/success counts | No tracking -- stateless evaluation |
| Statistics | `get_strategy_stats()` returns usage, success_rate, effectiveness per strategy | No equivalent |

**GAP**: C++ strategy evaluation is much simpler. It lacks:
1. Explicit strategy enum/catalog
2. Per-strategy effectiveness tracking with EMA
3. Usage/success statistics
4. Strategy selection based on difficulty + gaps (it uses performance + trend instead)

#### 5. Learning Plan Generation

| Aspect | Python (assessor.py) | C++ |
|---|---|---|
| Method name | `plan_next_learning(task)` | `generate_learning_plan(gaps)` |
| Output | `{gaps, difficulty, strategy, readiness}` | `List<InformationNeed>` with query, context, priority, source_type |
| Readiness assessment | `assess_readiness(required_topics)` -- averages confidence | No equivalent |

#### 6. Assessment Update

| Aspect | Python | C++ |
|---|---|---|
| Method name | `update_assessment(topic, success)` | `record_outcome(topic, correct)` |
| Behavior | Adjusts entity confidence in KG (+0.1/-0.05), rebuilds KnowledgeAssessment | Increments times_used/times_correct, recalibrates |

#### 7. High-Level Facade

| Aspect | Python (assessor.py) | C++ |
|---|---|---|
| Class | `MetaAssessor` combines Monitor + StrategySelector | `MetacognitionEngine` is a monolithic class |
| `self_evaluate(topics)` | Returns confidence per topic | No equivalent (closest: `assess_batch()`) |
| Serialization | `save_state()` / `load_state()` | No serialization |

#### 8. Additional C++ Capabilities (not in Python)

The C++ `MetacognitionEngine` has features NOT present in the Python module:
- `should_seek_info(topic)` -- determines if more info needed
- `generate_query(gap)` -- creates an information-seeking query
- `generate_report(learner)` -- comprehensive metacognitive report with dimension_scores
- `knows_about(topic)` -- boolean confidence check
- `topic_coverage(domain, required_topics, learner)` -- per-topic coverage map
- `estimate_relatedness_(topic_a, topic_b, learner)` -- semantic similarity

### Summary of Gaps

**Python features missing from C++ (port gaps):**

1. **Strategy catalog with effectiveness tracking** -- Python has 6 named strategies with EMA-based effectiveness updates and usage statistics. C++ has hardcoded strings in `evaluate_strategy()`.

2. **Knowledge-graph-driven confidence** -- Python computes confidence from KG topology (entity + neighbor + relation). C++ uses simple usage-based tracking. These are complementary approaches.

3. **`estimate_difficulty()`** -- Python can estimate task difficulty from text. C++ has no equivalent.

4. **`assess_readiness()`** -- Python can assess preparation level for a set of topics. C++ has no equivalent.

5. **High-level `MetaAssessor` facade** -- Python has a clean facade combining monitor + strategy. C++ is monolithic.

6. **Serialization** -- Python has `save_state()`/`load_state()`. C++ has no serialization.

**C++ features not in Python (extras in C++):**
- `should_seek_info()`, `generate_query()`, `generate_report()`, `knows_about()`, `topic_coverage()`

### Suggested C++ Implementation Priority

**Priority: MEDIUM** (value: MEDIUM, complexity: LOW-MEDIUM)

The C++ metacognition module already exists and covers the core functionality. The main gaps are:
1. Strategy catalog + effectiveness tracking (straightforward to add)
2. KG-based confidence computation (requires KnowledgeGraph integration)
3. Difficulty estimation (moderate -- needs text-to-entity matching)
4. Readiness assessment (simple averaging)

These can be added incrementally to the existing `MetacognitionEngine` class.
