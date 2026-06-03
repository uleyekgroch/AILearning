# Research: Python Goals Module (src/goals/)

- **Query**: Analyze the goals system in parallel-learning/src/goals/ for C++ port gap analysis
- **Scope**: internal
- **Date**: 2026-06-03

## Findings

### Module Overview

The `src/goals/` package implements a learning goal management system with 4 files, **591 lines total** (excluding `__init__.py`).

| File | Lines | Description |
|---|---|---|
| `parallel-learning/src/goals/__init__.py` | 4 | Re-exports: Goal, GoalStatus, GoalDecomposer, LearningPlan, LearningPlanner, GoalManager |
| `parallel-learning/src/goals/goal.py` | 63 | Goal data model with serialization |
| `parallel-learning/src/goals/decomposer.py` | 129 | Goal decomposition engine |
| `parallel-learning/src/goals/planner.py` | 187 | Learning plan generation with topological sort |
| `parallel-learning/src/goals/manager.py` | 208 | Full lifecycle management of goals |

### Class: `Goal` (goal.py:18-63)

A `@dataclass` representing a learning goal.

**Fields:**
- `id: str` -- unique identifier
- `description: str` -- natural language description
- `status: GoalStatus` -- enum: PENDING / IN_PROGRESS / COMPLETED / BLOCKED / ABANDONED
- `sub_goals: List[str]` -- IDs of child goals
- `parent_goal: Optional[str]` -- parent goal ID (tree structure)
- `required_knowledge: List[str]` -- entity IDs needed to complete
- `priority: float` -- 0.0 to 1.0
- `progress: float` -- 0.0 to 1.0
- `created_step: int` -- creation timestamp

**Methods:** `to_dict()`, `from_dict(data)` -- serialization.

### Class: `GoalDecomposer` (decomposer.py:9-129)

Decomposes a large goal into sub-goals using the knowledge graph.

**Dependencies:** `KnowledgeGraph`, optional `MetaAssessor` (metacognition module).

**Key Methods:**
- `decompose_goal(goal) -> List[Goal]` -- Extracts entities from description via knowledge graph matching, groups by entity type, creates one sub-goal per domain. Falls back to punctuation splitting if no entities found.
- `identify_required_knowledge(goal) -> List[str]` -- Finds knowledge gaps (entities not in KG or confidence < 0.3). Uses metacognition.monitor.identify_knowledge_gaps() if available.

**Internal methods:**
- `_extract_entities(text)` -- Matches KG entity IDs in text (longest-first greedy).
- `_fallback_decompose(goal)` -- Splits by `[,，。、；;]`.

### Class: `LearningPlanner` (planner.py:18-187)

Generates ordered learning steps for a goal.

**Dependencies:** `KnowledgeGraph`, optional `MetaAssessor`.

**Key Methods:**
- `plan_learning(goal) -> LearningPlan` -- Full pipeline: identify gaps -> topological sort -> choose strategy per gap -> generate step list.
- `_topological_sort(gaps)` -- Kahn's algorithm (BFS). Prioritizes higher-confidence gaps first. Handles cycles by appending unsorted.
- `_estimate_gap_difficulty(gap_id)` -- `(1 - entity.confidence) - neighbor_bonus`. Returns 0.9 for unknown entities.
- `_choose_strategy(difficulty, gap_count)` -- Returns one of: `decompose` / `analogize` / `practice` / `explore`.

**Data class: `LearningPlan`** -- fields: `goal_id`, `steps: List[Dict]`, `estimated_effort: float`.

### Class: `GoalManager` (manager.py:11-208)

Full lifecycle manager: create -> decompose -> plan -> execute -> track -> complete.

**Dependencies:** `KnowledgeGraph`, `GoalDecomposer`, `LearningPlanner`, optional `MetaAssessor`.

**Key Methods:**
- `create_goal(description, priority)` -- Creates Goal, auto-identifies required_knowledge via decomposer.
- `decompose_and_plan(goal_id)` -- Decomposes into sub-goals, generates LearningPlan.
- `update_progress(goal_id, progress)` -- Updates progress, propagates to parent via averaging.
- `get_next_action()` -> `Optional[Dict]` -- Priority: 1) in-progress plan step, 2) pending goal (highest priority first).
- `get_active_goals()` -- Returns PENDING + IN_PROGRESS goals.
- `check_completion(goal_id)` -- Checks if all sub-goals completed.
- `save_state()` / `load_state(state)` -- Full serialization/deserialization.

### Inter-Module Dependencies

```
GoalManager --> GoalDecomposer --> KnowledgeGraph
           |                  \-> MetaAssessor (optional)
           \--> LearningPlanner --> KnowledgeGraph
                                \-> MetaAssessor (optional)
```

All three classes depend on `src.knowledge.graph.KnowledgeGraph`. The optional `MetaAssessor` dependency connects to the metacognition module.

### C++ Port Status

**No equivalent exists in C++.** There are no `goals*` or `planner*` files in `ai-learning-cpp/`. This is a completely missing module.

### Suggested C++ Implementation Priority

**Priority: HIGH** (value: HIGH, complexity: MEDIUM)

Rationale:
- Moderate code volume (591 lines, all straightforward logic)
- Provides critical autonomous learning capability (goal-driven behavior)
- Clear separation of concerns (4 files map to 4 classes)
- Dependencies are well-understood (KnowledgeGraph + optional MetacognitionEngine)
- The topological sort in planner is the most complex algorithm but is well-documented
