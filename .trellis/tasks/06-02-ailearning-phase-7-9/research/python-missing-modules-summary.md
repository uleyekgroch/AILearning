# Research: Python Modules Missing from C++ Port

- **Query**: Identify all Python modules in parallel-learning/src/ that have no C++ equivalent
- **Scope**: internal
- **Date**: 2026-06-03

## Methodology

Compared the directory structure and file names of:
- Python: `parallel-learning/src/` (20 top-level module directories)
- C++: `ai-learning-cpp/include/ai_learning/` + `ai-learning-cpp/src/` (6 domain + 5 top-level directories)

A module is considered "missing" if there is no file in C++ with a matching or obviously equivalent name.

## Python Module Inventory vs C++ Status

### Core Infrastructure

| Python Module | Files | Lines | C++ Status | Notes |
|---|---|---|---|---|
| `src/core/` | 16 | 8,450 | **PARTIAL** | C++ has `learner.hpp/cpp`, `config.hpp`, `evolver.hpp/cpp`. Missing: `human_learning_system.py` (422), `human_learning_refactor.py` (178), `human_refactor_modules.py` (211), `production_system.py` (350), `learning_engine.py` (321), `nonstationary.py` (310), `motivation.py` (411), `plasticity.py` (155), `serializable.py` (42), `registry.py` (40), `device.py` (25), `interfaces.py` (131), `learner_streamlined.py` (146) |
| `src/data/` | 2 | 3,859 | **MISSING** | `oxford_words.py` (2705 lines word list), `source.py` (1154 lines text sources). Data-only module. |

### Domain Modules (with C++ equivalents)

| Python Module | Files | Lines | C++ Status | Notes |
|---|---|---|---|---|
| `src/knowledge/` | 10 | 3,225 | **PARTIAL** | C++ has entity, relation, knowledge_graph, knowledge_unit. Missing: `commonsense.py` (525), `crud.py` (617), `graph_db.py` (424), `schema.py` (660), `vector_db.py` (318), `bridge.py` (124), `unit.py` (203) |
| `src/language/` | 12 | 4,927 | **PARTIAL** | C++ has grounding, development, dialog_manager, llm_provider. Missing: `pragmatics.py` (836), `emergence.py` (556), `communication.py` (730), `regional.py` (469), `narrative.py` (408), `memory_scaffold.py` (351), `attention.py` (278), `continuous_concepts.py` (293), `crossmodal.py` (310), `inner_speech.py` (258), `grammar.py` (182) |
| `src/learning/` | 47 | 17,447 | **PARTIAL** | C++ has ~20 learning modules. Missing ~27 files. See separate analysis below. |
| `src/memory/` | 5 | 893 | **PARTIAL** | C++ has episodic_memory, hippocampal, consolidation. Missing: `system.py` (120), `semantic.py` (152), `working.py` (72). These are small. |
| `src/reasoning/` | 27 | 8,921 | **PARTIAL** | C++ has simulation, activation_spread, unified_engine, world_model. Missing ~23 files. See below. |
| `src/social/` | 9 | 1,739 | **PARTIAL** | C++ has social_agent, social_learning, society. Missing: `adversarial.py` (401), `planning.py` (364), `cultural.py` (269), `collaboration.py` (165), `teaching.py` (159), `observation.py` (167), `interaction.py` (99), `norms.py` (59). |
| `src/perception/` | 9 | 2,497 | **PARTIAL** | C++ has multimodal_encoder, knowledge_extractor. Missing: `learnable_encoder.py` (465), `object_world_model.py` (329), `embodied_grounding.py` (327), `multimodal_engine.py` (222), `encoder.py` (219), `text_encoder.py` (214), `auditory.py` (167), `visual.py` (146). |
| `src/environment/` | 6 | 2,335 | **PARTIAL** | C++ has simple_environment. Missing: `world_3d.py` (1238), `physics_3d.py` (496), `physics.py` (145), `world.py` (166), `objects.py` (84), `objects_3d.py` (206). |
| `src/curriculum/` | 3 | 382 | **PARTIAL** | C++ has i_curriculum. Missing: `evaluator.py` (136), `scheduler.py` (115), `stages.py` (131). |

### Domain Modules (completely missing from C++)

| Python Module | Files | Lines | C++ Status | Notes |
|---|---|---|---|---|
| `src/goals/` | 4 | 591 | **MISSING** | Entire goals system. See dedicated research file. |
| `src/validation/` | 1 | 246 | **MISSING** | Neuroscience validation. See dedicated research file. |
| `src/assessment/` | 2 | 627 | **MISSING** | `mastery.py` (318), `proficiency.py` (309). Assessment/tracking of learning mastery and proficiency levels. |
| `src/content/` | 3 | 206 | **MISSING** | `analyzer.py` (119), `loader.py` (68), `text_unit.py` (19). Text content loading and analysis. |
| `src/skills/` | 4 | 628 | **MISSING** | `grammar_ex.py` (201), `listening.py` (154), `reading.py` (139), `writing.py` (134). Language skill exercises. |
| `src/true_ai/` | 1 | 190 | **MISSING** | `perception.py` (190). True AI perception module. |

### Production System (no C++ equivalent intended)

| Python Module | Files | Lines | C++ Status | Notes |
|---|---|---|---|---|
| `src/production/` | 65 | 13,207 | **N/A** | Full production application layer with REST APIs, microservices, persistence, distributed infrastructure. Not a C++ port target. |
| `src/ai/` | 6 | 1,282 | **N/A** | Higher-level AI system orchestration. Likely not a port target. |

### Significant Missing Learning Sub-modules

The Python `src/learning/` has 47 files (17,447 lines). C++ has ~20 learning modules. Key missing ones:

| Python File | Lines | Description | C++ Equivalent? |
|---|---|---|---|
| `core_knowledge.py` | 967 | Core knowledge systems (Spelke) | No |
| `statistical_learner.py` | 948 | Statistical learning (C++ has statistical_learner.hpp) | **Yes, exists** |
| `contrastive_trainer.py` | 601 | Contrastive learning | No |
| `universal_system.py` | 604 | Universal learning system | C++ has `universal_learner.py` (401) -- partial |
| `language_acquisition.py` | 531 | Language acquisition | No |
| `lifelong_learning.py` | 570 | Lifelong learning | C++ has `lifelong_learning.hpp` -- **exists** |
| `concept_former.py` | 289 | Concept formation | No |
| `concept_space.py` | 422 | Concept space | No |
| `cognitive_mechanisms.py` | 523 | Cognitive mechanisms | No |
| `dendritic_computation.py` | 382 | Dendritic computation | No |
| `sleep_replay.py` | 425 | Sleep replay consolidation | No |
| `active_inference.py` | 361 | Active inference learning | No |
| `metacognitive_regulation.py` | 277 | Metacognitive regulation | Partially in C++ metacognition |
| `metacognitive_system.py` | 352 | Metacognitive system | Partially in C++ metacognition |
| `simulation_reasoning.py` | 401 | Simulation-based reasoning | C++ has simulation.hpp -- **exists** |
| `complementary_learning.py` | 468 | Complementary learning systems | No |
| `neuro_symbolic.py` | 407 | Neuro-symbolic integration | No |
| `perception_learning_loop.py` | 553 | Perception-learning loop | No |
| `self_evolution.py` | 295 | Self-evolution | C++ has self_modifier.hpp -- partial |
| `world_model.py` | 504 | World model | C++ has world_model.hpp -- **exists** |

### Significant Missing Reasoning Sub-modules

Python `src/reasoning/` has 27 files (8,921 lines). C++ has 4 reasoning files. Key missing:

| Python File | Lines | Description |
|---|---|---|
| `integrated_reasoning.py` | 601 | Integrated reasoning engine |
| `gnn_engine.py` | 677 | Graph neural network reasoning |
| `unified_causal.py` | 613 | Unified causal reasoning |
| `probabilistic.py` | 485 | Probabilistic reasoning |
| `metaphor.py` | 459 | Metaphor understanding |
| `integrated_engine.py` | 337 | Integrated reasoning engine |
| `unified_system.py` | 380 | Unified reasoning system |
| `causal_unified.py` | 344 | Unified causal reasoning |
| `hypothesis.py` | 320 | Hypothesis management |
| `questioning.py` | 328 | Question generation |
| `code_generator.py` | 257 | Code generation |
| `theory_of_mind.py` | 217 | Theory of mind |
| `tool_use.py` | 236 | Tool use reasoning |
| `creativity_engine.py` | 277 | Creative reasoning |
| `counterfactual.py` | 207 | Counterfactual reasoning |

## Prioritized Implementation Recommendations

Based on value (core learning capability) vs complexity (code volume, dependencies):

### Priority 1 -- HIGH value, manageable complexity

| Module | Lines | Rationale |
|---|---|---|
| **goals/** (entire module) | 591 | Goal-driven learning is core to autonomous behavior. Clean 4-class design, clear dependencies. |
| **assessment/** (mastery + proficiency) | 627 | Essential for tracking learning progress. Self-contained, no exotic dependencies. |

### Priority 2 -- MEDIUM-HIGH value

| Module | Lines | Rationale |
|---|---|---|
| **content/** (analyzer + loader + text_unit) | 206 | Text input pipeline. Small module, needed for learn_from_text. |
| **skills/** (reading + writing + listening + grammar) | 628 | Language skill exercises. Medium size, important for language development. |
| **curriculum/** evaluator + scheduler + stages | 382 | Curriculum management. C++ has the interface but no implementations. |

### Priority 3 -- MEDIUM value

| Module | Lines | Rationale |
|---|---|---|
| **validation/neuroscience.py** | 246 | Validation tool. Low complexity but requires scipy equivalent. |
| **memory/** (system + semantic + working) | 344 | Memory system completion. Small files. |

### Priority 4 -- Lower priority (large volume, specialized)

| Module | Lines | Rationale |
|---|---|---|
| Various learning/ sub-modules | ~8000+ | Many are specialized algorithms. Port selectively based on need. |
| Various reasoning/ sub-modules | ~6000+ | Many reasoning variants. Port selectively. |

## Caveats / Not Found

- The `src/production/` directory (13,207 lines) is a full production application layer and was excluded from the gap analysis as it appears to be a separate deployment target.
- The `src/ai/` directory (1,282 lines) is a high-level orchestration layer that may or may not be a port target.
- Some Python files in learning/ and reasoning/ may duplicate functionality that already exists in C++ under different names. A detailed function-level comparison was not performed for these large modules.
- Line counts include comments and blank lines, so actual code volume is lower.
