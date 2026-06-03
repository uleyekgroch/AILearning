# AI Learning C++20 Refactoring Plan

## 项目概况

| 项目 | Python 版 | C++20 版 |
|------|-----------|----------|
| 文件数 | 273 | ~200 (目标) |
| 总行数 | 72,507 | ~40,000 (更紧凑) |
| 核心类 | Learner (5140行 God Class) | ~30 个职责清晰的类 |
| 测试 | 无正式测试 | Catch2 TDD |

## 架构设计：DDD 限界上下文

```
┌─────────────────────────────────────────────────────┐
│                    Application Layer                  │
│              (autonomous_learn, HTTP API)             │
├─────────┬────────┬─────────┬──────────┬─────────────┤
│  Core   │ Know-  │ Learn-  │ Reason-  │   Social    │
│  Learner│ ledge  │ ing     │ ing      │             │
│  (thin) │ Graph  │ Engine  │ Engine   │  Agents     │
│         │ Entity │ Text-   │ Causal   │             │
│ Config  │ Relation│Learner │ Analogy  │ Observation │
│ Registry│ Common │ Stat-   │ Simul-   │ Teaching    │
│ Events  │ sense  | istical | ation    │ Collaborate │
├─────────┴────────┴─────────┴──────────┴─────────────┤
│            Domain Interfaces (7 个抽象接口)            │
├─────────────────────────────────────────────────────┤
│              Infrastructure Layer                      │
│         (Tensor ops, Persistence, Encoding)           │
└─────────────────────────────────────────────────────┘
```

## 7 个限界上下文 (Bounded Context)

| # | 上下文 | 命名空间 | 职责 |
|---|--------|----------|------|
| 1 | Core | `ai_learning::core` | 配置、注册表、事件、Learner 编排 |
| 2 | Perception | `ai_learning::perception` | 多模态编码、可学习编码器、知识提取 |
| 3 | Memory | `ai_learning::memory` | 工作记忆、情景记忆、语义记忆、巩固 |
| 4 | Knowledge | `ai_learning::knowledge` | 知识图谱、实体、关系、常识 |
| 5 | Learning | `ai_learning::learning` | 预测编码、统计学习、BTSP、对比学习 |
| 6 | Language | `ai_learning::language` | 接地、沟通、语言习得、发展阶段 |
| 7 | Reasoning | `ai_learning::reasoning` | 因果推理、类比、模拟推理、统一引擎 |

## TDD 开发计划（按依赖顺序）

### Phase 1: 基础设施层 (✅ 已完成)
- [x] `core/types.hpp` — Tensor, Properties, Stats
- [x] `core/config.hpp` — LearnerConfig, TrainerConfig
- [x] `core/domain_concepts.hpp` — C++20 concepts
- [x] `domain/domain_events.hpp` — 领域事件
- [x] `domain/knowledge/*` — Entity, Relation, KnowledgeGraph, KnowledgeUnit
- [x] `domain/memory/i_memory.hpp` — 记忆接口

### Phase 2: 核心引擎层 (✅ 已完成)
- [x] `core/module_registry.hpp` — 懒初始化模块注册表
- [x] `learning/predictive_coding_engine.hpp` — 预测编码引擎
- [x] `memory/episodic_memory.hpp` — 情景记忆（存储/检索/巩固）
- [x] `learning/stdp_learning.hpp` — STDP 赫布学习
- [x] `reasoning/activation_spread.hpp` — 激活扩散推理
- [x] `learning/text_learner.hpp` — 文本学习管线
- [x] `knowledge/knowledge_extractor.hpp` — 实体/关系提取
- [x] `core/learner.hpp` — 瘦 Learner（组合入口）

### Phase 3: 学习管线 (✅ 已完成)
- [x] `learning/verification.hpp` — 知识验证器（问题生成 + KG 验证）

### Phase 4: 推理与思考 (✅ 已完成)
- [x] `reasoning/simulation.hpp` — 模拟推理（场景构建 + 因果链 + 反事实 + 类比）

### Phase 5: 高阶能力 (✅ 已完成)
- [x] `learning/statistical_learner.hpp` — 统计学习（N-gram + PMI 概念涌现 + 共现关系 + 序列预测）
- [x] `reasoning/unified_engine.hpp` — 统一推理引擎（直接查询 + 因果 + 归纳 + 类比 + 反事实 + 概率推理）

### Phase 6: 语言与进化 (✅ 已完成)
- [x] `language/grounding.hpp` — 符号接地（感知聚类 + 社会标注 + 相似概念检索）
- [x] `language/development.hpp` — 语言发展阶段（感知运动→单字→双字→复杂→读写）
- [x] `core/evolver.hpp` — 自主进化（能力评估 + 弱项识别 + 针对性改进）

### Phase 7: 记忆巩固 (✅ 已完成)
- [x] `learning/hippocampal.hpp` — 海马记忆（快速单次学习 + 容量限制 + 实体索引 + 强度衰减）
- [x] `learning/consolidation.hpp` — 睡眠巩固（海马→皮层转移 + 弱记忆遗忘 + 皮层衰减）
- [x] 集成到 Learner 构造函数

## SOLID 合规性检查

| 原则 | 实现 |
|------|------|
| **S** — 单一职责 | 每个类一个职责，Learner 拆分为 30+ 类 |
| **O** — 开闭原则 | 通过接口扩展，不修改已有代码 |
| **L** — 里氏替换 | 所有接口实现可互换 |
| **I** — 接口隔离 | 7 个小接口，不是一个大接口 |
| **D** — 依赖倒置 | 依赖抽象接口，不依赖具体实现 |

## 行数限制

| 类别 | 限制 | 策略 |
|------|------|------|
| 头文件 | ≤ 800 行 | 前向声明 + 实现分离 |
| 实现文件 | ≤ 800 行 | 拆分大方法为辅助函数 |
| 函数 | ≤ 50 行 | 提取子步骤为独立方法 |
| 方法 | ≤ 30 行 | 单一职责 |
| 类 | ≤ 800 行 | 提取子职责为新类 |

---

## Phase 8-9+ 新增模块 (✅ 已完成)

### Phase 8: CUDA 性能优化
- [x] 8 个 CUDA 内核：tensor_ops, flash_attention, embedding_trainer, stdp_learning, activation_spread, analogical_transfer, active_experimenter, knowledge_graph
- [x] FP16 半精度, Flash Attention 2, sm_89 架构适配
- [x] CPU stub 模式（无 CUDA 时自动降级）

### Phase 9: 多 Agent + LLM + 持续学习
- [x] Society + AgentHandle（多 Agent 社会学习）
- [x] LLM Provider（通义千问 HTTP API）
- [x] DialogManager（多轮对话管理）
- [x] ContinuousLoop（持续在线学习引擎）
- [x] SelfModifier（自修改学习体）
- [x] IntrinsicMotivation（内在动机系统）

### REST API + Web 控制台
- [x] 47 个 REST 端点（Crow 框架）
- [x] WebSocket 实时事件推送
- [x] Vue 3 + ECharts Web 控制台
- [x] 路由组拆分：core_routes, advanced_routes, society_routes, chat_routes, runtime_routes

### Embedding 词向量集成
- [x] DistributionalSemantics (PPMI) → Learner 管线
- [x] EmbeddingTrainer (SGNS) → Learner 管线
- [x] PredictiveCodingEngine 嵌入预测学习
- [x] Wiki 语料验证（15,820 概念，语义相似度 0.91+）

### 语言无关分词
- [x] Tokenizer 模块（Chinese/English/Auto 检测）
- [x] StatisticalLearner::observe_tokens() 预分词接口
- [x] EmbeddingTrainer::add_tokens() 预分词接口

## 当前代码规模 (2026-06-03)

| 维度 | 数据 |
|------|------|
| 源文件 | 55 .cpp + 56 .hpp + 8 .cu |
| CUDA 代码 | 4,388 行 |
| 测试文件 | 24 个，7,000+ 行 |
| REST 端点 | 47 个 |
| 最大文件 | route_groups.hpp (89行，server 拆分后) |
| learner.cpp | 675 行（拆分自 964 行） |

## 待改进

### 监控项
- `distributional_semantics.cpp` (675行) — 接近限制
- `abstract_concept.cpp` (651行) — 接近限制

### 功能扩展（Python 有/C++ 无）
- `goals/` — 目标系统 (goal, decomposer, planner, manager)
- `metacognition/` — 独立元认知 (monitor, strategy, assessor)
- `validation/neuroscience` — 神经科学验证
