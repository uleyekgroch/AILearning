# 项目交接文档 — AILearning 项目全貌

> 本文档供 Claude 接手后续开发使用，包含完整项目上下文、架构、已完成工作、下一步计划。

## 一、项目概览

**项目名称**: AILearning（仿人类学习系统）  
**目标**: 从第一性原理出发，构建一个模拟人类认知发展的 AI 学习系统  
**核心理念**: 不是训练 LLM，而是构建一个能像人类婴儿→儿童→成人那样**自主认知发展**的学习体  
**仓库位置**: `D:\mayAi\AILearning_v0527`

### 项目结构

```
AILearning_v0527/
├── CLAUDE.md              # 开发规范（DDD/TDD/SOLID，800行限制）
├── AGENTS.md              # Agent 工作流配置
├── README.md              # 项目说明
├── ai-learning-cpp/       # ★ C++ 核心实现（主要开发重心）
│   ├── CMakeLists.txt     # CMake 构建（MinGW Makefiles / CUDA）
│   ├── include/ai_learning/
│   │   ├── core/          # 核心层：Learner、Config、Evolver
│   │   ├── domain/        # 领域层：知识图谱、环境、感知
│   │   ├── learning/      # 学习层：所有学习算法
│   │   ├── memory/        # 记忆层：情景记忆
│   │   ├── reasoning/     # 推理层：激活扩散、模拟推理
│   │   └── language/      # 语言层：接地、发展追踪
│   ├── src/               # 实现文件（与 include 镜像）
│   ├── tests/             # Catch2 v3 测试（276 个测试用例）
│   ├── build-win/         # Windows 构建（MinGW）
│   └── build-cuda/        # CUDA 构建（WSL/Linux）
├── parallel-learning/     # Python 原型（已完成，不再主要开发）
└── tasks/                 # 任务跟踪文档
```

---

## 二、C++ 架构（核心）

### 2.1 架构模式

- **DDD 分层架构**: `domain → learning → reasoning → memory → language → core`
- **聚合根**: `Learner` 类是唯一的事务边界（`core/learner.hpp`）
- **组合优于继承**: 所有子系统通过值语义成员变量持有
- **C++20 标准**: 使用 Concepts、`std::variant`、`std::optional`、范围 for

### 2.2 Learner 类（编排器）

`Learner` 是约 440 行的瘦编排器，持有 **25 个子系统**：

| 阶段 | 子系统 | 头文件 |
|------|--------|--------|
| **基础** | KnowledgeGraph, PredictiveCodingEngine, TextLearner, STDP, EpisodicMemory | `domain/`, `learning/`, `memory/` |
| **推理** | ActivationSpread, SimulationReasoning, UnifiedReasoningEngine, WorldModel | `reasoning/` |
| **语言** | GroundingModule, DevelopmentTracker, MultiModalEncoder | `language/`, `perception/` |
| **记忆** | HippocampalMemory, CorticalMemory, SleepConsolidation | `learning/` |
| **自主** | IntrinsicMotivationEngine, SkillTree, DevelopmentMilestones, ProblemSolver | `learning/` |
| **核心能力** | SelfModifier, MetacognitionEngine, LocalProcessSandbox | `core/`, `learning/` |
| **Phase 3** | AnalogicalTransferEngine, ContinualLearner, AbstractConceptEngine | `learning/` |
| **Phase 4** | SocialLearningEngine, EmotionEngine, InsightEngine | `learning/` |
| **Phase 5** | MetaLearner, ActiveExperimenter | `learning/` |

### 2.3 核心学习闭环

```
observe → perceive → predict → choose_action → learn → remember
```

---

## 三、五大阶段完成详情

### Phase 1-2: 基础学习 + 自主学习

| 模块 | 核心论文/理论 | 关键能力 |
|------|-------------|---------|
| PredictiveCodingEngine | Friston 自由能原理 | 预测-误差驱动学习 |
| TextLearner | — | 从文本提取知识 |
| StatisticalLearner | 概率学习 | 共现统计、概念涌现 |
| HippocampalMemory + CorticalMemory | Complementary Learning Systems | 快速编码 + 慢速巩固 |
| SleepConsolidation | 记忆重放 | 睡眠期间记忆整合 |
| IntrinsicMotivationEngine | Deci & Ryan 自我决定论 | 好奇心、掌握欲、自主性 |
| SkillTree | — | 技能依赖图、解锁机制 |
| DevelopmentMilestones | 发展心理学 | 阶段性里程碑检测 |
| SelfModifier | Darwin Gödel Machine (Sakana AI, 2025) | 自我修改学习参数 |
| MetacognitionEngine | 元认知理论 | 知道自己知道什么 |

**Commit**: `e7d19b4` feat: 实现自主学习六大模块

### Phase 3: 高级认知能力

| 模块 | 核心论文 | 关键能力 |
|------|---------|---------|
| **AnalogicalTransferEngine** | Gentner 结构映射理论 (1983) | 属性匹配、关系对齐、知识迁移 |
| **ContinualLearner** | EWC 弹性权重巩固 (Kirkpatrick, 2017) | Fisher 重要性、遗忘检测、经验回放 |
| **AbstractConceptEngine** | Rosch 原型理论 (1978) | 原型抽象、层次化概念、概念分化 |

**核心算法**:
- 类比迁移: Jaccard 属性重叠 + 关系动词匹配 → 贪心对齐
- 持续学习: `importance = confidence × (1 + log(1+usage)/5)`, 四级保护
- 抽象概念: 核心属性（≥60%实例出现）→ 原型特征 → 强度评分

**测试**: 215 测试 / 1233 断言全绿  
**Commit**: 记录在 `tasks/todo.md`

### Phase 4: 增强智能

| 模块 | 核心论文 | 关键能力 |
|------|---------|---------|
| **SocialLearningEngine** | Bandura 社会学习理论 (1977) | 观察→注意→保持→再现→动机五阶段 |
| **EmotionEngine** | McGaugh 情绪-记忆 (2004) + Schultz 多巴胺 (1997) | 效价×唤醒度×支配度, RPE, Yerkes-Dodson |
| **InsightEngine** | Ohlsson 顿悟 (2011) + Mednick 远程联想 (1962) | 约束释放、远程联想、视角转换 |

**核心算法**:
- 社会学习: 榜样评估(EMA α=0.3) + 模式提取(≥50%频率) + 策略模仿
- 情感引擎: `encoding_boost = 1 + arousal×0.5 + |valence|×0.3`, 倒U型绩效
- 顿悟: 远程联想检测 + 约束释放(strength < threshold) → 创造性评估(新颖度×实用性)

**跨Phase集成**: observe_behavior() → ContinualLearner 保护; try_insight() → EmotionEngine 兴奋事件  
**测试**: 251 测试 / 1301 断言全绿  
**Commit**: `f7b3330` feat: 实现Phase 4增强智能三大模块

### Phase 5: 高级元认知 ✅

| 模块 | 核心论文 | 关键能力 |
|------|---------|---------|
| **MetaLearner** | MAML (Finn, 2017) + RL² (Duan, 2016) | 策略评估推荐、学习率调度、跨域迁移、自我反思 |
| **ActiveExperimenter** | Bayesian Exp Design (Chaloner, 1995) | 假设生成、实验设计、贝叶斯更新、理论构建、探索策略 |

**核心算法**:
- 元学习: ε-贪心策略选择 + 任务-策略匹配评分 + 领域亲和度追踪
- 学习率调度: `rate × (1 - difficulty×0.5) × (1 - novelty×0.3) × (0.8 + urgency×0.4)`
- 跨域迁移: Jaccard 策略相似度评估迁移潜力
- 主动实验: 信息论实验设计 → 贝叶斯假设置信度更新 → 理论归纳
- 信息增益: `H(prior) - E[H(posterior)]` 二值熵优化

**文件清单**:

| 类型 | 文件 |
|------|------|
| 头文件 | `include/ai_learning/learning/meta_learner.hpp` |
| 头文件 | `include/ai_learning/learning/active_experimenter.hpp` |
| 实现 | `src/learning/meta_learner.cpp` |
| 实现 | `src/learning/active_experimenter.cpp` |
| 测试 | `tests/learning/test_phase5_meta_cognition.cpp` |

**Commit**: `e160ff7`

### Phase 6: 深度整合 ✅ 最新完成

| 整合场景 | 编排方法 | 串联模块 |
|----------|---------|---------|
| **全流水线闭环** | `run_full_pipeline()` | observe→learn→analogize→experiment→reflect→insight |
| **元学习驱动** | `meta_guided_session()` | IntrinsicMotivation + MetaLearner + EmotionEngine |
| **情感调制** | `compute_system_params()` | EmotionEngine → 全系统学习率/编码/探索参数 |
| **实验驱动探索** | `experiment_driven_exploration()` | ActiveExperimenter → IntrinsicMotivation 目标 |
| **社会加速类比** | `social_analogical_transfer()` | SocialLearning → AnalogicalTransfer → ContinualLearner |

**核心设计**:
- `IntegratedLearner` 持有 9 个子系统引用（Phase 3-5 全部模块）
- 六步闭环中每步成功/失败独立追踪，不因单步失败中断
- 顿悟自动触发兴奋情绪（正反馈循环）
- Yerkes-Dodson 倒U型学习率调制
- 社会观察提取的策略自动转化为类比迁移源概念

**文件清单**:

| 类型 | 文件 | 行数 |
|------|------|------|
| 头文件 | `include/ai_learning/learning/integrated_learner.hpp` | 187 |
| 实现 | `src/learning/integrated_learner.cpp` | 371 |
| 测试 | `tests/learning/test_phase6_integration.cpp` | 458 |
| 集成 | `include/ai_learning/core/learner.hpp`（修改） | +33 行 |
| 集成 | `src/core/learner.cpp`（修改） | +43 行 |

**Commit**: `58cb1b3`

---

## 四、构建与运行

### 构建命令（Windows / MinGW）

```bash
cd ai-learning-cpp/build-win
cmake .. -G "MinGW Makefiles"
cmake --build .
```

### 运行测试

```bash
# 全量测试
.\ai_learning_tests.exe --reporter compact

# 只跑 Phase 5
.\ai_learning_tests.exe "[phase5]"

# 只跑 Phase 4
.\ai_learning_tests.exe "[phase4]"

# 只跑 Phase 3
.\ai_learning_tests.exe "[phase3]"
```

### CUDA 构建（WSL/Linux）

```bash
cd ai-learning-cpp/build-cuda
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 五、测试标签体系

| 标签 | 覆盖 |
|------|------|
| `[phase3]` | 类比迁移、持续学习、抽象概念 (29 测试) |
| `[phase4]` | 社会学习、情感引擎、顿悟引擎 (36 测试) |
| `[phase5]` | 元学习、主动实验设计 (25 测试) |
| `[integration]` | 跨模块集成测试 |
| `[meta_learner]` | MetaLearner 单元测试 |
| `[experimenter]` | ActiveExperimenter 单元测试 |

---

## 六、开发注意事项（踩过的坑）

### 6.1 C++20 关键字冲突
- `concept` 是 C++20 保留关键字，变量名必须用 `cpt` 或 `prototype` 代替
- 在 `abstract_concept.hpp/cpp` 中已处理此问题

### 6.2 成员变量与成员函数同名
- `next_hypothesis_id_` 与 `next_hyp_id_()` 方法名冲突
- 解决：ID 计数器命名为 `hypothesis_id_counter_`、`experiment_id_counter_`、`theory_id_counter_`

### 6.3 CMake GLOB_RECURSE 需要重新 configure
- 新增 `.cpp` 文件后，必须重新运行 `cmake ..` 才会被发现
- 只 `cmake --build .` 不会编译新文件

### 6.4 Catch2 浮点断言
- 使用 `WithinAbs(value, epsilon)` 和 `WithinRel(value, tolerance)` 代替 `==`
- 不要用 `CHECK(a == b)` 比较浮点数

### 6.5 代码规范
- 单文件不超过 800 行
- 值语义成员变量（不用指针/unique_ptr）
- 所有方法返回 `auto` + 尾置返回类型
- 私有方法以 `_` 后缀命名（如 `bayesian_update_`）

---

## 七、下一步可做的工作

Phase 1-6 全部完成。以下是未来方向：

### 7.1 应用层（推荐优先）

1. **Python 绑定** — pybind11 暴露 C++ API 给 Python
2. **REST API** — 暴露学习接口为 HTTP 服务
3. **可视化** — 学习曲线、知识图谱、情感状态可视化
4. **实际语料学习** — 接入 Wikipedia/教科书语料进行真实学习实验

### 7.2 性能优化

1. **CUDA 加速核心算法** — 类比对齐、贝叶斯更新、信息增益计算
2. **批量经验处理** — MetaLearner 批量更新策略统计
3. **并行假设评估** — ActiveExperimenter 并行计算多个假设的信息价值

### 7.3 高级扩展

1. **多 Agent 协作** — 多个 Learner 实例组成社会，互相观察
2. **语言接口** — 自然语言输入→学习→自然语言输出
3. **持续在线学习** — 永不停止的学习循环
4. **元认知透明化** — 让系统解释"为什么选择这个策略"

---

## 八、Git 提交历史（最近 6 个）

```
58cb1b3 feat: 实现Phase 6深度整合 - 五大跨模块协同编排器
e160ff7 feat: 实现Phase 5高级元认知 - 元学习引擎+主动实验设计
f7b3330 feat: 实现Phase 4增强智能三大模块 - 社会学习/情感驱动/顿悟
e7d19b4 feat: 实现自主学习六大模块 - 让系统主动学习万物
7754b61 feat: 实现四大人类核心学习能力 — 动手做/自我修改/世界模型/元认知
97da6e5 refactor: 从第一性原理重构 - 真正的AI系统
```

---

## 九、项目统计

| 指标 | 数值 |
|------|------|
| C++ 头文件 | 54 个 |
| C++ 源文件 | 40 个 |
| 测试文件 | 17 个 |
| 测试用例 | **291** 个 |
| 断言 | **1450** 个 |
| Learner 持有子系统 | 26 个 |
| Git 提交数 | ~20+ |

---

## 十、关键文件快速索引

| 需求 | 文件 |
|------|------|
| 看整体架构 | `ai-learning-cpp/include/ai_learning/core/learner.hpp` |
| 看构建配置 | `ai-learning-cpp/CMakeLists.txt` |
| 看试试规范 | `CLAUDE.md` |
| 看 Phase 3 详情 | `tasks/todo.md` |
| 看 Phase 4 详情 | `tasks/todo_phase4.md` |
| 看最新模块 | `ai-learning-cpp/include/ai_learning/learning/meta_learner.hpp` |
| 看最新模块 | `ai-learning-cpp/include/ai_learning/learning/active_experimenter.hpp` |
| 看最新测试 | `ai-learning-cpp/tests/learning/test_phase5_meta_cognition.cpp` |
