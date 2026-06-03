# 从学习的本源出发的人工智能

> 不是靠堆参数和数据的"假智能"，而是真正理解"学习"本身是什么。

## 项目愿景

深入研究人类婴儿（0-18岁）和鹦鹉的语言学习过程，提炼学习的本质机制，然后设计一个从学习本源出发的人工智能系统。

## 核心发现

学习的本质是**预测误差最小化**（自由能原理）。语言从交流压力中自发涌现，符号接地不限于视觉——所有感知模态都能接地。

## 项目结构

```
ai-learning-cpp/     ← C++20 生产系统（活跃开发）
├── src/             — 70 个 .cpp 实现文件
├── include/         — 66 个 .hpp 接口定义
├── tests/           — 27 个 Catch2 单元测试
└── CMakeLists.txt   — CMake 构建（CUDA 自动检测）

data/                — Wiki 语料数据资产
archive/             — 历史实验代码（已废弃）
  ├── parallel-learning/  — Python 早期生产代码
  └── mvl/                — Python 实验验证脚本
```

> **Python 代码已归档**：`parallel-learning/` 与 `mvl/` 为历史实验代码，于 2026-06-03 移入 `archive/`。所有生产开发已迁移至 `ai-learning-cpp/`（C++20 实现）。

## 系统架构（C++20）

```
ai_learning::core::Learner — 统一学习体（529 行瘦编排器）
├── 感知层 — MultiModalEncoder（视觉/听觉/位置）
├── 预测编码引擎 — Hebbian 学习（无反向传播）
├── 知识图谱 — Entity-Relation 聚合根（DDD）
├── 记忆系统 — 海马快速 + 皮层慢速 + 睡眠巩固
├── 推理引擎 — 6 种模式（直接/因果/归纳/类比/反事实/概率）
├── 元认知 — 自我评估 / 知识空白检测
├── 语言接地 — GroundingModule + DevelopmentTracker
├── 发展阶段 — Piaget 式 5 阶段自动晋升
├── 目标系统 — GoalManager（分解 + 规划 + 追踪）
├── 嵌入学习 — DistributionalSemantics + EmbeddingTrainer
├── 社会学习 — Society + AgentHandle（多 Agent 协作）
├── REST API — 47 端点 + WebSocket 实时推送
└── CUDA 加速 — 8 个内核（自动 CPU stub 回退）
```

## 项目规模

| 维度 | 数值 |
|------|------|
| C++ 源文件 | 70 .cpp |
| C++ 头文件 | 66 .hpp |
| CUDA 内核 | 8 .cu |
| 测试文件 | 27 .cpp |
| **估算总行数** | **~34,400** |
| 核心编排器 | 529 行 (`learner.hpp`) |
| 最大单文件 | < 800 行（全部合规） |
| 构建系统 | CMake 3.22+ |

## 核心能力

### 1. C++20 预测编码引擎

```cpp
ai_learning::learning::PredictiveCodingEngine engine(config);

// 从预测误差学习（Hebbian，无反向传播）
double error = engine.learn(obs, action, actual_next_obs);

// 好奇心 = 预测误差 × 可学习性
double curiosity = engine.get_curiosity();
```

### 2. 分布语义 + 稠密嵌入

```cpp
// 从文本统计学习概念共现
learner.observe_text("数学是研究数量和结构的学科");

// 训练 Skip-gram 嵌入（自动 CPU/CUDA）
auto result = learner.embedding_trainer().train();

// 语义相似度查询
auto similar = learner.distributional_semantics().similarity("数学", "物理");
```

**预训练模型集成（llama.cpp，可选）：**

```cpp
// 配置预训练嵌入模型路径（bge-small-zh-v1.5 Q4_K_M, ~15MB）
config.embedding_model_path = "/path/to/bge-small-zh-v1.5-q4_k_m.gguf";

// 构造 Learner 时自动加载模型
ai_learning::core::Learner learner(config);

// embedding_trainer() 优先使用预训练模型，无需从零训练
auto vec = learner.embedding_trainer().get_embedding("人工智能");
```

> **推荐**: `bge-small-zh-v1.5 Q4_K_M` (15MB) 是专门的中文嵌入模型，语义判别能力远优于通用 LLM (如 Qwen) 的隐藏层状态。详见 [`LLAMA_CPP_VERIFICATION.md`](ai-learning-cpp/LLAMA_CPP_VERIFICATION.md)。

> 启用：`cmake -DAI_LEARNING_WITH_LLAMA_CPP=ON ..`
> 不启用时现有代码完全不受影响，SGNS 训练照常工作。

### 3. 文本学习与知识图谱

```cpp
// 从文本提取实体、关系、因果、数值
auto result = learner.learn_from_text("牛顿发现了万有引力定律");

// 知识图谱自动增长
std::cout << learner.knowledge_graph().entity_count() << " entities\n";
std::cout << learner.knowledge_graph().relation_count() << " relations\n";

// 思考/回答
std::string answer = learner.think("什么是人工智能");
```

### 4. REST API 服务

```bash
# 启动服务（默认 PC 引擎）
./ai_learning_server --port 8080 --threads 4

# 使用 MLP 引擎 + 本地 LLM（llama.cpp）
./ai_learning_server --engine mlp --llm-model /path/to/qwen2.5-3b-instruct-q4.gguf

# GPU offload 加速（将前 N 层卸载到 GPU，需 CUDA 环境）
# RTX 4060 Laptop 8GB 推荐 10-15 层，可获得 3.5x 加速
./ai_learning_server --llm-model /path/to/model.gguf --gpu-layers 15

# 文本学习
curl -X POST http://localhost:8080/api/learn/text \
  -H "Content-Type: application/json" \
  -d '{"text": "人工智能是计算机科学的一个分支"}'

# 提问
curl -X POST http://localhost:8080/api/think \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是人工智能"}'

# 多轮对话（自动选择后端：llama.cpp > OpenAI API > Stub）
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "教我一首唐诗", "session_id": "demo"}'

# WebSocket 实时统计
ws://localhost:8080/ws/stats
```

### 5. 自主学习循环

```cpp
// 在环境中运行完整闭环
auto result = learner.autonomous_learn(env, 1000, 0, 100);
std::cout << "steps: " << result.total_steps << "\n";
std::cout << "avg_error: " << result.avg_error << "\n";
```

### 6. 多 Agent 社会

```cpp
ai_learning::society::Society society(config);
auto agent1 = society.create_agent();
auto agent2 = society.create_agent();

// 社会观察学习
society.trigger_observation(agent2, agent1, "mathematics");

// 广播知识
society.broadcast_knowledge(agent1, "physics");
```

## 构建

### 要求

- CMake 3.22+
- C++20 编译器（GCC 11+, MSVC 2022+, Clang 14+）
- CUDA Toolkit 12.x（可选，无则自动纯 CPU 构建）

### 快速开始

#### Linux / WSL (Ubuntu 22.04)

```bash
cd ai-learning-cpp
mkdir build-wsl && cd build-wsl
cmake ..
make -j$(nproc)

# 运行主程序
./ai_learning_main

# 运行测试（422 个测试用例，1775 断言）
./ai_learning_tests

# 启动 REST 服务
./ai_learning_server --port 8080
```

> **WSL 已知问题**：`wsl: A localhost proxy configuration was detected but not mirrored into WSL.` 为 WSL NAT 模式代理警告，不影响编译和运行。

## 引擎选择

系统支持三种预测引擎，可通过 `LearnerFactory` 运行时切换：

```cpp
#include "ai_learning/core/learner_factory.hpp"

// 完整预测编码（默认）— 迭代推理 + Hebbian 更新
auto learner1 = ai_learning::core::LearnerFactory::create_default(config);

// 标准 MLP — SGD 反向传播，无迭代推理
auto learner2 = ai_learning::core::LearnerFactory::create_with_engine(
    config, ai_learning::core::LearnerFactory::make_engine("mlp", config));

// 轻量 PC — 单层隐藏层，最快
auto learner3 = ai_learning::core::LearnerFactory::create_with_engine(
    config, ai_learning::core::LearnerFactory::make_engine("light", config));
```

| 引擎 | 隐藏层 | 迭代推理 | 学习算法 | 适用场景 |
|------|--------|----------|----------|----------|
| **pc** | 2 | 是 | Hebbian | 生物启发、好奇心驱动 |
| **mlp** | 2 | 否 | SGD 反向传播 | 标准深度学习基线 |
| **light** | 1 | 否 | Hebbian | 快速、轻量、嵌入式 |

**神经推理增强（llama.cpp，可选）：**

```cpp
// 配置本地 LLM 用于推理增强
config.llm_model_path = "/path/to/qwen2.5-3b-instruct-q4.gguf";

// UnifiedReasoningEngine 自动使用 LLM 补充符号推理
// 当最高符号推理置信度 < 0.5 时，调用 LLM 进行语义推理
auto results = learner.reason("为什么天空是蓝色的");
// results[0].method == "neural" (LLM 补充)
```

**验证状态：** llama.cpp 三阶段集成已通过 Qwen2.5-3B-Instruct Q4_K_M 验证：
- ✅ Phase 1 (Embedding): 语义相似度合理
- ✅ Phase 2 (Dialog): 对话生成连贯
- ✅ Phase 3 (Reasoning): 符号-神经混合推理

详见 [`LLAMA_CPP_VERIFICATION.md`](ai-learning-cpp/LLAMA_CPP_VERIFICATION.md)

**性能基准（WSL, CPU-only）：**

| 引擎 | predict(ns) | learn(ns) | ops/sec (obs=128) |
|------|-------------|-----------|-------------------|
| PC | 5,543 | 501,480 | 2,082 |
| MLP | 5,626 | 12,368 | 40,855 |
| Light | 7,803 | 13,661 | 40,022 |

> PC 因迭代推理（3-10 步收敛）比 MLP/Light 慢 ~20-240x，适合需要精化预测的场景；
> MLP 与 Light 吞吐量在同一量级，Light 结构更简单，MLP 双层表达力更强。
> 运行 `./ai_learning_benchmark_engines` 获取完整数据。

#### Windows (MinGW)

```powershell
cd ai-learning-cpp
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build . -j4
```

## 测试结果

| 问题 | 答案 |
|------|------|
| 什么是人工智能 | 人工智能是计算机科学的一个分支。 |
| 牛顿发现了什么 | 牛顿发现了万有引力定律。 |
| 为什么地面湿了 | 下雨导致地面湿了。 |
| 水在多少度沸腾 | 水的温度是100摄氏度沸腾。 |

## 人类学习研究集成

| 研究发现 | 核心思想 | 论文来源 |
|---------|---------|---------|
| BTSP单次学习 | 资格痕迹+差异化增强 | Quanta Magazine 2026 |
| 学习进度监控 | 选择"够得着的挑战" | Nature Communications |
| 选择性重放 | 高不确定性记忆优先 | 海马体研究 |
| 预测编码Light | 抑制可预测信号 | Nature Communications 2025 |
| 奖励表征后移 | 信用分配时间迁移 | Nature 2026 哈佛 |
| 组合泛化 | 分离what/how+共享子空间 | Nature 2023 MLC |
| 社会偶联学习 | 即时反馈循环 | Royal Society 2026 |
| 符号接地 | 交互式环境因果学习 | arXiv 2026 USC |
| 认知预测路由 | 区分低级/高级误差 | Annual Review 2026 |
| GHL全局调制 | 神经调质信号 | arXiv 2026 |
| 学习进展好奇心 | 探索甜蜜区 | Oudeyer 2026 |
| 先类别后语言 | 感知分类先于语言 | Nature Neuroscience 2026 |
| 元学习组合规则 | 学习如何组合 | Nature 2023 MLC |

## 研究阶段

| 阶段 | 主题 | 状态 |
|------|------|------|
| 1-4 | 婴儿/鹦鹉研究 + 学习本质 + AI缺陷分析 | ✅ |
| 5-6 | 架构设计 + 最小可行学习体 | ✅ |
| 7-8 | 不确定性感知 + 自适应模型选择 | ✅ |
| 9-10 | 语言涌现 + 大规模语言涌现 | ✅ |
| 11-12 | 多Agent社会 + 跨代知识传递 | ✅ |
| 13-15 | 符号接地深化 + 复杂语法 + 否定涌现 | ✅ |
| 16-18 | 从句 + 时间压力 + 叙事 + 元认知 | ✅ |
| 19-20 | 因果推理 + 心智理论 | ✅ |
| 20b-23 | 视角标记 + 抽象推理 + 工具使用 + 跨模态 | ✅ |
| 24 | 统一语言系统（所有符号组合）| ✅ |
| 25 | 自适应策略选择（语言系统学习）| ✅ |
| 26 | 语言与环境探索整合 | ✅ |
| 27 | 多 Agent 共同探索与语言通信 | ✅ |
| 28 | 从婴儿到青少年的完整认知发展路径（8阶段）| ✅ |
| 29 | 真实感官输入（从手工特征到原始像素/声音）| ✅ |
| 30 | 大规模社会（100+ Agent 语言演化）| ✅ |
| 31 | 大概念空间语言分化（874,800 对象 + 区域化 + 噪声）| ✅ |
| 32 | 组合语法涌现（词序冗余性发现）| ✅ |
| 33 | 3D 物理世界（重力、碰撞、工具使用）| ✅ |
| 34 | 复杂感官环境（多形状、连续动作、丰富材质）| ✅ |
| 35 | 社会学习（多 Agent 共享 3D 环境协作学习）| ✅ |
| 36 | 语言 Grounding 深化（从通信游戏到真实指令执行）| ✅ |
| 37 | 指令执行闭环（物理执行 + 协作 + 反馈学习）| ✅ |
| 38 | 迁移学习（符号迁移 + 新概念适应 + 跨域迁移）| ✅ |
| 39 | 词汇引导行为（大概念空间 + 迁移优势 + 学习效率）| ✅ |
| 40 | 维度级词汇引导（维度演化 + 引导对比 + 维度迁移）| ✅ |
| 41 | 复合符号生成（涌现 + 描述效率 + 迁移）| ✅ |
| 42 | 3+ 复合符号 + 符号淘汰（递归复合 + 生命周期）| ✅ |
| 43 | 内部语言（语言作为思维工具）| ✅ |
| 44 | 主动教学（根据学习者调整描述）| ✅ |
| 45 | 文化演化（语言跨代变化）| ✅ |
| 46 | 元语言（语言谈论语言本身）| ✅ |
| 47 | CUDA 加速 + 1000+ Agent 大社会 | ✅ |
| 48 | 流体与软体物理（SPH 粒子 + 弹簧-质点）| ✅ |
| 49 | 超大规模社会（2000-5000 Agent）| ✅ |
| 50 | 跨语言迁移（不同环境的语言互译）| ✅ |
| 51 | 自主目标设定（Agent 自己决定学什么）| ✅ |
| 52 | 课程涌现（歧义度驱动的难度阶梯）| ✅ |
| 53 | 神经科学验证（与真实婴儿脑成像数据对比）| ✅ |
| 54 | 主动推理验证（Expected Free Energy 动作选择）| ✅ |
| 55 | 同伴互学（对称知识交换）| ✅ |
| 56 | 开放式学习（纯内驱力探索）| ✅ |
| 57 | 累积文化创新（踩在巨人肩膀上）| ✅ |
| 58 | 具身隐喻接地（Lakoff 跨域映射测试）| ✅ |
| 59 | 关键期关闭机制（突触可塑性衰减）| ✅ |
| 60 | 理论-代码一致性修正（Hebbian 预测编码替换反向传播）| ✅ |
| 61 | 类比推理驱动的隐喻涌现（结构映射 + 情感空间）| ✅ |
| 62 | 多模态身体经验接地（物理感觉→情感→抽象概念）| ✅ |
| 63 | 反事实推理（if/then/would 标记涌现）| ✅ |
| 64 | 多 Agent 协作规划（角色标记 + 顺序指令）| ✅ |
| 65 | 连续概念空间（模糊类别 + online k-means）| ✅ |
| 66 | 语言驱动记忆（语言作为记忆支架）| ✅ |
| 67 | 语言引导注意力（自上而下感知调制）| ✅ |
| 68 | 对抗性通信与欺骗（信任校准 + 声誉系统）| ✅ |
| 69 | 层级语法（词序承载语义角色 + 递归嵌入）| ✅ |
| 70 | 量化语言（数字 + 计数 + more/less 涌现）| ✅ |
| 71 | 空间关系语言（介词 left/right/above/below 涌现）| ✅ |
| 72 | 好奇心驱动的提问（why/what/how 涌现）| ✅ |
| 73 | 社会规范语言（礼貌 + 禁忌 + should/please 涌现）| ✅ |
| 74 | 非平稳环境适应（词汇淘汰 + 新词涌现 + 记忆保留）| ✅ |
| 75 | 多 Agent 辩论与说服（论证标记 because/but/so 涌现）| ✅ |
| 76 | 道德语言涌现（公平 + 利他信号 + 声誉系统）| ✅ |
| 77 | 幽默与游戏语言（非工具性通信 + 社交凝聚）| ✅ |
| 78 | 通用学习AI — 7个Phase完整实现 | ✅ |
| 79 | 连接孤岛模块 + 打通端到端梯度流 | ✅ |
| 80 | 自改进系统 + 经验反思学习 | ✅ |
| 81 | 统一决策框架 + 世界模型接入 | ✅ |
| 82 | 测试时训练 (In-Place TTT, ICLR 2026) | ✅ |
| 83 | BTSP单次学习 (Quanta Magazine 2026) | ✅ |
| 84 | 修正8个愿景差距 — 7个新引擎 | ✅ |
| 85 | 集成5个人类学习机制 | ✅ |
| 86 | 实现5个最新研究机制 | ✅ |

## 关键论文

1. Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*.
2. Harnad, S. (1990). The symbol grounding problem. *Physica D*.
3. Piaget, J. (1952). *The Origins of Intelligence in Children*.
4. Vygotsky, L. S. (1978). *Mind in Society*.
5. Tomasello, M. (2014). *A Natural History of Human Thinking*.
6. Pepperberg, I. M. (2009). *Alex & Me*.
7. Kuhl, P. K. (2007). Is speech learning 'gated' by the social brain? *Developmental Science*.
8. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
9. Hafner, D. et al. (2025). Mastering Diverse Control Tasks through World Models. *Nature*.
10. Heins et al. (2025). AXIOM: Learning to Play Games in Minutes. *arXiv:2505.24784*.
11. Behrouz et al. (2025). Nested Learning. *NeurIPS 2025*.
12. Scholkopf et al. (2021). Toward Causal Representation Learning. *Proceedings of IEEE*.
13. Mantiuk et al. (2025). From Curiosity to Competence. *arXiv:2507.08210*.
14. Xu et al. (2025). Symbol Grounding. *Nature Human Behaviour*.
15. Lake & Baroni (2023). Human-like systematic generalization through MLC. *Nature*.
16. Oudeyer, P. (2026). Curiosity. *HAL-Inria*.
17. Furutachi & Hofer (2026). Rethinking Predictive Processing. *Annual Review of Neuroscience*.
18. arXiv:2601.21367 (2026). Hebbian Learning with Global Direction.
19. Quanta Magazine (2026). A New Type of Neuroplasticity Rewires the Brain After a Single Experience.
20. Nature Neuroscience (2026). Two-month-old babies are already making sense of the world.
