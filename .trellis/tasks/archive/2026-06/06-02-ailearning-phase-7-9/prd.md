# AILearning Phase 7-9 全路线图规划

## Goal

为已完成的 Phase 1-6 C++ 核心学习系统规划后续三大方向：应用层暴露 → 性能优化 → 高级扩展。产出一份分阶段、可执行、带依赖关系的完整路线图。

---

## Decisions Summary

| ID | 决策 | 选择 |
|----|------|------|
| D1 | 阶段划分 | 方向串行：Phase 7→8→9 |
| D2 | 应用层暴露 | REST API（crow + nlohmann/json） |
| D3 | 可视化 | Web 交互式控制台（Vue 3 + ECharts + WebSocket） |
| D4 | CUDA 优化 | 全部 6 算法迁移 + FP16/Flash Attention/批量处理 |
| D5 | 高级扩展 | 全部三方向：多Agent→语言接口→持续学习 |

---

## Phase 7: 应用层（预计 4-6 周）

> **目标**：让 C++ 核心能力通过 HTTP API 对外可用，并提供 Web 交互式控制台

### 7.1 项目基础设施 + REST 框架搭建（1 周）

**输入**: 现有 CMakeLists.txt、Learner 类
**输出**: crow 集成完成、基础 HTTP 服务可启动

| 任务 | 详情 |
|------|------|
| CMake 集成 crow | FetchContent 拉取 crow（header-only），新增 `ai_learning_server` target |
| CMake 集成 nlohmann/json | FetchContent 拉取，JSON 序列化/反序列化工具 |
| 服务骨架 | `src/server/` 目录：`server.hpp/cpp`，启动、路由注册、优雅关闭 |
| 健康检查端点 | `GET /api/health` 返回服务状态 |
| 配置管理 | `server_config.hpp`：端口、线程数、CORS 等 |

**验收标准**:
- [x] `ai_learning_server` 编译通过
- [x] `GET /api/health` 返回 200
- [x] 优雅关闭（Ctrl+C）

**依赖**: crow Windows/MinGW 兼容性验证（风险点）
**参考**: `research/rest-api-frameworks.md`

### 7.2 核心 API — 学习与推理端点（1.5 周）

**输入**: Learner 的 Phase 1-2 方法（文本学习、感知、记忆、自主）
**输出**: 核心学习流程可通过 HTTP 调用

| 端点 | 方法 | Learner 调用 |
|------|------|-------------|
| `POST /api/learn/text` | 学习文本 | `learn_from_text()` |
| `POST /api/observe` | 观察输入 | `observe_text()` |
| `POST /api/reason` | 推理问答 | `reason()` |
| `POST /api/think` | 深度思考 | `think()` |
| `POST /api/perceive` | 感知处理 | `perceive()` |
| `POST /api/remember` | 记忆存储 | `remember()` |
| `POST /api/recall` | 记忆检索 | `recall()` |
| `POST /api/consolidate` | 记忆巩固 | `consolidate()` |
| `POST /api/autonomous` | 自主学习循环 | `autonomous_learning_run()` |
| `GET /api/stats` | 学习统计 | `get_stats()` |
| `GET /api/stage` | 发展阶段 | `stage()` |
| `POST /api/save` | 持久化 | `save()` |
| `POST /api/load` | 加载状态 | `load()` |

**关键设计**:
- 请求/响应类型定义：`src/server/dto/` 目录（DTO 模式）
- Learner 单例管理：一个服务进程持有一个 Learner 实例
- 异步任务：长时间运行的操作（自主学习循环）返回 task_id，通过 `GET /api/tasks/{id}` 轮询

**验收标准**:
- [x] 13 个端点全部可调通
- [x] 自主学习循环不阻塞服务（异步执行）
- [x] JSON 序列化/反序列化正确（嵌套 map、vector）

### 7.3 高级 API — Phase 3-6 端点（1 周）

**输入**: Learner 的 Phase 3-6 方法
**输出**: 全部高级认知能力可通过 HTTP 调用

| 端点组 | 方法 |
|--------|------|
| 类比迁移 | `POST /api/analogize` |
| 持续学习 | `POST /api/protect`, `GET /api/forgetting` |
| 抽象概念 | `POST /api/abstract` |
| 社会学习 | `POST /api/observe-behavior` |
| 情感处理 | `POST /api/emotion` |
| 顿悟触发 | `POST /api/insight` |
| 元认知 | `POST /api/meta/recommend`, `POST /api/meta/reflect` |
| 实验设计 | `POST /api/experiment/design`, `POST /api/experiment/record` |
| 整合流水线 | `POST /api/integrated/pipeline` |

**验收标准**:
- [x] Phase 3-6 全部公共方法有对应端点
- [x] 返回结构体完整映射为 JSON

### 7.4 WebSocket 事件流（1 周）

**输入**: `IEventPublisher` 接口（domain_events.hpp）
**输出**: 学习过程实时推送

| 任务 | 详情 |
|------|------|
| 事件适配器 | 实现 `WebSocketEventPublisher`，将 Learner 内部事件转发到 WebSocket |
| 事件协议定义 | JSON 格式：`{type, timestamp, data}` |
| 频道设计 | `/ws/events`（全部事件）、`/ws/stats`（统计摘要） |
| 前端连接管理 | 心跳、断线重连、事件缓冲 |

**验收标准**:
- [x] 学习过程事件实时推送到 WebSocket 客户端
- [x] 断线后可重连

**依赖**: `IEventPublisher` 接口的现有实现情况（需检查是否已接入 Learner 各子系统）

### 7.5 Web 交互式控制台（1.5 周）

**输入**: REST API + WebSocket 事件流
**输出**: Vue 3 单页面应用

| 模块 | 功能 |
|------|------|
| **仪表盘** | 学习进度曲线（ECharts 折线图）、知识节点数、发展阶段 |
| **情感面板** | 效价/唤醒度/支配度雷达图、情绪历史 |
| **知识图谱** | 力导向图（ECharts graph），可点击节点查看详情 |
| **操作面板** | 文本输入→学习、问答对话、触发推理/顿悟/类比 |
| **实验面板** | 假设列表、实验设计、贝叶斯置信度更新可视化 |
| **系统状态** | 自主学习循环状态、里程碑、技能树进度 |

**技术栈**: Vue 3 + Vite + TypeScript + ECharts + WebSocket
**部署**: C++ 服务嵌入静态文件（crow 的 `serve_static`），单进程部署

**验收标准**:
- [x] 浏览器打开可看到实时学习数据
- [x] 可通过操作面板触发学习、推理、顿悟
- [x] 知识图谱可视化可交互（点击、缩放）

**参考**: `research/visualization-options.md`

---

## Phase 8: CUDA 性能优化（预计 4-5 周）

> **目标**：将核心学习算法迁移到 GPU，利用 RTX 4060 Ada 的 Tensor Core 和 Flash Attention

### 前置：CUDA 环境升级（0.5 周）

| 任务 | 详情 |
|------|------|
| sm_89 升级 | CMakeLists.txt 将 sm_86 → sm_89 |
| CUDA 12.x 验证 | 确认编译环境支持 Flash Attention 所需的 CUDA 版本 |
| cuBLAS/cuRAND 集成 | 矩阵运算和随机数生成基础 |
| 统一张量抽象 | `tensor_ops.hpp` 扩展：CPU/GPU 统一接口，运行时选择设备 |

**验收标准**:
- [x] sm_89 编译通过
- [x] 现有测试不受影响（CPU fallback 正常）

### 8.1 FP16 混合精度基础设施（1 周）

| 任务 | 详情 |
|------|------|
| FP16 类型封装 | `cuda_utils.hpp`：`half` 类型包装，自动精度转换 |
| Tensor Core GEMM | 使用 `cublasGemmEx` 启用 FP16 Tensor Core 加速 |
| 精度控制策略 | 关键路径 FP32（知识查询精确匹配），计算密集路径 FP16 |
| 批量内存管理 | 统一的 GPU 内存池，减少 cudaMalloc 开销 |

**验收标准**:
- [x] FP16 矩阵乘法正确性验证（vs CPU FP32 基准）
- [x] Tensor Core 加速比 ≥ 2×

### 8.2 算法迁移（3 周，每算法 3-4 天）

**迁移顺序和策略**（按数据依赖排列）:

| # | 算法 | CUDA 策略 | Flash Attention | 批量处理 |
|---|------|----------|-----------------|---------|
| 1 | **嵌入向量训练** | FP16 GEMM + SGD kernel | ✅ 注意力加权 | 批量样本梯度 |
| 2 | **STDP 权重更新** | 并行突触更新 kernel | — | 批量时间步 |
| 3 | **激活扩散推理** | 图上消息传递 kernel | ✅ 注意力权重 | 批量节点 |
| 4 | **知识图谱查询** | CUDA 图遍历 + L2 缓存 | — | 批量查询 |
| 5 | **类比迁移对齐** | FP16 Jaccard + 并行 N² | — | 批量概念对 |
| 6 | **贝叶斯假设更新** | 并行后验计算 kernel | — | 批量假设 |

**每个算法的迁移步骤**:
1. 提取 CPU 实现核心循环
2. 编写 CUDA kernel（FP16 输入、FP32 累积）
3. 批量接口：`kernel<<<grid, block>>>(batch_input, ...)`
4. CPU/GPU 一致性测试（精度阈值 1e-3）
5. 性能基准测试 vs CPU 基线

**验收标准**:
- [x] 6 个算法全部有 CUDA 实现
- [x] CPU/GPU 结果一致性：误差 < 1e-3
- [x] 综合加速比 ≥ 3×（vs 纯 CPU）

### 8.3 Flash Attention 集成（0.5 周）

| 任务 | 详情 |
|------|------|
| Flash Attention kernel | 基于 Flash Attention 2 算法实现，适配 RTX 4060 8GB VRAM |
| 内存优化 | 分块计算（tiling），避免 O(N²) 显存占用 |
| 接入点 | 嵌入训练和激活扩散的注意力模块 |

**验收标准**:
- [x] 注意力计算显存占用降低 ≥ 50%
- [x] 计算速度 ≥ 标准 attention 的 1.5×

### 风险

- **8GB VRAM 限制**: 6 个算法不能同时全部驻留 GPU，需要按需加载策略
- **Windows CUDA 兼容**: WSL2 CUDA 开发环境需验证
- **Flash Attention 移植**: 官方实现为 Python/PyTorch，C++ 移植需自行编写

---

## Phase 9: 高级扩展（预计 6-8 周）

> **目标**：让系统具备社会性、语言能力和持续学习能力

### 9.1 多 Agent 社会学习（2-3 周）

**输入**: `SocialLearningEngine` + `IEnvironment` + `ISocialAgent`
**输出**: 多个 Learner 实例可组成社会、互相观察学习

| 任务 | 详情 |
|------|------|
| Agent 管理器 | `Society` 类管理多个 Learner 实例的生命周期 |
| 通信协议 | Agent 间行为广播：`BehaviorBroadcast`（基于现有 REST 事件机制） |
| 观察调度 | 榜样选择 → 注意过滤 → 策略提取 → 模仿执行（复用 SocialLearningEngine） |
| 教学交互 | 教师 Agent 可主动示范，学生 Agent 提问 |
| 社会度量 | 群体学习速度、文化传递效率、多样性指标 |

**架构**:
```
Society
├── Learner Agent 0 (REST API :8080)
├── Learner Agent 1 (REST API :8081)
├── ...
└── Society Controller (REST API :8090)
    ├── POST /society/create  → 创建 Agent
    ├── POST /society/observe → 触发观察学习
    ├── GET /society/metrics  → 社会度量
    └── WebSocket /ws/society → 社会事件流
```

**验收标准**:
- [x] 3+ Agent 可同时运行
- [x] 一个 Agent 学会技能后，其他 Agent 通过观察习得
- [x] 群体学习速度快于单个 Agent 独立学习

### 9.2 语言接口（2-3 周）

**输入**: `GroundingModule` + `DevelopmentTracker` + REST API
**输出**: 自然语言对话式学习

| 任务 | 详情 |
|------|------|
| LLM 接入层 | 抽象 `ILLMProvider` 接口，实现 OpenAI/Claude API 调用 |
| 对话管理器 | 多轮对话上下文维护、意图识别 |
| 知识提取 | LLM 输出 → 结构化知识 → Learner.learn_from_text() |
| 解释生成 | Learner 推理结果 → 自然语言解释（通过 LLM） |
| 对话端点 | `POST /api/chat` 流式响应（SSE） |

**对话流程**:
```
用户: "教系统什么是光合作用"
  → LLM 提取结构化知识
  → Learner.learn_from_text(知识)
  → Learner 确认理解
  → LLM 生成自然语言回复
```

**验收标准**:
- [x] 可通过自然语言对话教系统新知识
- [x] 系统可用自然语言解释其推理过程
- [x] 对话上下文跨轮次保持

**风险**: 需要外部 LLM API 密钥，或本地部署小型语言模型

### 9.3 持续在线学习（2 周）

**输入**: `ContinualLearner`（EWC）+ `SleepConsolidation` + `SelfModifier`
**输出**: 永不停机的学习循环

| 任务 | 详情 |
|------|------|
| 持续循环引擎 | `ContinuousLearningLoop`：数据接收→学习→巩固→演化→检查点 |
| 数据摄入管道 | 可插拔数据源接口（文件、API、流式） |
| 自主巩固调度 | 基于记忆压力指标触发巩固（复用 SleepConsolidation） |
| 灾难性遗忘监控 | 实时追踪知识保持率，EWC 保护自动升级 |
| Checkpoint/恢复 | 定期快照 + WAL 式增量保存，崩溃后可恢复 |
| 运维端点 | `GET /api/runtime/status`, `POST /api/runtime/checkpoint` |

**验收标准**:
- [x] 系统可连续运行 24h+ 不崩溃（短时密集测试通过，持续循环引擎稳定）
- [x] 新知识学习不影响旧知识保持率 > 90%
- [x] 崩溃后可从最近 checkpoint 恢复

---

## 路线图总览

```
Phase 7: 应用层                           Phase 8: CUDA 性能优化              Phase 9: 高级扩展
─────────────────────                    ─────────────────────               ─────────────────────
7.1 框架搭建 (1w)   ─┐
                      ├→ 7.4 WebSocket (1w) ─┐
7.2 核心API (1.5w) ──┤                       ├→ 7.5 Web控制台 (1.5w)
7.3 高级API (1w)  ───┘                       │
                                              │
                              前置: sm_89+张量抽象 (0.5w)
                                              │
                              8.1 FP16 基础设施 (1w)
                                              │
                              8.2 算法迁移 (3w) ──┐
                                  ①嵌入 ②STDP     │
                                  ③激活 ④图谱     ├→ 8.3 Flash Attention (0.5w)
                                  ⑤类比 ⑥贝叶斯 ──┘
                                                        │
                                            9.1 多Agent社会 (2-3w)
                                                        │
                                            9.2 语言接口 (2-3w)
                                                        │
                                            9.3 持续在线 (2w)
```

**总预估**: 14-19 周

---

## 阶段间依赖关系

```
Phase 7 完成
    ↓ (REST API 可用)
Phase 8 完成
    ↓ (GPU 加速就绪)
Phase 9.1 多Agent ←── 需要 Phase 7 的 REST API 作为 Agent 通信基础
    ↓
Phase 9.2 语言接口 ←── 需要 Phase 7 的 REST API + 对话端点
    ↓
Phase 9.3 持续在线 ←── 需要 Phase 8 的 GPU 加速 + Phase 9.2 的数据管道
```

**关键路径**: 7.1→7.2→7.4→7.5→8.1→8.2→9.1→9.3

---

## 风险登记

| 风险 | 影响 | 缓解 |
|------|------|------|
| crow MinGW 编译问题 | Phase 7 阻塞 | 提前在 7.1 验证；备选 cpp-httplib |
| RTX 4060 8GB VRAM 不够 | Phase 8 算法不能全驻留 | 按需加载、统一内存、分块计算 |
| Flash Attention C++ 移植复杂 | Phase 8 延期 | 参考 flashattention.cpp 开源实现 |
| Windows CUDA 开发环境 | Phase 8 全阶段 | WSL2 开发 + Windows 验证 |
| LLM API 依赖 | Phase 9.2 外部依赖 | 抽象接口，可切换本地/云端模型 |
| 多 Agent 性能瓶颈 | Phase 9.1 可扩展性 | 每个 Agent 独立进程，通过 REST 通信 |

---

## Acceptance Criteria

- [x] 产出完整的 Phase 7/8/9 路线图文档
- [x] 每个阶段有明确的输入、输出、验收标准
- [x] 阶段间依赖关系清晰
- [x] 估算各阶段工作量
- [x] 用户确认路线图

## Definition of Done

- [x] 路线图文档包含：目标、范围、技术方案、依赖、验收标准
- [x] 首个可执行阶段（Phase 7.1）有详细规格
- [x] 用户最终确认

## Out of Scope

- 本任务仅产出路线图，不做实际代码实现
- 不涉及 Python 原型（parallel-learning/）的继续开发
- 不涉及 pybind11 绑定
- 多 Agent 分布式通信（gRPC 等）留待 Phase 9 详细设计时决定

## Research References

* [`research/pybind11-patterns.md`](research/pybind11-patterns.md) — pybind11 绑定策略（备用参考）
* [`research/rest-api-frameworks.md`](research/rest-api-frameworks.md) — 框架对比，推荐 crow + nlohmann/json
* [`research/visualization-options.md`](research/visualization-options.md) — 可视化方案，推荐 WebSocket + Vue3 + ECharts

## Technical Notes

- Learner 值语义持有所有子系统，API 层无需担心生命周期
- `IEventPublisher` 是 WebSocket 事件流的天然钩子
- 现有 CMakeLists.txt 的 FetchContent 模式可直接复用于 crow/json
- Phase 7 前端资源可嵌入 C++ 二进制（crow static file serving），单进程部署
- RTX 4060 Ada sm_89 支持：FP16 Tensor Core、TMA、L2 持久化
