# 🧠 Parallel Learning — 从学习本源出发的人工智能

> 探究人类学习的本质机制，从0岁婴儿的学习方式出发，实现真正类人的智能系统。

## 核心理念

不是靠堆参数、堆算力的"假智能"，而是从**学习的本源**出发：

- **婴儿如何学习**：从感知到概念，从具体到抽象，从模仿到创造
- **鹦鹉如何学语言**：社会偶联、即时反馈、好奇心驱动
- **大脑如何工作**：预测编码、Hebbian学习、睡眠巩固、互补学习系统

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   统一学习体 (Learner)                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Layer 0: 核心知识先验 (Spelke 6个核心系统)        │    │
│  │  物体持久性 · 近似数感知 · 代理检测 · 几何 ·       │    │
│  │  社会 · 因果 — 为学习缩小假设空间100倍              │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Layer 1: 感知-预测循环 (0-12月对应)               │    │
│  │  感知 → 预测 → 误差 → 学习 → 概念涌现             │    │
│  │  概念从预测误差中涌现，不是从正则中提取              │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Layer 2: 概念形成 (12-24月对应)                   │    │
│  │  功能性表征：感知特征 + 可供性 + 使用场景           │    │
│  │  快速映射：互斥性 + 整体对象偏差 → 一次接触就理解    │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Layer 3: 语言习得 (2-4岁对应)                     │    │
│  │  标签附着在感知概念上 · 语法从使用中涌现             │    │
│  │  组合性表达（非模板拼接）· 过度规则化检测            │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Layer 4: 模拟推理 (4-6岁+对应)                    │    │
│  │  场景构建 → 因果链追踪 → 反事实模拟 → 类比发现      │    │
│  │  Pearl因果阶梯 L1-L3 · 自然语言表达                │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  记忆层   │  │  认知层   │  │  进化层   │              │
│  │          │  │          │  │          │              │
│  │ 互补学习  │  │ 树突计算  │  │ 自改进    │              │
│  │ 睡眠回放  │  │ 主动推理  │  │ 反思学习  │              │
│  │ 多时间尺度│  │ 认知路由  │  │ 自进化    │              │
│  │ 知识图谱  │  │ 元学习    │  │ 情感驱动  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

## 架构重构：从模式匹配到真正学习

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| **学习机制** | 正则提取 → 存储 | 感知 → 预测 → 误差 → 学习闭环 |
| **概念本质** | 字符串 + 向量 | 功能性表征（感知特征 + 可供性 + 场景） |
| **推理方式** | 查表 + 模板拼接 | 场景模拟 + 因果链追踪 + 反事实 |
| **核心知识先验** | 无 | Spelke 6个核心系统 |
| **概念涌现** | 从正则提取 | 从预测误差聚类中涌现 |
| **未训练数据** | 完全无法回答 | 概念泛化 + 因果推理部分回答 |

## 已实现的27+个人类学习机制

### 核心知识先验 (架构重构)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 1 | **Spelke核心知识** | Spelke & Kinzler 2007 | `core_knowledge.py` |
| 2 | **感知-预测学习循环** | Friston 2022 (Active Inference) | `perception_learning_loop.py` |
| 3 | **功能性概念表征** | Gibson 1977, Paivio 1971 | `functional_concept.py` |
| 4 | **快速映射** | Carey & Bartlett 1978 | `functional_concept.py` |
| 5 | **语言习得(使用基础)** | Tomasello 2003 | `language_acquisition.py` |
| 6 | **模拟推理** | Barsalou 1999, Pearl 2009 | `simulation_reasoning.py` |

### 基础学习 (Phase 1-3)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 7 | Hebbian学习 | Hebb 1949 | `learner.py` |
| 8 | 预测编码 | Whittington & Bogacz 2017 | `predictive_coding_engine.py` |
| 9 | 好奇心驱动探索 | Oudeyer et al. 2007 | `empowerment_exploration.py` |

### 核心认知 (Phase 4-7)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 10 | BTSP单次学习 | Bhatt et al. 2022 | `btsp_learning.py` |
| 11 | In-Place TTT | Sun et al. 2024 | `test_time_training.py` |
| 12 | 预测编码Light | Rao & Ballard 1999 | `predictive_coding_light.py` |
| 13 | 奖励表征后移 | Schultz 1998 | `predictive_coding_light.py` |
| 14 | 组合泛化 | Fodor & Pylyshyn 1988 | `predictive_coding_light.py` |
| 15 | 社会偶联学习 | Csibra & Gergely 2009 | `predictive_coding_light.py` |
| 16 | 符号接地 | Harnad 1990 | `predictive_coding_light.py` |

### 高级认知 (Phase 8-9)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 17 | 认知预测路由 | Clark 2013 | `cognitive_mechanisms.py` |
| 18 | GHL全局调制Hebbian | Fremaux & Gerstner 2016 | `cognitive_mechanisms.py` |
| 19 | 感知类别先于语言 | Harnad 1987 | `cognitive_mechanisms.py` |

### 最新研究集成 (2024-2025)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 20 | 树突计算 | Chavlis & Poirazi 2025 | `dendritic_computation.py` |
| 21 | 睡眠回放巩固 | Nature Comms 2022 | `sleep_replay.py` |
| 22 | 主动推理 | Friston 2022, Parr et al. 2024 | `active_inference.py` |
| 23 | 互补学习系统 | McClelland 1995 | `complementary_learning.py` |

### 发展学习 (Phase 10)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 24 | 语言发展阶段 | Piaget/Vygotsky | `language_development.py` |
| 25 | 具身接地 | Barsalou 2008 | `embodied_grounding.py` |
| 26 | 社会反馈学习 | Tomasello 2003 | `social_distillation.py` |
| 27 | 知识蒸馏 | Hinton 2015 | `social_distillation.py` |

### 认知机制 (Phase 11)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 28 | 层次概念 | Sigala & Logothetis 2002 | `hierarchical_concepts.py` |
| 29 | 时序预测 | Schapiro 2013 | `temporal_sequence.py` |
| 30 | 注意力门控 | Posner 1980 | `attention_gating.py` |

### 高级认知机制 (Phase 12)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 31 | 图式学习 | Rumelhart 1980 | `schema_learning.py` |
| 32 | 元认知调控 | Flavell 1979 | `metacognitive_regulation.py` |
| 33 | 跨域迁移 | Gentner 1983 | `cross_domain_transfer.py` |

## 学习流程

每次 `learn_from_text()` 执行完整的学习闭环：

```
核心知识先验评估
  → 感知-预测循环（文本作为虚拟感知输入）
  → 预测误差分析 → 概念涌现检测
  → 编码(BPE+Transformer)
  → 实体提取 → 关系提取 → 知识图谱注入
  → BTSP标记 → 矛盾检测 → 因果提取
  → 概念形成（带感知特征+可供性）
  → Hebbian训练 → 反馈验证 → 负向学习
  → 记忆存储 → 多时间尺度
  → 自改进 → 反思学习 → 因果引擎
  → 情感驱动 → 创造性引擎
  → 预测编码Light → 奖励后移
  → 组合泛化 → 社会偶联 → 符号接地
  → 认知路由 → GHL调制
  → 学习进展 → 感知类别 → 元学习组合
  → 树突计算 → 睡眠记录
  → 主动推理 → 互补学习存储
```

## 推理流程

`think()` 的多路径推理：

```
问题输入
  → 路径0: 模拟推理（场景构建 → 因果追踪 → 反事实 → 类比）
  → 路径1: 概念空间激活扩散（含可供性匹配）
  → 路径2: 多跳推理（知识图谱遍历）
  → 路径3: Legacy回退
```

## 运行方式

### 学习语料 + HTTP问答

```bash
# 方法1: 优化学习脚本
python optimized_learn.py --limit 1000 --test --start-http

# 方法2: 完整学习流程（包含元学习模块）
python optimized_learn.py --mode full --limit 500 --test

# 方法3: 快速服务器（预构建知识）
python fast_server.py
```

### HTTP API

```bash
# 提问
curl "http://localhost:8080/ask?q=什么是人工智能"

# 学习新知识
curl -X POST "http://localhost:8080/learn" -d "地球绕太阳公转一周需要365天"

# 状态查询
curl "http://localhost:8080/status"
```

### 测试

```bash
# 架构重构验证（Phase 0-4）
python test_core_knowledge.py          # 核心知识先验（7项测试）
python test_perception_loop.py         # 感知-预测循环（7项测试）
python test_functional_concept.py      # 功能性概念（6项测试）
python test_language_acquisition.py    # 语言习得（5项测试）
python test_simulation_reasoning.py    # 模拟推理（6项测试）

# 系统回归验证
python test_phase7.py                  # Phase 7 综合验证
```

## 技术规格

| 维度 | 规格 |
|------|------|
| 编码器 | BPE分词 + Transformer(2层, 4头) |
| 知识提取 | 感知-预测循环 + 统计学习 |
| 推理引擎 | 模拟推理 + 6种模式(直接/因果/归纳/类比/反事实/概率) |
| 知识图谱 | 实体-关系三元组 + 向量检索 |
| 概念表征 | 功能性：向量 + 感知特征 + 可供性 + 使用场景 |
| 核心知识 | Spelke 6系统（物体/数/代理/几何/社会/因果） |
| GPU加速 | CUDA FP16混合精度 (RTX 4060 8GB) |
| 学习速度 | 88条/秒 (优化模式), ~5条/秒 (完整模式) |
| 模块总数 | 157个Python文件, ~47,600行代码 |

## 项目结构

```
parallel-learning/
├── src/
│   ├── core/              # 核心学习体
│   │   ├── learner.py            # 统一学习体 (4800+行, 完整学习闭环)
│   │   ├── config.py             # 配置
│   │   ├── enhanced_learner.py   # 增强学习体
│   │   ├── learning_engine.py    # 预测编码引擎
│   │   ├── motivation.py         # 内在动机系统
│   │   └── device.py             # 设备管理
│   ├── perception/        # 感知层
│   │   ├── learnable_encoder.py  # BPE + Transformer编码器
│   │   ├── knowledge_extractor.py# 可微分知识提取
│   │   ├── object_world_model.py # 对象中心世界模型
│   │   └── embodied_grounding.py # 具身接地
│   ├── learning/          # 学习层 (33个机制模块)
│   │   ├── ★ core_knowledge.py         # Spelke核心知识先验 (967行)
│   │   ├── ★ perception_learning_loop.py # 感知-预测循环 (553行)
│   │   ├── ★ functional_concept.py     # 功能性概念系统 (479行)
│   │   ├── ★ language_acquisition.py   # 语言习得系统 (531行)
│   │   ├── ★ simulation_reasoning.py   # 模拟推理系统 (388行)
│   │   ├── concept_space.py            # 概念空间 + 激活扩散 (422行)
│   │   ├── statistical_learner.py      # 统计学习 (939行)
│   │   ├── btsp_learning.py            # BTSP单次学习
│   │   ├── test_time_training.py       # 测试时训练
│   │   ├── predictive_coding_light.py  # 预测编码+4个子机制
│   │   ├── cognitive_mechanisms.py     # 认知路由+GHL+进展+类别+元学习
│   │   ├── dendritic_computation.py    # 树突计算
│   │   ├── sleep_replay.py             # 睡眠回放巩固
│   │   ├── active_inference.py         # 主动推理
│   │   ├── complementary_learning.py   # 互补学习系统
│   │   ├── world_model.py             # 世界模型
│   │   └── ... (共42个学习模块)
│   ├── reasoning/         # 推理层
│   │   ├── unified_engine.py     # 统一推理引擎
│   │   ├── causal.py             # 因果推理
│   │   ├── analogical.py         # 类比推理
│   │   ├── counterfactual.py     # 反事实推理
│   │   ├── metaphor.py           # 隐喻推理
│   │   └── ... (共14个推理模块)
│   ├── knowledge/         # 知识层
│   │   ├── graph.py              # 知识图谱
│   │   ├── entity.py             # 实体
│   │   └── relation.py           # 关系
│   ├── memory/            # 记忆层
│   │   ├── system.py             # 三层记忆系统
│   │   ├── consolidation.py      # 记忆巩固
│   │   └── ...
│   ├── language/          # 语言层
│   ├── environment/       # 环境层
│   ├── social/            # 社会层
│   ├── metacognition/     # 元认知层
│   └── skills/            # 技能层
├── data/                  # 语料数据
├── server.py              # HTTP服务器
├── fast_server.py         # 快速服务器
├── optimized_learn.py     # 优化学习脚本
└── README.md
```

★ = 架构重构新增核心模块

## 认知科学理论基石

| 理论 | 学者 | 应用 |
|------|------|------|
| 核心知识系统 | Spelke & Kinzler 2007 | 6个进化先验约束学习 |
| 主动推理 / 自由能原理 | Friston 2022 | 预测-误差-学习闭环 |
| 可供性理论 | Gibson 1977 | 概念 = "你能用它做什么" |
| 快速映射 | Carey & Bartlett 1978 | 一次接触即理解新概念 |
| 双重编码 | Paivio 1971 | 感知+语言双重编码 |
| 因果阶梯 | Pearl 2009 | L1关联→L2干预→L3反事实 |
| 心理模拟理论 | Barsalou 1999 | 推理 = 内部场景模拟 |
| 结构映射理论 | Gentner 1983 | 类比匹配关系结构 |
| 使用基础理论 | Tomasello 2003 | 语法从使用中涌现 |
| 互补学习系统 | McClelland 1995 | 海马(快学)+皮层(慢学) |

## 许可

研究项目，用于探索人类学习本质和人工智能的根本路径。
