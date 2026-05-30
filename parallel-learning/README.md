# 🧠 Parallel Learning — 从学习本源出发的人工智能

> 探究人类学习的本质机制，从0岁婴儿的学习方式出发，实现真正类人的智能系统。

## 核心理念

不是靠堆参数、堆算力的"假智能"，而是从**学习的本源**出发：

- **婴儿如何学习**：从感知到概念，从具体到抽象，从模仿到创造
- **鹦鹉如何学语言**：社会偶联、即时反馈、好奇心驱动
- **大脑如何工作**：预测编码、Hebbian学习、睡眠巩固、互补学习系统

## 系统架构

```
┌─────────────────────────────────────────────┐
│              统一学习体 (Learner)              │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  感知层   │  │  学习层   │  │  推理层   │  │
│  │          │  │          │  │          │  │
│  │ BPE编码器 │  │ BTSP     │  │ 统一推理  │  │
│  │ Transformer│ │ TTT      │  │ 因果DAG  │  │
│  │ 知识提取  │  │ 预测编码  │  │ 多跳推理  │  │
│  │ 多模态    │  │ GHL      │  │ 类比推理  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  记忆层   │  │  认知层   │  │  进化层   │  │
│  │          │  │          │  │          │  │
│  │ 互补学习  │  │ 树突计算  │  │ 自改进    │  │
│  │ 睡眠回放  │  │ 主动推理  │  │ 反思学习  │  │
│  │ 多时间尺度│  │ 认知路由  │  │ 自进化    │  │
│  │ 知识图谱  │  │ 元学习    │  │ 情感驱动  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

## 已实现的17个人类学习机制

### 基础学习 (Phase 1-3)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 1 | Hebbian学习 | Hebb 1949 | `learner.py` |
| 2 | 预测编码 | Whittington & Bogacz 2017 | `predictive_coding_engine.py` |
| 3 | 好奇心驱动探索 | Oudeyer et al. 2007 | `empowerment_exploration.py` |

### 核心认知 (Phase 4-7)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 4 | BTSP单次学习 | Bhatt et al. 2022 | `btsp_learning.py` |
| 5 | In-Place TTT | Sun et al. 2024 | `test_time_training.py` |
| 6 | 预测编码Light | Rao & Ballard 1999 | `predictive_coding_light.py` |
| 7 | 奖励表征后移 | Schultz 1998 | `predictive_coding_light.py` |
| 8 | 组合泛化 | Fodor & Pylyshyn 1988 | `predictive_coding_light.py` |
| 9 | 社会偶联学习 | Csibra & Gergely 2009 | `predictive_coding_light.py` |
| 10 | 符号接地 | Harnad 1990 | `predictive_coding_light.py` |

### 高级认知 (Phase 8-9)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 11 | 认知预测路由 | Clark 2013 | `cognitive_mechanisms.py` |
| 12 | GHL全局调制Hebbian | Fremaux & Gerstner 2016 | `cognitive_mechanisms.py` |
| 13 | 感知类别先于语言 | Harnad 1987 | `cognitive_mechanisms.py` |

### 最新研究集成 (2024-2025)
| # | 机制 | 论文来源 | 模块 |
|---|------|----------|------|
| 14 | **树突计算** | Chavlis & Poirazi 2025 | `dendritic_computation.py` |
| 15 | **睡眠回放巩固** | Nature Comms 2022, NeuroDream 2025 | `sleep_replay.py` |
| 16 | **主动推理** | Friston 2022, Parr et al. 2024 | `active_inference.py` |
| 17 | **互补学习系统** | McClelland 1995, Nature Neurosci 2023 | `complementary_learning.py` |

## 学习流程 (30步)

每次`learn_from_text()`执行完整的30步学习闭环：

```
编码(BPE+Transformer) → 实体提取 → 关系提取 → 知识图谱注入
  → BTSP标记 → 矛盾检测 → 因果提取 → 概念形成 → 数值提取
  → Hebbian训练 → 反馈验证 → 负向学习 → 不确定性更新
  → 记忆存储 → 多时间尺度 → 自改进 → 反思学习 → 因果引擎
  → 情感驱动 → 创造性引擎 → 预测编码Light → 奖励后移
  → 组合泛化 → 社会偶联 → 符号接地 → 认知路由 → GHL调制
  → 学习进展 → 感知类别 → 元学习组合 → 树突计算 → 睡眠记录
  → 主动推理 → 互补学习存储
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

### 浏览器

打开 `http://localhost:8080` 使用Web界面提问。

## 技术规格

| 维度 | 规格 |
|------|------|
| 编码器 | BPE分词 + Transformer(2层, 4头) |
| 知识提取 | 混合(正则冷启动 + 学习模型) |
| 推理引擎 | 6种模式(直接/因果/归纳/类比/反事实/概率) |
| 知识图谱 | 实体-关系三元组 + 向量检索 |
| GPU加速 | CUDA FP16混合精度 (RTX 4060 8GB) |
| 学习速度 | 88条/秒 (优化模式), ~5条/秒 (完整模式) |

## 项目结构

```
parallel-learning/
├── src/
│   ├── core/           # 核心学习体
│   │   ├── learner.py         # 统一学习体 (3300+行, 30步学习闭环)
│   │   ├── config.py          # 配置
│   │   └── device.py          # 设备管理
│   ├── perception/     # 感知层
│   │   ├── learnable_encoder.py   # BPE + Transformer编码器
│   │   ├── knowledge_extractor.py # 可微分知识提取
│   │   ├── object_world_model.py  # 对象中心世界模型
│   │   └── embodied_grounding.py  # 具身接地
│   ├── learning/       # 学习层 (17个机制)
│   │   ├── btsp_learning.py           # BTSP单次学习
│   │   ├── test_time_training.py      # 测试时训练
│   │   ├── predictive_coding_light.py # 预测编码+4个子机制
│   │   ├── cognitive_mechanisms.py    # 认知路由+GHL+进展+类别+元学习
│   │   ├── dendritic_computation.py   # 树突计算 (NEW)
│   │   ├── sleep_replay.py            # 睡眠回放巩固 (NEW)
│   │   ├── active_inference.py        # 主动推理 (NEW)
│   │   ├── complementary_learning.py  # 互补学习系统 (NEW)
│   │   ├── self_improvement.py        # 自改进系统
│   │   ├── reflective_learning.py     # 反思学习
│   │   ├── multiscale_learning.py     # 多时间尺度
│   │   ├── empowerment_exploration.py # 好奇心+赋能
│   │   └── self_evolution.py          # 自进化引擎
│   ├── reasoning/      # 推理层
│   │   ├── unified_engine.py    # 统一推理引擎
│   │   ├── causal_engine.py     # 因果推理
│   │   ├── tool_engine.py       # 工具调用
│   │   ├── code_generator.py    # 代码生成
│   │   └── creativity_engine.py # 创造力引擎
│   └── knowledge/      # 知识层
│       ├── graph.py            # 知识图谱
│       ├── entity.py           # 实体
│       └── relation.py         # 关系
├── data/               # 语料数据
│   └── extracted/
│       ├── baike/           # 百科问答 (1.5GB, 140万条)
│       ├── wiki/            # 维基百科
│       ├── news/            # 新闻
│       └── webtext/         # 网页文本
├── server.py           # HTTP服务器
├── fast_server.py      # 快速服务器
├── optimized_learn.py  # 优化学习脚本
└── README.md
```

## 许可

研究项目，用于探索人类学习本质和人工智能的根本路径。
