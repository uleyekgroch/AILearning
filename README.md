# 从学习的本源出发的人工智能

> 不是靠堆参数和数据的"假智能"，而是真正理解"学习"本身是什么。

## 项目愿景

深入研究人类婴儿（0-18岁）和鹦鹉的语言学习过程，提炼学习的本质机制，然后设计一个从学习本源出发的人工智能系统。

## 核心发现

学习的本质是**预测误差最小化**（自由能原理）。语言从交流压力中自发涌现，符号接地不限于视觉——所有感知模态都能接地。

### 涌现的符号系统

| 类别 | 符号 | 涌现条件 |
|------|------|----------|
| 组合性 | 颜色+形状+大小+材质 | 单符号不足以区分场景 |
| 否定 | "not" | 子集关系（多值特征） |
| 时态 | "past"/"present"/"future" | 时间维度成为独立特征 |
| 因果 | "because" | 混杂场景（虚假相关+真正因果） |
| 类比 | "like" | 跨领域特征隔离+关系共享 |
| 工具 | "use"/"for" | 外观歧义时的功能描述需求 |
| 视角 | "know"/"think"/"believe" | 置信度差异 |
| 跨模态 | "loud"/"rough" 等 20 个 | 视觉模糊时的听觉/触觉区分 |

## 项目结构

```
mvl/
├── 核心系统
│   ├── agent.py / agent_fep.py          # 学习体
│   ├── environment.py / environment_3d.py # 环境
│   ├── language_emergence.py             # 语言涌现核心
│   └── main.py                           # 入口
│
├── 符号接地模块（13 个 grounding_*.py）
│   ├── grounding_actions.py              # 动作
│   ├── grounding_emotions.py             # 情感
│   ├── grounding_causal_reasoning.py     # 因果推理
│   ├── grounding_theory_of_mind.py       # 心智理论
│   ├── grounding_abstraction.py          # 抽象推理
│   ├── grounding_tool_use.py             # 工具使用
│   └── grounding_crossmodal.py           # 跨模态
│
├── 高级语言模块
│   ├── narrative.py                      # 叙事与篇章
│   ├── metacognition.py                  # 元认知
│   ├── language_society.py               # 多 Agent 社会
│   └── generational_transfer.py          # 跨代知识传递
│
└── 实验模块（24 个 experiment_*.py）
```

## 快速开始

```bash
cd mvl

# 运行语言涌现实验
python experiment_language.py

# 运行否定涌现实验
python experiment_compound_negation.py

# 运行因果推理实验
python experiment_causal_reasoning.py

# 运行跨模态语言实验
python experiment_crossmodal.py

# 运行全部实验
python experiment_tool_use.py
```

## 理论框架

详见 [theory_framework.md](theory_framework.md)，包含 5.1-5.31 节的完整理论分析。

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

## 关键论文

1. Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*.
2. Harnad, S. (1990). The symbol grounding problem. *Physica D*.
3. Piaget, J. (1952). *The Origins of Intelligence in Children*.
4. Vygotsky, L. S. (1978). *Mind in Society*.
5. Tomasello, M. (2014). *A Natural History of Human Thinking*.
6. Pepperberg, I. M. (2009). *Alex & Me*.
7. Kuhl, P. K. (2007). Is speech learning 'gated' by the social brain? *Developmental Science*.
