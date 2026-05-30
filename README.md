# 从学习的本源出发的人工智能

> 不是靠堆参数和数据的"假智能"，而是真正理解"学习"本身是什么。

## 项目愿景

深入研究人类婴儿（0-18岁）和鹦鹉的语言学习过程，提炼学习的本质机制，然后设计一个从学习本源出发的人工智能系统。

## 核心发现

学习的本质是**预测误差最小化**（自由能原理）。语言从交流压力中自发涌现，符号接地不限于视觉——所有感知模态都能接地。

## 系统架构

```
src/core/learner.py — 主学习体（2000+行）
├── 感知层 — 多模态编码
├── 预测编码引擎 — Hebbian学习
├── 知识图谱 — 实体-关系存储
├── 记忆系统 — 工作/情景/语义记忆
├── 推理引擎 — 演绎/归纳/类比
├── 元认知 — 自我评估/知识空白
├── 语言接地 — 符号↔世界模型
└── 发展阶段 — Piaget式课程
```

## 核心能力

### 1. Hebbian学习 + 预测编码

```python
# Hebbian规则: ΔW = η × pre × post
weight[idx] += hebbian_lr * entity_activation

# 预测编码
epsilon = actual - predicted  # 预测误差
W += lr * np.outer(pre_synaptic, post_synaptic_error)
```

### 2. 离线整合（睡眠）

```python
def consolidate(self):
    self._offline_replay(memories)  # 重组记忆
    self._extract_abstractions(memories)  # 提取抽象
    self._integrate_knowledge()  # 整合知识
```

### 3. 组合泛化

```python
# 组合概念
compose_concepts('红', '球') → '红球'
decompose_concept('红球') → ['红', '球']

# 类比迁移
analogical_transfer('水流', '电流', {'水': '电', '管道': '导线'})
```

### 4. 矛盾检测与修正

```python
# 检测矛盾
conflict = _check_contradiction(subject, relation, obj)

# 解决矛盾
_resolve_contradiction(subject, relation, obj, conflict, source)
```

### 5. 间隔重复

```python
# 难以回忆的项目 → 加强巩固
# 容易回忆的项目 → 延长间隔
update_uncertainty(key, success)
```

## 测试结果

| 问题 | 答案 |
|------|------|
| 什么是人工智能 | 计算机科学的一个分支 |
| 牛顿发现了什么 | 万有引力定律 |
| 为什么地面湿了 | 下雨 → 地面湿了 |
| 水在多少度沸腾 | 100.0摄氏度 |
| 水流像什么 | 水流 像 电流 |

## 性能

- 学习速度: 1935条/秒 (CUDA加速)
- 基准分数: 0.94 (A级)
- GPU: RTX 4060, 8GB显存

## 与传统系统的区别

| 维度 | 传统LLM | 本系统 |
|------|---------|--------|
| 学习方式 | 预训练+微调 | 持续学习 |
| 推理方式 | 统计关联 | 因果推理 |
| 记忆方式 | 静态参数 | 动态重构 |
| 进化能力 | 无 | 自我改进 |
| 可解释性 | 低 | 高 |

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
├── 感官编码模块
│   ├── encoder_sensory.py              # 可学习感官编码器
│   ├── environment_sensory.py          # 感官网格世界
│   └── agent_sensory.py               # 端到端感官Agent
│
├── 大规模社会模块
│   ├── language_society_large.py       # 100+ Agent 语言社会（GPU 加速相似度）
│   └── multi_agent_env.py             # 自适应网格环境 + 空间索引

├── CUDA 加速模块
│   └── cuda_utils.py                  # 设备管理、NumPy↔PyTorch、批量操作
│
├── 3D 多 Agent 模块
│   ├── multi_agent_3d_env.py          # 多 Agent 共享 3D 环境
│   ├── agent_social.py               # 社会学习 Agent
│   └── instruction_grounding.py      # 指令 Grounding（动词+物体描述）
│
├── 跨语言翻译模块
│   └── language_translator.py        # 双语 Agent + 跨环境翻译
│
├── 自主学习模块
│   └── self_directed_learning.py     # 知识评估 + 目标选择 + 自主学习
│
├── 课程学习模块
│   └── curriculum_learning.py        # 歧义度控制 + 渐进课程 + 自主节奏
│
├── 神经科学验证模块
│   └── neuroscience_validation.py    # 文献数据 + 曲线拟合 + 关键期测试
│
├── 开放式学习模块
│   └── open_ended_learning.py        # 内驱力探索 + 结构发现
│
├── 累积文化创新模块
│   └── cultural_innovation.py        # 棘轮效应 + 代际传承 + 创新涌现
│
├── 具身隐喻模块
│   └── embodied_metaphor.py          # 跨域映射涌现 + 隐喻接地测试
│
├── 关键期模块
│   └── critical_period.py            # 可塑性衰减 + 关键期关闭 + 重新打开
│
├── 预测编码模块
│   └── experiment_predictive_coding.py  # Hebbian vs 反向传播对比 + 局部性验证
│
├── 类比隐喻模块
│   └── analogy_metaphor.py             # 结构映射检测 + 情感空间 + 具身隐喻
│
├── 认知扩展模块
│   ├── experiment_counterfactual.py    # 反事实推理（if/then/would）
│   ├── experiment_cooperative_planning.py # 协作规划（角色 + 顺序标记）
│   ├── experiment_continuous_concepts.py  # 连续概念空间（模糊类别）
│   ├── experiment_language_memory.py     # 语言驱动记忆（记忆支架）
│   ├── experiment_language_attention.py  # 语言引导注意力（感知调制）
│   ├── experiment_adversarial.py         # 对抗性通信（欺骗 + 信任）
│   ├── experiment_hierarchical_syntax.py # 层级语法（词序 + 递归嵌入）
│
├── 量化与空间模块
│   ├── experiment_quantitative_language.py  # 量化语言（数字 + 计数 + 多/少）
│   └── experiment_spatial_language.py       # 空间关系语言（介词涌现）
│
├── 好奇心与规范模块
│   ├── experiment_curiosity_question.py     # 好奇心提问（why/what/how 涌现）
│   └── experiment_social_norms.py           # 社会规范语言（礼貌 + 禁忌）
│
├── 适应与辩论模块
│   ├── experiment_nonstationary.py          # 非平稳环境适应（词汇淘汰 + 新词涌现）
│   └── experiment_debate.py                 # 辩论与说服（论证标记涌现）
│
├── 道德与游戏模块
│   ├── experiment_moral_language.py         # 道德语言涌现（公平 + 利他信号）
│   └── experiment_humor_play.py             # 幽默与游戏（非工具性通信）
│
├── 对话与认知模块
│   ├── experiment_conversational_repair.py  # 对话修复（huh/again/different 澄清标记）
│   ├── experiment_sleep_consolidation.py    # 睡眠式记忆巩固（离线重播 + 遗忘曲线）
│   ├── experiment_politeness.py             # 礼貌与面子语言（please/sorry 社交距离）
│   └── experiment_empathy.py                # 共情与视角采择（happy/sad 情感标记）
│
├── 社会与文化模块
│   ├── experiment_ownership.py              # 所有权与财产（mine/yours/share 涌现）
│   ├── experiment_negotiation.py            # 谈判与讨价还价（fair/deal/compromise）
│   ├── experiment_dialect.py                # 方言分化与语言接触（pidgin → creole）
│   └── experiment_cryptolect.py             # 秘密语言（群体内部密码词汇）

└── 实验模块（46 个 experiment_*.py）
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

## 关键论文

1. Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*.
2. Harnad, S. (1990). The symbol grounding problem. *Physica D*.
3. Piaget, J. (1952). *The Origins of Intelligence in Children*.
4. Vygotsky, L. S. (1978). *Mind in Society*.
5. Tomasello, M. (2014). *A Natural History of Human Thinking*.
6. Pepperberg, I. M. (2009). *Alex & Me*.
7. Kuhl, P. K. (2007). Is speech learning 'gated' by the social brain? *Developmental Science*.
