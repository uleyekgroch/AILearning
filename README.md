# 从学习的本源出发的人工智能

> 不是靠堆参数和数据的"假智能"，而是真正理解"学习"本身是什么。

## 项目愿景

深入研究人类婴儿（0-18岁）和鹦鹉的语言学习过程，提炼学习的本质机制，然后设计一个从学习本源出发的人工智能系统。

## 核心发现

学习的本质是**预测误差最小化**（自由能原理）。语言从交流压力中自发涌现，符号接地不限于视觉——所有感知模态都能接地。

## 系统架构

```
src/core/learner.py — 主学习体（3282行）
├── 感知层 — BPE分词 + Transformer编码器
├── 预测编码引擎 — Hebbian学习 + 预测编码Light
├── 知识图谱 — 实体-关系存储
├── 记忆系统 — 工作/情景/语义记忆 + 多时间尺度
├── 推理引擎 — 6种推理模式（直接/因果/归纳/类比/反事实/概率）
├── 元认知 — 自我评估/知识空白
├── 语言接地 — 符号↔世界模型
├── 发展阶段 — Piaget式课程
├── 自改进 — 脚手架+权重更新
├── 反思学习 — 经验反思+策略合成
├── 测试时训练 — In-Place TTT (ICLR 2026)
├── BTSP学习 — 单次学习（资格痕迹+平台电位）
└── 认知机制 — 认知路由+GHL+学习进展+类别先于语言+元学习组合
```

## 项目规模

- **源文件**: 135个Python文件
- **代码量**: 36,741行
- **主系统**: 3,282行 (learner.py)
- **学习模块**: 21个
- **推理模块**: 20个
- **感知模块**: 10个
- **Git提交**: 20次迭代

## 核心能力

### 1. BPE分词 + Transformer编码器

```python
# 从字符级升级到子词级
encoder = LearnableTextEncoder(d_model=128, n_heads=4, n_layers=2)
encoder.train_tokenizer(corpus)  # 从语料学习BPE合并规则
embedding = encoder(text)  # Transformer编码
```

### 2. 测试时训练 (In-Place TTT)

```python
# 推理时原地更新快权重
ttt = TestTimeTrainer(encoder, lr=1e-5)
ttt.adapt_to_query(query, query_repr, relevant_entities)
# 自适应学习率：相似度高→小更新，相似度低→大更新
```

### 3. BTSP单次学习

```python
# 行为时间尺度突触可塑性
btsp = BTSPLearningSystem()
btsp.mark_eligible(entity, embedding)  # 资格痕迹
btsp.trigger_plateau(trigger_strength=1.0)  # 平台电位
# 差异化增强：远离其他实体中心
```

### 4. 认知预测路由

```python
# 区分低级感觉误差和高级认知误差
router = CognitivePredictiveRouter()
routing = router.route_error(low_error, high_error)
# 动态调整路由权重：低级0.3, 高级0.7
```

### 5. GHL全局调制Hebbian学习

```python
# 神经调质信号调制局部学习
ghl = GlobalModulatedHebbian()
global_signal = ghl.compute_global_signal(reward, novelty, uncertainty)
delta = ghl.hebbian_update(pre, post, global_signal)
# Δw = η × sign(global_signal) × pre × post
```

### 6. 学习进展好奇心

```python
# 探索甜蜜区（不太简单也不太难）
progress = LearningProgressCuriosity()
progress.update_progress(domain, performance)
sweet_spot = progress.get_sweet_spot_domain()
```

### 7. 先类别后语言

```python
# 感知分类先于语言涌现
categories = PerceptualCategorySystem()
category = categories.discover_category(entity, representation)
# 自动聚类：12个实体→1个类别
```

### 8. 元学习组合规则

```python
# 学习如何组合，而非记住什么组合
composition = MetaLearningComposition()
composition.learn_rule(components, result, success)
predicted = composition.apply_rule(new_components)
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
