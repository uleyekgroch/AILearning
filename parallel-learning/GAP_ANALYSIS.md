# 通用学习AI — 差距分析与改进路线图

## 一、当前系统 vs 通用学习AI：7大差距

### 差距1：文本编码 — 字符级 → 语义级

| 维度 | 当前系统 | 通用学习AI需要 |
|------|---------|--------------|
| 编码方式 | `ord(c) % 10000` 字符映射 | 学习到的子词/词汇单元 |
| 词序信息 | 丢失（平均池化） | 保留（注意力机制） |
| 上下文 | 无 | 动态上下文建模 |
| 跨语言 | 仅中文字符 | 语言无关的语义表示 |

**根因**：没有学习到的分词器（BPE/WordPiece），没有位置编码，没有自注意力。

**论文依据**：
- LeCun (2022) "A Path Towards Autonomous Machine Intelligence" — 嵌入空间预测优于像素/字符空间
- Balestriero & LeCun (2025) "LeJEPA" — 在抽象表示空间做预测

**改进方案**：实现学习到的子词分词器 + 位置编码 + 多头注意力

---

### 差距2：知识提取 — 正则模式 → 学习驱动

| 维度 | 当前系统 | 通用学习AI需要 |
|------|---------|--------------|
| 实体识别 | `re.findall(r'[一-鿿]{2,6}')` | 学习到的序列标注 |
| 关系抽取 | 8个硬编码正则 | 开放信息提取（OpenIE） |
| 因果发现 | 4个固定模式 | 从干预中学习因果 |
| 数值理解 | 4种单位模式 | 通用数值推理 |

**根因**：知识提取流水线不可微分，无法通过梯度优化。

**论文依据**：
- Scholkopf et al. (2021) "Toward Causal Representation Learning" — 因果表示学习
- arXiv:2603.17405 — 高维数据的因果表示学习

**改进方案**：可微分的实体-关系提取器 + 因果干预机制

---

### 差距3：推理机制 — 图遍历 → 逻辑演绎

| 维度 | 当前系统 | 通用学习AI需要 |
|------|---------|--------------|
| 直接查询 | ✓ 知识图谱遍历 | ✓ |
| 链式推理 | ✓ 两跳 | 多跳 + 路径推理 |
| 逻辑推理 | ✗ | 一阶逻辑、蕴涵、否定 |
| 归纳推理 | ✗ | 从实例归纳规律 |
| 反事实 | ✗ 注册但未调用 | do-calculus 干预推理 |
| 概率推理 | ✗ | 贝叶斯推理 |

**根因**：`think()` 方法只调用 `_reason_from_entities`，未集成已注册的 CounterfactualModule、TheoryOfMindModule 等。

**论文依据**：
- Heins et al. (2025) "AXIOM" — 对象中心的世界模型 + 符号推理
- NeurIPS 2024 CRL Workshop — 因果表示学习

**改进方案**：统一推理引擎，集成逻辑/因果/反事实/概率推理

---

### 差距4：记忆系统 — 静态存储 → 动态巩固

| 维度 | 当前系统 | 通用学习AI需要 |
|------|---------|--------------|
| 工作记忆 | FIFO 队列 | 容量受限的注意力窗口 |
| 情景记忆 | 带衰减的trace | 情节化 + 时空索引 |
| 语义记忆 | 向量存储 | 结构化知识图谱 |
| 巩固 | 随机配对 + 字符串前缀 | 选择性重播 + 泛化 + 遗忘 |
| 索引 | O(n) 遍历 | 向量索引（HNSW/IVF） |

**根因**：`_extract_abstractions` 用 `os.path.commonprefix()` 提取抽象，不是真正的知识压缩。

**论文依据**：
- Behrouz et al. (2025) "Nested Learning" (NeurIPS 2025) — 多时间尺度学习
- Nature 2025 "Mitigating catastrophic forgetting" — 持续学习综述

**改进方案**：多时间尺度记忆 + 选择性巩固 + 遗忘曲线

---

### 差距5：世界模型 — 线性预测 → 对象中心

| 维度 | 当前系统 | 通用学习AI需要 |
|------|---------|--------------|
| 预测器 | 线性层 `W_obs @ obs + W_action @ action` | 潜在空间预测 |
| 表示 | 整体观测向量 | 对象中心（属性+关系） |
| 物理 | 无 | 直觉物理引擎 |
| 规划 | 无 | 想象式轨迹rollout |

**根因**：`ProbabilisticPredictor` 是单层线性模型，没有层次结构，没有对象分解。

**论文依据**：
- Hafner et al. (2025) DreamerV3 (Nature) — 世界模型中做梦
- Heins et al. (2025) AXIOM — 对象中心世界模型

**改进方案**：层级预测编码 + 对象中心表示 + 想象式规划

---

### 差距6：学习机制 — 单一更新 → 多时间尺度

| 维度 | 当前系统 | 通用学习AI需要 |
|------|---------|--------------|
| 感知层 | 梯度更新 | 快速适应（分钟级） |
| 知识层 | Hebbian更新 | 慢速固化（天级） |
| 元学习 | 无 | 学习如何学习 |
| 课程 | 硬编码阶段 | 自适应难度调节 |

**根因**：所有参数以相同速率更新，没有分层学习率。

**论文依据**：
- ORBIT (2026) — 跨episode元RL
- MAML-en-LLM (2024) — 元学习作为LLM训练技术

**改进方案**：多时间尺度参数更新 + 元学习信号

---

### 差距7：模块集成 — 孤岛 → 统一框架

| 维度 | 当前系统 | 通用学习AI需要 |
|------|---------|--------------|
| 注册模块 | 40+ | 同样多，但全部串联 |
| think()调用 | knowledge, causal_dag, concept_formation | 全部模块协同 |
| 反事实 | CounterfactualModule 存在但未调用 | 集成到推理链 |
| 理论心智 | TheoryOfMindModule 存在但未调用 | 集成到决策 |
| 隐喻 | MetaphorTracker 存在但未调用 | 集成到语言理解 |

**根因**：核心方法（`think`, `learn_from_text`）没有设计为调用所有已注册模块。

**改进方案**：统一推理管道，所有模块按需参与

---

## 二、改进路线图（按优先级排序）

### Phase 1: 学习到的文本编码器（差距1）

**目标**：从字符级 bag-of-characters 升级为学习到的子词编码器

**实现**：
1. 实现 BPE 分词器（从语料学习子词）
2. 添加位置编码（正弦/可学习）
3. 实现多头自注意力层
4. 替换 `_encode_text` 的字符级方案

**验证**：编码 "狗咬人" 和 "人咬狗" 应产生不同向量

**论文**：Vaswani et al. (2017) "Attention Is All You Need"

---

### Phase 2: 可微分知识提取（差距2）

**目标**：从正则模式升级为学习驱动的实体-关系提取

**实现**：
1. 实现序列标注模型（BiLSTM-CRF 或 Transformer-based）
2. 实现关系分类器（给定两个实体，预测关系类型）
3. 实现开放关系抽取（不限于预定义关系类型）
4. 将提取结果反馈到编码器训练

**验证**：学习 "水在100摄氏度沸腾" 后，能提取 "水-温度-100°C"，无需硬编码正则

**论文**：
- Miwa & Bansal (2016) "End-to-End Relation Extraction"
- Stanovsky et al. (2018) "OpenIE"

---

### Phase 3: 统一推理引擎（差距3）

**目标**：将所有已注册模块集成到推理链

**实现**：
1. 重构 `think()` 方法为推理管道
2. 集成 CounterfactualModule（反事实推理）
3. 集成 TheoryOfMindModule（心智推理）
4. 集成 MetaphorTracker（隐喻理解）
5. 实现多跳路径推理
6. 实现归纳推理（从实例到规则）

**验证**：
- "如果所有鸟都会飞，企鹅是鸟，企鹅会飞吗？" → 正确回答
- "为什么地面湿了？" → 能推理出多个可能原因

**论文**：
- Heins et al. (2025) AXIOM — 符号推理 + 世界模型
- Pearl (2009) "Causality" — do-calculus

---

### Phase 4: 对象中心世界模型（差距5）

**目标**：从整体向量预测升级为对象级预测

**实现**：
1. 实现对象检测器（从观测中分解对象）
2. 实现对象属性编码器（颜色、形状、大小、位置）
3. 实现对象间关系编码器（接触、支撑、包含）
4. 实现对象级预测（预测每个对象的下一状态）
5. 实现想象式规划（在世界模型中rollout）

**验证**：
- 学习物理规则后，能预测 "球从桌子边缘滚落会掉到地上"
- 能规划 "要拿到高处的箱子，需要先站在椅子上"

**论文**：
- Heins et al. (2025) AXIOM — 对象中心世界模型
- Hafner et al. (2025) DreamerV3 — 想象式规划

---

### Phase 5: 多时间尺度学习（差距6）

**目标**：不同模块以不同速率学习

**实现**：
1. 实现分层学习率（感知层 1e-3，知识层 1e-5，元层 1e-7）
2. 实现选择性巩固（高不确定性记忆优先巩固）
3. 实现遗忘曲线（Ebbinghaus 曲线）
4. 实现元学习信号（记录适应速度，强化有效迁移）

**验证**：
- 学习新领域时，旧领域知识不被破坏
- 高频使用的知识保持更久，低频知识逐渐遗忘

**论文**：
- Behrouz et al. (2025) "Nested Learning" — 多时间尺度
- Kirkpatrick et al. (2017) "EWC" — 弹性权重巩固

---

### Phase 6: Empowerment驱动探索（差距4补充）

**目标**：从预测误差好奇心升级为能力感驱动

**实现**：
1. 实现 Empowerment 计算（agent 对环境的控制力）
2. 实现好奇心-能力平衡（prediction error + competence）
3. 实现多尺度好奇心（短期新颖性 + 长期规则变化）

**验证**：
- Agent 优先探索它能控制的区域
- 避免在随机噪声区域浪费时间

**论文**：
- Mantiuk et al. (2025) "From Curiosity to Competence"
- arXiv:2503.23631 "Intrinsically-Motivated Open-World Exploration"

---

### Phase 7: 感觉运动接地（差距2补充）

**目标**：概念不仅通过语言学习，还通过物理交互接地

**实现**：
1. 实现抓取、碰撞、移动等基本动作
2. 实现触觉/力反馈编码
3. 实现概念的渐进接地（具体名词→动作词→抽象概念）

**验证**：
- "红色大球" 不仅是语言描述，还有视觉+触觉+运动的多模态表示
- "推" 不仅是文字符号，还有运动序列的接地

**论文**：
- Xu et al. (2025) Nature HBM — 符号接地
- 清华大学 (2025) "Embodied AI: From LLMs to World Models"

---

## 三、核心架构变更

### 当前架构（问题）

```
learner.py (2650行)
├── _encode_text()          → 字符级，不可学习
├── _extract_entities()     → 正则，硬编码
├── _extract_relations()    → 正则，8个模式
├── _extract_causal()       → 正则，4个模式
├── _reason_from_entities() → 图遍历，2跳
├── think()                 → 只调用3个模块
└── 40+注册模块             → 大多未被调用
```

### 目标架构

```
learner.py (重构)
├── 编码层
│   ├── SubwordTokenizer    → BPE分词
│   ├── PositionalEncoding  → 位置编码
│   ├── SelfAttention       → 多头注意力
│   └── _encode_text()      → 学习到的语义表示
│
├── 提取层
│   ├── EntityExtractor     → 序列标注（可微分）
│   ├── RelationClassifier  → 关系分类（可微分）
│   ├── CausalDiscovery     → 因果发现（干预驱动）
│   └── learn_from_text()   → 端到端学习
│
├── 推理层
│   ├── DeductiveEngine     → 演绎推理
│   ├── InductiveEngine     → 归纳推理
│   ├── CounterfactualEngine→ 反事实推理
│   ├── ProbabilisticEngine → 概率推理
│   └── think()             → 统一推理管道
│
├── 世界模型
│   ├── ObjectDetector      → 对象分解
│   ├── ObjectEncoder       → 属性+关系编码
│   ├── HierarchicalPredictor→ 层级预测
│   └── ImaginationPlanner  → 想象式规划
│
├── 记忆系统
│   ├── WorkingMemory       → 注意力窗口
│   ├── EpisodicMemory      → 情节化存储
│   ├── SemanticMemory      → 结构化知识
│   ├── ConsolidationEngine → 选择性巩固
│   └── ForgettingCurve     → 遗忘曲线
│
└── 元学习
    ├── AdaptationTracker   → 适应速度记录
    ├── TransferLearner     → 跨任务迁移
    └── CurriculumScheduler → 自适应课程
```

---

## 四、关键论文清单

### 必读论文（核心理论）

1. Friston (2010) "The free-energy principle" — 自由能原理
2. LeCun (2022) "A Path Towards Autonomous Machine Intelligence" — JEPA架构
3. Scholkopf et al. (2021) "Toward Causal Representation Learning" — 因果表示
4. Hafner et al. (2025) DreamerV3 (Nature) — 世界模型
5. Heins et al. (2025) AXIOM (arXiv:2505.24784) — 对象中心推理

### 最新进展（2024-2026）

6. Behrouz et al. (2025) "Nested Learning" (NeurIPS 2025) — 持续学习
7. Mantiuk et al. (2025) "From Curiosity to Competence" — 内在动机
8. Xu et al. (2025) Nature HBM — 符号接地
9. ORBIT (2026) — 元学习
10. arXiv:2601.18858 — 组合泛化

### 实现参考

11. Vaswani et al. (2017) "Attention Is All You Need" — Transformer
12. Kipf et al. (2018) "Contrastive Learning of Structured World Models" — 对象表示
13. Whittington & Bogacz (2017) — 预测编码算法
14. Kirkpatrick et al. (2017) "EWC" — 弹性权重巩固

---

## 五、下一步行动

立即开始 Phase 1（学习到的文本编码器），因为这是所有后续改进的基础。
