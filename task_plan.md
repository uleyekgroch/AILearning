# 任务计划：探究学习的本质 — 从本源出发的人工智能

## 目标

深入研究人类婴儿（0-18岁）和鹦鹉的语言学习过程，提炼学习的本质机制，然后设计一个从学习本源出发的人工智能系统——不是靠堆参数和数据的"假智能"，而是真正理解"学习"本身是什么。

---

## Phase 1: 婴儿语言学习的深度研究 (0-6岁)
**Status:** complete ✅

研究婴儿从出生到6岁的完整学习路径：
- 0-6个月：感知觉醒期——声音辨别、视觉聚焦、社会性微笑
- 6-12个月：牙牙学语期——音素过滤、母语锁定、手势交流
- 12-24个月：单词爆发期——第一个词、词汇爆炸、单词组合
- 24-48个月：语法涌现期——语法规则的隐式习得、叙事能力
- 48-72个月：元认知萌芽——心智理论、抽象思维、自我意识

**关键问题：**
- 婴儿大脑的"学习算法"到底是什么？
- 不是监督学习，不是强化学习——那是什么？
- 社会交互在学习中的不可替代作用是什么？

---

## Phase 2: 鹦鹉语言学习的对比研究
**Status:** complete ✅

鹦鹉（特别是非洲灰鹦鹉Alex）的学习能力分析：
- 鹦鹉的声学学习机制——模仿vs理解
- Alex实验：标签、颜色、数量、类比推理
- 鹦鹉学习的限制——为什么停在了那里？
- 与人类婴儿的关键差异——是什么让人类突破了？

**关键问题：**
- 鹦鹉的"理解"和人类的"理解"本质区别是什么？
- 社会嵌入度如何影响学习深度？

---

## Phase 3: 学习的本质——提炼核心机制
**Status:** complete ✅

从Phase 1和Phase 2中提炼学习的本质：

1. **具身认知 (Embodied Cognition)** — 学习不是抽象计算，是身体与环境的交互
2. **社会构建 (Social Construction)** — 维果茨基的最近发展区、共同注意、社会参照
3. **主动建构 (Active Construction)** — 皮亚杰的建构主义，婴儿是主动的科学家
4. **预测编码 (Predictive Coding)** — 大脑是一个预测机器，学习 = 减少预测误差
5. **符号接地 (Symbol Grounding)** — 符号必须连接到感知和行动，否则就是空洞的
6. **发展阶梯 (Developmental Staging)** — 学习有不可跳跃的阶段，每阶段为下一阶段奠基

**关键问题：**
- 如果只能保留一个机制，哪个最根本？
- 现有AI缺失了哪些关键机制？

---

## Phase 4: 现有AI的根本缺陷分析
**Status:** complete ✅

为什么现有AI是"假智能"：
- GPT/LLM：统计模式匹配，没有接地，没有具身，没有社会
- 深度学习：需要海量数据，婴儿不需要
- 强化学习：奖励信号从哪来？婴儿的"奖励"是什么？
- 缺失的关键：主动性、好奇心、社会性、发展阶梯

---

## Phase 5: 从本源出发的AI架构设计
**Status:** complete ✅

基于学习本质设计新的AI系统：
- 核心架构选择
- 感知层设计
- 社会交互层
- 发展阶段引擎
- 好奇心/预测误差驱动的学习
- 符号接地机制

---

## Phase 6: 实现路线图
**Status:** complete ✅

从最简单的原型开始：
- 最小可行学习体 (Minimum Viable Learner)
- 阶段性验证标准
- 迭代计划

---

## Phase 8: 自适应模型选择（解决稳定性-可塑性困境）
**Status:** complete ✅

修复自适应模型选择系统，解决 v1 的四个失败模式。

- 新增 ErrorDrivenSelector（基于预测误差 + 迟滞 + 冷却期）
- 修改知识迁移：跨架构跳过权重注入，只迁移符号知识
- 关键发现：只升级不降级——降级导致从零开始的学习震荡
- 结果：自适应模型优于所有固定模型（改善 18.7%）

---

## Phase 7: 不确定性感知决策
**Status:** complete ✅

利用 FEP 的精度估计设计风险敏感任务，验证"做错有代价"的环境中不确定性估计的价值。

- 实现三个风险敏感任务（悬崖导航、危险探索、风险-收益权衡）
- 首次实现 `check_failure()=True` 的任务
- 对比 FEP Risk-OFF vs Risk-ON（风险权重扫描 0.0-2.0）
- 结果：悬崖导航中轻度风险惩罚（0.1）有效（成功率 10%→33.3%），但过强惩罚有害

**关键发现：** 风险惩罚的效果取决于任务结构。当环境有安全通道需要被发现时，轻度风险惩罚有效；当 agent 已经能很好地处理任务时，额外的风险惩罚反而有害。

---

## Phase 9: 语言涌现（从符号到语法的自发涌现）
**Status:** complete ✅

实现参照游戏，验证语法从交流需求中自发涌现。

- 创建 language_emergence.py（EmergingLanguage, Speaker, Listener, CommunicationGame）
- 将语言能力集成到 agent.py（language_game, communicate 方法）
- 创建 experiment_language.py（单符号 vs 组合系统对比）
- 结果：组合系统成功率 100% vs 单符号 22.6%，词序一致性 100%

**关键发现：** 语法不是预定义的规则，而是从交流压力中涌现的固化习惯。

---

## Phase 10: 大规模语言涌现（4 属性维度 + 3+ 符号组合）
**Status:** complete ✅

扩展到 4 个属性维度（颜色、形状、大小、材质），验证更复杂的语法结构涌现。

- 扩展 language_emergence.py：支持 3+ 符号组合、n-gram 模式、形容词层级偏好
- 创建 experiment_language_rich.py：4 种复杂度独立对比 + 渐进复杂度实验
- 结果：所有场景 100% 成功率，3符号率 100%（extreme），n-gram 模式 75-113 种

**关键发现：** 3+ 符号组合、形容词层级排序（大小>颜色>材质>形状）、n-gram 语法模式均从交流压力中涌现。

---

## Phase 11: 多 Agent 社会（方言分化与语言融合）
**Status:** complete ✅

实现多 agent 社会，每个 agent 拥有独立的语言，观察方言分化和语言融合。

- 修改 language_emergence.py：新增 LanguageAgent 类、cross_language_round、compute_language_similarity
- 新建 language_society.py：LanguageSociety 类，支持 full/star/line/groups 拓扑
- 新建 experiment_language_society.py：4 个实验
- 结果：方言分化度 13.4%（组内 0.901 vs 组间 0.781），拓扑影响趋同速度，人口越大差异越大

**关键发现：** 方言是语言演化的自然产物；网络拓扑决定语言传播速度；方言的本质是频率差异而非词汇差异。

---

## Phase 12: 跨代知识传递（文化的积累与传承）
**Status:** complete ✅

实现跨代知识传递，观察知识传承和文化积累。

- 新建 generational_transfer.py：GenerationalChain 类、teach_directly、seed_knowledge
- 新建 experiment_generational.py：4 个实验（单代传递、多代积累、机制对比、教学时长）
- 结果：知识播种是唯一能实现逐代积累的机制（6→9→10→11），直接交流无积累（6→7→7→6）

**关键发现：** 知识播种 > 直接交流；文化积累需要显式传递机制；词汇量随教学时长线性增长至饱和。

---

## Phase 13: 符号接地深化（动作、情感、因果关系）
**Status:** complete ✅

将符号接地从静态属性扩展到三个新维度：动作（动词）、情感（状态词）、因果关系（连接词）。

- 修改 language_emergence.py：新增 ACTIONS/EMOTIONS/CAUSAL_MARKERS 常量
- 新建 grounding_actions.py：ActionEvent, ActionCommunicationGame
- 新建 grounding_emotions.py：EmotionMapper, EmotionState, EmotionCommunicationGame
- 新建 grounding_causal.py：CausalRule, CausalGroundingModule, CausalCommunicationGame
- 新建 grounding_unified.py：统一场景生成
- 新建 experiment_grounding.py：4 个实验
- 结果：动作符号 10 个涌现，情感符号 8 个涌现，因果标记 1 个涌现

**关键发现：** 符号接地的统一机制（Dict[str, str]）；情感符号需要差异化接地才能涌现；接地深度的层级：静态 < 动作 < 情感 < 因果。

---

## Phase 14: 复杂语法结构（否定、时态、递归从句）
**Status:** complete ✅

将语言能力从简单组合扩展到复杂语法结构。

- 修改 language_emergence.py：新增否定/时态/从句常量，修改 Listener 支持否定和从句匹配，修改 Speaker 支持否定和从句生成
- 新建 grounding_temporal.py：TemporalEvent, TemporalCommunicationGame
- 新建 grounding_negation.py：NegationCommunicationGame
- 新建 grounding_recursive.py：RecursiveCommunicationGame
- 新建 experiment_complex_grammar.py：4 个实验
- 结果：时态符号 3 个涌现（past/present/future），否定和从句未涌现

**关键发现：** 时态作为独立特征维度天然涌现；否定和从句在单值特征系统中永远不是最优解（正向描述 1 符号 < 否定 2 符号 < 从句 3 符号）；机制已正确实现，但场景复杂度不足。

---

## Phase 15: 复合特征让否定涌现（多值特征维度）
**Status:** complete ✅

解决 Phase 14 的核心问题：在单值特征系统中否定永远不是最优解。

**核心发现：** 子集关系是否定涌现的充要条件。当物体 A 的特征是物体 B 的子集时，正向描述永远无法区分它们——只有否定可以。

**实现：**
- 新建 grounding_compound.py：多值特征系统（Dict[str, Set[str]]），CompoundSpeaker, CompoundListener
- 新建 experiment_compound_negation.py：4 个实验（简单子集、涌现、复杂度对比、稳定性）
- 更新理论框架（5.22 复合特征与否定涌现）

**实验结果：**
- 否定涌现率：100%（5/5 次运行）
- 平均否定使用率：35.5%
- 否定成功率：100%（185/185 次使用全部成功）
- "not" 符号进入词汇表

**关键发现：**
1. 否定不是"更复杂的语法"，而是处理子集关系的必要工具
2. 多值特征是否定涌现的充要条件
3. 效率驱动的语法选择：否定（2符号）比正向（3+符号）更短时被选择
4. 与人类发展一致：否定习得（2-3岁）对应特征理解从"列表"到"集合"的跃迁

---

## Phase 16a: 从句在多值系统中涌现
**Status:** complete ✅

实现从句机制，验证在多值特征系统中从句是否涌现。

- 修改 grounding_compound.py：新增 _try_relative_clause、_interpret_with_relative、generate_clause_scene
- 新建 experiment_compound_clause.py：5 个实验
- 结果：从句在无约束条件下不涌现（2符号组合总是比3符号从句更高效）

**关键发现：** 从句 "X that Y" 的匹配语义等价于 "X AND Y"。在离散符号系统中，2符号组合总是比3符号从句更高效。从句需要时间压力（max_len约束）才能成为最优策略。机制已正确实现。

---

## Phase 16b: 时间压力下的语言效率
**Status:** complete ✅

限时交流，迫使 agent 使用更短（可能是否定/从句）的表达。

- 修改 grounding_compound.py：CompoundSpeaker.describe 增加 max_len 参数
- 新建 experiment_time_pressure.py：4 个实验
- 结果：max_len=2 时否定成为最高效策略（成功率 41.8%→83.0%），从句仍不涌现

**关键发现：** 时间压力显著影响语法策略选择。否定（2符号）是最紧凑的区分策略。从句（3符号）在当前系统中永远不是最优解。

---

## Phase 17: 叙事与篇章（超越句子）
**Status:** complete ✅

将语言能力从句子扩展到叙事和篇章。

- 新建 narrative.py：Event, Narrative, NarrativeSpeaker, NarrativeListener, NarrativeGame
- 新建 experiment_narrative.py：4 个实验
- 结果："then" 立即涌现（100% 使用率），"because" 未涌现（需要因果推理机制）

**关键发现：** 时序连接词 "then" 从多事件交流中自然涌现（涌现轮次 0）。因果连接词 "because" 需要因果推理机制才能涌现。叙事是语言从句子到篇章的自然扩展。

---

## Phase 18: 元认知与自我反思
**Status:** complete ✅

Agent 对自身学习过程的觉察。

- 新建 metacognition.py：MetacognitiveAgent, MetacognitiveTeacher, MetacognitiveGame
- 新建 experiment_metacognition.py：4 个实验
- 结果：元认知机制有效（+9% 成功率），但信号是内部的，不进入词汇表

**关键发现：** 元认知通过调整行为（而非通信）来改善学习。信号是内部评估机制，不需要"说出来"。这与人类元认知一致——我们内心知道"我不确定"，但不一定说出来。

---

## Phase 19: 因果推理（让 "because" 涌现）
**Status:** complete ✅

解决 Phase 17 的核心问题："because" 未涌现是因为系统无法区分因果关系和时序关系。

- 新建 grounding_causal_reasoning.py：CausalWorld, CausalModel, CausalReasoningAgent, CausalCommunicationGame
- 新建 experiment_causal_reasoning.py：4 个实验
- 修改 language_emergence.py：添加 CAUSAL_REASONING_MARKERS, TEMPORAL_MARKERS, PERSPECTIVE_MARKERS, ABSTRACT_MARKERS, TOOL_MARKERS，更新 _symbol_category()

**实验结果：**
- "because" 涌现率：100%（5/5 次运行）
- 平均 "because" 使用：3.2 次/200 轮
- 因果推理改善：+2.3%（有因果推理 vs 无因果推理）
- "because" 成功率：30.8%（虚假相关场景）

**关键发现：**
1. "because" 从因果推理需求中涌现：100% 涌现率证明机制有效
2. 混杂场景是涌现的关键：虚假相关和真正因果共存时，"because" 提供预测优势
3. 标记选择本身成为通信成功的关键因素：这是第一次让标记选择有语义差异
4. 使用率较低说明需要更复杂的场景来增加 "because" 的通信压力

---

## Phase 20: 心智理论（理解他人的信念和意图）
**Status:** complete ✅

理解他人的信念和意图，实现视角切换。

- 新建 grounding_theory_of_mind.py：Perspective, TheoryOfMindAgent, AsymmetricCommunicationGame
- 新建 experiment_theory_of_mind.py：5 个实验

**实验结果：**
- 视角调整使用：140-219 次/300 轮
- 视角调整成功率：82.2%
- 视角标记涌现率：0%（"know", "think", "believe" 未涌现）
- 有心智理论 vs 无心智理论：-9.0%（反而更差）

**关键发现：**
1. 视角调整机制有效：Agent 能根据 Listener 的知识调整描述
2. 视角标记词未涌现：当前场景不需要用语言传达认知状态
3. 有心智理论反而差：视角调整策略过于保守，过滤掉了有用信息
4. 需要更复杂的场景（如需要解释自己的信念状态）来驱动视角标记涌现

---

## Phase 21: 抽象推理（类比、隐喻、概念迁移）
**Status:** complete ✅

验证 "like" 从跨领域交流压力中涌现。

- 新建 grounding_abstraction.py：Concept, AnalogyMapping, AbstractionModule, AbstractCommunicationGame
- 新建 experiment_abstraction.py：6 个实验
- 核心设计：特征隔离 + 关系共享，Listener 只理解已知领域的特征
- 结果："like" 涌现率 100%（5/5），类比使用 500/500，成功率 100%

**关键发现：** 特征隔离是类比涌现的充要条件。不同领域的特征值完全不同，但共享通用关系类型（acts_on, avoided_by, located_in）。类比通过关系结构（而非特征）建立理解桥梁。

---

## Phase 22: 工具使用（问题解决和规划）
**Status:** complete ✅

验证 "use" 和 "for" 从工具选择压力中涌现。

- 新建 grounding_tool_use.py：Tool, Goal, ToolCommunicationGame, BaselineToolGame
- 新建 experiment_tool_use.py：5 个实验
- 核心设计：目标依赖的物体重释，外观唯一性检测，功能描述策略
- 结果："use" 和 "for" 涌现率 100%（5/5），功能描述使用 202/500，成功率 100%

**关键发现：** 功能描述是外观歧义时的最优策略。系统自动检测外观唯一性，切换到功能描述。"use" 和 "for" 从这种切换需求中涌现。

---

## Phase 20b: 置信度驱动的视角标记涌现
**Status:** complete ✅

修复 Phase 20 的视角标记未涌现问题。

- 修复 grounding_theory_of_mind.py bug（line 58: probability → proposition）
- 新增 generate_confidence_scenario()（identical_visual 模式）
- 新增 ConfidenceCommunicationGame, BaselineConfidenceGame
- 修改 play_round()：接入 choose_perspective_marker()，低置信度时添加噪声
- 修改 _listener_interpret()：检测标记并应用置信度加分
- 新增 experiment_theory_of_mind.py 实验 6-10
- 结果："know"/"think"/"believe" 涌现率 100%（5/5），涌现轮次 11

**实验结果：**
- 实验 7（置信度差异）：成功率 69.0%，标记使用 462/500
- "know" 成功率 100%，"think" 77.2%，"believe" 35.7%
- 实验 8（对比）：-0.5%（标记提供元信息但不直接改善准确率）
- 实验 10（稳定性）：100% 涌现率（5/5）

**关键发现：**
1. 视角标记从置信度差异中涌现：100% 涌现率
2. "know" = 高置信度（总是正确），"think" = 中置信度，"believe" = 低置信度
3. 标记提供元信息（置信度），但不直接改善通信准确率
4. 与人类发展一致：心智理论在 4-5 岁发展

---

## Phase 23: 跨模态语言（视觉、听觉、触觉整合）
**Status:** complete ✅

验证听觉/触觉符号从视觉模糊中涌现。

- 新建 grounding_crossmodal.py：CrossModalObject, CrossModalGame, BaselineCrossModalGame
- 新建 experiment_crossmodal.py：5 个实验
- 修改 language_emergence.py：添加 AUDITORY_SYMBOLS, TACTILE_SYMBOLS
- 结果：跨模态符号涌现率 100%（5/5），20 个符号全部涌现，成功率 100%

**实验结果：**
- 实验 2（视觉模糊）：100% 成功率，跨模态使用 500/500
- 实验 3（纯跨模态）：100% 成功率，20 个符号涌现
- 实验 4（对比）：跨模态 100% vs 纯视觉 48.7% → +51.3% 改善
- 实验 5（稳定性）：100% 涌现率（5/5）

**关键发现：**
1. 跨模态符号从视觉模糊中立即涌现（第 0 轮）
2. 听觉/触觉特征在视觉无法区分时成为必要描述
3. 跨模态描述比纯视觉描述准确率高 51.3%
4. 符号接地不限于视觉——所有感知模态都能接地

---

## 决策记录

| 决策 | 理由 | 日期 |
|------|------|------|
| 从婴儿发展心理学入手 | 这是人类学习的最完整记录 | 2026-05-27 |
| 对比鹦鹉学习 | 提供控制变量，找出人类独有的关键 | 2026-05-27 |

## 错误记录

| 错误 | 尝试 | 解决 |
|------|------|------|
| (暂无) | | |
