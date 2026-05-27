# 从学习本源出发的AI架构设计

## 核心理念

**不是更大的模型，而是不同的学习范式。**

当前AI：数据 → 模型 → 输出（被动接收）
本系统：环境 ← Agent → 环境（主动探索）

类比：
- GPT = 一个人背了整本字典，但从未见过真实世界
- 本系统 = 一个婴儿在房间里爬来爬去，通过探索建立对世界的理解

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    社会环境 (Social)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                  │
│  │ Teacher  │  │ Peer    │  │ Human   │                  │
│  │ Agent    │  │ Agent   │  │ User    │                  │
│  └────┬────┘  └────┬────┘  └────┬────┘                  │
│       └────────────┼────────────┘                        │
│                    ▼                                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │              社会交互层 (Social Layer)             │    │
│  │  • 共同注意 (Joint Attention)                     │    │
│  │  • 脚手架 (Scaffolding)                          │    │
│  │  • 示范学习 (Demonstration Learning)              │    │
│  └────────────────────┬────────────────────────────┘    │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │              学习体 (Learner Agent)               │    │
│  │                                                   │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │         发展引擎 (Development Engine)     │    │    │
│  │  │  Stage 1: 感知运动 (Sensorimotor)        │    │    │
│  │  │  Stage 2: 前运算 (Pre-operational)       │    │    │
│  │  │  Stage 3: 具体运算 (Concrete Ops)        │    │    │
│  │  └────────────────┬────────────────────────┘    │    │
│  │                   ▼                              │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │       预测模型 (Predictive World Model)   │    │    │
│  │  │  • 观测预测: f(obs, action) → next_obs   │    │    │
│  │  │  • 奖励预测: f(obs, action) → reward     │    │    │
│  │  │  • 不确定性估计: σ(prediction)            │    │    │
│  │  └────────────────┬────────────────────────┘    │    │
│  │                   ▼                              │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │       好奇心模块 (Curiosity Module)       │    │    │
│  │  │  内在奖励 = 预测误差 × 可学习性          │    │    │
│  │  │  探索策略 = 朝向最大可学习不确定性        │    │    │
│  │  └────────────────┬────────────────────────┘    │    │
│  │                   ▼                              │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │       符号接地模块 (Grounding Module)     │    │    │
│  │  │  • 感知聚类 → 原始概念                    │    │    │
│  │  │  • 概念组合 → 复合符号                    │    │    │
│  │  │  • 社会标注 → 语言符号                    │    │    │
│  │  └────────────────┬────────────────────────┘    │    │
│  │                   ▼                              │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │       行动策略 (Action Policy)            │    │    │
│  │  │  • 当前阶段可用动作空间                   │    │    │
│  │  │  • 内在动机驱动的探索                     │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │              物理环境 (Environment)               │    │
│  │  • 2D/3D 网格世界                               │    │
│  │  • 可交互物体（颜色、形状、材质）                │    │
│  │  • 物理规律（重力、碰撞）                        │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 六大模块详细设计

### 1. 预测模型 (Predictive World Model)

**核心：** 大脑是一个分层生成模型

```python
class PredictiveModel:
    """
    学习体的"大脑"——一个不断预测下一刻的生成模型

    不是分类器，不是判别器——是预测器。
    学习信号 = 预测误差，不是标签。
    """

    def __init__(self):
        self.encoder = PerceptionEncoder()      # 感知编码
        self.predictor = NextStatePredictor()    # 下一状态预测
        self.uncertainty = UncertaintyEstimator() # 不确定性估计

    def predict(self, observation, action):
        """给定当前观测和动作，预测下一时刻的观测"""
        latent = self.encoder(observation)
        next_latent = self.predictor(latent, action)
        uncertainty = self.uncertainty(latent, action)
        return next_latent, uncertainty

    def learn(self, observation, action, actual_next_obs):
        """学习 = 减少预测误差"""
        predicted = self.predict(observation, action)
        error = self.compute_error(predicted, actual_next_obs)
        self.update(error)  # 预测误差驱动的更新
        return error
```

**关键设计决策：**
- 不使用固定架构，而是随发展阶段**生长**网络
- Stage 1：简单的感知-动作映射
- Stage 2：加入时序记忆（LSTM/GRU）
- Stage 3：加入关系推理模块

### 2. 好奇心模块 (Curiosity Module)

**核心：** 好奇心 = 可学习的预测误差

```python
class CuriosityModule:
    """
    Schmidhuber式好奇心：寻找"可学习的不确定性"

    太简单（预测误差=0）→ 无聊
    太复杂（预测误差不可减少）→ 放弃
    甜蜜点（可以学会但还没学会）→ 好奇
    """

    def compute_intrinsic_reward(self, prediction_error, error_reduction_rate):
        """
        内在奖励 = 预测误差 × 可学习性

        - prediction_error: 当前预测误差
        - error_reduction_rate: 误差减少的速率（学习进度）
        """
        # 好奇心甜蜜点：误差存在但在减少
        curiosity = prediction_error * sigmoid(error_reduction_rate)
        return curiosity

    def select_exploration_target(self, world_model, candidates):
        """选择最有学习价值的探索目标"""
        scores = []
        for candidate in candidates:
            error = world_model.estimate_error(candidate)
            learnability = world_model.estimate_learnability(candidate)
            scores.append(error * learnability)
        return candidates[argmax(scores)]
```

### 3. 发展引擎 (Development Engine)

**核心：** 学习有不可跳跃的阶段

```python
class DevelopmentEngine:
    """
    管理学习体的发展阶段

    Piaget的阶段论实现：
    - 每个阶段有特定的能力和限制
    - 高阶能力依赖低阶能力的充分发展
    - 阶段转换由认知冲突驱动
    """

    STAGES = {
        'sensorimotor': {
            'age': '0-2 equivalent',
            'abilities': ['basic_perception', 'simple_action', 'object_permanence'],
            'limitations': ['no_symbolic', 'no_abstract'],
            'promotion_criteria': {
                'prediction_accuracy': 0.8,  # 基本预测能力
                'object_permanence_test': True,  # 客体永久性
            }
        },
        'pre_operational': {
            'age': '2-7 equivalent',
            'abilities': ['symbolic_representation', 'simple_language', 'egocentric_thinking'],
            'limitations': ['no_reversible_ops', 'no_abstract_logic'],
            'promotion_criteria': {
                'symbol_grounding_score': 0.7,
                'social_reference_test': True,
            }
        },
        'concrete_operational': {
            'age': '7-11 equivalent',
            'abilities': ['logical_reasoning', 'classification', 'conservation'],
            'limitations': ['concrete_only'],
            'promotion_criteria': {
                'classification_accuracy': 0.8,
                'conservation_test': True,
            }
        }
    }

    def __init__(self):
        self.current_stage = 'sensorimotor'
        self.stage_progress = 0.0

    def check_promotion(self, agent_stats):
        """检查是否满足进入下一阶段的条件"""
        criteria = self.STAGES[self.current_stage]['promotion_criteria']
        for criterion, threshold in criteria.items():
            if agent_stats[criterion] < threshold:
                return False
        return True

    def promote(self):
        """进入下一阶段"""
        stages = list(self.STAGES.keys())
        current_idx = stages.index(self.current_stage)
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            self.stage_progress = 0.0
            # 解锁新能力
            return self.STAGES[self.current_stage]['abilities']
        return None
```

### 4. 符号接地模块 (Grounding Module)

**核心：** 符号从感知-行动经验中涌现

```python
class GroundingModule:
    """
    符号接地：从感知到抽象的渐进过程

    不是预定义的词汇表，而是从经验中涌现的概念：
    1. 感知聚类 → 原始概念（"红色"="这种视觉模式"）
    2. 概念组合 → 复合符号（"红球"="红色"+"球形"）
    3. 社会标注 → 语言符号（教师说"红球"→关联）
    """

    def __init__(self):
        self.perceptual_clusters = {}  # 感知聚类 → 原始概念
        self.symbol_mappings = {}      # 符号 → 概念映射
        self.concept_hierarchy = {}    # 概念层级

    def ground_from_perception(self, observation, action, outcome):
        """从感知经验中建立概念"""
        # 聚类相似的感知经验
        cluster_id = self.perceptual_cluster(observation)
        if cluster_id not in self.perceptual_clusters:
            self.perceptual_clusters[cluster_id] = {
                'centroid': observation,
                'examples': [],
                'associated_actions': [],
                'outcomes': []
            }
        self.perceptual_clusters[cluster_id]['examples'].append(observation)
        self.perceptual_clusters[cluster_id]['associated_actions'].append(action)

    def ground_from_social(self, symbol, referent, context):
        """从社会交互中接地符号（教师命名）"""
        # 找到与referent最匹配的感知聚类
        cluster_id = self.find_matching_cluster(referent)
        if cluster_id:
            self.symbol_mappings[symbol] = {
                'referent_cluster': cluster_id,
                'confidence': 1.0,
                'contexts': [context]
            }

    def compose_symbols(self, symbol1, symbol2):
        """概念组合：将两个符号组合为新概念"""
        # 交集语义：找到同时匹配两个符号的感知经验
        clusters1 = self.get_referent_clusters(symbol1)
        clusters2 = self.get_referent_clusters(symbol2)
        intersection = self.find_intersection(clusters1, clusters2)
        return intersection
```

### 5. 社会交互层 (Social Layer)

**核心：** 学习发生在社会交互中

```python
class SocialLayer:
    """
    社会交互：Vygotsky的ZPD + Tomasello的共同意图性

    不是简单的"从人类反馈学习"(RLHF)，
    而是真正的社会学习：共同注意、脚手架、示范。
    """

    def __init__(self):
        self.joint_attention = JointAttentionManager()
        self.scaffolding = ScaffoldingEngine()
        self.demonstration = DemonstrationRecorder()

    def joint_attention_episode(self, teacher, learner, object):
        """
        共同注意：教师和学习者同时关注同一物体

        这是词汇学习的关键框架——
        在共同注意情境中学到的词，才是真正接地的。
        """
        # 教师指向物体
        teacher.point_to(object)
        # 学习者跟随注视
        learner.follow_gaze(teacher.gaze)
        # 建立共享注意力焦点
        self.joint_attention.establish(teacher, learner, object)
        return self.joint_attention.current_focus

    def scaffold(self, teacher, learner, task):
        """
        脚手架：教师在学习者的ZPD内提供支持

        不是直接告诉答案，而是提供适当的支持，
        随着学习者能力增长逐步撤除。
        """
        zpd = self.assess_zpd(learner, task)
        if zpd == 'too_easy':
            return None  # 不需要脚手架
        elif zpd == 'too_hard':
            return self.scaffolding.simplify_task(task)
        else:  # in ZPD
            return self.scaffolding.provide_support(learner, task)

    def demonstrate(self, teacher, task):
        """
        示范学习：教师展示如何完成任务

        鹦鹉Alex的模型/竞争对手法的核心——
        通过观察他人的正确行为来学习。
        """
        demonstration = teacher.perform(task)
        self.demonstration.record(demonstration)
        return demonstration
```

### 6. 身体/环境 (Body/Environment)

**核心：** 具身认知——认知分布在身体-环境中

```python
class Environment:
    """
    物理环境：学习体的"身体"所在地

    不是静态数据集，而是可以交互的动态世界。
    学习体通过行动改变环境，环境通过反馈改变学习体。
    """

    def __init__(self, width=10, height=10):
        self.grid = np.zeros((width, height))
        self.objects = []  # 可交互物体
        self.physics = SimplePhysics()  # 物理规律

    def step(self, agent, action):
        """
        执行动作，返回结果

        关键：行动-感知闭环
        agent.act() → environment.step() → agent.perceive()
        """
        # 执行动作
        result = self.physics.apply(agent, action)
        # 返回新的观测
        new_obs = self.get_observation(agent)
        # 计算预测误差（学习信号）
        prediction_error = agent.compute_prediction_error(new_obs)
        return new_obs, prediction_error

    def add_object(self, obj):
        """添加可交互物体"""
        self.objects.append(obj)
        # 物体有属性：颜色、形状、材质、重量
        # 这些属性是学习体通过交互逐步发现的
```

---

## 最小可行学习体 (Minimum Viable Learner)

### 设计目标

一个可以在简单环境中**从零开始学习**的Agent：
- 不预训练，不预定义词汇
- 通过好奇心驱动探索
- 经历至少2个发展阶段
- 从感知经验中涌现符号
- 可以与简单的教师交互

### 实现计划

**Phase 1: 基础框架** (当前)
- 简单2D网格世界
- 基础预测模型（MLP）
- 好奇心驱动的探索
- 感知聚类 → 原始概念

**Phase 2: 发展阶段**
- 实现2个发展阶段
- 阶段转换机制
- 能力解锁

**Phase 3: 社会交互**
- 简单的教师Agent
- 共同注意机制
- 符号命名

**Phase 4: 验证与展示**
- 学习过程可视化
- 与现有方法对比
- 概念涌现展示

---

## 与现有方法的本质区别

| 维度 | 本系统 | 深度学习 | 强化学习 | 发展机器人学 |
|------|--------|---------|---------|-------------|
| 学习信号 | 预测误差（内在） | 标签（外在） | 奖励（外在） | 混合 |
| 数据来源 | 主动探索 | 被动接收 | 环境反馈 | 预设任务 |
| 发展阶段 | 有 | 无 | 无 | 部分有 |
| 社会性 | 核心模块 | 无 | 无 | 有限 |
| 符号接地 | 涌现 | 预定义 | 无 | 有限 |
| 持续学习 | 自然 | 需额外机制 | 需额外机制 | 部分支持 |

---

*架构设计完成时间：2026-05-27*
