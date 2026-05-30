"""
学习体模块：从学习本源出发的AI Agent

核心理念：
- 学习信号 = 预测误差（不是标签）
- 驱动力 = 好奇心（不是损失函数）
- 数据来源 = 主动探索（不是被动接收）
- 发展 = 阶段性（不是一次性训练）

这是系统的核心——一个会"好奇"的Agent，
通过不断预测世界来学习，通过预测失败来成长。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import json

# 导入神经网络预测模型
from predictive_nn import NeuralNetworkPredictor, AdaptiveLearningRatePredictor

# 导入自适应模型选择器
from complexity_estimator import AdaptiveModelSelector, ErrorDrivenSelector

# 导入知识迁移系统
from knowledge_transfer import KnowledgeTransferSystem

# 导入语言涌现系统
from language_emergence import CommunicationGame, EmergingLanguage
from environment_language_bridge import EnvironmentLanguageBridge, ExperienceDrivenScenarioGenerator
from adaptive_strategy import AdaptiveCommunicationGame


@dataclass
class Experience:
    """一次经验：观测→动作→结果"""
    observation: np.ndarray
    action: int
    next_observation: np.ndarray
    prediction_error: float
    timestamp: int


class PredictiveModel:
    """
    预测模型：学习体的"大脑"

    不是分类器，不是判别器——是预测器。
    学习信号 = 预测误差，不是标签。

    这是自由能原理的实现：
    大脑不断预测下一刻，当预测失败时更新模型。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 简单的线性预测模型（后续可以替换为神经网络）
        # 预测: f(obs, action) → next_obs
        self.W_obs = np.random.randn(obs_dim, hidden_dim) * 0.01
        self.W_action = np.random.randn(action_dim, hidden_dim) * 0.01
        self.W_out = np.random.randn(hidden_dim, obs_dim) * 0.01

        # 学习率
        self.lr = 0.01

        # 预测误差历史（用于计算学习进度）
        self.error_history = deque(maxlen=100)

    def predict(self, obs: np.ndarray, action: int) -> np.ndarray:
        """
        预测下一时刻的观测

        给定当前观测和动作，预测会发生什么。
        这是学习体对世界的"心智模型"。
        """
        # 编码动作为one-hot
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0

        # 前向传播
        hidden = np.tanh(obs @ self.W_obs + action_vec @ self.W_action)
        prediction = hidden @ self.W_out

        return prediction

    def learn(self, obs: np.ndarray, action: int, actual_next_obs: np.ndarray) -> float:
        """
        学习 = 预测编码 + 局部 Hebbian 更新

        实现 Whittington & Bogacz (2017) 的预测编码算法：
        1. 迭代推理：每层计算局部残差 ε = μ - f(W × μ_below)
        2. 反馈连接传输上层误差（不是链式法则）
        3. 权重更新：ΔW = η × ε_post × f' × μ_pre^T（Hebbian）

        与反向传播的根本区别：
        - 反向传播：δ_l = δ_{l+1} @ W_{l+1}.T × f'（链式法则，非局部）
        - 预测编码：ε_l = μ_l - f(W_l × μ_{l-1})（局部残差）
        收敛后两者等价，但每步操作都是局部的。
        """
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0

        # === Step 1: 前向传播初始化 μ（信念）===
        z_hidden = obs @ self.W_obs + action_vec @ self.W_action
        mu_hidden = np.tanh(z_hidden)

        # 常量：输入不变，tanh 结果在迭代中恒定
        predicted_hidden = mu_hidden.copy()

        # === Step 2: 迭代推理（自适应停止）===
        max_steps = 50
        inference_lr = 0.1
        convergence_threshold = 1e-4
        for _ in range(max_steps):
            prev_hidden = mu_hidden.copy()

            # 局部残差：每层只用自己的预测和实际值
            epsilon_out = actual_next_obs - mu_hidden @ self.W_out
            epsilon_hidden = mu_hidden - predicted_hidden

            # 反馈信号：通过反馈连接传输，不是链式法则
            feedback = self.W_out @ epsilon_out * (1 - mu_hidden ** 2)

            # 更新隐藏层信念
            mu_hidden -= inference_lr * (-epsilon_hidden + feedback)

            # 自适应停止
            if np.mean((mu_hidden - prev_hidden) ** 2) < convergence_threshold:
                break

        # === Step 3: 局部 Hebbian 权重更新 ===
        epsilon_out = actual_next_obs - mu_hidden @ self.W_out
        epsilon_hidden = mu_hidden - predicted_hidden
        f_prime = 1 - mu_hidden ** 2

        # ΔW_out = η × μ_hidden × ε_out^T （Hebbian：突触后误差 × 突触前活动）
        self.W_out += self.lr * np.outer(mu_hidden, epsilon_out)

        # ΔW_obs = η × obs × (ε_hidden × f')^T （Hebbian）
        self.W_obs += self.lr * np.outer(obs, epsilon_hidden * f_prime)
        self.W_action += self.lr * np.outer(action_vec, epsilon_hidden * f_prime)

        prediction_error = float(np.mean(epsilon_out ** 2))
        self.error_history.append(prediction_error)

        return prediction_error

    def get_learning_progress(self) -> float:
        """
        计算学习进度

        学习进度 = 误差减少的速率
        这是好奇心的关键指标——
        如果误差在减少，说明正在学习，应该继续探索。
        """
        if len(self.error_history) < 10:
            return 0.0

        recent = list(self.error_history)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])

        # 学习进度 = (旧误差 - 新误差) / 旧误差
        if first_half > 0:
            progress = (first_half - second_half) / first_half
        else:
            progress = 0.0

        return max(0.0, progress)


class CuriosityModule:
    """
    好奇心模块：Schmidhuber式好奇心

    好奇心 = 可学习的预测误差

    太简单（预测误差=0）→ 无聊
    太复杂（预测误差不可减少）→ 放弃
    甜蜜点（可以学会但还没学会）→ 好奇

    这是学习的内在驱动力——
    不需要外在奖励，学习本身就是奖励。
    """

    def __init__(self):
        # 好奇心参数
        self.alpha = 0.5  # 预测误差权重
        self.beta = 0.5   # 学习进度权重

        # 内在奖励历史
        self.reward_history = deque(maxlen=100)

    def compute_intrinsic_reward(self,
                                  prediction_error: float,
                                  learning_progress: float) -> float:
        """
        计算内在奖励

        内在奖励 = 预测误差 × 可学习性

        - prediction_error: 当前预测误差（越大越"新奇"）
        - learning_progress: 学习进度（越大越"可学"）

        甜蜜点：误差存在但在减少
        """
        # 好奇心 = 误差 × 可学习性
        # 如果误差很大但不可学习（太复杂），好奇心低
        # 如果误差很小（太简单），好奇心低
        # 如果误差中等且在减少（正在学习），好奇心高
        curiosity = self.alpha * prediction_error + self.beta * learning_progress

        # 归一化到[0, 1]
        curiosity = np.clip(curiosity, 0.0, 1.0)

        self.reward_history.append(curiosity)

        return curiosity

    def should_explore(self, current_error: float, threshold: float = 0.1) -> bool:
        """判断是否应该继续探索"""
        if len(self.reward_history) < 10:
            return True

        avg_reward = np.mean(self.reward_history)
        return avg_reward > threshold


class DevelopmentEngine:
    """
    发展引擎：管理学习体的发展阶段

    Piaget的阶段论实现：
    - 每个阶段有特定的能力和限制
    - 高阶能力依赖低阶能力的充分发展
    - 阶段转换由认知冲突驱动

    这是学习的"脚手架"——
    不是一次性训练所有能力，
    而是按照发展阶段逐步解锁。
    """

    STAGES = {
        'sensorimotor': {
            'name': '感知运动阶段',
            'description': '通过感觉和运动与世界交互',
            'age': '0-2',
            'abilities': ['basic_perception', 'simple_action', 'object_permanence'],
            'limitations': ['no_symbolic', 'no_abstract'],
            'promotion_criteria': {
                'prediction_accuracy': 0.5,
                'exploration_diversity': 0.3,
                'total_steps': 50,
            }
        },
        'early_preoperational': {
            'name': '前运算早期',
            'description': '符号功能出现，简单沟通',
            'age': '2-4',
            'abilities': ['symbolic_representation', 'simple_communication', 'egocentric_thinking'],
            'limitations': ['no_reversible_ops', 'no_abstract_logic', 'no_grammar'],
            'promotion_criteria': {
                'symbol_count': 2,
                'social_reference': True,
                'total_steps': 200,
            }
        },
        'late_preoperational': {
            'name': '前运算后期',
            'description': '语法、否定、时态、基本叙事',
            'age': '4-6',
            'abilities': ['grammar', 'negation', 'tense', 'basic_narrative', 'simple_classification'],
            'limitations': ['no_conservation', 'no_transitivity'],
            'promotion_criteria': {
                'symbol_count': 4,
                'experience_count': 300,
                'classification_accuracy': 0.3,
                'total_steps': 500,
            }
        },
        'early_concrete': {
            'name': '具体运算早期',
            'description': '守恒、分类、序列化',
            'age': '6-8',
            'abilities': ['conservation', 'classification', 'seriation', 'logical_reasoning'],
            'limitations': ['concrete_only', 'no_hypothetical'],
            'promotion_criteria': {
                'classification_accuracy': 0.5,
                'conservation_test': True,
                'seriation_score': 0.3,
                'symbol_count': 3,
                'experience_count': 500,
                'total_steps': 800,
            }
        },
        'late_concrete': {
            'name': '具体运算后期',
            'description': '传递性、多步规划、类比推理',
            'age': '8-11',
            'abilities': ['transitivity', 'multi_step_planning', 'analogy', 'abstract_reasoning'],
            'limitations': ['no_hypothetical_deductive'],
            'promotion_criteria': {
                'abstract_reasoning_score': 0.3,
                'planning_score': 0.4,
                'classification_accuracy': 0.6,
                'experience_count': 800,
                'total_steps': 1200,
            }
        },
        'early_formal': {
            'name': '形式运算早期',
            'description': '假设检验、反事实推理、系统性推理',
            'age': '11-13',
            'abilities': ['hypothesis_testing', 'counterfactual_thinking', 'systematic_reasoning'],
            'limitations': [],
            'promotion_criteria': {
                'hypothesis_confirmed': 0.5,
                'counterfactual_diversity': 0.3,
                'abstract_reasoning_score': 0.4,
                'total_steps': 1800,
            }
        },
        'late_formal': {
            'name': '形式运算后期',
            'description': '科学推理、视角协调、元认知',
            'age': '13-15',
            'abilities': ['scientific_reasoning', 'perspective_coordination', 'meta_cognition'],
            'limitations': [],
            'promotion_criteria': {
                'meta_cognition': 0.5,
                'perspective_coordination': 0.4,
                'hypothesis_confirmed': 0.6,
                'total_steps': 2500,
            }
        },
        'adolescent': {
            'name': '青少年阶段',
            'description': '抽象问题解决、道德推理、身份认同',
            'age': '15-17',
            'abilities': ['abstract_problem_solving', 'moral_reasoning', 'identity_formation',
                          'hypothetical_deductive', 'scientific_thinking'],
            'limitations': [],
            'promotion_criteria': {}
        }
    }

    def __init__(self):
        self.current_stage = 'sensorimotor'
        self.stage_progress = 0.0
        self.stage_history = [('sensorimotor', 0)]

    def get_current_abilities(self) -> List[str]:
        """获取当前阶段的能力"""
        return self.STAGES[self.current_stage]['abilities']

    def get_current_limitations(self) -> List[str]:
        """获取当前阶段的限制"""
        return self.STAGES[self.current_stage]['limitations']

    def check_promotion(self, agent_stats: Dict) -> bool:
        """检查是否满足进入下一阶段的条件"""
        criteria = self.STAGES[self.current_stage]['promotion_criteria']

        for criterion, threshold in criteria.items():
            if criterion not in agent_stats:
                return False

            value = agent_stats[criterion]
            if isinstance(threshold, bool):
                if value != threshold:
                    return False
            else:
                if value < threshold:
                    return False

        return True

    def promote(self) -> Optional[List[str]]:
        """进入下一阶段"""
        stages = list(self.STAGES.keys())
        current_idx = stages.index(self.current_stage)

        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            self.stage_progress = 0.0
            self.stage_history.append((self.current_stage, len(self.stage_history)))

            # 返回新解锁的能力
            return self.STAGES[self.current_stage]['abilities']

        return None

    def get_stage_info(self) -> Dict:
        """获取当前阶段信息"""
        return {
            'stage': self.current_stage,
            'name': self.STAGES[self.current_stage]['name'],
            'description': self.STAGES[self.current_stage]['description'],
            'abilities': self.STAGES[self.current_stage]['abilities'],
            'limitations': self.STAGES[self.current_stage]['limitations'],
            'progress': self.stage_progress
        }


class GroundingModule:
    """
    符号接地模块：从感知到抽象的渐进过程

    不是预定义的词汇表，而是从经验中涌现的概念：
    1. 感知聚类 → 原始概念（"红色"="这种视觉模式"）
    2. 概念组合 → 复合符号（"红球"="红色"+"球形"）
    3. 社会标注 → 语言符号（教师说"红球"→关联）

    这是符号接地问题的实现——
    符号的意义来自感知-行动经验，不是来自其他符号。
    """

    def __init__(self):
        # 感知聚类 → 原始概念
        self.perceptual_clusters: Dict[int, Dict] = {}
        self.cluster_counter = 0

        # 符号 → 概念映射
        self.symbol_mappings: Dict[str, Dict] = {}

        # 概念层级
        self.concept_hierarchy: Dict[str, List[str]] = {}

    def ground_from_perception(self, observation: np.ndarray) -> int:
        """
        从感知经验中建立概念

        将相似的感知经验聚类为"概念"。
        这是符号接地的第一步——
        先有感知范畴，才有符号标签。
        """
        # 简单的聚类：找到最接近的已有聚类
        best_cluster = None
        best_distance = float('inf')

        for cluster_id, cluster in self.perceptual_clusters.items():
            # 确保维度匹配
            min_len = min(len(observation), len(cluster['centroid']))
            distance = np.linalg.norm(observation[:min_len] - cluster['centroid'][:min_len])
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster_id

        # 如果距离太远，创建新聚类
        if best_distance > 0.5 or best_cluster is None:
            cluster_id = self.cluster_counter
            self.cluster_counter += 1
            self.perceptual_clusters[cluster_id] = {
                'centroid': observation.copy(),
                'count': 1,
                'examples': [observation.copy()]
            }
        else:
            # 更新已有聚类
            cluster_id = best_cluster
            cluster = self.perceptual_clusters[cluster_id]
            cluster['count'] += 1
            cluster['examples'].append(observation.copy())
            # 更新中心（处理维度不匹配的情况）
            min_len = min(len(observation), len(cluster['centroid']))
            new_centroid = cluster['centroid'].copy()
            new_centroid[:min_len] = (
                cluster['centroid'][:min_len] * (cluster['count'] - 1) + observation[:min_len]
            ) / cluster['count']
            cluster['centroid'] = new_centroid

        return cluster_id

    def ground_from_social(self, symbol: str, referent_obs: np.ndarray, context: str = ""):
        """
        从社会交互中接地符号

        当教师命名一个物体时，
        将符号与当前感知经验关联。
        这是符号接地的第二步——
        符号的意义来自社会交互中的共同注意。
        """
        # 找到与referent最匹配的感知聚类
        cluster_id = self.ground_from_perception(referent_obs)

        # 建立符号映射
        if symbol not in self.symbol_mappings:
            self.symbol_mappings[symbol] = {
                'referent_clusters': [cluster_id],
                'confidence': 1.0,
                'contexts': [context] if context else [],
                'usage_count': 1
            }
        else:
            mapping = self.symbol_mappings[symbol]
            if cluster_id not in mapping['referent_clusters']:
                mapping['referent_clusters'].append(cluster_id)
            mapping['usage_count'] += 1
            mapping['confidence'] = min(1.0, mapping['confidence'] + 0.1)
            if context and context not in mapping['contexts']:
                mapping['contexts'].append(context)

    def get_symbol_meaning(self, symbol: str) -> Optional[Dict]:
        """获取符号的含义"""
        return self.symbol_mappings.get(symbol)

    def get_grounded_symbols(self) -> List[str]:
        """获取所有已接地的符号"""
        return list(self.symbol_mappings.keys())

    def find_similar_concepts(self, observation: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
        """找到与给定观测最相似的概念"""
        similarities = []

        for cluster_id, cluster in self.perceptual_clusters.items():
            distance = np.linalg.norm(observation - cluster['centroid'])
            similarity = 1.0 / (1.0 + distance)
            similarities.append((cluster_id, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_object_features(self, obj) -> Dict[str, str]:
        """
        从物体提取特征符号

        将物体的属性转换为符号标签，
        用于语言描述和交流。
        """
        features = {}
        if hasattr(obj, 'color') and obj.color:
            features['color'] = obj.color
        if hasattr(obj, 'shape') and obj.shape:
            features['shape'] = obj.shape
        if hasattr(obj, 'weight'):
            if obj.weight < 1.0:
                features['size'] = 'small'
            elif obj.weight > 1.2:
                features['size'] = 'big'
        return features


class LearningAgent:
    """
    学习体：从学习本源出发的AI Agent

    这是系统的核心——一个会"好奇"的Agent，
    通过不断预测世界来学习，通过预测失败来成长。

    与现有AI的本质区别：
    1. 学习信号 = 预测误差（不是标签）
    2. 驱动力 = 好奇心（不是损失函数）
    3. 数据来源 = 主动探索（不是被动接收）
    4. 发展 = 阶段性（不是一次性训练）
    """

    def __init__(self, obs_dim: int = 10, action_dim: int = 5,
                 model_type: str = 'linear'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model_type = model_type

        # 自适应模型选择器
        self.model_selector = AdaptiveModelSelector()
        self.error_selector = ErrorDrivenSelector()

        # 知识迁移系统
        self.knowledge_transfer = KnowledgeTransferSystem()

        # 核心模块 - 支持可切换的预测模型
        self._init_predictive_model(model_type)

        self.curiosity = CuriosityModule()
        self.development = DevelopmentEngine()
        self.grounding = GroundingModule()

        # 语言涌现系统
        self.language_game = CommunicationGame()
        self.language = self.language_game.language

        # Phase 26: 环境-语言整合
        self.bridge = EnvironmentLanguageBridge()
        self.scenario_gen = ExperienceDrivenScenarioGenerator(self.bridge)
        self.adaptive_game = AdaptiveCommunicationGame()
        self.communication_stats = {'total': 0, 'success': 0}

    def _init_predictive_model(self, model_type: str):
        """初始化预测模型"""
        if model_type == 'neural_network':
            self.predictive_model = NeuralNetworkPredictor(self.obs_dim, self.action_dim)
        elif model_type == 'adaptive_nn':
            self.predictive_model = AdaptiveLearningRatePredictor(self.obs_dim, self.action_dim)
        elif model_type == 'adaptive':
            # 自适应模式：初始使用线性模型，后续根据复杂度切换
            self.predictive_model = PredictiveModel(self.obs_dim, self.action_dim)
        else:
            self.predictive_model = PredictiveModel(self.obs_dim, self.action_dim)

        # 经验记忆
        self.experiences: List[Experience] = []
        self.step_count = 0

        # 统计信息
        self.stats = {
            'total_steps': 0,
            'total_prediction_error': 0.0,
            'symbols_learned': 0,
            'stage_changes': 0,
            'social_interactions': 0,
        }

        # 模仿学习状态
        self._last_teacher_action = None
        self._last_teacher_demo_action = None

    def perceive(self, observation: dict) -> np.ndarray:
        """
        感知：将环境观测转换为内部表征

        这是感知编码——
        将原始的感觉输入转换为可处理的向量。
        """
        # 提取agent位置
        agent_pos = observation['agent_position']

        # 提取可见物体的特征
        visible_objects = observation['visible_objects']
        if visible_objects:
            # 取最近的物体
            closest = min(visible_objects, key=lambda x: x['distance'])
            obj_obs = closest['object'].to_observation()
        else:
            # 没有可见物体时的默认观测
            obj_obs = np.zeros(10)

        # 组合观测
        obs = np.concatenate([agent_pos, obj_obs])

        # 确保维度正确
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return obs

    def act(self, observation: dict) -> int:
        """
        行动：选择动作

        动作选择策略：
        1. 模仿学习：以一定概率复制教师的示范动作
        2. 好奇心驱动：选择最能减少预测误差的动作
        """
        obs = self.perceive(observation)

        # 获取当前阶段可用的动作空间
        available_actions = self._get_available_actions()

        # 模仿学习：以一定概率复制教师示范
        if self._last_teacher_demo_action is not None:
            imitation_probs = {
                'sensorimotor': 0.3,
                'early_preoperational': 0.2,
                'late_preoperational': 0.15,
                'early_concrete': 0.1,
                'late_concrete': 0.05,
                'early_formal': 0.03,
                'late_formal': 0.02,
                'adolescent': 0.01,
            }
            imitate_prob = imitation_probs.get(
                self.development.current_stage, 0.1
            )
            if np.random.random() < imitate_prob:
                demo_action = self._last_teacher_demo_action
                if demo_action in available_actions:
                    return demo_action

        # 好奇心驱动的动作选择
        best_action = None
        best_curiosity = -float('inf')

        for action in available_actions:
            # 预测执行该动作后的结果
            predicted_next = self.predictive_model.predict(obs, action)

            # 计算该动作的好奇心值
            # 好奇心 = 预测不确定性 × 可学习性
            prediction_uncertainty = np.std(predicted_next)  # 不确定性
            learning_progress = self.predictive_model.get_learning_progress()

            curiosity = self.curiosity.compute_intrinsic_reward(
                prediction_uncertainty, learning_progress
            )

            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best_action = action

        # 如果没有明确的好奇心驱动，随机探索
        if best_action is None:
            best_action = np.random.choice(available_actions)

        return best_action

    def _get_available_actions(self) -> List[int]:
        """获取当前阶段可用的动作"""
        base_actions = [0, 1, 2, 3]  # 上下左右

        stage = self.development.current_stage
        if stage in ('early_preoperational', 'late_preoperational',
                     'early_concrete', 'late_concrete'):
            base_actions.append(4)  # 推
        elif stage in ('early_formal', 'late_formal', 'adolescent'):
            base_actions = list(range(self.action_dim))  # 全部动作

        return base_actions

    def adapt_model_to_environment(self, objects: List[Dict], grid_size: Tuple[int, int]):
        """
        自适应切换模型（兼容旧接口）

        新方案使用预测误差驱动（ErrorDrivenSelector），
        objects/grid_size 参数保留但不再使用。
        """
        if self.model_type != 'adaptive':
            return

        self._adapt_by_error()

    def _adapt_by_error(self, last_error: float = None):
        """
        基于预测误差的自适应切换

        使用 ErrorDrivenSelector 判断是否需要切换模型。
        跨架构切换时直接替换（不渐进过渡），只迁移符号知识。
        """
        # 使用最后一次学习的误差
        if last_error is None:
            if self.predictive_model.error_history:
                last_error = self.predictive_model.error_history[-1]
            else:
                return

        # 获取当前模型类型
        current_type_name = type(self.predictive_model).__name__
        current_type = {
            'PredictiveModel': 'linear',
            'NeuralNetworkPredictor': 'neural_network',
            'AdaptiveLearningRatePredictor': 'adaptive_nn',
        }.get(current_type_name, 'linear')

        # 判断是否需要切换
        should_switch, target_type = self.error_selector.should_switch(
            last_error, current_type
        )

        if should_switch:
            stats = self.error_selector.get_stats()
            print(f"\n模型切换 #{stats['switch_count']}: "
                  f"{current_type} → {target_type} "
                  f"(平均误差={stats['avg_error']:.4f})")

            # 跨架构切换：直接替换，不渐进过渡
            # 提取符号知识
            old_grounding = {
                'type': 'grounding',
                'perceptual_clusters': {},
                'symbol_mappings': {}
            }
            for cid, cluster in self.grounding.perceptual_clusters.items():
                old_grounding['perceptual_clusters'][cid] = {
                    'centroid': cluster['centroid'].copy(),
                    'count': cluster['count']
                }
            for sym, mapping in self.grounding.symbol_mappings.items():
                old_grounding['symbol_mappings'][sym] = {
                    'referent_clusters': mapping['referent_clusters'].copy(),
                    'confidence': mapping['confidence'],
                    'usage_count': mapping['usage_count']
                }

            # 创建新模型
            from predictive_nn import NeuralNetworkPredictor
            if target_type == 'neural_network':
                self.predictive_model = NeuralNetworkPredictor(self.obs_dim, self.action_dim)
            else:
                self.predictive_model = PredictiveModel(self.obs_dim, self.action_dim)

            # 注入符号知识（完全迁移）
            self.knowledge_transfer.injector.inject_to_grounding(
                self.grounding, old_grounding, injection_rate=1.0
            )

    def learn_from_experience(self, observation: dict, action: int,
                              next_observation: dict,
                              extrinsic_reward: float = 0.0) -> float:
        """
        从经验中学习

        学习 = 减少预测误差
        这是学习的核心机制——
        当预测与实际不符时，调整模型。

        参数：
            extrinsic_reward: 外在奖励（来自任务），与内在好奇心奖励融合
        """
        obs = self.perceive(observation)
        next_obs = self.perceive(next_observation)

        # 学习：更新预测模型
        prediction_error = self.predictive_model.learn(obs, action, next_obs)

        # 记录经验
        experience = Experience(
            observation=obs,
            action=action,
            next_observation=next_obs,
            prediction_error=prediction_error,
            timestamp=self.step_count
        )
        self.experiences.append(experience)

        # 符号接地：从感知经验中建立概念
        self.grounding.ground_from_perception(obs)

        # 更新统计
        self.step_count += 1
        self.stats['total_steps'] += 1
        self.stats['total_prediction_error'] += prediction_error

        # 计算内在奖励（好奇心）
        intrinsic_reward = self.curiosity.compute_intrinsic_reward(
            prediction_error, self.predictive_model.get_learning_progress()
        )

        # 融合奖励：70% 好奇心 + 30% 任务
        self.stats['last_intrinsic_reward'] = intrinsic_reward
        self.stats['last_extrinsic_reward'] = extrinsic_reward
        self.stats['last_total_reward'] = 0.7 * intrinsic_reward + 0.3 * extrinsic_reward

        # 检查是否应该进入下一阶段
        self._check_development()

        # 自适应模型切换（基于预测误差）
        if self.model_type == 'adaptive':
            self._adapt_by_error(prediction_error)

        return prediction_error

    def record_social_interaction(self):
        """记录一次社会交互事件"""
        self.stats['social_interactions'] = self.stats.get('social_interactions', 0) + 1

    def communicate(self, partner: 'LearningAgent',
                    scene_objects: list, target_idx: int) -> bool:
        """
        与伙伴进行参照游戏（兼容接口）

        两个 agent 通过语言描述和识别物体，
        交流成功促进语言的组合性和语法涌现。
        """
        # 提取所有物体的特征
        scene_features = []
        for obj in scene_objects:
            features = self.grounding.get_object_features(obj)
            scene_features.append(features)

        # 用 speaker 的语言游戏进行一轮
        success = self.language_game.play_round(scene_features, target_idx)

        # 记录社会交互
        self.record_social_interaction()
        partner.record_social_interaction()

        return success

    def communicate_from_observation(self, partner: 'LearningAgent',
                                      observation: dict) -> bool:
        """
        从环境观测直接通信（Phase 26 核心方法）

        将环境观测转换为统一场景，用自适应语言系统通信。
        语言从真实探索经验中涌现。
        """
        scene = self.bridge.observation_to_scene(observation)
        if scene is None:
            return False

        # 记录经验（用于经验驱动的场景生成）
        self.scenario_gen.record_observation(observation)

        # 用自适应语言系统通信
        success = self.adaptive_game.play_round(scene)

        self.communication_stats['total'] += 1
        if success:
            self.communication_stats['success'] += 1

        self.record_social_interaction()
        partner.record_social_interaction()

        return success

    def explore_and_communicate(self, env, partner: 'LearningAgent',
                                 steps: int = 100) -> dict:
        """
        探索环境，遇到多物体场景时触发通信

        返回：通信统计
        """
        comm_stats = {'total': 0, 'success': 0}

        for _ in range(steps):
            obs = env.get_observation()

            # 随机动作（探索）
            import random as _rand
            action = _rand.randint(0, self.action_dim - 1)
            env.step(action)

            # 视野中有 2+ 物体时触发通信
            visible = obs.get('visible_objects', [])
            if len(visible) >= 2:
                success = self.communicate_from_observation(partner, obs)
                comm_stats['total'] += 1
                if success:
                    comm_stats['success'] += 1

        return comm_stats

    def observe_teacher_demo(self, action: int):
        """观察教师的示范动作（用于模仿学习）"""
        self._last_teacher_demo_action = action

    def _check_development(self):
        """检查发展阶段"""
        # 基于实际预测误差计算准确率（不是学习进度）
        error_history = list(self.predictive_model.error_history)
        if error_history:
            mean_error = np.mean(error_history[-50:])
            pred_accuracy = 1.0 / (1.0 + mean_error)
        else:
            pred_accuracy = 0.0

        agent_stats = {
            'prediction_accuracy': pred_accuracy,
            'exploration_diversity': self._compute_exploration_diversity(),
            'symbol_count': len(self.grounding.get_grounded_symbols()),
            'social_reference': self.stats.get('social_interactions', 0) > 0,
            'classification_accuracy': self._compute_classification_accuracy(),
            'conservation_test': self._test_conservation(),
            'total_steps': self.stats.get('total_steps', 0),
            'experience_count': len(self.experiences),
            'abstract_reasoning_score': self._compute_abstract_reasoning(),
            'hypothesis_confirmed': self._compute_hypothesis_confirmed(),
            'counterfactual_diversity': self._compute_counterfactual_diversity(),
            'meta_cognition': self._compute_meta_cognition(),
            'seriation_score': self._compute_seriation_score(),
            'planning_score': self._compute_planning_score(),
            'perspective_coordination': self._compute_perspective_coordination(),
            'moral_reasoning': self._compute_moral_reasoning(),
        }

        # 检查是否满足晋升条件
        if self.development.check_promotion(agent_stats):
            new_abilities = self.development.promote()
            if new_abilities:
                self.stats['stage_changes'] += 1
                print(f"\n{'='*50}")
                print(f"发展晋升！进入: {self.development.current_stage}")
                print(f"新能力: {new_abilities}")
                print(f"{'='*50}\n")

    def _compute_exploration_diversity(self) -> float:
        """计算探索多样性"""
        if len(self.experiences) < 10:
            return 0.0

        recent_actions = [e.action for e in self.experiences[-10:]]
        unique_actions = len(set(recent_actions))
        return unique_actions / self.action_dim

    def _compute_classification_accuracy(self) -> float:
        """
        计算分类准确率

        衡量 agent 对相似物体的预测是否一致。
        如果同类物体（相同物体特征）的预测误差低且一致，
        说明 agent 具备分类能力。
        """
        if len(self.experiences) < 30:
            return 0.0

        recent = self.experiences[-80:]

        # 按物体特征分组
        groups = {}
        for exp in recent:
            obj_feat = tuple(np.round(exp.observation[3:7], 0))
            if obj_feat not in groups:
                groups[obj_feat] = []
            groups[obj_feat].append(exp.prediction_error)

        if len(groups) < 2:
            return min(1.0, len(self.grounding.get_grounded_symbols()) / 5.0)

        # 各组内的预测误差一致性（组内方差越小 = 分类越准确）
        group_scores = []
        for feat, errors in groups.items():
            if len(errors) >= 3:
                mean_err = np.mean(errors)
                consistency = 1.0 / (1.0 + np.std(errors))
                accuracy = 1.0 / (1.0 + mean_err)
                group_scores.append(consistency * accuracy)

        if not group_scores:
            return 0.0

        return np.mean(group_scores)

    def _test_conservation(self) -> bool:
        """守恒测试（简化版）"""
        # 需要更多的经验才能通过守恒测试
        return len(self.experiences) > 100

    def _compute_abstract_reasoning(self) -> float:
        """
        抽象推理分数：跨物体类型的预测准确率一致性

        如果 agent 在不同类型的物体上都能准确预测，
        说明它学到了跨域的抽象规律，而非具体物体的模式。
        分数 = 1.0 - 各类型预测误差的归一化方差
        """
        if len(self.experiences) < 50:
            return 0.0

        # 跳过 agent 位置（前3维），用物体特征（3-15维）作为类型标识
        recent = self.experiences[-100:]
        type_errors = {}
        for exp in recent:
            # 用物体特征部分（跳过agent位置）作为类型标识
            obj_features = exp.observation[3:7]  # 物体的颜色/形状特征
            type_key = tuple(np.round(obj_features, 0))
            if type_key not in type_errors:
                type_errors[type_key] = []
            type_errors[type_key].append(exp.prediction_error)

        # 需要至少2种不同类型才能计算跨域一致性
        if len(type_errors) < 2:
            return 0.0

        # 各类型的平均误差
        type_means = [np.mean(errs) for errs in type_errors.values()]
        overall_mean = np.mean(type_means)

        if overall_mean < 1e-6:
            return 1.0

        # 方差越小 = 各类型表现越一致 = 抽象推理越强
        variance = np.var(type_means)
        normalized_var = variance / (overall_mean ** 2 + 1e-6)
        score = max(0.0, 1.0 - normalized_var)

        return score

    def _compute_hypothesis_confirmed(self) -> float:
        """
        假设验证正确率：预测被实际结果确认的比例

        当 agent 对某个动作做出预测，然后执行并观察实际结果，
        如果预测误差低于阈值，认为假设被确认。
        """
        if len(self.experiences) < 20:
            return 0.0

        recent = self.experiences[-50:]
        confirmed = sum(1 for e in recent if e.prediction_error < 0.1)
        return confirmed / len(recent)

    def _compute_counterfactual_diversity(self) -> float:
        """
        反事实多样性：尝试过的假设动作比例

        agent 是否探索了"如果我做不同的动作会怎样"——
        即对同一状态尝试多种不同动作的程度。
        """
        if len(self.experiences) < 20:
            return 0.0

        recent = self.experiences[-50:]
        unique_actions = len(set(e.action for e in recent))
        return unique_actions / self.action_dim

    def _compute_meta_cognition(self) -> float:
        """
        元认知：学习策略的自调整能力

        通过检测学习进度的变化模式来衡量：
        如果 agent 能在学习停滞时改变策略（探索多样性增加），
        说明具备元认知能力。
        """
        if len(self.experiences) < 60:
            return 0.0

        recent = self.experiences[-60:]
        first_half = recent[:30]
        second_half = recent[30:]

        # 前半段和后半段的探索多样性
        first_diversity = len(set(e.action for e in first_half)) / self.action_dim
        second_diversity = len(set(e.action for e in second_half)) / self.action_dim

        # 前半段和后半段的平均误差
        first_error = np.mean([e.prediction_error for e in first_half])
        second_error = np.mean([e.prediction_error for e in second_half])

        # 元认知 = 误差增加时多样性是否也增加（策略调整）
        if first_error < second_error:
            # 误差增加时，检查是否调整了策略
            return 1.0 if second_diversity > first_diversity else 0.3
        else:
            # 误差在减少，说明策略有效
            return 0.7

    def _compute_seriation_score(self) -> float:
        """
        序列化能力：沿维度排序的预测一致性

        检查 agent 对不同重量物体的预测是否呈现单调关系。
        如果重量大的物体预测位移也大，说明具备序列化能力。
        """
        if len(self.experiences) < 30:
            return 0.0

        recent = self.experiences[-60:]

        # 按物体重量分组，计算各组的平均预测误差
        # 观测向量结构: [agent_pos(2), color(4), shape(3), pos_weight(3)]
        # 重量在 obs[11]（pos_weight 的第3个元素）
        weight_groups = {}
        for exp in recent:
            weight_key = round(float(exp.observation[11]) * 2) / 2 if len(exp.observation) > 11 else 1.0
            if weight_key not in weight_groups:
                weight_groups[weight_key] = []
            weight_groups[weight_key].append(exp.prediction_error)

        if len(weight_groups) < 3:
            return 0.0

        # 检查重量与预测误差的排序一致性
        sorted_weights = sorted(weight_groups.keys())
        mean_errors = [np.mean(weight_groups[w]) for w in sorted_weights]

        # 计算排序一致性（相邻比较的单调性）
        correct_orderings = 0
        total_pairs = 0
        for i in range(len(mean_errors)):
            for j in range(i + 1, len(mean_errors)):
                total_pairs += 1
                # 重量越大，预测应该越准确（误差越小）
                if mean_errors[i] >= mean_errors[j]:
                    correct_orderings += 1

        return correct_orderings / max(1, total_pairs)

    def _compute_planning_score(self) -> float:
        """
        规划能力：多步目标导向行为

        检测 agent 是否执行了连续相同动作序列（目标追求）
        然后切换动作（目标完成）。这是多步规划的标志。
        """
        if len(self.experiences) < 50:
            return 0.0

        recent = self.experiences[-100:]
        actions = [e.action for e in recent]

        # 寻找连续相同动作序列（长度>=3）后跟切换
        goal_sequences = 0
        i = 0
        while i < len(actions) - 3:
            # 检查连续相同动作
            seq_len = 1
            while i + seq_len < len(actions) and actions[i + seq_len] == actions[i]:
                seq_len += 1

            if seq_len >= 3:
                goal_sequences += 1
                i += seq_len
            else:
                i += 1

        # 归一化：最多可能的序列数
        max_sequences = len(actions) / 3
        return min(1.0, goal_sequences / max(1, max_sequences))

    def _compute_perspective_coordination(self) -> float:
        """
        视角协调：区分自己和他人的视角

        基于通信成功率和社会交互频率。
        成功的跨视角沟通需要理解他人的知识状态。
        """
        total_comm = self.communication_stats.get('total', 0)
        success_comm = self.communication_stats.get('success', 0)

        if total_comm < 10:
            return 0.0

        comm_rate = success_comm / total_comm
        # 社会交互因子：交互越多，视角协调越成熟
        interaction_factor = min(1.0, total_comm / 100)

        return comm_rate * interaction_factor

    def _compute_moral_reasoning(self) -> float:
        """
        道理推理：动作分布的均衡度

        在多 Agent 场景中，检测 agent 的行为是否平衡
        （不是总是采取"自私"的动作）。
        基于动作分布的熵。
        """
        if len(self.experiences) < 50:
            return 0.0

        recent = self.experiences[-100:]
        action_counts = {}
        for exp in recent:
            action_counts[exp.action] = action_counts.get(exp.action, 0) + 1

        total = len(recent)
        entropy = 0.0
        for count in action_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # 归一化熵（最大熵 = log2(action_dim)）
        max_entropy = np.log2(self.action_dim) if self.action_dim > 1 else 1.0
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['current_stage'] = self.development.current_stage
        stats['grounded_symbols'] = len(self.grounding.get_grounded_symbols())
        stats['avg_prediction_error'] = (
            stats['total_prediction_error'] / max(1, stats['total_steps'])
        )
        stats['learning_progress'] = self.predictive_model.get_learning_progress()
        return stats

    def report(self) -> str:
        """生成学习报告"""
        stats = self.get_stats()
        stage_info = self.development.get_stage_info()

        report = f"""
学习体状态报告
{'='*50}

发展阶段: {stage_info['name']} ({stage_info['stage']})
阶段描述: {stage_info['description']}

当前能力: {', '.join(stage_info['abilities'])}
当前限制: {', '.join(stage_info['limitations'])}

学习统计:
- 总步数: {stats['total_steps']}
- 平均预测误差: {stats['avg_prediction_error']:.4f}
- 学习进度: {stats['learning_progress']:.2%}
- 已接地符号: {stats['grounded_symbols']}
- 发展晋升次数: {stats['stage_changes']}

好奇心状态:
- 近期好奇心奖励: {np.mean(list(self.curiosity.reward_history)) if self.curiosity.reward_history else 0:.4f}
- 探索多样性: {self._compute_exploration_diversity():.2%}

已学习的符号:
"""
        for symbol, meaning in self.grounding.symbol_mappings.items():
            report += f"  - {symbol}: 置信度 {meaning['confidence']:.2f}, 使用次数 {meaning['usage_count']}\n"

        return report
