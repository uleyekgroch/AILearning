"""
知识迁移系统

解决稳定性-可塑性困境：
- 太稳定：无法适应新环境
- 太可塑：忘记之前学到的知识

解决方案：模型切换时保留部分知识
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class KnowledgeExtractor:
    """
    知识提取器

    从模型中提取学到的知识。

    知识类型：
    1. 权重知识：模型参数
    2. 经验知识：历史经验
    3. 概念知识：符号映射
    """

    def extract_from_predictive_model(self, model) -> Dict:
        """
        从预测模型中提取知识

        提取：
        1. 权重矩阵
        2. 误差历史
        3. 学习进度
        """
        knowledge = {
            'type': 'predictive_model',
            'weights': {},
            'error_history': list(model.error_history),
            'learning_progress': model.get_learning_progress()
        }

        # 提取权重
        if hasattr(model, 'W_obs'):
            knowledge['weights']['W_obs'] = model.W_obs.copy()
            knowledge['weights']['W_action'] = model.W_action.copy()
            knowledge['weights']['W_out'] = model.W_out.copy()
        elif hasattr(model, 'W1'):
            knowledge['weights']['W1'] = model.W1.copy()
            knowledge['weights']['b1'] = model.b1.copy()
            knowledge['weights']['W2'] = model.W2.copy()
            knowledge['weights']['b2'] = model.b2.copy()
            knowledge['weights']['W3'] = model.W3.copy()
            knowledge['weights']['b3'] = model.b3.copy()

        return knowledge

    def extract_from_grounding(self, grounding) -> Dict:
        """
        从符号接地模块中提取知识

        提取：
        1. 感知聚类
        2. 符号映射
        """
        knowledge = {
            'type': 'grounding',
            'perceptual_clusters': {},
            'symbol_mappings': {}
        }

        # 提取感知聚类
        for cluster_id, cluster in grounding.perceptual_clusters.items():
            knowledge['perceptual_clusters'][cluster_id] = {
                'centroid': cluster['centroid'].copy(),
                'count': cluster['count']
            }

        # 提取符号映射
        for symbol, mapping in grounding.symbol_mappings.items():
            knowledge['symbol_mappings'][symbol] = {
                'referent_clusters': mapping['referent_clusters'].copy(),
                'confidence': mapping['confidence'],
                'usage_count': mapping['usage_count']
            }

        return knowledge

    def extract_from_curiosity(self, curiosity) -> Dict:
        """
        从好奇心模块中提取知识

        提取：
        1. 奖励历史
        2. 参数设置
        """
        knowledge = {
            'type': 'curiosity',
            'reward_history': list(curiosity.reward_history),
            'alpha': curiosity.alpha,
            'beta': curiosity.beta
        }

        return knowledge


class KnowledgeInjector:
    """
    知识注入器

    将知识注入到新模型中。

    注入策略：
    1. 完全注入：直接复制所有知识
    2. 部分注入：只注入部分知识
    3. 渐进注入：逐步注入知识
    """

    def inject_to_predictive_model(self, model, knowledge: Dict, injection_rate: float = 0.5):
        """
        将知识注入到预测模型

        参数：
            model: 目标模型
            knowledge: 知识字典
            injection_rate: 注入率（0-1）
        """
        if knowledge['type'] != 'predictive_model':
            return

        # 注入权重
        if 'weights' in knowledge:
            weights = knowledge['weights']

            # 线性模型
            if hasattr(model, 'W_obs') and 'W_obs' in weights:
                # 确保维度匹配
                if model.W_obs.shape == weights['W_obs'].shape:
                    model.W_obs = model.W_obs * (1 - injection_rate) + weights['W_obs'] * injection_rate
                    model.W_action = model.W_action * (1 - injection_rate) + weights['W_action'] * injection_rate
                    model.W_out = model.W_out * (1 - injection_rate) + weights['W_out'] * injection_rate

            # 神经网络模型
            elif hasattr(model, 'W1') and 'W1' in weights:
                if model.W1.shape == weights['W1'].shape:
                    model.W1 = model.W1 * (1 - injection_rate) + weights['W1'] * injection_rate
                    model.b1 = model.b1 * (1 - injection_rate) + weights['b1'] * injection_rate
                    model.W2 = model.W2 * (1 - injection_rate) + weights['W2'] * injection_rate
                    model.b2 = model.b2 * (1 - injection_rate) + weights['b2'] * injection_rate
                    model.W3 = model.W3 * (1 - injection_rate) + weights['W3'] * injection_rate
                    model.b3 = model.b3 * (1 - injection_rate) + weights['b3'] * injection_rate

        # 注入误差历史
        if 'error_history' in knowledge:
            for error in knowledge['error_history'][-50:]:  # 只注入最近的50个
                model.error_history.append(error)

    def inject_to_grounding(self, grounding, knowledge: Dict, injection_rate: float = 0.5):
        """
        将知识注入到符号接地模块

        参数：
            grounding: 目标模块
            knowledge: 知识字典
            injection_rate: 注入率（0-1）
        """
        if knowledge['type'] != 'grounding':
            return

        # 注入感知聚类
        if 'perceptual_clusters' in knowledge:
            for cluster_id, cluster_data in knowledge['perceptual_clusters'].items():
                if cluster_id not in grounding.perceptual_clusters:
                    grounding.perceptual_clusters[cluster_id] = {
                        'centroid': cluster_data['centroid'].copy(),
                        'count': cluster_data['count'],
                        'examples': [cluster_data['centroid'].copy()]
                    }
                else:
                    # 合并聚类
                    existing = grounding.perceptual_clusters[cluster_id]
                    total_count = existing['count'] + cluster_data['count']
                    existing['centroid'] = (
                        existing['centroid'] * existing['count'] +
                        cluster_data['centroid'] * cluster_data['count']
                    ) / total_count
                    existing['count'] = total_count

        # 注入符号映射
        if 'symbol_mappings' in knowledge:
            for symbol, mapping_data in knowledge['symbol_mappings'].items():
                if symbol not in grounding.symbol_mappings:
                    grounding.symbol_mappings[symbol] = {
                        'referent_clusters': mapping_data['referent_clusters'].copy(),
                        'confidence': mapping_data['confidence'] * injection_rate,
                        'usage_count': int(mapping_data['usage_count'] * injection_rate)
                    }
                else:
                    # 合并映射
                    existing = grounding.symbol_mappings[symbol]
                    existing['confidence'] = max(existing['confidence'], mapping_data['confidence'])
                    existing['usage_count'] += int(mapping_data['usage_count'] * injection_rate)
                    for cluster_id in mapping_data['referent_clusters']:
                        if cluster_id not in existing['referent_clusters']:
                            existing['referent_clusters'].append(cluster_id)

    def inject_to_curiosity(self, curiosity, knowledge: Dict, injection_rate: float = 0.5):
        """
        将知识注入到好奇心模块

        参数：
            curiosity: 目标模块
            knowledge: 知识字典
            injection_rate: 注入率（0-1）
        """
        if knowledge['type'] != 'curiosity':
            return

        # 注入奖励历史
        if 'reward_history' in knowledge:
            for reward in knowledge['reward_history'][-50:]:  # 只注入最近的50个
                curiosity.reward_history.append(reward)

        # 注入参数（加权平均）
        if 'alpha' in knowledge:
            curiosity.alpha = curiosity.alpha * (1 - injection_rate) + knowledge['alpha'] * injection_rate
        if 'beta' in knowledge:
            curiosity.beta = curiosity.beta * (1 - injection_rate) + knowledge['beta'] * injection_rate


class GradualSwitcher:
    """
    渐进切换器

    实现平滑的模型切换。

    策略：
    1. 双模型并行：新旧模型同时运行
    2. 混合预测：新旧模型的预测加权平均
    3. 逐步过渡：逐步增加新模型的权重
    """

    def __init__(self, transition_steps: int = 50):
        self.transition_steps = transition_steps
        self.current_step = 0
        self.old_model = None
        self.new_model = None
        self.mixing_weight = 0.0  # 0 = 完全使用旧模型，1 = 完全使用新模型

    def start_transition(self, old_model, new_model):
        """开始过渡"""
        self.old_model = old_model
        self.new_model = new_model
        self.current_step = 0
        self.mixing_weight = 0.0

    def step(self) -> float:
        """推进一步，返回当前混合权重"""
        if self.old_model is None or self.new_model is None:
            return 1.0

        self.current_step += 1
        self.mixing_weight = min(1.0, self.current_step / self.transition_steps)

        return self.mixing_weight

    def predict(self, obs: np.ndarray, action: int) -> np.ndarray:
        """混合预测"""
        if self.old_model is None or self.new_model is None:
            raise ValueError("未开始过渡")

        # 旧模型预测
        old_pred = self.old_model.predict(obs, action)

        # 新模型预测
        new_pred = self.new_model.predict(obs, action)

        # 混合预测
        mixed_pred = old_pred * (1 - self.mixing_weight) + new_pred * self.mixing_weight

        return mixed_pred

    def is_complete(self) -> bool:
        """检查过渡是否完成"""
        return self.mixing_weight >= 1.0


class KnowledgeTransferSystem:
    """
    知识迁移系统

    整合知识提取、注入和渐进切换。
    """

    def __init__(self):
        self.extractor = KnowledgeExtractor()
        self.injector = KnowledgeInjector()
        self.switcher = GradualSwitcher()

        # 知识缓存
        self.knowledge_cache = {}

    def extract_knowledge(self, agent) -> Dict:
        """
        从Agent中提取所有知识

        参数：
            agent: 学习体

        返回：
            知识字典
        """
        knowledge = {
            'predictive_model': self.extractor.extract_from_predictive_model(agent.predictive_model),
            'grounding': self.extractor.extract_from_grounding(agent.grounding),
            'curiosity': self.extractor.extract_from_curiosity(agent.curiosity)
        }

        return knowledge

    def inject_knowledge(self, agent, knowledge: Dict, injection_rate: float = 0.5):
        """
        将知识注入到Agent

        参数：
            agent: 学习体
            knowledge: 知识字典
            injection_rate: 注入率（0-1）
        """
        # 注入预测模型知识
        if 'predictive_model' in knowledge:
            self.injector.inject_to_predictive_model(
                agent.predictive_model,
                knowledge['predictive_model'],
                injection_rate
            )

        # 注入符号接地知识
        if 'grounding' in knowledge:
            self.injector.inject_to_grounding(
                agent.grounding,
                knowledge['grounding'],
                injection_rate
            )

        # 注入好奇心知识
        if 'curiosity' in knowledge:
            self.injector.inject_to_curiosity(
                agent.curiosity,
                knowledge['curiosity'],
                injection_rate
            )

    def transfer_with_gradual_switch(self, agent, new_model_type: str,
                                      transition_steps: int = 50) -> bool:
        """
        使用渐进切换进行知识迁移

        跨架构迁移策略：
        - 不迁移权重（线性模型的 W_obs 和神经网络的 W1 形状不同，混合无意义）
        - 只迁移符号知识（grounding, curiosity）——让新模型从头学权重但保留概念
        - 同架构迁移仍保留权重注入

        参数：
            agent: 学习体
            new_model_type: 新模型类型
            transition_steps: 过渡步数

        返回：
            是否成功
        """
        # 提取旧模型知识
        old_knowledge = self.extract_knowledge(agent)

        # 创建新模型
        from predictive_nn import NeuralNetworkPredictor, AdaptiveLearningRatePredictor
        from agent import PredictiveModel

        if new_model_type == 'neural_network':
            new_model = NeuralNetworkPredictor(agent.obs_dim, agent.action_dim)
        elif new_model_type == 'adaptive_nn':
            new_model = AdaptiveLearningRatePredictor(agent.obs_dim, agent.action_dim)
        else:
            new_model = PredictiveModel(agent.obs_dim, agent.action_dim)

        # 判断是否同架构
        old_type = type(agent.predictive_model).__name__
        new_type = type(new_model).__name__
        same_architecture = (old_type == new_type)

        if same_architecture:
            # 同架构：注入权重（有意义）
            self.injector.inject_to_predictive_model(
                new_model,
                old_knowledge['predictive_model'],
                injection_rate=0.5
            )
        # 跨架构：跳过权重注入，只迁移符号知识

        # 注入符号知识（grounding, curiosity）——总是迁移
        if 'grounding' in old_knowledge:
            self.injector.inject_to_grounding(
                agent.grounding,
                old_knowledge['grounding'],
                injection_rate=1.0  # 符号知识完全迁移
            )
        if 'curiosity' in old_knowledge:
            self.injector.inject_to_curiosity(
                agent.curiosity,
                old_knowledge['curiosity'],
                injection_rate=0.5
            )

        # 开始渐进切换
        self.switcher.start_transition(agent.predictive_model, new_model)

        # 缓存知识
        self.knowledge_cache['old'] = old_knowledge

        return True

    def step_transition(self, agent) -> float:
        """
        推进过渡

        参数：
            agent: 学习体

        返回：
            当前混合权重
        """
        weight = self.switcher.step()

        # 如果过渡完成，更新Agent的模型
        if self.switcher.is_complete():
            agent.predictive_model = self.switcher.new_model
            return 1.0

        return weight
