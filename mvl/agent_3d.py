"""
3D环境的学习体

这是从2D到3D的升级。
核心改进：
1. 3D感知：处理3D位置和方向
2. 多模态融合：视觉+触觉+距离
3. 3D动作：跳跃、下蹲、推动、拉动
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from agent import LearningAgent, PredictiveModel, CuriosityModule, DevelopmentEngine, GroundingModule
from predictive_nn import NeuralNetworkPredictor, AdaptiveLearningRatePredictor
from knowledge_transfer import KnowledgeTransferSystem


class LearningAgent3D(LearningAgent):
    """
    3D环境的学习体

    继承自LearningAgent，增加：
    1. 3D感知处理
    2. 多模态融合
    3. 3D动作空间
    """

    def __init__(self, obs_dim: int = 20, action_dim: int = 8,
                 model_type: str = 'neural_network'):
        # 3D环境需要更大的观测空间
        # 观测包含：agent位置(3) + 最近物体特征(12) + 距离(1) + 触觉(1) + 其他(3) = 20
        super().__init__(obs_dim, action_dim, model_type)

        # 3D动作空间
        self.action_dim = 8  # 前后左右+跳蹲+推拉

    def perceive(self, observation: dict) -> np.ndarray:
        """
        3D感知：处理多模态输入

        观测包含：
        1. 视觉：物体的颜色、形状、大小
        2. 触觉：是否接触物体
        3. 距离：与物体的距离
        4. 位置：Agent的3D位置
        """
        # Agent位置
        agent_pos = observation['agent_position']

        # 可见物体
        visible_objects = observation['visible_objects']
        if visible_objects:
            # 取最近的物体
            closest = min(visible_objects, key=lambda x: x['distance'])
            obj_obs = closest['object'].to_observation()
            distance = np.array([closest['distance']])
            direction = closest['direction']
        else:
            # 没有可见物体时的默认观测
            obj_obs = np.zeros(12)
            distance = np.array([10.0])  # 最大距离
            direction = np.zeros(3)

        # 触觉感知
        touch_objects = observation['touch_objects']
        if touch_objects:
            # 是否接触任何物体
            is_touching = any(t['is_touching'] for t in touch_objects)
            touch_signal = np.array([1.0 if is_touching else 0.0])
        else:
            touch_signal = np.array([0.0])

        # 组合观测
        # agent位置(3) + 物体特征(12) + 距离(1) + 方向(3) + 触觉(1) = 20
        obs = np.concatenate([agent_pos, obj_obs, distance, direction, touch_signal])

        # 确保维度正确
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return obs

    def act(self, observation: dict) -> int:
        """
        3D动作选择

        动作空间：
        0: 前进 (y+)
        1: 后退 (y-)
        2: 左移 (x-)
        3: 右移 (x+)
        4: 跳跃 (z+)
        5: 下蹲 (z-)
        6: 推动
        7: 拉动
        """
        obs = self.perceive(observation)
        available_actions = self._get_available_actions()

        # Epsilon-greedy: 以10%概率随机探索（模拟幼儿的随机试错）
        # 随着发展推进，随机探索概率降低
        stage_epsilon = {
            'sensorimotor': 0.15,
            'pre_operational': 0.10,
            'concrete_operational': 0.05,
            'formal_operational': 0.02,
        }
        epsilon = stage_epsilon.get(self.development.current_stage, 0.10)

        if np.random.random() < epsilon:
            return np.random.choice(available_actions)

        # 好奇心驱动的动作选择
        best_action = None
        best_curiosity = -float('inf')

        for action in available_actions:
            predicted_next = self.predictive_model.predict(obs, action)
            prediction_uncertainty = np.std(predicted_next)
            learning_progress = self.predictive_model.get_learning_progress()

            curiosity = self.curiosity.compute_intrinsic_reward(
                prediction_uncertainty, learning_progress
            )

            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best_action = action

        if best_action is None:
            best_action = np.random.choice(available_actions)

        return best_action

    def _get_available_actions(self) -> List[int]:
        """获取当前阶段可用的动作"""
        stage = self.development.current_stage

        if stage == 'sensorimotor':
            return [0, 1, 2, 3, 4, 5]  # 前后左右+跳蹲
        elif stage in ['pre_operational', 'concrete_operational']:
            return [0, 1, 2, 3, 4, 5, 6, 7]  # +推动、拉动
        else:  # formal_operational
            return list(range(self.action_dim))  # 全部动作


class MultiModalPredictiveModel:
    """
    多模态预测模型

    处理多种感知通道：
    1. 视觉通道：物体特征
    2. 触觉通道：接触信息
    3. 空间通道：位置和距离
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 视觉通道处理
        self.visual_dim = 12  # 物体特征维度
        self.visual_encoder = np.random.randn(self.visual_dim, hidden_dim // 2) * 0.01

        # 触觉通道处理
        self.touch_dim = 1  # 触觉信号维度
        self.touch_encoder = np.random.randn(self.touch_dim, hidden_dim // 4) * 0.01

        # 空间通道处理
        self.spatial_dim = 4  # 位置+距离维度
        self.spatial_encoder = np.random.randn(self.spatial_dim, hidden_dim // 4) * 0.01

        # 动作编码
        self.action_encoder = np.random.randn(action_dim, hidden_dim // 4) * 0.01

        # 融合层
        fusion_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4 + hidden_dim // 4
        self.fusion_weights = np.random.randn(fusion_dim, obs_dim) * 0.01

        # 学习率
        self.lr = 0.001

        # 预测误差历史
        self.error_history = deque(maxlen=100)

    def predict(self, obs: np.ndarray, action: int) -> np.ndarray:
        """
        多模态预测

        融合视觉、触觉、空间信息进行预测。
        """
        # 分离不同模态
        visual_obs = obs[:self.visual_dim]
        touch_obs = obs[self.visual_dim:self.visual_dim + self.touch_dim]
        spatial_obs = obs[self.visual_dim + self.touch_dim:self.visual_dim + self.touch_dim + self.spatial_dim]

        # 编码动作
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0

        # 编码各模态
        visual_encoded = visual_obs @ self.visual_encoder
        touch_encoded = touch_obs @ self.touch_encoder
        spatial_encoded = spatial_obs @ self.spatial_encoder
        action_encoded = action_vec @ self.action_encoder

        # 融合
        fused = np.concatenate([visual_encoded, touch_encoded, spatial_encoded, action_encoded])

        # 预测
        prediction = fused @ self.fusion_weights

        return prediction

    def learn(self, obs: np.ndarray, action: int, actual_next_obs: np.ndarray) -> float:
        """学习 = 减少预测误差"""
        # 预测
        prediction = self.predict(obs, action)

        # 计算误差
        error = actual_next_obs - prediction
        prediction_error = np.mean(error ** 2)

        # 简化的梯度更新
        # 这里可以实现更复杂的反向传播
        # 为了简洁，使用简单的更新规则

        # 记录误差
        self.error_history.append(prediction_error)

        return prediction_error

    def get_learning_progress(self) -> float:
        """计算学习进度"""
        if len(self.error_history) < 10:
            return 0.0

        recent = list(self.error_history)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])

        if first_half > 0:
            progress = (first_half - second_half) / first_half
        else:
            progress = 0.0

        return max(0.0, progress)
