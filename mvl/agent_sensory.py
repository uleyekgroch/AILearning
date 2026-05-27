"""
端到端感官 Agent：从原始像素/声音学习

核心机制：
- 编码器将原始感官输入（8x8x4 图像 + 7维音频 + 2维位置）编码为 40 维向量
- 预测模型在编码空间中预测下一时刻的状态
- 编码器和预测模型联合训练，但目标编码 detach（防止表示坍缩）

这是从"手工特征"到"学习感知"的关键转变。
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque

from encoder_sensory import SensoryEncoder
from predictive_nn import NeuralNetworkPredictor


class SensoryAgent:
    """
    端到端感官学习 Agent

    学习流程：
    1. 原始感官输入 → 编码器 → 40维编码
    2. 编码 + 动作 → 预测模型 → 40维预测
    3. 预测误差反向传播：预测模型 → 编码器
    4. 好奇心驱动动作选择
    """

    def __init__(self, visual_shape=(8, 8, 4), audio_dim=7, pos_dim=2,
                 action_dim=5, learning_rate=0.001):
        # 编码器
        self.encoder = SensoryEncoder(
            visual_shape=visual_shape,
            audio_dim=audio_dim,
            pos_dim=pos_dim
        )
        encoded_dim = self.encoder.output_dim  # 40

        # 预测模型（在编码空间中工作）
        self.predictor = NeuralNetworkPredictor(
            obs_dim=encoded_dim,
            action_dim=action_dim,
            hidden1_dim=128,
            hidden2_dim=64
        )
        self.predictor.lr = learning_rate

        # 参数
        self.action_dim = action_dim
        self.encoded_dim = encoded_dim

        # 好奇心参数
        self.curiosity_beta = 0.1
        self.curiosity_eta = 0.5

        # 学习率
        self.encoder.lr = learning_rate

        # 经验历史
        self.experiences = deque(maxlen=1000)

        # 统计
        self.step_count = 0
        self.stats = {
            'total_steps': 0,
            'total_prediction_error': 0.0,
            'encoder_weight_norm': 0.0,
            'predictor_weight_norm': 0.0,
        }

        # 动作历史（用于探索多样性）
        self._action_history = deque(maxlen=100)

    def perceive(self, observation: dict) -> np.ndarray:
        """
        感知：用编码器将原始感官数据编码为向量

        Args:
            observation: dict with 'visual', 'audio', 'agent_position'

        Returns:
            (40,) 编码向量
        """
        visual = observation.get('visual', np.zeros((8, 8, 4)))
        audio = observation.get('audio', np.zeros(7))
        position = observation.get('agent_position', np.zeros(2))

        # 确保维度正确
        if visual.shape != (8, 8, 4):
            visual = np.zeros((8, 8, 4))
        if audio.shape != (7,):
            audio = np.zeros(7)
        if position.shape != (2,):
            position = np.zeros(2)

        encoded = self.encoder.forward(visual, audio, position)
        return encoded

    def perceive_detached(self, observation: dict) -> np.ndarray:
        """
        感知（detach 版本）：不缓存中间结果，不参与编码器梯度计算

        用于编码目标观测——编码器不会从这个路径接收梯度。
        """
        visual = observation.get('visual', np.zeros((8, 8, 4)))
        audio = observation.get('audio', np.zeros(7))
        position = observation.get('agent_position', np.zeros(2))

        if visual.shape != (8, 8, 4):
            visual = np.zeros((8, 8, 4))
        if audio.shape != (7,):
            audio = np.zeros(7)
        if position.shape != (2,):
            position = np.zeros(2)

        # 独立前向传播（不使用缓存，不参与反向传播）
        # Conv forward
        conv_out = self.encoder._conv2d_forward(visual)
        conv_relu = np.maximum(0, conv_out)
        conv_flat = conv_relu.flatten()
        vis_feat = np.maximum(0, conv_flat @ self.encoder.vis_W + self.encoder.vis_b)

        # Audio forward
        aud_feat = np.maximum(0, audio @ self.encoder.aud_W + self.encoder.aud_b)

        # Position forward
        pos_feat = position @ self.encoder.pos_W + self.encoder.pos_b

        # Fusion
        encoded = np.concatenate([vis_feat, aud_feat, pos_feat])
        return encoded

    def act(self, observation: dict) -> int:
        """
        好奇心驱动的动作选择

        选择最能减少预测误差的动作（信息量最大的动作）。
        """
        obs = self.perceive(observation)

        # 探索多样性：偶尔随机选择
        if np.random.random() < 0.1:
            action = np.random.randint(self.action_dim)
            self._action_history.append(action)
            return action

        # 好奇心驱动：选择预测误差最大的动作
        best_action = 0
        best_error = -1

        for a in range(self.action_dim):
            # 预测下一时刻
            predicted = self.predictor.predict(obs, a)

            # 计算预测不确定性（用历史误差的方差估计）
            if len(self.predictor.error_history) > 5:
                recent_errors = list(self.predictor.error_history)[-10:]
                uncertainty = np.std(recent_errors) + 0.01
            else:
                uncertainty = 1.0

            # 信息量 = 不确定性 * 预测误差的绝对值
            info_gain = uncertainty * np.mean(np.abs(predicted))

            if info_gain > best_error:
                best_error = info_gain
                best_action = a

        self._action_history.append(best_action)
        return best_action

    def learn_from_experience(self, observation: dict, action: int,
                              next_observation: dict) -> float:
        """
        端到端学习：编码器 + 预测模型联合训练

        关键设计：目标编码使用 detach（停止梯度），
        防止编码器退化为恒等映射。

        Args:
            observation: 当前观测（含 visual, audio, agent_position）
            action: 执行的动作
            next_observation: 下一时刻观测

        Returns:
            prediction_error: 预测误差
        """
        # Step 1: 编码当前观测（保留计算图用于编码器梯度）
        encoded_obs = self.perceive(observation)

        # Step 2: 编码下一时刻观测（detach！不参与编码器梯度）
        encoded_next_target = self.perceive_detached(next_observation)

        # Step 3: 预测模型学习 + 获取输入梯度
        prediction_error, d_encoded = self.predictor.learn_and_get_input_gradient(
            encoded_obs, action, encoded_next_target
        )

        # Step 4: 编码器学习（通过梯度回传）
        self.encoder.backward(d_encoded, self.encoder.lr)

        # Step 5: 记录经验
        self.experiences.append({
            'observation': observation,
            'action': action,
            'next_observation': next_observation,
            'prediction_error': prediction_error,
            'step': self.step_count,
        })

        # Step 6: 更新统计
        self.step_count += 1
        self.stats['total_steps'] += 1
        self.stats['total_prediction_error'] += prediction_error

        # 定期更新权重范数
        if self.step_count % 50 == 0:
            self.stats['encoder_weight_norm'] = np.sqrt(
                np.sum(self.encoder.conv_W ** 2) +
                np.sum(self.encoder.vis_W ** 2) +
                np.sum(self.encoder.aud_W ** 2) +
                np.sum(self.encoder.pos_W ** 2)
            )
            self.stats['predictor_weight_norm'] = np.sqrt(
                np.sum(self.predictor.W1 ** 2) +
                np.sum(self.predictor.W2 ** 2) +
                np.sum(self.predictor.W3 ** 2)
            )

        return prediction_error

    def compute_exploration_diversity(self) -> float:
        """计算探索多样性（动作分布的熵）"""
        if len(self._action_history) < 10:
            return 0.0

        counts = np.zeros(self.action_dim)
        for a in self._action_history:
            counts[a] += 1
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(self.action_dim)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_average_error(self, window: int = 50) -> float:
        """获取最近 window 步的平均预测误差"""
        if len(self.predictor.error_history) == 0:
            return 0.0
        recent = list(self.predictor.error_history)[-window:]
        return np.mean(recent)

    def get_learning_progress(self) -> float:
        """获取学习进度"""
        return self.predictor.get_learning_progress()
