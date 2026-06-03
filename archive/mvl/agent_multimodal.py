"""
多模态学习体

从结构化特征 → 原始感官输入的范式跃迁。

核心改进：
1. 视觉编码器：处理 8x8x4 视觉场（深度+RGB）
2. 听觉编码器：处理声音事件（类型+强度+方向）
3. 触觉编码器：处理接触信号
4. 融合层：将多模态信息压缩为统一表征

类比：婴儿不是接收"物体特征向量"，
而是从原始光影和声音中自己学会"看到"和"听到"。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from agent import LearningAgent, PredictiveModel, CuriosityModule, DevelopmentEngine, GroundingModule
from predictive_nn import NeuralNetworkPredictor
from perception_visual import VisualField
from perception_auditory import AuditorySystem


class MultimodalEncoder:
    """
    多模态编码器

    将原始感官数据编码为紧凑的特征向量。
    每个模态有独立的编码器，
    最后通过融合层整合。
    """

    def __init__(self, visual_shape: Tuple[int, int, int] = (8, 8, 4),
                 audio_dim: int = 56,
                 tactile_dim: int = 1,
                 position_dim: int = 3,
                 fusion_dim: int = 64):
        """
        Args:
            visual_shape: 视觉场形状 (H, W, C)
            audio_dim: 听觉编码维度
            tactile_dim: 触觉维度
            position_dim: 位置维度
            fusion_dim: 融合后的特征维度
        """
        self.visual_shape = visual_shape
        self.audio_dim = audio_dim
        self.tactile_dim = tactile_dim
        self.position_dim = position_dim
        self.fusion_dim = fusion_dim

        # 视觉编码器：256 → 128 → 64 → 32
        visual_flat = visual_shape[0] * visual_shape[1] * visual_shape[2]  # 256
        self.W_v1 = np.random.randn(visual_flat, 128) * np.sqrt(2.0 / visual_flat)
        self.b_v1 = np.zeros(128)
        self.W_v2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b_v2 = np.zeros(64)
        self.W_v3 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b_v3 = np.zeros(32)

        # 听觉编码器：56 → 32 → 16
        self.W_a1 = np.random.randn(audio_dim, 32) * np.sqrt(2.0 / audio_dim)
        self.b_a1 = np.zeros(32)
        self.W_a2 = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b_a2 = np.zeros(16)

        # 触觉编码器：1 → 8
        self.W_t = np.random.randn(tactile_dim, 8) * np.sqrt(2.0 / tactile_dim)
        self.b_t = np.zeros(8)

        # 位置编码器：3 → 8
        self.W_p = np.random.randn(position_dim, 8) * np.sqrt(2.0 / position_dim)
        self.b_p = np.zeros(8)

        # 融合层：32 + 16 + 8 + 8 = 64 → fusion_dim
        concat_dim = 32 + 16 + 8 + 8  # 64
        self.W_fusion = np.random.randn(concat_dim, fusion_dim) * np.sqrt(2.0 / concat_dim)
        self.b_fusion = np.zeros(fusion_dim)

        # 学习率
        self.lr = 0.001

        # 缓存
        self._cache = {}

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def encode(self, visual_field: np.ndarray, audio_events: np.ndarray,
               tactile: np.ndarray, position: np.ndarray) -> np.ndarray:
        """
        编码多模态输入

        Args:
            visual_field: (8, 8, 4) 视觉张量
            audio_events: (56,) 听觉编码
            tactile: (1,) 触觉信号
            position: (3,) 位置向量

        Returns:
            fused: (fusion_dim,) 融合特征向量
        """
        # 视觉编码
        v_flat = visual_field.flatten()
        v_h1 = self._relu(v_flat @ self.W_v1 + self.b_v1)
        v_h2 = self._relu(v_h1 @ self.W_v2 + self.b_v2)
        v_out = self._relu(v_h2 @ self.W_v3 + self.b_v3)  # (32,)

        # 听觉编码
        a_h1 = self._relu(audio_events @ self.W_a1 + self.b_a1)
        a_out = self._relu(a_h1 @ self.W_a2 + self.b_a2)  # (16,)

        # 触觉编码
        t_out = self._relu(tactile @ self.W_t + self.b_t)  # (8,)

        # 位置编码
        p_out = self._relu(position @ self.W_p + self.b_p)  # (8,)

        # 融合
        concat = np.concatenate([v_out, a_out, t_out, p_out])  # (64,)
        fused = self._relu(concat @ self.W_fusion + self.b_fusion)  # (fusion_dim,)

        # 缓存用于反向传播
        self._cache = {
            'v_flat': v_flat, 'v_h1': v_h1, 'v_h2': v_h2, 'v_out': v_out,
            'a_h1': a_h1, 'a_out': a_out,
            't_out': t_out, 'p_out': p_out,
            'concat': concat, 'fused': fused,
            'audio_events': audio_events, 'tactile': tactile, 'position': position,
        }

        return fused

    def get_encoding_dim(self) -> int:
        return self.fusion_dim


class MultimodalPredictiveModel:
    """
    多模态预测模型

    接收融合后的多模态特征 + 动作，
    预测下一时刻的多模态特征。
    """

    def __init__(self, fusion_dim: int = 64, action_dim: int = 8,
                 hidden_dim: int = 128):
        self.fusion_dim = fusion_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 输入维度 = fusion_dim + action_dim
        input_dim = fusion_dim + action_dim

        # 2层MLP
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim // 2) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim // 2)
        self.W3 = np.random.randn(hidden_dim // 2, fusion_dim) * np.sqrt(2.0 / (hidden_dim // 2))
        self.b3 = np.zeros(fusion_dim)

        self.lr = 0.001
        self.error_history = deque(maxlen=100)
        self._cache = {}

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def predict(self, fused_obs: np.ndarray, action: int) -> np.ndarray:
        """预测下一时刻的融合特征"""
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0

        x = np.concatenate([fused_obs, action_vec])

        z1 = x @ self.W1 + self.b1
        h1 = self._relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = self._relu(z2)
        output = h2 @ self.W3 + self.b3

        self._cache = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
        return output

    def learn(self, fused_obs: np.ndarray, action: int,
              actual_next_fused: np.ndarray) -> float:
        """学习 = 减少预测误差"""
        prediction = self.predict(fused_obs, action)
        error = actual_next_fused - prediction
        prediction_error = np.mean(error ** 2)

        # 反向传播
        d_out = -2 * error / len(error)
        d_W3 = np.outer(self._cache['h2'], d_out)
        d_b3 = d_out
        d_h2 = d_out @ self.W3.T
        d_z2 = d_h2 * self._relu_deriv(self._cache['z2'])
        d_W2 = np.outer(self._cache['h1'], d_z2)
        d_b2 = d_z2
        d_h1 = d_z2 @ self.W2.T
        d_z1 = d_h1 * self._relu_deriv(self._cache['z1'])
        d_W1 = np.outer(self._cache['x'], d_z1)
        d_b1 = d_z1

        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

        self.error_history.append(prediction_error)
        return prediction_error

    def get_learning_progress(self) -> float:
        if len(self.error_history) < 10:
            return 0.0
        recent = list(self.error_history)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        if first_half > 0:
            return max(0.0, (first_half - second_half) / first_half)
        return 0.0


class MultimodalAgent:
    """
    多模态学习体

    从原始感官数据（视觉场、声音事件、触觉信号）中学习，
    而不是从预处理的结构化特征中学习。

    核心区别：
    - 旧方式：环境 → 结构化特征 → agent（环境做了感知）
    - 新方式：环境 → 原始感官 → agent 自己做感知（agent 做感知）
    """

    def __init__(self, action_dim: int = 8, fusion_dim: int = 64):
        self.action_dim = action_dim
        self.fusion_dim = fusion_dim

        # 多模态编码器
        self.encoder = MultimodalEncoder(
            visual_shape=(8, 8, 4),
            audio_dim=56,
            tactile_dim=1,
            position_dim=3,
            fusion_dim=fusion_dim
        )

        # 多模态预测模型
        self.predictive_model = MultimodalPredictiveModel(
            fusion_dim=fusion_dim,
            action_dim=action_dim,
            hidden_dim=128
        )

        # 好奇心模块
        self.curiosity = CuriosityModule()

        # 发展引擎
        self.development = DevelopmentEngine()

        # 符号接地模块
        self.grounding = GroundingModule()

        # 经验记忆
        self.experiences = deque(maxlen=1000)
        self.step_count = 0

        # 统计
        self.stats = {
            'total_steps': 0,
            'total_prediction_error': 0.0,
        }

    def perceive(self, observation: dict) -> np.ndarray:
        """
        多模态感知：从原始感官数据中提取特征

        这是与 LearningAgent3D 的核心区别：
        - 不再使用预处理的结构化特征
        - 直接处理视觉场和声音事件
        """
        # 视觉场 (8, 8, 4)
        visual = observation.get('visual_field', np.zeros((8, 8, 4)))

        # 听觉事件 (56,)
        audio = observation.get('audio_events', np.zeros(56))

        # 触觉信号
        touch_objects = observation.get('touch_objects', [])
        if touch_objects:
            is_touching = any(t.get('is_touching', False) for t in touch_objects)
            tactile = np.array([1.0 if is_touching else 0.0])
        else:
            tactile = np.array([0.0])

        # 位置
        position = observation.get('agent_position', np.zeros(3))

        # 编码
        fused = self.encoder.encode(visual, audio, tactile, position)
        return fused

    def act(self, observation: dict) -> int:
        """
        好奇心驱动的动作选择

        基于融合后的多模态表征选择动作。
        """
        obs = self.perceive(observation)

        best_action = None
        best_curiosity = -float('inf')

        for action in range(self.action_dim):
            predicted_next = self.predictive_model.predict(obs, action)
            uncertainty = np.std(predicted_next)
            progress = self.predictive_model.get_learning_progress()

            curiosity = self.curiosity.compute_intrinsic_reward(uncertainty, progress)
            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best_action = action

        if best_action is None:
            best_action = np.random.randint(self.action_dim)

        return best_action

    def learn_from_experience(self, observation: dict, action: int,
                              next_observation: dict,
                              extrinsic_reward: float = 0.0) -> float:
        """
        从经验中学习

        编码 → 预测 → 误差 → 更新
        """
        obs = self.perceive(observation)
        next_obs = self.perceive(next_observation)

        # 更新预测模型
        prediction_error = self.predictive_model.learn(obs, action, next_obs)

        # 符号接地
        self.grounding.ground_from_perception(obs)

        # 更新统计
        self.step_count += 1
        self.stats['total_steps'] += 1
        self.stats['total_prediction_error'] += prediction_error

        # 内在奖励
        intrinsic = self.curiosity.compute_intrinsic_reward(
            prediction_error, self.predictive_model.get_learning_progress()
        )
        self.stats['last_intrinsic'] = intrinsic
        self.stats['last_extrinsic'] = extrinsic_reward
        self.stats['last_total'] = 0.7 * intrinsic + 0.3 * extrinsic_reward

        # 检查发展
        self._check_development()

        return prediction_error

    def _check_development(self):
        """检查发展阶段"""
        agent_stats = {
            'prediction_accuracy': 1.0 - self.predictive_model.get_learning_progress(),
            'exploration_diversity': self._compute_exploration_diversity(),
            'symbol_count': len(self.grounding.get_grounded_symbols()),
            'social_reference': self.stats.get('social_interactions', 0) > 0,
            'classification_accuracy': min(1.0, len(self.grounding.get_grounded_symbols()) / 10.0),
            'conservation_test': self.step_count > 100,
        }
        if self.development.check_promotion(agent_stats):
            new_abilities = self.development.promote()
            if new_abilities:
                print(f"\n发展晋升！进入: {self.development.current_stage}")

    def _compute_exploration_diversity(self) -> float:
        if len(self.experiences) < 10:
            return 0.0
        recent = [e['action'] for e in list(self.experiences)[-10:]]
        return len(set(recent)) / self.action_dim

    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        stats['current_stage'] = self.development.current_stage
        stats['grounded_symbols'] = len(self.grounding.get_grounded_symbols())
        stats['avg_prediction_error'] = (
            stats['total_prediction_error'] / max(1, stats['total_steps'])
        )
        stats['learning_progress'] = self.predictive_model.get_learning_progress()
        return stats
