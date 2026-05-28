"""
3D 物理世界中的 Agent v2（连续动作 + 增强感官）

核心机制：
- 在 PhysicsWorld3D 中执行连续动作（6 维向量）
- 感官编码器将 3D 观察编码为 36 维向量
- 预测模型在编码空间中预测下一时刻的状态
- 预测误差驱动学习（自由能原理）
- 好奇心驱动动作选择

复用：SensoryEncoder、NeuralNetworkPredictor、EmergingLanguage
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from environment_3d import PhysicsWorld3D, Observation3D, AudioEvent3D
from encoder_sensory import SensoryEncoder
from predictive_nn import NeuralNetworkPredictor
from language_emergence import EmergingLanguage


class SensoryEncoder3D(SensoryEncoder):
    """
    3D 感官编码器

    扩展 SensoryEncoder 支持 3D 输入：
    - 视觉：64x64x3 RGB 图像
    - 音频：碰撞事件特征
    - 本体感知：8 维（速度、朝向、持有物体）
    """

    def __init__(self, visual_shape=(64, 64, 3), audio_dim=10, pos_dim=10,
                 visual_out=16, audio_out=10, pos_out=10):
        # 调用父类初始化（但会覆盖一些参数）
        self.visual_shape = visual_shape
        self.audio_dim = audio_dim
        self.pos_dim = pos_dim
        self.visual_out = visual_out
        self.audio_out = audio_out
        self.pos_out = pos_out

        self.output_dim = visual_out + audio_out + pos_out  # 36

        # Conv2D: 3x3 kernel, stride=4, 8 output channels
        # Input: (64,64,3) → Output: (16,16,8)
        self.conv_kernel_size = 3
        self.conv_stride = 4
        self.conv_in_channels = visual_shape[2]  # 3
        self.conv_out_channels = 8
        self.conv_out_h = (visual_shape[0] - self.conv_kernel_size) // self.conv_stride + 1  # 16
        self.conv_out_w = (visual_shape[1] - self.conv_kernel_size) // self.conv_stride + 1  # 16
        self.conv_flat_dim = self.conv_out_h * self.conv_out_w * self.conv_out_channels  # 2048

        # Conv weights
        fan_in = self.conv_kernel_size * self.conv_kernel_size * self.conv_in_channels
        self.conv_W = np.random.randn(
            self.conv_kernel_size, self.conv_kernel_size,
            self.conv_in_channels, self.conv_out_channels
        ) * np.sqrt(2.0 / fan_in)
        self.conv_b = np.zeros(self.conv_out_channels)

        # Visual linear: 2048 → 16
        self.vis_W = np.random.randn(self.conv_flat_dim, visual_out) * np.sqrt(2.0 / self.conv_flat_dim)
        self.vis_b = np.zeros(visual_out)

        # Audio linear: 8 → 16
        self.aud_W = np.random.randn(audio_dim, audio_out) * np.sqrt(2.0 / audio_dim)
        self.aud_b = np.zeros(audio_out)

        # Proprioception linear: 8 → 8
        self.pos_W = np.random.randn(pos_dim, pos_out) * np.sqrt(2.0 / pos_dim)
        self.pos_b = np.zeros(pos_out)

        # Cache for backward
        self._cache = {}
        self.lr = 0.001

    def encode(self, observation: Observation3D) -> np.ndarray:
        """
        编码 3D 观察为向量

        Args:
            observation: Observation3D（visual, depth, audio_events, proprioception）

        Returns:
            (40,) 编码向量
        """
        # 1. 视觉编码
        visual = observation.visual  # (64, 64, 3)
        vis_encoded = self._encode_visual(visual)

        # 2. 音频编码
        audio_vec = self._encode_audio(observation.audio_events)
        aud_encoded = audio_vec @ self.aud_W + self.aud_b
        aud_encoded = np.maximum(0, aud_encoded)  # ReLU

        # 3. 本体感知编码
        proprio = observation.proprioception  # (8,)
        pos_encoded = proprio @ self.pos_W + self.pos_b
        pos_encoded = np.maximum(0, pos_encoded)  # ReLU

        # 4. 融合
        return np.concatenate([vis_encoded, aud_encoded, pos_encoded])

    def _encode_visual(self, visual: np.ndarray) -> np.ndarray:
        """视觉编码：Conv2D → ReLU → Linear → ReLU"""
        h, w, c = visual.shape
        kh, kw = self.conv_kernel_size, self.conv_kernel_size
        oh, ow = self.conv_out_h, self.conv_out_w

        # Conv2D forward
        conv_out = np.zeros((oh, ow, self.conv_out_channels))
        for i in range(oh):
            for j in range(ow):
                patch = visual[i*self.conv_stride:i*self.conv_stride+kh,
                               j*self.conv_stride:j*self.conv_stride+kw, :]
                for k in range(self.conv_out_channels):
                    conv_out[i, j, k] = np.sum(patch * self.conv_W[:, :, :, k]) + self.conv_b[k]

        # ReLU
        conv_out = np.maximum(0, conv_out)

        # Flatten
        flat = conv_out.flatten()

        # Linear
        vis_encoded = flat @ self.vis_W + self.vis_b
        vis_encoded = np.maximum(0, vis_encoded)  # ReLU

        return vis_encoded

    def _encode_audio(self, audio_events: List[AudioEvent3D]) -> np.ndarray:
        """
        音频编码：将音频事件列表编码为 10 维向量

        编码方式：
        - 前 7 维：材质 one-hot（metal/wood/plastic/glass/rubber/stone/fabric）
        - 后 3 维：事件类型聚合（collision, interaction, movement）
        """
        from environment_3d import MATERIAL_NAMES

        vec = np.zeros(self.audio_dim)

        # 事件类型分组
        collision_types = {'collision', 'ground_hit', 'wall_hit'}
        interaction_types = {'grab', 'drop', 'throw'}
        movement_types = {'push'}

        for event in audio_events[:5]:  # 最多 5 个事件
            # 材质 one-hot（前 7 维）
            if event.material in MATERIAL_NAMES:
                mat_idx = MATERIAL_NAMES.index(event.material)
                vec[mat_idx] += event.amplitude

            # 事件类型聚合（后 3 维）
            if event.event_type in collision_types:
                vec[7] += event.amplitude
            elif event.event_type in interaction_types:
                vec[8] += event.amplitude
            elif event.event_type in movement_types:
                vec[9] += event.amplitude

        # 归一化
        vec = np.clip(vec, 0, 1)

        return vec

    def get_param_count(self) -> int:
        return (self.conv_W.size + self.conv_b.size +
                self.vis_W.size + self.vis_b.size +
                self.aud_W.size + self.aud_b.size +
                self.pos_W.size + self.pos_b.size)


class Agent3D:
    """
    3D 物理世界中的 Agent

    学习流程：
    1. 执行物理动作
    2. 编码感官输入
    3. 预测误差驱动学习
    4. 好奇心驱动动作选择
    """

    def __init__(self, physics: PhysicsWorld3D,
                 learning_rate: float = 0.001,
                 use_language: bool = True):
        self.physics = physics

        # 感官编码器
        self.encoder = SensoryEncoder3D()

        # 预测模型（在编码空间中工作）
        self.predictor = NeuralNetworkPredictor(
            obs_dim=self.encoder.output_dim,  # 36
            action_dim=6,                     # 连续动作 6 维
            hidden1_dim=128,
            hidden2_dim=64,
        )
        self.predictor.lr = learning_rate
        self.encoder.lr = learning_rate

        # 语言系统（可选）
        self.language = EmergingLanguage() if use_language else None

        # 好奇心参数
        self.curiosity_beta = 0.1
        self.curiosity_eta = 0.5

        # 经验历史
        self.experiences = deque(maxlen=1000)
        self._action_history = deque(maxlen=100)

        # 统计
        self.step_count = 0
        self.stats = {
            'total_steps': 0,
            'total_prediction_error': 0.0,
            'avg_prediction_error': 0.0,
        }

        # 上一步编码（用于预测误差）
        self._last_encoded = None
        self._last_action = None

    def step(self, action) -> Dict:
        """
        执行一步动作 + 学习

        Args:
            action: int（离散动作）或 np.ndarray（6 维连续动作）

        Returns:
            dict: {
                'observation': Observation3D,
                'encoded': np.ndarray,
                'prediction_error': float,
                'curiosity_reward': float,
            }
        """
        # 1. 执行物理动作
        obs = self.physics.step(action)

        # 2. 编码感官输入
        encoded = self.encoder.encode(obs)

        # 3. 预测误差驱动学习
        prediction_error = 0.0
        if self._last_encoded is not None and self._last_action is not None:
            # 用上一步的编码 + 动作预测当前编码
            prediction_error = self.predictor.learn(
                self._last_encoded, self._last_action, encoded
            )

        # 4. 好奇心奖励
        uncertainty = np.std(encoded)
        learning_progress = self.predictor.get_learning_progress()
        curiosity = self._compute_curiosity(uncertainty, learning_progress)

        # 5. 更新统计
        self.step_count += 1
        self.stats['total_steps'] += 1
        self.stats['total_prediction_error'] += prediction_error
        self.stats['avg_prediction_error'] = (
            self.stats['total_prediction_error'] / self.stats['total_steps']
        )

        # 6. 记录经验
        self.experiences.append({
            'encoded': encoded.copy(),
            'action': action,
            'prediction_error': prediction_error,
            'curiosity': curiosity,
        })
        self._action_history.append(action)

        # 7. 更新上一步状态
        self._last_encoded = encoded.copy()
        self._last_action = action

        return {
            'observation': obs,
            'encoded': encoded,
            'prediction_error': prediction_error,
            'curiosity_reward': curiosity,
        }

    def choose_action(self, epsilon: float = 0.1) -> np.ndarray:
        """
        好奇心驱动的连续动作选择

        以 epsilon 概率随机探索，否则采样多个候选动作，选择最高好奇心的。
        返回 6 维连续向量 [force_x, force_y, force_z, torque_z, grip, throw_speed]
        """
        if np.random.random() < epsilon:
            return self._random_continuous_action()

        if self._last_encoded is None:
            return self._random_continuous_action()

        # 采样候选动作，评估好奇心
        num_candidates = 8
        best_action = None
        best_curiosity = -float('inf')

        for _ in range(num_candidates):
            candidate = self._random_continuous_action()
            # 用离散化动作索引进行预测（将连续动作映射到最近的离散动作）
            action_idx = self._continuous_to_discrete(candidate)
            predicted = self.predictor.predict(self._last_encoded, action_idx)
            uncertainty = np.std(predicted)
            learning_progress = self.predictor.get_learning_progress()
            curiosity = self._compute_curiosity(uncertainty, learning_progress)

            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best_action = candidate

        return best_action

    def _random_continuous_action(self) -> np.ndarray:
        """生成随机连续动作"""
        action = np.random.uniform(-1, 1, 6)
        # grip 和 throw_speed 在 [0, 1]
        action[4] = np.abs(action[4])  # grip >= 0
        action[5] = np.abs(action[5])  # throw_speed >= 0
        return action

    def _continuous_to_discrete(self, action: np.ndarray) -> int:
        """将连续动作映射到最近的离散动作（用于预测模型）"""
        # 取前 3 维 force 的最大方向作为离散动作
        force = action[:3]
        abs_force = np.abs(force)
        axis = np.argmax(abs_force)
        if abs_force[axis] < 0.2:
            return 0  # FORWARD（静止时默认前进）
        if axis == 0:
            return 0 if force[0] > 0 else 1  # FORWARD / BACKWARD
        elif axis == 1:
            return 2 if force[1] > 0 else 3  # LEFT / RIGHT
        else:
            return 4 if force[2] > 0 else 5  # UP / DOWN

    def _compute_curiosity(self, uncertainty: float, learning_progress: float) -> float:
        """
        好奇心 = β * uncertainty + η * learning_progress

        - uncertainty：预测的不确定性（高 → 好奇）
        - learning_progress：学习进度（正 → 好奇）
        """
        return self.curiosity_beta * uncertainty + self.curiosity_eta * learning_progress

    def explore(self, num_steps: int = 100, epsilon: float = 0.15) -> Dict:
        """
        自主探索

        Args:
            num_steps: 探索步数
            epsilon: 随机探索概率

        Returns:
            探索统计
        """
        errors = []
        curiosities = []

        for _ in range(num_steps):
            action = self.choose_action(epsilon=epsilon)
            result = self.step(action)
            errors.append(result['prediction_error'])
            curiosities.append(result['curiosity_reward'])

        return {
            'avg_error': np.mean(errors) if errors else 0.0,
            'avg_curiosity': np.mean(curiosities) if curiosities else 0.0,
            'final_error': errors[-1] if errors else 0.0,
            'error_trend': (np.mean(errors[len(errors)//2:]) - np.mean(errors[:len(errors)//2]))
                           if len(errors) > 10 else 0.0,
        }

    def describe_object(self, obj_id: int) -> List[str]:
        """用语言描述物体（如果语言系统启用）"""
        if self.language is None:
            return []

        features = self.physics.get_object_features(obj_id)
        if not features:
            return []

        # 直接使用特征值作为符号
        symbols = []
        for key, value in features.items():
            if key == 'id' or key == 'distance':
                continue
            symbols.append(value)

        return symbols

    def reset(self):
        """重置 agent 状态"""
        self._last_encoded = None
        self._last_action = None
        self._action_history.clear()
