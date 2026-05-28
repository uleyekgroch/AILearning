"""
社会学习 Agent

继承 Agent3D，新增：
- 观察学习：观察附近 agent 的动作和结果
- 模仿：选择与成功 agent 相似的动作
- 指令能力：生成和执行指令（Phase 36）
- 社会统计：通信成功率、观察学习次数
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

from agent_3d import Agent3D, SensoryEncoder3D
from environment_3d import Observation3D
from predictive_nn import NeuralNetworkPredictor
from language_emergence import EmergingLanguage


class SocialAgent3D(Agent3D):
    """
    社会学习 Agent

    在 Agent3D 基础上新增：
    1. 观察学习：从附近 agent 的动作-结果对中学习
    2. 模仿动作：参考附近 agent 的成功动作
    3. 通信接口：描述物体、理解描述
    """

    def __init__(self, agent_id: int, learning_rate: float = 0.001,
                 use_language: bool = True):
        # 不传 physics（在共享环境中由 MultiAgent3DEnv 管理）
        # 手动初始化 Agent3D 的组件
        self.agent_id = agent_id

        # 感官编码器
        self.encoder = SensoryEncoder3D()

        # 预测模型
        self.predictor = NeuralNetworkPredictor(
            obs_dim=self.encoder.output_dim,  # 36
            action_dim=6,                     # 连续动作 6 维
            hidden1_dim=128,
            hidden2_dim=64,
        )
        self.predictor.lr = learning_rate
        self.encoder.lr = learning_rate

        # 语言系统
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
            'observation_learning_count': 0,
            'communication_successes': 0,
            'communication_attempts': 0,
        }

        # 上一步编码
        self._last_encoded = None
        self._last_action = None

        # 观察学习缓存
        self._observed_experiences = deque(maxlen=200)

    def step_with_observation(self, observation: Observation3D, action) -> Dict:
        """
        使用给定观察执行学习（用于共享环境）

        与 Agent3D.step 类似，但不调用 physics.step（由 MultiAgent3DEnv 管理）
        """
        # 编码感官输入
        encoded = self.encoder.encode(observation)

        # 预测误差驱动学习
        prediction_error = 0.0
        if self._last_encoded is not None and self._last_action is not None:
            prediction_error = self.predictor.learn(
                self._last_encoded, self._last_action, encoded
            )

        # 好奇心奖励
        uncertainty = np.std(encoded)
        learning_progress = self.predictor.get_learning_progress()
        curiosity = self.curiosity_beta * uncertainty + self.curiosity_eta * learning_progress

        # 更新统计
        self.step_count += 1
        self.stats['total_steps'] += 1
        self.stats['total_prediction_error'] += prediction_error
        self.stats['avg_prediction_error'] = (
            self.stats['total_prediction_error'] / self.stats['total_steps']
        )

        # 记录经验
        self.experiences.append({
            'encoded': encoded.copy(),
            'action': action,
            'prediction_error': prediction_error,
            'curiosity': curiosity,
        })

        # 更新上一步状态
        self._last_encoded = encoded.copy()
        self._last_action = action

        # 从观察缓存中学习
        self._learn_from_observations()

        return {
            'encoded': encoded,
            'prediction_error': prediction_error,
            'curiosity_reward': curiosity,
        }

    def observe_other(self, other_action, other_obs_before: np.ndarray,
                      other_obs_after: np.ndarray):
        """
        观察其他 agent 的动作和结果

        将观察到的 (状态, 动作, 下一状态) 加入观察缓存
        """
        self._observed_experiences.append({
            'obs_before': other_obs_before.copy(),
            'action': other_action,
            'obs_after': other_obs_after.copy(),
        })
        self.stats['observation_learning_count'] += 1

    def _learn_from_observations(self):
        """从观察缓存中学习（每 5 步学习一次）"""
        if len(self._observed_experiences) < 5:
            return
        if self.step_count % 5 != 0:
            return

        # 随机选一个观察经验学习
        idx = np.random.randint(len(self._observed_experiences))
        exp = self._observed_experiences[idx]
        self.predictor.learn(exp['obs_before'], exp['action'], exp['obs_after'])

    def choose_action_social(self, epsilon: float = 0.1,
                              nearby_actions: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        社会感知的动作选择

        除了好奇心驱动，还参考附近 agent 的成功动作
        """
        if np.random.random() < epsilon:
            return self._random_continuous_action()

        if self._last_encoded is None:
            return self._random_continuous_action()

        # 好奇心驱动
        num_candidates = 8
        best_action = None
        best_curiosity = -float('inf')

        for _ in range(num_candidates):
            candidate = self._random_continuous_action()
            action_idx = self._continuous_to_discrete(candidate)
            predicted = self.predictor.predict(self._last_encoded, action_idx)
            uncertainty = np.std(predicted)
            learning_progress = self.predictor.get_learning_progress()
            curiosity = self.curiosity_beta * uncertainty + self.curiosity_eta * learning_progress

            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best_action = candidate

        # 如果有附近 agent 的成功动作，以小概率模仿
        if nearby_actions and np.random.random() < 0.2:
            best_action = np.random.choice(len(nearby_actions))
            best_action = nearby_actions[best_action]

        return best_action

    def _random_continuous_action(self) -> np.ndarray:
        """生成随机连续动作"""
        action = np.random.uniform(-1, 1, 6)
        action[4] = np.abs(action[4])
        action[5] = np.abs(action[5])
        return action

    def _continuous_to_discrete(self, action: np.ndarray) -> int:
        """连续动作 → 离散动作索引"""
        force = action[:3]
        abs_force = np.abs(force)
        axis = np.argmax(abs_force)
        if abs_force[axis] < 0.2:
            return 0
        if axis == 0:
            return 0 if force[0] > 0 else 1
        elif axis == 1:
            return 2 if force[1] > 0 else 3
        else:
            return 4 if force[2] > 0 else 5

    def describe_object(self, features: Dict) -> List[str]:
        """用语言描述物体特征"""
        if self.language is None:
            return []
        symbols = []
        for key, value in features.items():
            if key in ('id', 'distance'):
                continue
            symbols.append(value)
        return symbols

    def try_communicate(self, other: 'SocialAgent3D',
                        scene_features: Dict) -> bool:
        """
        与另一个 agent 尝试通信

        协议：
        1. self 描述物体
        2. other 在自己的视野中寻找匹配
        3. 成功 → 双方更新语言
        """
        self.stats['communication_attempts'] += 1
        other.stats['communication_attempts'] += 1

        # self 描述
        symbols = self.describe_object(scene_features)
        if not symbols:
            return False

        # other 尝试理解
        success = False
        # 简化：检查 other 的词汇中是否有匹配的符号
        if other.language:
            other_vocab = set(other.language.vocabulary.keys())
            matched = sum(1 for s in symbols if s in other_vocab)
            success = matched > 0

        # 更新语言
        if self.language:
            self.language.total_games += 1
            if success:
                self.language.total_successes += 1
            self.language.record_usage(symbols, success)

        if other.language:
            other.language.total_games += 1
            if success:
                other.language.total_successes += 1
            other.language.record_usage(symbols, success)

        if success:
            self.stats['communication_successes'] += 1
            other.stats['communication_successes'] += 1

        return success

    def reset_social(self):
        """重置社会学习状态"""
        self._last_encoded = None
        self._last_action = None
        self._action_history.clear()
        self._observed_experiences.clear()

    # ============================================================
    # 指令能力
    # ============================================================

    def generate_instruction(self, target_features: Dict[str, str],
                              scene_objects: List[Dict],
                              verb: str = 'grab') -> List[str]:
        """
        为另一个 agent 生成指令

        Args:
            target_features: 目标物体的特征
            scene_objects: 场景中所有物体
            verb: 动作动词

        Returns:
            指令符号列表，如 ['grab', 'sphere']
        """
        from instruction_grounding import InstructionGenerator
        gen = InstructionGenerator(self.language)
        return gen.generate_for_target(target_features, scene_objects, verb)

    def execute_instruction(self, symbols: List[str],
                             scene_objects: List[Dict],
                             agent_pos: np.ndarray,
                             agent_facing: float,
                             held_object: Optional[int] = None) -> List[int]:
        """
        执行指令，返回动作序列

        Args:
            symbols: 指令符号列表
            scene_objects: 场景中可见物体
            agent_pos: agent 当前位置
            agent_facing: agent 当前朝向
            held_object: 当前持有的物体 id

        Returns:
            离散动作列表
        """
        from instruction_grounding import InstructionParser, InstructionFollower
        parser = InstructionParser()
        follower = InstructionFollower()

        instruction = parser.parse(symbols)
        if instruction is None:
            return []

        return follower.plan_actions(
            instruction, agent_pos, agent_facing,
            scene_objects, held_object=held_object
        )
