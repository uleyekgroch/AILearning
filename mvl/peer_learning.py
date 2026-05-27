"""
同伴学习系统

两个 agent 在同一环境中，互相观察和学习。

Vygotsky 的社会文化理论指出：
高级认知功能首先在社会层面出现，然后内化为个体能力。
同伴学习是社会学习的重要形式——
不是教师-学生的单向传递，
而是平等个体之间的相互学习。

类比：两个婴儿一起玩耍时，一个学会了拍手，
另一个观察并模仿，也学会了拍手。
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

from agent import LearningAgent


class PeerAgent(LearningAgent):
    """
    同伴 Agent

    与主 agent 在同一环境中学习。
    可以观察同伴的动作和结果，
    从同伴的成功中学习。
    """

    def __init__(self, obs_dim: int = 10, action_dim: int = 8,
                 model_type: str = 'neural_network'):
        super().__init__(obs_dim, action_dim, model_type)

        # 同伴观察记录
        self.peer_observations = deque(maxlen=50)

        # 社会学习倾向：观察同伴成功动作后增加该动作的倾向
        self.action_tendencies = np.ones(action_dim) / action_dim

        # 同伴影响权重
        self.peer_influence = 0.1  # 同伴经验的影响程度

    def observe_peer(self, peer_action: int, peer_obs: np.ndarray,
                     peer_next_obs: np.ndarray, peer_prediction_error: float):
        """
        观察同伴的动作和结果

        如果同伴的动作导致预测误差降低（成功），
        则增加该动作的倾向。
        """
        self.peer_observations.append({
            'action': peer_action,
            'obs': peer_obs,
            'next_obs': peer_next_obs,
            'error': peer_prediction_error,
        })

        # 更新动作倾向
        # 如果同伴的预测误差低（成功），增加该动作倾向
        success_score = max(0, 1.0 - peer_prediction_error)
        self.action_tendencies[peer_action] += self.peer_influence * success_score

        # 归一化
        self.action_tendencies /= self.action_tendencies.sum()

    def act(self, observation: dict) -> int:
        """
        动作选择：结合好奇心和社会学习

        以一定概率选择同伴验证过的成功动作，
        否则使用好奇心驱动选择。
        """
        obs = self.perceive(observation)

        available_actions = self._get_available_actions()

        # 社会学习：以一定概率选择同伴的成功动作
        social_prob = 0.15  # 社会学习概率
        if np.random.random() < social_prob and len(self.peer_observations) > 5:
            # 选择同伴最成功的动作
            action_scores = np.zeros(self.action_dim)
            for record in self.peer_observations:
                a = record['action']
                score = max(0, 1.0 - record['error'])
                action_scores[a] += score

            # 只考虑可用动作
            for a in range(self.action_dim):
                if a not in available_actions:
                    action_scores[a] = 0

            if action_scores.sum() > 0:
                probs = action_scores / action_scores.sum()
                return np.random.choice(self.action_dim, p=probs)

        # 好奇心驱动的动作选择
        best_action = None
        best_curiosity = -float('inf')

        for action in available_actions:
            predicted_next = self.predictive_model.predict(obs, action)
            uncertainty = np.std(predicted_next)
            progress = self.predictive_model.get_learning_progress()

            curiosity = self.curiosity.compute_intrinsic_reward(uncertainty, progress)

            # 叠加社会学习倾向
            curiosity *= (1.0 + self.action_tendencies[action])

            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best_action = action

        if best_action is None:
            best_action = np.random.choice(available_actions)

        return best_action


class PeerLearningEnvironment:
    """
    同伴学习环境

    两个 agent 在同一物理环境中学习。
    他们可以互相观察动作和结果。
    """

    def __init__(self, env):
        """
        Args:
            env: PhysicsEnvironment 实例
        """
        self.env = env

        # 两个 agent 的位置
        self.agent_a_x = env.width * 0.4
        self.agent_a_y = env.height / 2
        self.agent_a_z = 1.0

        self.agent_b_x = env.width * 0.6
        self.agent_b_y = env.height / 2
        self.agent_b_z = 1.0

        # 记录上一步的观测和动作
        self._prev_obs_a = None
        self._prev_obs_b = None
        self._prev_action_a = None
        self._prev_action_b = None

    def step(self, action_a: int, action_b: int, dt: float = 0.01):
        """
        执行两个 agent 的动作

        Returns:
            obs_a, obs_b, reward_a, reward_b, done
        """
        # 保存前一步观测
        self._prev_obs_a = self._get_obs_for_peer('a')
        self._prev_obs_b = self._get_obs_for_peer('b')
        self._prev_action_a = action_a
        self._prev_action_b = action_b

        # 执行 agent_a 的动作
        self._execute_action(action_a, 'a')
        # 执行 agent_b 的动作
        self._execute_action(action_b, 'b')

        # 更新物理
        self.env._update_physics(dt)
        self.env.step_count += 1

        # 获取新观测
        obs_a = self._get_obs_for_peer('a')
        obs_b = self._get_obs_for_peer('b')

        # 简单奖励
        reward_a = 0.0
        reward_b = 0.0
        done = self.env.step_count >= 500

        return obs_a, obs_b, reward_a, reward_b, done

    def _execute_action(self, action: int, peer: str):
        """执行单个 agent 的动作"""
        move_speed = 1.0
        jump_speed = 2.0

        if peer == 'a':
            x, y, z = self.agent_a_x, self.agent_a_y, self.agent_a_z
        else:
            x, y, z = self.agent_b_x, self.agent_b_y, self.agent_b_z

        if action == 0:
            y = min(self.env.height - 1, y + move_speed)
        elif action == 1:
            y = max(0, y - move_speed)
        elif action == 2:
            x = max(0, x - move_speed)
        elif action == 3:
            x = min(self.env.width - 1, x + move_speed)
        elif action == 4:
            z = min(self.env.depth - 1, z + jump_speed)
        elif action == 5:
            z = max(1.0, z - move_speed)
        elif action == 6:
            # 推动：临时移动 agent 位置执行推动
            old_x, old_y, old_z = self.env.agent_x, self.env.agent_y, self.env.agent_z
            self.env.agent_x, self.env.agent_y, self.env.agent_z = x, y, z
            self.env._push_objects()
            self.env.agent_x, self.env.agent_y, self.env.agent_z = old_x, old_y, old_z
        elif action == 7:
            old_x, old_y, old_z = self.env.agent_x, self.env.agent_y, self.env.agent_z
            self.env.agent_x, self.env.agent_y, self.env.agent_z = x, y, z
            self.env._pull_objects()
            self.env.agent_x, self.env.agent_y, self.env.agent_z = old_x, old_y, old_z

        if peer == 'a':
            self.agent_a_x, self.agent_a_y, self.agent_a_z = x, y, z
        else:
            self.agent_b_x, self.agent_b_y, self.agent_b_z = x, y, z

    def _get_obs_for_peer(self, peer: str) -> dict:
        """获取某个 peer 的观测（包含同伴信息）"""
        # 获取环境基础观测
        base_obs = self.env.get_observation()

        # 添加同伴位置信息
        if peer == 'a':
            base_obs['peer_position'] = np.array([self.agent_b_x, self.agent_b_y, self.agent_b_z])
            base_obs['self_position'] = np.array([self.agent_a_x, self.agent_a_y, self.agent_a_z])
        else:
            base_obs['peer_position'] = np.array([self.agent_a_x, self.agent_a_y, self.agent_a_z])
            base_obs['self_position'] = np.array([self.agent_b_x, self.agent_b_y, self.agent_b_z])

        return base_obs

    def get_peer_action(self, peer: str) -> Optional[int]:
        """获取某个 peer 上一步的动作"""
        if peer == 'a':
            return self._prev_action_a
        else:
            return self._prev_action_b

    def reset(self):
        """重置环境"""
        self.env.reset()
        self.agent_a_x = self.env.width * 0.4
        self.agent_a_y = self.env.height / 2
        self.agent_a_z = 1.0
        self.agent_b_x = self.env.width * 0.6
        self.agent_b_y = self.env.height / 2
        self.agent_b_z = 1.0
        self._prev_obs_a = None
        self._prev_obs_b = None
        self._prev_action_a = None
        self._prev_action_b = None
