"""
自由能原理 Agent

将 FEP 的三个核心组件集成到一个 agent 中：
1. ProbabilisticPredictor — 概率生成模型
2. VariationalBelief — 变分推断
3. ActiveInferenceModule — 主动推理

与 LearningAgent 的区别：
- LearningAgent：最小化 MSE 预测误差
- FEPAgent：最小化变分自由能 F

这是从"误差最小化"到"自由能最小化"的范式转换。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from free_energy import ProbabilisticPredictor
from variational_inference import VariationalBelief
from active_inference import ActiveInferenceModule
from agent import DevelopmentEngine, GroundingModule


class FEPAgent:
    """
    自由能原理 Agent

    核心理念：
    - 学习 = 最小化变分自由能 F = 复杂度 - 准确度
    - 动作 = 最小化期望自由能 G = 信息增益 + 工具价值
    - 注意力 = 精度 π = 1/σ²（不确定性倒数）
    """

    def __init__(self, obs_dim: int = 20, action_dim: int = 8):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # FEP 核心组件
        self.predictor = ProbabilisticPredictor(obs_dim, action_dim, hidden_dim=64)
        self.belief = VariationalBelief(obs_dim)
        self.active_inference = ActiveInferenceModule(action_dim)

        # 保留发展引擎和符号接地
        self.development = DevelopmentEngine()
        self.grounding = GroundingModule()

        # 经验记录
        self.experiences = deque(maxlen=200)
        self.step_count = 0

        # 统计
        self.stats = {
            'total_steps': 0,
            'total_free_energy': 0.0,
            'symbols_learned': 0,
            'stage_changes': 0,
            'social_interactions': 0,
        }

        # 模仿学习
        self._last_teacher_demo_action = None

    def perceive(self, observation: dict) -> np.ndarray:
        """感知编码（与 LearningAgent3D 相同）"""
        agent_pos = observation['agent_position']

        visible_objects = observation['visible_objects']
        if visible_objects:
            closest = min(visible_objects, key=lambda x: x['distance'])
            obj_obs = closest['object'].to_observation()
            distance = np.array([closest['distance']])
            direction = closest['direction']
        else:
            obj_obs = np.zeros(12)
            distance = np.array([10.0])
            direction = np.zeros(3)

        touch_objects = observation['touch_objects']
        if touch_objects:
            is_touching = any(t['is_touching'] for t in touch_objects)
            touch_signal = np.array([1.0 if is_touching else 0.0])
        else:
            touch_signal = np.zeros(1)

        obs = np.concatenate([agent_pos, obj_obs, distance, direction, touch_signal])

        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return obs

    def act(self, observation: dict) -> int:
        """
        主动推理动作选择

        使用期望自由能 G 而非好奇心选择动作。
        """
        obs = self.perceive(observation)
        available_actions = self._get_available_actions()

        # Epsilon-greedy（保留少量随机探索）
        stage_epsilon = {
            'sensorimotor': 0.15,
            'pre_operational': 0.10,
            'concrete_operational': 0.05,
            'formal_operational': 0.02,
        }
        epsilon = stage_epsilon.get(self.development.current_stage, 0.10)

        if np.random.random() < epsilon:
            return np.random.choice(available_actions)

        # 模仿学习
        if self._last_teacher_demo_action is not None:
            imitation_probs = {
                'sensorimotor': 0.3,
                'pre_operational': 0.15,
                'concrete_operational': 0.05,
                'formal_operational': 0.02,
            }
            imitate_prob = imitation_probs.get(self.development.current_stage, 0.1)
            if np.random.random() < imitate_prob:
                if self._last_teacher_demo_action in available_actions:
                    return self._last_teacher_demo_action

        # 主动推理：选择最小化期望自由能的动作（批量计算）
        action = self.active_inference.select_action_batch(
            obs, self.predictor, self.belief, available_actions
        )

        return action

    def _get_available_actions(self) -> List[int]:
        """根据发展阶段获取可用动作"""
        stage = self.development.current_stage
        if stage == 'sensorimotor':
            return [0, 1, 2, 3, 4, 5]
        elif stage in ['pre_operational', 'concrete_operational']:
            return [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            return list(range(self.action_dim))

    def learn_from_experience(self, observation: dict, action: int,
                              next_observation: dict,
                              extrinsic_reward: float = 0.0) -> Tuple[float, float]:
        """
        自由能最小化学习

        Returns:
            free_energy: 变分自由能
            precision_weighted_error: 精度加权预测误差
        """
        obs = self.perceive(observation)
        next_obs = self.perceive(next_observation)

        # 1. 概率预测
        predicted_mean, predicted_log_var = self.predictor.predict(obs, action)

        # 2. 变分信念更新
        self.belief.update_belief(next_obs, predicted_mean, predicted_log_var)

        # 3. 自由能最小化学习
        free_energy, pw_error = self.predictor.learn(obs, action, next_obs)

        # 4. 符号接地
        self.grounding.ground_from_perception(obs)

        # 5. 记录经验
        self.experiences.append({
            'obs': obs, 'action': action, 'next_obs': next_obs,
            'free_energy': free_energy, 'pw_error': pw_error,
        })

        # 6. 更新统计
        self.step_count += 1
        self.stats['total_steps'] += 1
        self.stats['total_free_energy'] += free_energy

        # 7. 检查发展
        self._check_development()

        return free_energy, pw_error

    def record_social_interaction(self):
        """记录社会交互"""
        self.stats['social_interactions'] = self.stats.get('social_interactions', 0) + 1

    def observe_teacher_demo(self, action: int):
        """观察教师示范"""
        self._last_teacher_demo_action = action

    def set_risk_sensitive(self, weight: float = 1.0):
        """启用风险感知动作选择（精度加权风险惩罚）"""
        self.active_inference.set_risk_sensitivity(weight)

    def get_risk_assessment(self, obs: dict) -> dict:
        """返回当前不确定性状态（诊断用）"""
        obs_vec = self.perceive(obs)
        action_risks = {}
        for a in self._get_available_actions():
            mean, log_var = self.predictor.predict(obs_vec, a)
            var = np.exp(log_var)
            belief_prec = np.exp(-self.belief.log_var)
            action_risks[a] = float(np.sum(var * belief_prec))
        return {
            'belief_uncertainty': self.belief.get_belief_uncertainty().tolist(),
            'action_risks': action_risks,
            'exploration_drive': self.active_inference.get_exploration_drive(),
            'risk_drive': self.active_inference.get_risk_drive(),
        }

    def _check_development(self):
        """检查发展阶段"""
        # 使用原始 MSE（不是精度加权误差）来衡量预测准确度
        # 精度加权误差会因精度波动而不稳定
        error_history = list(self.predictor.error_history)
        if error_history:
            # error_history 存的是 np.mean(error**2)，即原始 MSE
            mean_error = np.mean(error_history[-50:])
            pred_accuracy = 1.0 / (1.0 + mean_error)
        else:
            pred_accuracy = 0.0

        # 抽象推理
        abstract_score = self._compute_abstract_reasoning()

        # 分类准确率
        classification = self._compute_classification_accuracy()

        agent_stats = {
            'prediction_accuracy': pred_accuracy,
            'exploration_diversity': self._compute_exploration_diversity(),
            'symbol_count': len(self.grounding.get_grounded_symbols()),
            'social_reference': self.stats.get('social_interactions', 0) > 0,
            'classification_accuracy': classification,
            'conservation_test': len(self.experiences) > 100,
            'total_steps': self.stats.get('total_steps', 0),
            'experience_count': len(self.experiences),
            'abstract_reasoning_score': abstract_score,
            'hypothesis_confirmed': self._compute_hypothesis_confirmed(),
            'counterfactual_diversity': self._compute_counterfactual_diversity(),
            'meta_cognition': self._compute_meta_cognition(),
        }

        if self.development.check_promotion(agent_stats):
            new_abilities = self.development.promote()
            if new_abilities:
                self.stats['stage_changes'] += 1
                print(f"\n{'='*50}")
                print(f"发展晋升！进入: {self.development.current_stage}")
                print(f"新能力: {new_abilities}")
                print(f"{'='*50}\n")

    def _compute_exploration_diversity(self) -> float:
        if len(self.experiences) < 10:
            return 0.0
        recent = list(self.experiences)[-10:]
        unique_actions = len(set(e['action'] for e in recent))
        return unique_actions / self.action_dim

    def _compute_abstract_reasoning(self) -> float:
        if len(self.experiences) < 50:
            return 0.0
        recent = list(self.experiences)[-100:]
        type_errors = {}
        for exp in recent:
            obj_feat = tuple(np.round(exp['obs'][3:7], 0))
            if obj_feat not in type_errors:
                type_errors[obj_feat] = []
            type_errors[obj_feat].append(exp['free_energy'])
        if len(type_errors) < 2:
            return 0.0
        type_means = [np.mean(errs) for errs in type_errors.values()]
        overall_mean = np.mean(type_means)
        if overall_mean < 1e-6:
            return 1.0
        variance = np.var(type_means)
        normalized_var = variance / (overall_mean**2 + 1e-6)
        return max(0.0, 1.0 - normalized_var)

    def _compute_hypothesis_confirmed(self) -> float:
        if len(self.experiences) < 20:
            return 0.0
        recent = list(self.experiences)[-50:]
        confirmed = sum(1 for e in recent if e['pw_error'] < 0.1)
        return confirmed / len(recent)

    def _compute_counterfactual_diversity(self) -> float:
        if len(self.experiences) < 20:
            return 0.0
        recent = list(self.experiences)[-50:]
        unique_actions = len(set(e['action'] for e in recent))
        return unique_actions / self.action_dim

    def _compute_meta_cognition(self) -> float:
        if len(self.experiences) < 60:
            return 0.0
        recent = list(self.experiences)[-60:]
        first_half = recent[:30]
        second_half = recent[30:]
        first_diversity = len(set(e['action'] for e in first_half)) / self.action_dim
        second_diversity = len(set(e['action'] for e in second_half)) / self.action_dim
        first_fe = np.mean([e['free_energy'] for e in first_half])
        second_fe = np.mean([e['free_energy'] for e in second_half])
        if first_fe < second_fe:
            return 1.0 if second_diversity > first_diversity else 0.3
        return 0.7

    def _compute_classification_accuracy(self) -> float:
        if len(self.experiences) < 30:
            return 0.0
        recent = list(self.experiences)[-80:]
        groups = {}
        for exp in recent:
            obj_feat = tuple(np.round(exp['obs'][3:7], 0))
            if obj_feat not in groups:
                groups[obj_feat] = []
            groups[obj_feat].append(exp['free_energy'])
        if len(groups) < 2:
            return min(1.0, len(self.grounding.get_grounded_symbols()) / 5.0)
        group_scores = []
        for feat, errors in groups.items():
            if len(errors) >= 3:
                consistency = 1.0 / (1.0 + np.std(errors))
                accuracy = 1.0 / (1.0 + np.mean(errors))
                group_scores.append(consistency * accuracy)
        return np.mean(group_scores) if group_scores else 0.0

    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        stats['current_stage'] = self.development.current_stage
        stats['grounded_symbols'] = len(self.grounding.get_grounded_symbols())
        stats['avg_free_energy'] = (
            stats['total_free_energy'] / max(1, stats['total_steps'])
        )
        stats['learning_progress'] = self.predictor.get_learning_progress()
        stats['avg_precision'] = self.predictor.get_avg_precision()
        stats['exploration_drive'] = self.active_inference.get_exploration_drive()
        stats['risk_drive'] = self.active_inference.get_risk_drive()
        return stats
