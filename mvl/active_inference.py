"""
主动推理：基于期望自由能的动作选择

FEP 的动作选择机制：
不是最大化奖励，而是最小化期望自由能。

期望自由能 G = 信息增益 + 工具价值
  - 信息增益（epistemic value）：选择能减少不确定性的动作
  - 工具价值（pragmatic value）：选择能实现偏好的动作

这统一了好奇心和目标导向行为：
- 好奇心 = 信息增益（探索）
- 目标 = 工具价值（利用）
- 主动推理 = 两者的最优平衡

与 Schmidhuber 好奇心的区别：
- Schmidhuber：curiosity = PE × learnability（加法）
- FEP：G = H[q] - E[ln p(o)]（信息论）
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import deque


class ActiveInferenceModule:
    """
    主动推理模块

    替代 CuriosityModule，实现 FEP 的动作选择。
    不是选择"最有趣"的动作，
    而是选择"最能减少期望自由能"的动作。
    """

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

        # 先验偏好（可由发展目标设定）
        self.preference_mu = None  # 偏好状态的均值
        self.preference_precision = None  # 偏好精度

        # 期望自由能历史
        self.gef_history = deque(maxlen=100)

        # 风险敏感度
        self.risk_penalty_weight = 0.0  # 默认关闭，向后兼容
        self.risk_history = deque(maxlen=100)

    def set_preference(self, target_state: np.ndarray, precision: float = 1.0):
        """
        设定先验偏好

        这是 agent 的"目标"——
        不是外部给的奖励函数，
        而是 agent 内在的状态偏好。
        """
        self.preference_mu = target_state.copy()
        self.preference_precision = precision

    def set_risk_sensitivity(self, weight: float):
        """
        设置风险敏感度

        weight=0: 无风险惩罚（默认，向后兼容）
        weight>0: 惩罚高不确定性动作
        weight越大，agent越规避风险
        """
        self.risk_penalty_weight = weight

    def compute_expected_free_energy(self,
                                      obs: np.ndarray,
                                      action: int,
                                      predictor,
                                      belief) -> float:
        """
        计算期望自由能 G

        G = 信息增益 + 工具价值

        信息增益 = 选择该动作后，信念不确定性减少的期望值
        工具价值 = 选择该动作后，到达偏好状态的期望程度
        """
        # 预测执行该动作后的结果
        mean, log_var = predictor.predict(obs, action)
        var = np.exp(log_var)

        # === 信息增益（epistemic value）===
        # 信息增益 = 当前不确定性 - 预测后不确定性
        # 如果该动作能大幅减少不确定性 → 信息增益大 → 值得探索
        current_uncertainty = np.exp(belief.log_var)
        predicted_uncertainty = var  # 预测的不确定性
        info_gain = np.sum(np.log(current_uncertainty + 1e-8) -
                          np.log(predicted_uncertainty + 1e-8))

        # === 工具价值（pragmatic value）===
        # 工具价值 = 预测结果与偏好的匹配程度
        pragmatic_value = 0.0
        if self.preference_mu is not None:
            pref_error = mean - self.preference_mu[:len(mean)]
            pref_precision = self.preference_precision if self.preference_precision else 1.0
            pragmatic_value = -0.5 * pref_precision * np.sum(pref_error**2)

        # === 风险惩罚（risk penalty）===
        # 惩罚预测方差大且信念精度高的维度
        # 当 agent 不确定（高方差）但关心该维度（高精度）时，风险最大
        risk_term = 0.0
        if self.risk_penalty_weight > 0:
            belief_precision = np.exp(-belief.log_var)
            risk_term = np.sum(var * belief_precision)

        # 期望自由能 = -信息增益 - 工具价值 + 风险惩罚
        # 最小化 G = 最大化信息增益 + 工具价值 - 最小化风险
        gef = -info_gain - pragmatic_value + self.risk_penalty_weight * risk_term

        return gef

    def select_action(self, obs: np.ndarray, predictor,
                      belief, available_actions: List[int],
                      exploration_weight: float = 1.0) -> int:
        """
        选择最小化期望自由能的动作

        这是 FEP 的核心决策机制——
        每个动作的"价值"由期望自由能决定。
        """
        best_action = available_actions[0]
        best_gef = float('inf')
        best_risk = 0.0

        for action in available_actions:
            gef = self.compute_expected_free_energy(obs, action, predictor, belief)

            # 计算该动作的风险项
            mean, log_var = predictor.predict(obs, action)
            var = np.exp(log_var)
            belief_precision = np.exp(-belief.log_var)
            risk = np.sum(var * belief_precision) if self.risk_penalty_weight > 0 else 0.0

            # 添加随机扰动（softmax 选择）
            gef_noisy = gef + np.random.normal(0, 0.1)

            if gef_noisy < best_gef:
                best_gef = gef_noisy
                best_action = action
                best_risk = risk

        self.gef_history.append(best_gef)
        if self.risk_penalty_weight > 0:
            self.risk_history.append(best_risk)
        return best_action

    def get_avg_gef(self) -> float:
        """获取平均期望自由能"""
        if not self.gef_history:
            return 0.0
        return np.mean(list(self.gef_history))

    def get_exploration_drive(self) -> float:
        """
        获取探索驱动力

        当信息增益项主导时 → 探索驱动力高
        当工具价值项主导时 → 探索驱动力低
        """
        if len(self.gef_history) < 10:
            return 1.0

        recent = list(self.gef_history)[-10:]
        variance = np.var(recent)
        return min(1.0, variance / (np.mean(np.abs(recent)) + 1e-8))

    def get_risk_drive(self) -> float:
        """获取风险驱动力（风险惩罚项的平均值）"""
        if not self.risk_history:
            return 0.0
        return np.mean(list(self.risk_history))
