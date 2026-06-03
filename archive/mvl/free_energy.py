"""
自由能原理：概率生成模型与变分自由能

Karl Friston 的自由能原理：
所有自组织系统都在最小化变分自由能，
这等价于最小化预测误差（当先验平坦时）。

核心区别：
- MSE：f(x) → μ, loss = (o - μ)²
- FEP：f(x) → (μ, σ²), F = (o - μ)²/σ² - ln(σ²)

精度 π = 1/σ² 是关键——
它让 agent 自动学会"关注什么"：
高精度维度 → 大梯度 → 快学习
低精度维度 → 小梯度 → 慢学习

这是注意力机制的贝叶斯解释。
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class ProbabilisticPredictor:
    """
    概率生成模型

    与 DeterministicPredictor 的区别：
    - 输出 (mean, log_variance) 而非单点预测
    - 使用精度加权的预测误差
    - 自由能 = 复杂度 - 准确度
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 均值预测网络
        self.W_obs = np.random.randn(obs_dim, hidden_dim) * 0.01
        self.W_action = np.random.randn(action_dim, hidden_dim) * 0.01
        self.W_mean = np.random.randn(hidden_dim, obs_dim) * 0.01

        # 对数方差预测网络（log σ² 保证数值稳定性）
        self.W_logvar = np.random.randn(hidden_dim, obs_dim) * 0.01
        self.b_logvar = np.zeros(obs_dim)  # 初始化为0 → σ²=1

        # 学习率
        self.lr = 0.01
        self.lr_precision = 0.005  # 精度网络的学习率

        # 历史记录
        self.error_history = deque(maxlen=100)
        self.free_energy_history = deque(maxlen=100)
        self.precision_history = deque(maxlen=100)

    def predict(self, obs: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        概率预测：输出均值和对数方差

        Returns:
            mean: 预测均值 μ
            log_var: 预测对数方差 log(σ²)
        """
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0

        hidden = np.tanh(obs @ self.W_obs + action_vec @ self.W_action)

        mean = hidden @ self.W_mean
        log_var = hidden @ self.W_logvar + self.b_logvar

        return mean, log_var

    def predict_mean_only(self, obs: np.ndarray, action: int) -> np.ndarray:
        """仅返回均值（兼容接口）"""
        mean, _ = self.predict(obs, action)
        return mean

    def learn(self, obs: np.ndarray, action: int,
              actual_next_obs: np.ndarray) -> Tuple[float, float]:
        """
        自由能最小化学习

        Returns:
            free_energy: 变分自由能 F
            prediction_error: 精度加权预测误差
        """
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0

        # 前向传播
        hidden = np.tanh(obs @ self.W_obs + action_vec @ self.W_action)
        mean = hidden @ self.W_mean
        log_var = hidden @ self.W_logvar + self.b_logvar

        # 数值稳定性：限制 log_var 范围
        # 下限 -4.6 ≈ σ²=0.01，上限 5.0 ≈ σ²=148
        log_var = np.clip(log_var, -4.6, 5.0)
        var = np.exp(log_var)
        precision = 1.0 / var  # 精度 π = 1/σ²

        # 预测误差
        error = actual_next_obs - mean

        # 精度加权误差
        precision_weighted_error = error * precision

        # 变分自由能 F = 复杂度 - 准确度
        # 简化版（假设先验平坦，复杂度项为常数）：
        # F ≈ -准确度 = 0.5 * Σ(π * ε² - ln π)
        accuracy = 0.5 * np.sum(precision * error**2 - log_var)
        free_energy = accuracy  # 复杂度项在先验平坦时为常数

        # === 精度加权梯度更新（带梯度裁剪）===

        # 准确度对 mean 的梯度（精度加权）
        d_accuracy_d_mean = -precision_weighted_error

        # 准确度对 log_var 的梯度
        # accuracy = 0.5 * Σ(π·ε² - log_var) = 0.5 * Σ(exp(-log_var)·ε² - log_var)
        # d/d(log_var) = 0.5 * (-exp(-log_var)·ε² - 1) = -0.5 * (π·ε² + 1)
        d_accuracy_d_logvar = -0.5 * (precision * error**2 + 1.0)

        # 梯度裁剪：防止精度放大导致梯度爆炸
        clip_val = 5.0
        d_accuracy_d_mean = np.clip(d_accuracy_d_mean, -clip_val, clip_val)
        d_accuracy_d_logvar = np.clip(d_accuracy_d_logvar, -clip_val, clip_val)

        # 反向传播到隐藏层
        # 均值网络梯度
        d_hidden_mean = d_accuracy_d_mean @ self.W_mean.T * (1 - hidden**2)
        self.W_mean -= self.lr * hidden.reshape(-1, 1) @ d_accuracy_d_mean.reshape(1, -1)
        self.W_obs -= self.lr * obs.reshape(-1, 1) @ d_hidden_mean.reshape(1, -1)
        self.W_action -= self.lr * action_vec.reshape(-1, 1) @ d_hidden_mean.reshape(1, -1)

        # 方差网络梯度
        d_hidden_logvar = d_accuracy_d_logvar @ self.W_logvar.T * (1 - hidden**2)
        self.W_logvar -= self.lr_precision * hidden.reshape(-1, 1) @ d_accuracy_d_logvar.reshape(1, -1)
        self.b_logvar -= self.lr_precision * d_accuracy_d_logvar
        self.W_obs -= self.lr_precision * 0.1 * obs.reshape(-1, 1) @ d_hidden_logvar.reshape(1, -1)

        # 记录
        self.error_history.append(np.mean(error**2))
        self.free_energy_history.append(free_energy)
        self.precision_history.append(np.mean(precision))

        return free_energy, np.mean(precision_weighted_error**2)

    def get_learning_progress(self) -> float:
        """学习进度 = 自由能下降速率"""
        if len(self.free_energy_history) < 10:
            return 0.0

        recent = list(self.free_energy_history)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])

        if first_half > 0:
            progress = (first_half - second_half) / abs(first_half)
        else:
            progress = 0.0

        return max(0.0, progress)

    def get_avg_precision(self) -> float:
        """获取平均精度"""
        if not self.precision_history:
            return 1.0
        return np.mean(list(self.precision_history))

    def get_free_energy(self) -> float:
        """获取当前自由能"""
        if not self.free_energy_history:
            return 0.0
        return self.free_energy_history[-1]


class PrecisionModulatedLearning:
    """
    精度调制学习

    FEP 的核心洞见：学习率应该由精度决定。
    高精度（高确定性）→ 学习快
    低精度（低确定性）→ 学习慢

    这实现了"选择性注意力"——
    agent 自动学会关注可预测的维度，忽略噪声。
    """

    def __init__(self, base_lr: float = 0.01):
        self.base_lr = base_lr
        self.lr_history = deque(maxlen=100)

    def compute_effective_lr(self, precision: np.ndarray) -> np.ndarray:
        """
        根据精度计算各维度的有效学习率

        高精度维度获得更高学习率。
        """
        # 归一化精度到 [0.1, 2.0] 范围
        normalized = precision / (np.mean(precision) + 1e-8)
        effective_lr = self.base_lr * np.clip(normalized, 0.1, 2.0)

        self.lr_history.append(np.mean(effective_lr))
        return effective_lr

    def get_avg_lr(self) -> float:
        """获取平均有效学习率"""
        if not self.lr_history:
            return self.base_lr
        return np.mean(list(self.lr_history))
