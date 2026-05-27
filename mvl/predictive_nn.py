"""
神经网络预测模型：学习体的"升级大脑"

核心改进：
- 2层隐藏层，ReLU激活
- 能学习非线性映射
- 更强的表达能力

这是从"线性世界理解"到"非线性世界理解"的关键升级。
"""

import numpy as np
from typing import List, Tuple
from collections import deque


class NeuralNetworkPredictor:
    """
    神经网络预测器

    架构：输入 → 隐藏层1 → 隐藏层2 → 输出
    激活函数：ReLU
    学习率：自适应

    与线性模型的区别：
    - 能学习非线性映射
    - 更强的表达能力
    - 更好的泛化
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden1_dim: int = 128, hidden2_dim: int = 64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        # 输入维度 = obs_dim + action_dim (one-hot)
        input_dim = obs_dim + action_dim

        # 初始化权重（He初始化）
        self.W1 = np.random.randn(input_dim, hidden1_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1_dim)

        self.W2 = np.random.randn(hidden1_dim, hidden2_dim) * np.sqrt(2.0 / hidden1_dim)
        self.b2 = np.zeros(hidden2_dim)

        self.W3 = np.random.randn(hidden2_dim, obs_dim) * np.sqrt(2.0 / hidden2_dim)
        self.b3 = np.zeros(obs_dim)

        # 学习率
        self.lr = 0.001

        # 预测误差历史
        self.error_history = deque(maxlen=100)

        # 缓存前向传播结果
        self._cache = {}

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (x > 0).astype(float)

    def predict(self, obs: np.ndarray, action: int) -> np.ndarray:
        """
        预测下一时刻的观测

        前向传播：
        1. 输入 = [obs, action_one_hot]
        2. hidden1 = relu(W1 @ input + b1)
        3. hidden2 = relu(W2 @ hidden1 + b2)
        4. output = W3 @ hidden2 + b3
        """
        # 编码动作
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0

        # 拼接输入
        x = np.concatenate([obs, action_vec])

        # 前向传播
        z1 = x @ self.W1 + self.b1
        h1 = self._relu(z1)

        z2 = h1 @ self.W2 + self.b2
        h2 = self._relu(z2)

        output = h2 @ self.W3 + self.b3

        # 缓存用于反向传播
        self._cache = {
            'x': x,
            'z1': z1,
            'h1': h1,
            'z2': z2,
            'h2': h2,
            'output': output
        }

        return output

    def learn(self, obs: np.ndarray, action: int, actual_next_obs: np.ndarray) -> float:
        """
        学习 = 减少预测误差

        反向传播：
        1. 计算输出误差
        2. 反向传播到隐藏层
        3. 更新权重
        """
        # 前向传播
        prediction = self.predict(obs, action)

        # 计算误差
        error = actual_next_obs - prediction
        prediction_error = np.mean(error ** 2)

        # 反向传播
        # 输出层梯度
        d_output = -2 * error / len(error)

        # W3, b3梯度
        d_W3 = np.outer(self._cache['h2'], d_output)
        d_b3 = d_output

        # 隐藏层2梯度
        d_h2 = d_output @ self.W3.T
        d_z2 = d_h2 * self._relu_derivative(self._cache['z2'])

        # W2, b2梯度
        d_W2 = np.outer(self._cache['h1'], d_z2)
        d_b2 = d_z2

        # 隐藏层1梯度
        d_h1 = d_z2 @ self.W2.T
        d_z1 = d_h1 * self._relu_derivative(self._cache['z1'])

        # W1, b1梯度
        d_W1 = np.outer(self._cache['x'], d_z1)
        d_b1 = d_z1

        # 更新权重
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

        # 记录误差
        self.error_history.append(prediction_error)

        return prediction_error

    def get_learning_progress(self) -> float:
        """
        计算学习进度

        学习进度 = 误差减少的速率
        """
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


class AdaptiveLearningRatePredictor(NeuralNetworkPredictor):
    """
    自适应学习率预测器

    根据学习进度自动调整学习率：
    - 学习快时：降低学习率，避免震荡
    - 学习慢时：提高学习率，加速收敛
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden1_dim: int = 128, hidden2_dim: int = 64):
        super().__init__(obs_dim, action_dim, hidden1_dim, hidden2_dim)

        # 自适应学习率参数
        self.lr_min = 0.0001
        self.lr_max = 0.01
        self.lr_decay = 0.999
        self.lr_growth = 1.001

        # 学习率历史
        self.lr_history = []

    def learn(self, obs: np.ndarray, action: int, actual_next_obs: np.ndarray) -> float:
        """学习并自适应调整学习率"""
        error = super().learn(obs, action, actual_next_obs)

        # 自适应调整学习率
        progress = self.get_learning_progress()
        if progress > 0.1:
            # 学习快，降低学习率
            self.lr = max(self.lr_min, self.lr * self.lr_decay)
        elif progress < 0.01:
            # 学习慢，提高学习率
            self.lr = min(self.lr_max, self.lr * self.lr_growth)

        self.lr_history.append(self.lr)

        return error
