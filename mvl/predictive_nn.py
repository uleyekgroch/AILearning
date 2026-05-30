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
    神经网络预测器 — 预测编码实现

    架构：输入 → 隐藏层1 → 隐藏层2 → 输出
    激活函数：ReLU
    学习规则：预测编码 + 局部 Hebbian 更新

    核心原理（Whittington & Bogacz, 2017）：
    1. 每层维护信念 μ 和局部残差 ε = μ - f(W × μ_below)
    2. 迭代推理让信念收敛到最优估计
    3. 权重更新 ΔW = η × ε_post × f' × μ_pre^T（纯局部 Hebbian）

    与反向传播的区别：
    - 反向传播需要链式法则穿层（非局部）
    - 预测编码每层只看相邻层的信号（局部）
    - 收敛后等价于反向传播的梯度
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden1_dim: int = 128, hidden2_dim: int = 64,
                 inference_lr: float = 0.05,
                 max_inference_steps: int = 50,
                 convergence_threshold: float = 1e-4):
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

        # 预测编码推理参数
        self.inference_lr = inference_lr
        self.max_inference_steps = max_inference_steps
        self.convergence_threshold = convergence_threshold
        self.clip_val = 3.0

        # 预测误差历史
        self.error_history = deque(maxlen=100)

        # 缓存前向传播结果
        self._cache = {}

        # 推理步数统计
        self._inference_steps_log = []

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (x > 0).astype(float)

    def predict(self, obs: np.ndarray, action) -> np.ndarray:
        """
        预测下一时刻的观测

        前向传播：
        1. 输入 = [obs, action_vec]
        2. hidden1 = relu(W1 @ input + b1)
        3. hidden2 = relu(W2 @ hidden1 + b2)
        4. output = W3 @ hidden2 + b3

        action: int（离散，one-hot）或 np.ndarray（连续，直接使用）
        """
        # 编码动作
        if isinstance(action, np.ndarray):
            action_vec = action[:self.action_dim]
        else:
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

    def learn(self, obs: np.ndarray, action, actual_next_obs: np.ndarray) -> float:
        """
        学习 = 预测编码 + 局部 Hebbian 更新

        三步走：
        1. 前向初始化 μ（信念）
        2. 迭代推理：局部残差 + 反馈连接 → 收敛
        3. Hebbian 权重更新：ΔW = η × ε_post × f' × μ_pre^T
        """
        # 编码动作
        if isinstance(action, np.ndarray):
            action_vec = action[:self.action_dim]
        else:
            action_vec = np.zeros(self.action_dim)
            action_vec[action] = 1.0

        x = np.concatenate([obs, action_vec])

        # === Step 1: 前向初始化 μ ===
        mu_h1 = self._relu(x @ self.W1 + self.b1)
        mu_h2 = self._relu(mu_h1 @ self.W2 + self.b2)
        # mu_output = actual_next_obs （clamp 目标）

        # === Step 2: 迭代推理（自适应停止）===
        steps_taken = 0
        cv = self.clip_val
        z1 = x @ self.W1 + self.b1
        z2 = mu_h1 @ self.W2 + self.b2
        for t in range(self.max_inference_steps):
            prev_h1 = mu_h1.copy()
            prev_h2 = mu_h2.copy()

            # 局部残差（每层独立计算）
            z1 = x @ self.W1 + self.b1
            z2 = mu_h1 @ self.W2 + self.b2
            pred_out = mu_h2 @ self.W3 + self.b3

            epsilon_out = np.clip(actual_next_obs - pred_out, -cv, cv)
            epsilon_h2 = np.clip(mu_h2 - self._relu(z2), -cv, cv)
            epsilon_h1 = np.clip(mu_h1 - self._relu(z1), -cv, cv)

            # 反馈信号（通过反馈连接，不是链式法则）
            feedback_h2 = np.clip(self.W3 @ epsilon_out, -cv, cv) * self._relu_derivative(z2)
            e_h2_scaled = np.clip(epsilon_h2 * self._relu_derivative(z2), -cv, cv)
            feedback_h1 = np.clip(self.W2 @ e_h2_scaled, -cv, cv) * self._relu_derivative(z1)

            # 更新信念
            mu_h1 = np.clip(mu_h1 - self.inference_lr * np.clip(-epsilon_h1 + feedback_h1, -cv, cv), -cv, cv)
            mu_h2 = np.clip(mu_h2 - self.inference_lr * np.clip(-epsilon_h2 + feedback_h2, -cv, cv), -cv, cv)

            steps_taken = t + 1

            # 自适应停止
            change = 0.5 * (np.mean((mu_h1 - prev_h1) ** 2) + np.mean((mu_h2 - prev_h2) ** 2))
            if change < self.convergence_threshold:
                break

        self._inference_steps_log.append(steps_taken)

        # === Step 3: 最终残差（复用循环最后的 z1/z2）===
        pred_out = mu_h2 @ self.W3 + self.b3
        epsilon_out = actual_next_obs - pred_out
        epsilon_h2 = mu_h2 - self._relu(z2)
        epsilon_h1 = mu_h1 - self._relu(z1)

        # === Step 4: 局部 Hebbian 权重更新 ===
        # ΔW3 = η × μ_h2 × ε_out^T
        self.W3 += self.lr * np.outer(mu_h2, epsilon_out)
        self.b3 += self.lr * epsilon_out

        # ΔW2 = η × μ_h1 × (ε_h2 × f')^T
        self.W2 += self.lr * np.outer(mu_h1, epsilon_h2 * self._relu_derivative(z2))
        self.b2 += self.lr * epsilon_h2 * self._relu_derivative(z2)

        # ΔW1 = η × x × (ε_h1 × f')^T
        self.W1 += self.lr * np.outer(x, epsilon_h1 * self._relu_derivative(z1))
        self.b1 += self.lr * epsilon_h1 * self._relu_derivative(z1)

        prediction_error = float(np.mean(epsilon_out ** 2))
        self.error_history.append(prediction_error)

        return prediction_error

    def learn_and_get_input_gradient(self, obs: np.ndarray, action,
                                      actual_next_obs: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        学习并返回输入梯度（端到端学习）

        与 learn() 相同的预测编码，但额外返回 ε_input 作为编码器的梯度信号。

        Returns: (prediction_error, d_obs)
        """
        # 编码动作
        if isinstance(action, np.ndarray):
            action_vec = action[:self.action_dim]
        else:
            action_vec = np.zeros(self.action_dim)
            action_vec[action] = 1.0

        x = np.concatenate([obs, action_vec])

        # === Step 1: 前向初始化 ===
        mu_h1 = self._relu(x @ self.W1 + self.b1)
        mu_h2 = self._relu(mu_h1 @ self.W2 + self.b2)

        # === Step 2: 迭代推理 ===
        cv = self.clip_val
        z1 = x @ self.W1 + self.b1
        z2 = mu_h1 @ self.W2 + self.b2
        for t in range(self.max_inference_steps):
            prev_h1 = mu_h1.copy()
            prev_h2 = mu_h2.copy()

            z1 = x @ self.W1 + self.b1
            z2 = mu_h1 @ self.W2 + self.b2
            pred_out = mu_h2 @ self.W3 + self.b3

            epsilon_out = np.clip(actual_next_obs - pred_out, -cv, cv)
            epsilon_h2 = np.clip(mu_h2 - self._relu(z2), -cv, cv)
            epsilon_h1 = np.clip(mu_h1 - self._relu(z1), -cv, cv)

            feedback_h2 = np.clip(self.W3 @ epsilon_out, -cv, cv) * self._relu_derivative(z2)
            e_h2_scaled = np.clip(epsilon_h2 * self._relu_derivative(z2), -cv, cv)
            feedback_h1 = np.clip(self.W2 @ e_h2_scaled, -cv, cv) * self._relu_derivative(z1)

            mu_h1 = np.clip(mu_h1 - self.inference_lr * np.clip(-epsilon_h1 + feedback_h1, -cv, cv), -cv, cv)
            mu_h2 = np.clip(mu_h2 - self.inference_lr * np.clip(-epsilon_h2 + feedback_h2, -cv, cv), -cv, cv)

            change = 0.5 * (np.mean((mu_h1 - prev_h1) ** 2) + np.mean((mu_h2 - prev_h2) ** 2))
            if change < self.convergence_threshold:
                break

        # === Step 3: 最终残差 + 输入梯度（复用循环最后的 z1/z2）===
        pred_out = mu_h2 @ self.W3 + self.b3
        epsilon_out = actual_next_obs - pred_out
        epsilon_h2 = mu_h2 - self._relu(z2)
        epsilon_h1 = mu_h1 - self._relu(z1)

        # 输入梯度：ε_input 传回编码器
        epsilon_input = epsilon_h1 * self._relu_derivative(z1) @ self.W1.T
        d_obs = epsilon_input[:self.obs_dim]

        # === Step 4: Hebbian 权重更新 ===
        self.W3 += self.lr * np.outer(mu_h2, epsilon_out)
        self.b3 += self.lr * epsilon_out
        self.W2 += self.lr * np.outer(mu_h1, epsilon_h2 * self._relu_derivative(z2))
        self.b2 += self.lr * epsilon_h2 * self._relu_derivative(z2)
        self.W1 += self.lr * np.outer(x, epsilon_h1 * self._relu_derivative(z1))
        self.b1 += self.lr * epsilon_h1 * self._relu_derivative(z1)

        prediction_error = float(np.mean(epsilon_out ** 2))
        self.error_history.append(prediction_error)

        return prediction_error, d_obs

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

    def learn(self, obs: np.ndarray, action, actual_next_obs: np.ndarray) -> float:
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
