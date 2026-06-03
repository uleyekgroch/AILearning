"""
预测编码学习引擎 — PyTorch 原生实现

从 mvl/predictive_nn.py 移植，核心改造：
- np.ndarray → torch.Tensor
- 裸权重数组 → torch.nn.Parameter
- np.outer → torch.ger
- torch.nn.Module 封装，支持 .to(device)

核心算法（Whittington & Bogacz, 2017）：
1. 前向初始化信念 μ
2. 迭代推理：局部残差 + 反馈连接 → 收敛
3. Hebbian 权重更新：ΔW = η × ε_post × f' × μ_pre^T

与反向传播的区别：每层只看相邻层信号（局部性），收敛后等价于 BP 梯度。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from collections import deque

from src.core.interfaces import ILearningEngine
from src.core.config import LearnerConfig
from src.core.device import get_device


class PredictiveCodingEngine(nn.Module, ILearningEngine):
    """
    预测编码引擎

    架构：input(obs+action) → hidden1 → hidden2 → output(obs')
    激活：ReLU
    学习：预测编码 + 局部 Hebbian 更新
    """

    def __init__(self, config: LearnerConfig):
        nn.Module.__init__(self)
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        h1, h2 = config.hidden_dims
        self.hidden1_dim = h1
        self.hidden2_dim = h2
        input_dim = config.obs_dim + config.action_dim

        self.device = get_device(config.device)

        # He 初始化权重
        self.W1 = nn.Parameter(torch.randn(input_dim, h1) * (2.0 / input_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(h1))
        self.W2 = nn.Parameter(torch.randn(h1, h2) * (2.0 / h1) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(h2))
        self.W3 = nn.Parameter(torch.randn(h2, config.obs_dim) * (2.0 / h2) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(config.obs_dim))

        # 超参数
        self.lr = config.learning_rate
        self.inference_lr = config.inference_lr
        self.max_inference_steps = config.max_inference_steps
        self.convergence_threshold = config.convergence_threshold
        self.clip_val = 3.0

        # 统计
        self.error_history = deque(maxlen=100)
        self._inference_steps_log: deque = deque(maxlen=200)

        # 好奇心状态
        self.curiosity_alpha = config.curiosity_alpha
        self.curiosity_beta = config.curiosity_beta
        self.curiosity_decay = config.curiosity_decay
        self._prediction_errors = deque(maxlen=50)
        self._learning_progress = 0.0

        self.to(self.device)

    @staticmethod
    def _relu(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)

    @staticmethod
    def _relu_derivative(x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """自动将张量移到引擎所在设备"""
        return t.to(self.device)

    def _encode_action(self, action) -> torch.Tensor:
        """将动作编码为向量"""
        if isinstance(action, torch.Tensor) and action.dim() == 1 and action.shape[0] > 1:
            return self._to_device(action[:self.action_dim])
        elif isinstance(action, torch.Tensor) and action.dim() == 0:
            idx = action.item()
            vec = torch.zeros(self.action_dim, device=self.device)
            vec[idx] = 1.0
            return vec
        else:
            return torch.zeros(self.action_dim, device=self.device)

    def predict(self, state: torch.Tensor, action=None) -> torch.Tensor:
        """前向传播：预测下一个状态"""
        state = self._to_device(state)
        if action is not None:
            action_vec = self._encode_action(action)
            x = torch.cat([state, action_vec])
        else:
            x = torch.cat([state, torch.zeros(self.action_dim, device=self.device)])

        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        output = h2 @ self.W3 + self.b3
        return output

    def predict_batch(self, states: torch.Tensor,
                      action_vecs: torch.Tensor) -> torch.Tensor:
        """批量前向传播：一次矩阵运算处理多个 (state, action) 对"""
        states = self._to_device(states)
        action_vecs = self._to_device(action_vecs)
        x = torch.cat([states, action_vecs], dim=-1)
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def learn(self, predicted: torch.Tensor, actual: torch.Tensor,
              obs: torch.Tensor = None, action=None) -> float:
        """
        预测编码学习

        三步走：
        1. 前向初始化信念 μ
        2. 迭代推理收敛
        3. 局部 Hebbian 权重更新
        """
        if obs is None:
            obs = predicted.detach()

        obs = self._to_device(obs)
        actual = self._to_device(actual)
        action_vec = self._encode_action(action) if action is not None else \
            torch.zeros(self.action_dim, device=self.device)
        x = torch.cat([obs, action_vec])

        cv = self.clip_val
        with torch.no_grad():
            # Step 1: 前向初始化
            mu_h1 = self._relu(x @ self.W1 + self.b1)
            mu_h2 = self._relu(mu_h1 @ self.W2 + self.b2)

            # Step 2: 迭代推理
            steps_taken = 0
            z1 = x @ self.W1 + self.b1
            z2 = mu_h1 @ self.W2 + self.b2
            for t in range(self.max_inference_steps):
                prev_h1 = mu_h1.clone()
                prev_h2 = mu_h2.clone()

                z1 = x @ self.W1 + self.b1
                z2 = mu_h1 @ self.W2 + self.b2
                pred_out = mu_h2 @ self.W3 + self.b3

                epsilon_out = torch.clamp(actual - pred_out, -cv, cv)
                epsilon_h2 = torch.clamp(mu_h2 - self._relu(z2), -cv, cv)
                epsilon_h1 = torch.clamp(mu_h1 - self._relu(z1), -cv, cv)

                feedback_h2 = torch.clamp(self.W3 @ epsilon_out, -cv, cv) * self._relu_derivative(z2)
                e_h2_scaled = torch.clamp(epsilon_h2 * self._relu_derivative(z2), -cv, cv)
                feedback_h1 = torch.clamp(self.W2 @ e_h2_scaled, -cv, cv) * self._relu_derivative(z1)

                mu_h1 = torch.clamp(
                    mu_h1 - self.inference_lr * torch.clamp(-epsilon_h1 + feedback_h1, -cv, cv),
                    -cv, cv)
                mu_h2 = torch.clamp(
                    mu_h2 - self.inference_lr * torch.clamp(-epsilon_h2 + feedback_h2, -cv, cv),
                    -cv, cv)

                steps_taken = t + 1
                change = 0.5 * ((mu_h1 - prev_h1).pow(2).mean() +
                                (mu_h2 - prev_h2).pow(2).mean())
                if change < self.convergence_threshold:
                    break

            self._inference_steps_log.append(steps_taken)

            # Step 3: 最终残差
            pred_out = mu_h2 @ self.W3 + self.b3
            epsilon_out = actual - pred_out
            epsilon_h2 = mu_h2 - self._relu(z2)
            epsilon_h1 = mu_h1 - self._relu(z1)

        # Step 4: Hebbian 权重更新（需要梯度）
        with torch.no_grad():
            self.W3.add_(self.lr * torch.ger(mu_h2, epsilon_out))
            self.b3.add_(self.lr * epsilon_out)

            grad_h2 = epsilon_h2 * self._relu_derivative(z2)
            self.W2.add_(self.lr * torch.ger(mu_h1, grad_h2))
            self.b2.add_(self.lr * grad_h2)

            grad_h1 = epsilon_h1 * self._relu_derivative(z1)
            self.W1.add_(self.lr * torch.ger(x, grad_h1))
            self.b1.add_(self.lr * grad_h1)

        prediction_error = float(epsilon_out.pow(2).mean())
        self.error_history.append(prediction_error)
        self._prediction_errors.append(prediction_error)

        # 更新学习进度
        self._learning_progress = self.get_learning_progress()

        return prediction_error

    def learn_with_input_gradient(self, obs: torch.Tensor, action,
                                   actual: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """学习并返回输入梯度（供编码器端到端学习）"""
        obs = self._to_device(obs)
        actual = self._to_device(actual)
        action_vec = self._encode_action(action)
        x = torch.cat([obs, action_vec])

        cv = self.clip_val
        with torch.no_grad():
            mu_h1 = self._relu(x @ self.W1 + self.b1)
            mu_h2 = self._relu(mu_h1 @ self.W2 + self.b2)

            z1 = x @ self.W1 + self.b1
            z2 = mu_h1 @ self.W2 + self.b2
            for t in range(self.max_inference_steps):
                prev_h1 = mu_h1.clone()
                prev_h2 = mu_h2.clone()

                z1 = x @ self.W1 + self.b1
                z2 = mu_h1 @ self.W2 + self.b2
                pred_out = mu_h2 @ self.W3 + self.b3

                epsilon_out = torch.clamp(actual - pred_out, -cv, cv)
                epsilon_h2 = torch.clamp(mu_h2 - self._relu(z2), -cv, cv)
                epsilon_h1 = torch.clamp(mu_h1 - self._relu(z1), -cv, cv)

                feedback_h2 = torch.clamp(self.W3 @ epsilon_out, -cv, cv) * self._relu_derivative(z2)
                e_h2_scaled = torch.clamp(epsilon_h2 * self._relu_derivative(z2), -cv, cv)
                feedback_h1 = torch.clamp(self.W2 @ e_h2_scaled, -cv, cv) * self._relu_derivative(z1)

                mu_h1 = torch.clamp(
                    mu_h1 - self.inference_lr * torch.clamp(-epsilon_h1 + feedback_h1, -cv, cv),
                    -cv, cv)
                mu_h2 = torch.clamp(
                    mu_h2 - self.inference_lr * torch.clamp(-epsilon_h2 + feedback_h2, -cv, cv),
                    -cv, cv)

                change = 0.5 * ((mu_h1 - prev_h1).pow(2).mean() +
                                (mu_h2 - prev_h2).pow(2).mean())
                if change < self.convergence_threshold:
                    break

            # 最终残差
            pred_out = mu_h2 @ self.W3 + self.b3
            epsilon_out = actual - pred_out
            epsilon_h2 = mu_h2 - self._relu(z2)
            epsilon_h1 = mu_h1 - self._relu(z1)

            # 输入梯度
            epsilon_input = (epsilon_h1 * self._relu_derivative(z1)) @ self.W1.T
            d_obs = epsilon_input[:self.obs_dim]

            # Hebbian 更新
            self.W3.add_(self.lr * torch.ger(mu_h2, epsilon_out))
            self.b3.add_(self.lr * epsilon_out)

            grad_h2 = epsilon_h2 * self._relu_derivative(z2)
            self.W2.add_(self.lr * torch.ger(mu_h1, grad_h2))
            self.b2.add_(self.lr * grad_h2)

            grad_h1 = epsilon_h1 * self._relu_derivative(z1)
            self.W1.add_(self.lr * torch.ger(x, grad_h1))
            self.b1.add_(self.lr * grad_h1)

        prediction_error = float(epsilon_out.pow(2).mean())
        self.error_history.append(prediction_error)
        self._prediction_errors.append(prediction_error)

        return prediction_error, d_obs.detach()

    def get_curiosity(self, state: torch.Tensor) -> float:
        """好奇心 = 预测误差 × 可学习性"""
        if len(self._prediction_errors) < 2:
            return 1.0

        # 预测误差分量
        recent_errors = list(self._prediction_errors)[-10:]
        avg_error = sum(recent_errors) / len(recent_errors)

        # 可学习性分量（学习进度越高，可学习性越低）
        learnability = max(0.0, 1.0 - self._learning_progress)

        curiosity = self.curiosity_alpha * min(avg_error, 1.0) + \
                    self.curiosity_beta * learnability

        return max(0.0, curiosity)

    def get_learning_progress(self) -> float:
        """学习进度 = 误差减少速率"""
        if len(self.error_history) < 10:
            return 0.0

        errors = list(self.error_history)
        half = len(errors) // 2
        first_half = sum(errors[:half]) / half
        second_half = sum(errors[half:]) / (len(errors) - half)

        if first_half > 0:
            progress = (first_half - second_half) / first_half
        else:
            progress = 0.0

        return max(0.0, min(1.0, progress))

    def get_avg_inference_steps(self) -> float:
        """平均推理步数（监测收敛效率）"""
        if not self._inference_steps_log:
            return 0.0
        return sum(self._inference_steps_log[-100:]) / len(self._inference_steps_log[-100:])
