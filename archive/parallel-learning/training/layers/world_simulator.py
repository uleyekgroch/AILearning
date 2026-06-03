"""世界模拟器 — DreamerV3风格的潜在动态网络

不是规则查找，是学习生成模型。

核心能力：
1. 潜在动态 — 学习状态转移函数
2. 心理模拟 — 在潜在空间想象未来
3. 奖励预测 — 预测行动的后果

运行方式：
    python training/layers/world_simulator.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


class Encoder(nn.Module):
    """编码器 — 将观测编码为潜在状态"""

    def __init__(self, obs_dim: int = 256, hidden_dim: int = 256, latent_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码观测"""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class DynamicsNetwork(nn.Module):
    """动态网络 — 学习状态转移函数"""

    def __init__(self, latent_dim: int = 32, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一个状态"""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.fc3(x)
        return next_state


class RewardNetwork(nn.Module):
    """奖励网络 — 预测状态的奖励"""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """预测奖励"""
        x = F.relu(self.fc1(state))
        reward = self.fc2(x)
        return reward


class WorldSimulator:
    """世界模拟器

    学习世界的内部模型，支持心理模拟。
    """

    def __init__(self, obs_dim: int = 256, action_dim: int = 8,
                 latent_dim: int = 32, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # 网络
        self.encoder = Encoder(obs_dim, 256, latent_dim).to(self.device)
        self.dynamics = DynamicsNetwork(latent_dim, action_dim, 256).to(self.device)
        self.reward = RewardNetwork(latent_dim, 64).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.reward.parameters()),
            lr=1e-4
        )

        # 经验缓冲
        self.buffer: List[Tuple[torch.Tensor, int, torch.Tensor, float]] = []

        # 统计
        self.stats = {
            'training_steps': 0,
            'simulations': 0,
            'avg_loss': 0.0,
        }

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """编码观测为潜在状态"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mean, logvar = self.encoder(obs)
        return self.encoder.reparameterize(mean, logvar)

    def predict_next(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """预测下一个状态"""
        action_onehot = torch.zeros(self.action_dim, device=self.device)
        action_onehot[action] = 1.0

        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_vec = action_onehot.unsqueeze(0)
        next_state = self.dynamics(state, action_vec)
        return next_state

    def predict_reward(self, state: torch.Tensor) -> float:
        """预测状态的奖励"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            reward = self.reward(state)
        return reward.item()

    def imagine(self, initial_state: torch.Tensor, actions: List[int]) -> List[Tuple[torch.Tensor, float]]:
        """心理模拟 — 想象行动的后果

        给定初始状态和一系列行动，想象会发生什么。
        """
        trajectory = []
        state = initial_state

        for action in actions:
            # 预测下一个状态
            next_state = self.predict_next(state, action)

            # 预测奖励
            reward = self.predict_reward(next_state)

            trajectory.append((next_state, reward))
            state = next_state

        self.stats['simulations'] += 1
        return trajectory

    def learn_from_experience(self, obs: torch.Tensor, action: int,
                             next_obs: torch.Tensor, reward: float):
        """从经验中学习"""
        # 存储经验
        self.buffer.append((obs, action, next_obs, reward))

        # 编码
        state = self.encode(obs)
        next_state_true = self.encode(next_obs)

        # 预测
        action_onehot = torch.zeros(self.action_dim, device=self.device)
        action_onehot[action] = 1.0

        next_state_pred = self.dynamics(state, action_onehot.unsqueeze(0))
        reward_pred = self.reward(state)

        # 计算损失
        state_loss = F.mse_loss(next_state_pred, next_state_true)
        reward_loss = F.mse_loss(reward_pred, torch.tensor([[reward]], device=self.device))
        loss = state_loss + reward_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.stats['training_steps'] += 1
        self.stats['avg_loss'] = loss.item()

    def plan(self, initial_obs: torch.Tensor, horizon: int = 5,
             num_samples: int = 10) -> List[int]:
        """规划 — 找到最佳行动序列

        通过想象多个可能的未来，选择最好的。
        """
        initial_state = self.encode(initial_obs)

        best_actions = []
        best_reward = float('-inf')

        # 随机采样多个行动序列
        for _ in range(num_samples):
            actions = [np.random.randint(self.action_dim) for _ in range(horizon)]
            trajectory = self.imagine(initial_state, actions)

            # 计算总奖励
            total_reward = sum(reward for _, reward in trajectory)

            if total_reward > best_reward:
                best_reward = total_reward
                best_actions = actions

        return best_actions

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'device': str(self.device),
            'buffer_size': len(self.buffer),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'latent_dim': self.latent_dim,
        }


def test_world_simulator():
    """测试世界模拟器"""
    print("=" * 70)
    print("世界模拟器测试")
    print("=" * 70)

    simulator = WorldSimulator(obs_dim=64, action_dim=4, latent_dim=16)

    # 测试编码
    print("\n1. 编码测试:")
    obs = torch.randn(64).to(simulator.device)
    state = simulator.encode(obs)
    print(f"  观测维度: {obs.shape}")
    print(f"  状态维度: {state.shape}")

    # 测试预测
    print("\n2. 预测测试:")
    next_state = simulator.predict_next(state, 0)
    reward = simulator.predict_reward(state)
    print(f"  下一状态维度: {next_state.shape}")
    print(f"  预测奖励: {reward:.3f}")

    # 测试心理模拟
    print("\n3. 心理模拟测试:")
    actions = [0, 1, 2, 3]
    trajectory = simulator.imagine(state, actions)
    print(f"  行动序列: {actions}")
    print(f"  轨迹长度: {len(trajectory)}")
    for i, (state, reward) in enumerate(trajectory):
        print(f"    步骤 {i}: 奖励={reward:.3f}")

    # 测试学习
    print("\n4. 学习测试:")
    for i in range(10):
        obs = torch.randn(64).to(simulator.device)
        next_obs = torch.randn(64).to(simulator.device)
        action = np.random.randint(4)
        reward = np.random.randn()
        simulator.learn_from_experience(obs, action, next_obs, reward)

    print(f"  训练步数: {simulator.stats['training_steps']}")
    print(f"  平均损失: {simulator.stats['avg_loss']:.4f}")

    # 测试规划
    print("\n5. 规划测试:")
    obs = torch.randn(64).to(simulator.device)
    best_actions = simulator.plan(obs, horizon=3, num_samples=5)
    print(f"  最佳行动序列: {best_actions}")

    # 统计
    print("\n统计:")
    stats = simulator.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_world_simulator()
