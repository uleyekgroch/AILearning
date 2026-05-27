"""
变分推断：贝叶斯信念更新

变分推断是 FEP 的核心计算机制：
agent 维护对隐藏状态的信念 q(s)，
通过最小化自由能来更新信念。

类比：
- 婴儿看到一个模糊的红色物体
- 先验：可能是球（常见）
- 观测：圆形轮廓 + 红色
- 后验：很可能是红球（先验 × 似然）

这不是"看到什么就是什么"，
而是"结合先验知识和当前观测的最优推断"。
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class VariationalBelief:
    """
    变分信念：agent 对隐藏状态的内部表征

    q(s) = N(μ_s, σ_s²)

    不是直接存储观测，而是维护一个
    关于"世界状态"的概率信念。
    """

    def __init__(self, state_dim: int):
        self.state_dim = state_dim

        # 信念均值和方差
        self.mu = np.zeros(state_dim)
        self.log_var = np.zeros(state_dim)  # log(σ²)，初始 σ²=1

        # 先验（可随发展变化）
        self.prior_mu = np.zeros(state_dim)
        self.prior_log_var = np.zeros(state_dim)  # 平坦先验

        # 信念更新历史
        self.update_history = deque(maxlen=50)
        self.surprise_history = deque(maxlen=100)

    def update_belief(self, observation: np.ndarray,
                      predicted_mean: np.ndarray,
                      predicted_log_var: np.ndarray,
                      likelihood_precision: float = 1.0):
        """
        精度加权贝叶斯信念更新

        后验 ∝ 先验 × 似然
        在高斯假设下：
          μ_post = (π_prior * μ_prior + π_obs * obs) / (π_prior + π_obs)
          1/σ²_post = 1/σ²_prior + 1/σ²_obs

        这是精度加权平均——
        高精度的信号对后验影响更大。
        """
        # 精度 = 1/σ²
        prior_precision = np.exp(-self.prior_log_var)
        obs_precision = np.exp(-predicted_log_var) * likelihood_precision

        # 后验精度 = 先验精度 + 观测精度
        posterior_precision = prior_precision + obs_precision
        posterior_var = 1.0 / (posterior_precision + 1e-8)

        # 后验均值 = 精度加权平均
        self.mu = (prior_precision * self.prior_mu +
                   obs_precision * observation[:self.state_dim]) / (posterior_precision + 1e-8)

        self.log_var = np.log(posterior_var + 1e-8)

        # 记录更新幅度
        update_magnitude = np.mean(np.abs(obs_precision * (observation[:self.state_dim] - self.mu)))
        self.update_history.append(update_magnitude)

    def compute_surprise(self, observation: np.ndarray,
                         predicted_mean: np.ndarray) -> float:
        """
        计算惊奇度（信息论意义）

        惊奇度 = -ln p(o) ≈ 预测误差的精度加权和
        高惊奇度 = 当前信念无法解释观测
        """
        error = observation[:self.state_dim] - predicted_mean[:self.state_dim]
        precision = np.exp(-self.log_var)

        surprise = 0.5 * np.sum(precision * error**2 - self.log_var)
        self.surprise_history.append(surprise)

        return surprise

    def get_belief_confidence(self) -> float:
        """信念置信度 = 平均精度"""
        precision = np.exp(-self.log_var)
        return np.mean(precision)

    def get_belief_uncertainty(self) -> np.ndarray:
        """信念不确定性 = 方差"""
        return np.exp(self.log_var)

    def set_prior(self, mu: np.ndarray, log_var: np.ndarray):
        """设置先验信念"""
        self.prior_mu = mu[:self.state_dim]
        self.prior_log_var = log_var[:self.state_dim]

    def reset(self):
        """重置信念"""
        self.mu = np.zeros(self.state_dim)
        self.log_var = np.zeros(self.state_dim)


class HierarchicalBelief:
    """
    层级信念系统

    实现 FEP 的层级生成模型：
    - 低层：感知特征（颜色、形状）
    - 中层：物体概念（红球、蓝方块）
    - 高层：抽象规则（物体会掉落）

    每层的预测误差向上传播，
    每层的信念向下约束。
    """

    def __init__(self, layer_dims: list):
        """
        Args:
            layer_dims: 各层状态维度，如 [20, 10, 5]
        """
        self.layers = [VariationalBelief(dim) for dim in layer_dims]
        self.num_layers = len(layer_dims)

        # 层间连接权重（从低到高的预测）
        self.bottom_up_weights = []
        self.top_down_weights = []
        for i in range(self.num_layers - 1):
            w_up = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.01
            w_down = np.random.randn(layer_dims[i+1], layer_dims[i]) * 0.01
            self.bottom_up_weights.append(w_up)
            self.top_down_weights.append(w_down)

    def update_hierarchical(self, observation: np.ndarray):
        """
        层级信念更新

        1. 底层接收观测
        2. 预测误差向上传播
        3. 高层更新信念
        4. 高层信念向下约束底层
        """
        # 底层更新
        self.layers[0].update_belief(
            observation,
            self.layers[0].mu,
            self.layers[0].log_var
        )

        # 自底向上传播
        for i in range(self.num_layers - 1):
            # 低层信念 → 高层预测
            bottom_up_signal = self.layers[i].mu @ self.bottom_up_weights[i]

            # 高层用低层信号更新
            self.layers[i+1].update_belief(
                bottom_up_signal,
                self.layers[i+1].mu,
                self.layers[i+1].log_var,
                likelihood_precision=0.5  # 低层信号精度较低
            )

    def get_layer_surprise(self, layer: int) -> float:
        """获取某层的惊奇度"""
        if layer < len(self.layers):
            return np.mean(list(self.layers[layer].surprise_history)) if self.layers[layer].surprise_history else 0.0
        return 0.0
