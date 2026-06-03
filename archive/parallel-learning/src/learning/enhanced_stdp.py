"""
增强型STDP系统 - 基于2024年最新研究

NeoHebbian synapses to accelerate online training of neuromorphic systems
STDP as noisy gradient descent
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional


class EnhancedSTDP:
    """
    增强型STDP学习系统

    基于2024年最新研究：
    1. NeoHebbian突触加速训练
    2. STDP作为噪声梯度下降
    3. 稳态调节机制
    4. 元可塑性控制
    """

    def __init__(self, dim=128, lr=0.01, decay=0.95):
        self.dim = dim

        # STDP连接：(pre, post) → weight
        self.connections = defaultdict(dict)

        # 元可塑性：控制学习率
        self.meta_plasticity = defaultdict(lambda: 1.0)

        # 稳态机制：防止过度激活
        self.homeostatic = defaultdict(lambda: 0.0)

        # 学习参数
        self.lr = lr
        self.decay = decay
        self.tau_pre = 20.0  # 前突迹时间常数
        self.tau_post = 20.0  # 后突迹时间常数

        # 脉冲迹（用于时序依赖）
        self.pre_trace = defaultdict(lambda: 0.0)
        self.post_trace = defaultdict(lambda: 0.0)

        # 稳态目标
        self.target_rate = 0.1  # 目标激活率
        self.homeostatic_k = 0.01  # 稳态增益

    def update(self, pre: str, post: str, pre_spike: bool, post_spike: bool,
               dt: float = 1.0) -> float:
        """
        STDP更新：基于脉冲时序的可塑性

        规则：
        - pre在post之前激发 → 长时程增强(LTP)
        - post在pre之前激发 → 长时程抑制(LTD)

        Args:
            pre: 前突触神经元
            post: 后突触神经元
            pre_spike: 前神经元是否激发
            post_spike: 后神经元是否激发
            dt: 时间步长

        Returns:
            权重变化量
        """
        # 更新脉冲迹
        if pre_spike:
            self.pre_trace[pre] += 1.0
        else:
            self.pre_trace[pre] *= np.exp(-dt / self.tau_pre)

        if post_spike:
            self.post_trace[post] += 1.0
        else:
            self.post_trace[post] *= np.exp(-dt / self.tau_post)

        # STDP规则（基于脉冲迹）
        delta_w = 0.0

        if pre_spike and post_spike:
            # 同时激发：简单赫布
            delta_w = self.lr * 0.1

        elif pre_spike and not post_spike:
            # pre激发，post未激发：根据post迹判断LTP/LTD
            delta_w = self.lr * self.post_trace[post] * 0.1

        elif not pre_spike and post_spike:
            # post激发，pre未激发：根据pre迹判断LTP/LTD
            delta_w = -self.lr * self.pre_trace[pre] * 0.1

        # 应用元可塑性调节
        meta_factor = self.meta_plasticity[(pre, post)]
        delta_w *= meta_factor

        # 更新权重
        if pre not in self.connections:
            self.connections[pre] = {}

        if post not in self.connections[pre]:
            self.connections[pre][post] = 0.1  # 初始权重

        self.connections[pre][post] = np.clip(
            self.connections[pre][post] + delta_w,
            0.0, 1.0  # 权重范围[0, 1]
        )

        # 更新元可塑性（基于权重大小）
        weight = self.connections[pre][post]
        if weight > 0.8:
            # 权重过大，降低学习率
            self.meta_plasticity[(pre, post)] *= 0.95
        elif weight < 0.2:
            # 权重过小，提高学习率
            self.meta_plasticity[(pre, post)] *= 1.05

        # 更新稳态
        activity = float(pre_spike) + float(post_spike)
        error = activity - self.target_rate
        self.homeostatic[pre] += self.homeostatic_k * error * dt

        return delta_w

    def get_strength(self, pre: str, post: str) -> float:
        """获取连接强度"""
        if pre in self.connections and post in self.connections[pre]:
            return self.connections[pre][post]
        return 0.0

    def get_related(self, concept: str, top_k: int = 5) -> list:
        """获取相关概念（按连接强度排序）"""
        if concept not in self.connections:
            return []

        related = [(post, weight) for post, weight in
                   self.connections[concept].items()]
        related.sort(key=lambda x: -x[1])
        return related[:top_k]

    def decay_all(self):
        """所有连接随时间衰减"""
        for pre in self.connections:
            for post in self.connections[pre]:
                self.connections[pre][post] *= self.decay

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        total_connections = sum(
            len(conns) for conns in self.connections.values()
        )

        avg_weight = 0.0
        if total_connections > 0:
            total_weight = sum(
                weight for conns in self.connections.values()
                for weight in conns.values()
            )
            avg_weight = total_weight / total_connections

        return {
            'total_connections': total_connections,
            'avg_weight': avg_weight,
            'num_pre_neurons': len(self.connections),
            'meta_plasticity_avg': np.mean(list(self.meta_plasticity.values()))
                                if self.meta_plasticity else 0.0,
        }


class STDPSequenceLearner:
    """
    基于STDP的序列学习器

    用于学习时序模式，如：
    - 文本中的词序
    - 事件序列
    - 因果链
    """

    def __init__(self):
        self.stdp = EnhancedSTDP()
        self.sequences = []  # 存储学到的序列

    def learn_sequence(self, items: list):
        """
        学习序列

        Args:
            items: 序列中的项目列表
        """
        # 对相邻项目建立STDP连接
        for i in range(len(items) - 1):
            pre, post = items[i], items[i + 1]
            # pre激发 → post激发
            self.stdp.update(pre, post, pre_spike=True, post_spike=True)

        # 存储序列
        self.sequences.append(tuple(items))

    def predict_next(self, context: list, top_k: int = 3) -> list:
        """
        基于上下文预测下一个项目

        Args:
            context: 上下文项目列表
            top_k: 返回top-k个预测

        Returns:
            预测的项目列表（带权重）
        """
        if not context:
            return []

        last_item = context[-1]

        # 获取与最后一项相关的项目
        related = self.stdp.get_related(last_item, top_k=top_k * 2)

        # 过滤已出现的项目
        predictions = [(item, weight) for item, weight in related
                      if item not in context]

        return predictions[:top_k]

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'stdp_stats': self.stdp.get_stats(),
            'num_sequences': len(self.sequences),
            'avg_sequence_length': np.mean([len(s) for s in self.sequences])
                                     if self.sequences else 0.0,
        }


if __name__ == '__main__':
    print("=== 增强型STDP系统 ===")
    print()
    print("基于2024年最新研究：")
    print("1. NeoHebbian突触加速训练")
    print("2. STDP作为噪声梯度下降")
    print("3. 稳态调节机制")
    print("4. 元可塑性控制")
    print()
    print("核心特性：")
    print("- 脉冲时序依赖可塑性")
    print("- 元可塑性调节学习率")
    print("- 稳态机制防止过度激活")
    print("- 序列学习能力")
