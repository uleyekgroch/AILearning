"""
生物学习机制 - 从第一性原理出发

人类学习的核心机制：
1. STDP（脉冲时序依赖可塑性）
2. Hebbian学习（一起激活的神经元会连接）
3. 预测编码（预测误差驱动学习）
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Synapse:
    """突触"""
    pre_neuron_id: str
    post_neuron_id: str
    weight: float
    last_pre_spike: float = 0.0
    last_post_spike: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Neuron:
    """神经元"""
    neuron_id: str
    activation: float = 0.0
    threshold: float = 0.5
    last_spike_time: float = 0.0
    spike_count: int = 0


class STDPLearning:
    """STDP学习（脉冲时序依赖可塑性）

    生物学基础：
    - 如果突触前神经元在突触后神经元之前激活，突触增强
    - 如果突触前神经元在突触后神经元之后激活，突触减弱
    - 这是学习和记忆的神经基础
    """

    def __init__(self, a_plus: float = 0.01, a_minus: float = 0.01,
                 tau_plus: float = 20.0, tau_minus: float = 20.0):
        """
        初始化STDP学习

        Args:
            a_plus: 增强幅度
            a_minus: 减弱幅度
            tau_plus: 增强时间常数
            tau_minus: 减弱时间常数
        """
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        # 突触列表
        self.synapses: Dict[Tuple[str, str], Synapse] = {}

    def add_synapse(self, pre_id: str, post_id: str, initial_weight: float = 0.5) -> Synapse:
        """
        添加突触

        Args:
            pre_id: 突触前神经元ID
            post_id: 突触后神经元ID
            initial_weight: 初始权重

        Returns:
            突触对象
        """
        synapse = Synapse(
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
            weight=initial_weight
        )
        self.synapses[(pre_id, post_id)] = synapse
        return synapse

    def update_weights(self, pre_id: str, post_id: str,
                      pre_spike_time: float, post_spike_time: float) -> float:
        """
        更新突触权重

        Args:
            pre_id: 突触前神经元ID
            post_id: 突触后神经元ID
            pre_spike_time: 突触前脉冲时间
            post_spike_time: 突触后脉冲时间

        Returns:
            权重变化量
        """
        key = (pre_id, post_id)
        if key not in self.synapses:
            return 0.0

        synapse = self.synapses[key]

        # 计算时间差
        dt = post_spike_time - pre_spike_time

        # STDP规则
        if dt > 0:
            # 突触前先激活 -> 增强（因果关系）
            dw = self.a_plus * np.exp(-dt / self.tau_plus)
        else:
            # 突触后先激活 -> 减弱（非因果关系）
            dw = -self.a_minus * np.exp(dt / self.tau_minus)

        # 更新权重
        synapse.weight += dw
        synapse.weight = np.clip(synapse.weight, 0.0, 1.0)

        # 更新脉冲时间
        synapse.last_pre_spike = pre_spike_time
        synapse.last_post_spike = post_spike_time

        return dw

    def get_weight(self, pre_id: str, post_id: str) -> float:
        """获取突触权重"""
        key = (pre_id, post_id)
        if key in self.synapses:
            return self.synapses[key].weight
        return 0.0


class HebbianLearning:
    """Hebbian学习

    生物学基础：
    - "一起激活的神经元会连接"（Neurons that fire together wire together）
    - 这是联想学习的基础
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        初始化Hebbian学习

        Args:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate

        # 共激活矩阵
        self.co_activation: Dict[Tuple[str, str], float] = {}

    def update(self, active_neurons: List[str]) -> Dict[Tuple[str, str], float]:
        """
        更新共激活

        Args:
            active_neurons: 活跃神经元列表

        Returns:
            权重变化
        """
        changes = {}

        # 计算所有活跃神经元对的共激活
        for i, neuron_i in enumerate(active_neurons):
            for j, neuron_j in enumerate(active_neurons):
                if i < j:  # 避免重复
                    key = (neuron_i, neuron_j)
                    if key not in self.co_activation:
                        self.co_activation[key] = 0.0

                    # 增强共激活
                    self.co_activation[key] += self.learning_rate
                    changes[key] = self.co_activation[key]

        return changes

    def get_association(self, neuron_i: str, neuron_j: str) -> float:
        """
        获取两个神经元的关联强度

        Args:
            neuron_i: 神经元I
            neuron_j: 神经元J

        Returns:
            关联强度
        """
        key = (neuron_i, neuron_j)
        return self.co_activation.get(key, 0.0)


class PredictiveCoding:
    """预测编码

    生物学基础：
    - 大脑不断预测下一刻会发生什么
    - 预测误差驱动学习
    - 这是学习的根本动力
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        """
        初始化预测编码

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 预测权重
        self.predict_weights = np.random.randn(input_dim, hidden_dim) * 0.01
        self.feedback_weights = np.random.randn(hidden_dim, input_dim) * 0.01

        # 学习率
        self.learning_rate = 0.01

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            input_data: 输入数据

        Returns:
            预测值
        """
        # 前向传播
        hidden = np.dot(input_data, self.predict_weights)
        prediction = np.dot(hidden, self.feedback_weights)
        return prediction

    def compute_error(self, prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """
        计算预测误差

        Args:
            prediction: 预测值
            actual: 实际值

        Returns:
            预测误差
        """
        return actual - prediction

    def learn(self, input_data: np.ndarray, actual: np.ndarray) -> float:
        """
        学习

        Args:
            input_data: 输入数据
            actual: 实际值

        Returns:
            预测误差大小
        """
        # 预测
        prediction = self.predict(input_data)

        # 计算误差
        error = self.compute_error(prediction, actual)
        error_magnitude = np.linalg.norm(error)

        # 更新权重
        hidden = np.dot(input_data, self.predict_weights)
        self.feedback_weights += self.learning_rate * np.outer(hidden, error)
        self.predict_weights += self.learning_rate * np.outer(input_data, np.dot(error, self.feedback_weights.T))

        return error_magnitude


class BiologicalLearning:
    """生物学习系统

    整合三种生物学习机制：
    1. STDP：脉冲时序依赖可塑性
    2. Hebbian：共激活学习
    3. 预测编码：预测误差驱动学习
    """

    def __init__(self, input_dim: int = 64):
        """
        初始化生物学习系统

        Args:
            input_dim: 输入维度
        """
        self.input_dim = input_dim

        # STDP学习
        self.stdp = STDPLearning()

        # Hebbian学习
        self.hebbian = HebbianLearning()

        # 预测编码
        self.predictive_coding = PredictiveCoding(input_dim)

        # 统计
        self.stats = {
            'stdp_updates': 0,
            'hebbian_updates': 0,
            'predictive_errors': [],
        }

    def learn_stdp(self, pre_id: str, post_id: str,
                  pre_spike_time: float, post_spike_time: float) -> float:
        """
        STDP学习

        Args:
            pre_id: 突触前神经元ID
            post_id: 突触后神经元ID
            pre_spike_time: 突触前脉冲时间
            post_spike_time: 突触后脉冲时间

        Returns:
            权重变化
        """
        dw = self.stdp.update_weights(pre_id, post_id, pre_spike_time, post_spike_time)
        self.stats['stdp_updates'] += 1
        return dw

    def learn_hebbian(self, active_neurons: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Hebbian学习

        Args:
            active_neurons: 活跃神经元列表

        Returns:
            权重变化
        """
        changes = self.hebbian.update(active_neurons)
        self.stats['hebbian_updates'] += 1
        return changes

    def learn_predictive(self, input_data: np.ndarray, actual: np.ndarray) -> float:
        """
        预测编码学习

        Args:
            input_data: 输入数据
            actual: 实际值

        Returns:
            预测误差
        """
        error = self.predictive_coding.learn(input_data, actual)
        self.stats['predictive_errors'].append(error)
        return error

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_error = np.mean(self.stats['predictive_errors']) if self.stats['predictive_errors'] else 0.0

        return {
            **self.stats,
            'avg_predictive_error': avg_error,
            'stdp_synapse_count': len(self.stdp.synapses),
            'hebbian_association_count': len(self.hebbian.co_activation),
        }
