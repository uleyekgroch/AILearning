"""
AttentionMechanism实体

注意力机制实体，实现多头注意力
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List
import numpy as np


@dataclass
class AttentionMechanism:
    """注意力机制实体

    注意力机制实现了多头注意力，用于增强推理能力。

    Attributes:
        mechanism_id: 机制唯一标识
        input_dim: 输入维度
        output_dim: 输出维度
        num_heads: 注意力头数
        weights: 权重矩阵
        created_at: 创建时间
    """

    mechanism_id: str
    input_dim: int
    output_dim: int
    num_heads: int
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证注意力机制参数"""
        # 验证机制ID
        if not self.mechanism_id:
            raise ValueError("Mechanism ID cannot be empty")

        # 验证维度
        if self.input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {self.input_dim}")

        if self.output_dim <= 0:
            raise ValueError(f"Output dimension must be positive, got {self.output_dim}")

        if self.num_heads <= 0:
            raise ValueError(f"Number of heads must be positive, got {self.num_heads}")

        # 初始化权重
        if not self.weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化权重矩阵"""
        # Query权重
        self.weights['W_q'] = np.random.randn(self.input_dim, self.output_dim) * 0.01
        # Key权重
        self.weights['W_k'] = np.random.randn(self.input_dim, self.output_dim) * 0.01
        # Value权重
        self.weights['W_v'] = np.random.randn(self.input_dim, self.output_dim) * 0.01

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """前向传播

        Args:
            query: 查询向量
            key: 键向量
            value: 值向量

        Returns:
            注意力输出
        """
        # 线性变换
        Q = np.dot(query, self.weights['W_q'])
        K = np.dot(key, self.weights['W_k'])
        V = np.dot(value, self.weights['W_v'])

        # 计算注意力分数
        d_k = self.output_dim
        scores = np.dot(Q, K.T) / np.sqrt(d_k)

        # Softmax
        attention_weights = self._softmax(scores)

        # 加权求和
        output = np.dot(attention_weights, V)

        return output

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数

        Args:
            x: 输入数组

        Returns:
            Softmax输出
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mechanism_id': self.mechanism_id,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_heads': self.num_heads,
            'created_at': self.created_at.isoformat(),
        }
