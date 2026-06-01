"""
DeepReasoningModel实体

深度推理模型实体，实现多层感知机推理
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List
import numpy as np


@dataclass
class DeepReasoningModel:
    """深度推理模型实体

    深度推理模型实现了多层感知机，用于增强推理能力。

    Attributes:
        model_id: 模型唯一标识
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_layers: 层数
        weights: 权重矩阵
        biases: 偏置向量
        created_at: 创建时间
    """

    model_id: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    weights: List[np.ndarray] = field(default_factory=list)
    biases: List[np.ndarray] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证深度推理模型参数"""
        # 验证模型ID
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")

        # 验证维度
        if self.input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {self.input_dim}")

        if self.hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {self.hidden_dim}")

        if self.output_dim <= 0:
            raise ValueError(f"Output dimension must be positive, got {self.output_dim}")

        if self.num_layers <= 0:
            raise ValueError(f"Number of layers must be positive, got {self.num_layers}")

        # 初始化权重
        if not self.weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化权重矩阵"""
        self.weights = []
        self.biases = []

        # 输入层到隐藏层
        self.weights.append(np.random.randn(self.input_dim, self.hidden_dim) * 0.01)
        self.biases.append(np.zeros(self.hidden_dim))

        # 隐藏层到隐藏层
        for _ in range(self.num_layers - 2):
            self.weights.append(np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01)
            self.biases.append(np.zeros(self.hidden_dim))

        # 隐藏层到输出层
        self.weights.append(np.random.randn(self.hidden_dim, self.output_dim) * 0.01)
        self.biases.append(np.zeros(self.output_dim))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """前向传播

        Args:
            input_data: 输入数据

        Returns:
            模型输出
        """
        x = input_data

        # 通过每一层
        for i in range(len(self.weights)):
            # 线性变换
            x = np.dot(x, self.weights[i]) + self.biases[i]

            # 激活函数（ReLU，除了最后一层）
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)

        return x

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_id': self.model_id,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'total_params': sum(w.size for w in self.weights) + sum(b.size for b in self.biases),
            'created_at': self.created_at.isoformat(),
        }

    @property
    def total_params(self) -> int:
        """获取总参数数量"""
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
