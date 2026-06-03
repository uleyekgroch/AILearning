"""
LearningStrategy值对象

学习策略值对象，定义学习方法
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from enum import Enum


class StrategyType(Enum):
    """策略类型枚举"""
    INCREMENTAL = "incremental"
    BATCH = "batch"
    ONLINE = "online"
    TRANSFER = "transfer"


@dataclass(frozen=True)
class LearningStrategy:
    """学习策略值对象

    学习策略是不可变的值对象，定义学习方法。

    Attributes:
        strategy_id: 策略唯一标识
        strategy_type: 策略类型
        parameters: 策略参数
        priority: 优先级（1-10）
        created_at: 创建时间
    """

    strategy_id: str
    strategy_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证学习策略参数"""
        # 验证策略类型
        valid_types = [st.value for st in StrategyType]
        if self.strategy_type not in valid_types:
            raise ValueError(
                f"Invalid strategy type: {self.strategy_type}. "
                f"Must be one of: {valid_types}"
            )

        # 验证策略ID
        if not self.strategy_id:
            raise ValueError("Strategy ID cannot be empty")

        # 验证优先级
        if not (1 <= self.priority <= 10):
            raise ValueError(
                f"Priority must be between 1 and 10, got {self.priority}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type,
            'parameters': self.parameters,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
        }

    @property
    def is_incremental(self) -> bool:
        """检查是否为增量学习策略"""
        return self.strategy_type == StrategyType.INCREMENTAL.value

    @property
    def is_batch(self) -> bool:
        """检查是否为批量学习策略"""
        return self.strategy_type == StrategyType.BATCH.value

    @property
    def is_online(self) -> bool:
        """检查是否为在线学习策略"""
        return self.strategy_type == StrategyType.ONLINE.value

    @property
    def is_transfer(self) -> bool:
        """检查是否为迁移学习策略"""
        return self.strategy_type == StrategyType.TRANSFER.value

    @property
    def learning_rate(self) -> float:
        """获取学习率"""
        return self.parameters.get("learning_rate", 0.01)

    @property
    def batch_size(self) -> int:
        """获取批量大小"""
        return self.parameters.get("batch_size", 32)
