"""
MotorCommand实体

运动命令实体，表示发送给执行器的命令
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from enum import Enum


class MotorType(Enum):
    """运动类型枚举"""
    MOVEMENT = "movement"
    GRASP = "grasp"
    MANIPULATION = "manipulation"
    GAZE = "gaze"
    SPEECH = "speech"


@dataclass
class MotorCommand:
    """运动命令实体

    运动命令是具身感知系统的核心实体，
    表示发送给执行器的控制命令。

    Attributes:
        command_id: 命令唯一标识
        motor_type: 运动类型
        parameters: 命令参数
        priority: 优先级（1-10）
        timestamp: 时间戳
        metadata: 额外元数据
    """

    command_id: str
    motor_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证运动命令参数"""
        # 验证运动类型
        valid_types = [mt.value for mt in MotorType]
        if self.motor_type not in valid_types:
            raise ValueError(
                f"Invalid motor type: {self.motor_type}. "
                f"Must be one of: {valid_types}"
            )

        # 验证命令ID
        if not self.command_id:
            raise ValueError("Command ID cannot be empty")

        # 验证优先级
        if not (1 <= self.priority <= 10):
            raise ValueError(
                f"Priority must be between 1 and 10, got {self.priority}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'command_id': self.command_id,
            'motor_type': self.motor_type,
            'parameters': self.parameters,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }

    @property
    def is_movement(self) -> bool:
        """检查是否为运动命令"""
        return self.motor_type == MotorType.MOVEMENT.value

    @property
    def is_grasp(self) -> bool:
        """检查是否为抓取命令"""
        return self.motor_type == MotorType.GRASP.value

    @property
    def is_high_priority(self) -> bool:
        """检查是否为高优先级命令"""
        return self.priority >= 8

    def __eq__(self, other):
        """运动命令相等性比较（基于ID）"""
        if not isinstance(other, MotorCommand):
            return False
        return self.command_id == other.command_id

    def __hash__(self):
        """运动命令哈希值（基于ID）"""
        return hash(self.command_id)
