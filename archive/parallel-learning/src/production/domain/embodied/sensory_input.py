"""
SensoryInput实体

感觉输入实体，表示来自传感器的输入
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from enum import Enum


class SensoryType(Enum):
    """感觉类型枚举"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"
    VESTIBULAR = "vestibular"


@dataclass
class SensoryInput:
    """感觉输入实体

    感觉输入是具身感知系统的基础实体，
    表示来自传感器的原始输入数据。

    Attributes:
        input_id: 输入唯一标识
        sensory_type: 感觉类型
        data: 传感器数据
        timestamp: 时间戳
        metadata: 额外元数据
    """

    input_id: str
    sensory_type: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证感觉输入参数"""
        # 验证感觉类型
        valid_types = [st.value for st in SensoryType]
        if self.sensory_type not in valid_types:
            raise ValueError(
                f"Invalid sensory type: {self.sensory_type}. "
                f"Must be one of: {valid_types}"
            )

        # 验证输入ID
        if not self.input_id:
            raise ValueError("Input ID cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'input_id': self.input_id,
            'sensory_type': self.sensory_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }

    @property
    def is_visual(self) -> bool:
        """检查是否为视觉输入"""
        return self.sensory_type == SensoryType.VISUAL.value

    @property
    def is_auditory(self) -> bool:
        """检查是否为听觉输入"""
        return self.sensory_type == SensoryType.AUDITORY.value

    @property
    def is_tactile(self) -> bool:
        """检查是否为触觉输入"""
        return self.sensory_type == SensoryType.TACTILE.value

    def __eq__(self, other):
        """感觉输入相等性比较（基于ID）"""
        if not isinstance(other, SensoryInput):
            return False
        return self.input_id == other.input_id

    def __hash__(self):
        """感觉输入哈希值（基于ID）"""
        return hash(self.input_id)
