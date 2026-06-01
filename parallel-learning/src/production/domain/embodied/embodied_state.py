"""
EmbodiedState聚合根

具身状态聚合根，管理具身系统的状态
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class EmbodiedState:
    """具身状态聚合根

    具身状态是具身感知系统的核心聚合根，
    负责管理具身系统的整体状态。

    Attributes:
        state_id: 状态唯一标识
        position: 位置坐标 (x, y, z)
        orientation: 姿态角度 (roll, pitch, yaw)
        energy_level: 能量水平 [0, 1]
        emotional_state: 情感状态
        created_at: 创建时间
        updated_at: 更新时间
    """

    state_id: str
    position: Dict[str, float]
    orientation: Dict[str, float]
    energy_level: float
    emotional_state: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证具身状态参数"""
        # 验证状态ID
        if not self.state_id:
            raise ValueError("State ID cannot be empty")

        # 验证能量水平
        if not (0.0 <= self.energy_level <= 1.0):
            raise ValueError(
                f"Energy level must be between 0 and 1, got {self.energy_level}"
            )

        # 验证位置坐标
        required_position_keys = ['x', 'y', 'z']
        for key in required_position_keys:
            if key not in self.position:
                raise ValueError(f"Position must contain key '{key}'")

        # 验证姿态角度
        required_orientation_keys = ['roll', 'pitch', 'yaw']
        for key in required_orientation_keys:
            if key not in self.orientation:
                raise ValueError(f"Orientation must contain key '{key}'")

    def update_position(self, new_position: Dict[str, float]) -> None:
        """更新位置

        Args:
            new_position: 新位置坐标
        """
        self.position.update(new_position)
        self.updated_at = datetime.now()

    def update_orientation(self, new_orientation: Dict[str, float]) -> None:
        """更新姿态

        Args:
            new_orientation: 新姿态角度
        """
        self.orientation.update(new_orientation)
        self.updated_at = datetime.now()

    def update_energy_level(self, new_energy_level: float) -> None:
        """更新能量水平

        Args:
            new_energy_level: 新能量水平

        Raises:
            ValueError: 如果能量水平无效
        """
        if not (0.0 <= new_energy_level <= 1.0):
            raise ValueError(
                f"Energy level must be between 0 and 1, got {new_energy_level}"
            )
        self.energy_level = new_energy_level
        self.updated_at = datetime.now()

    def update_emotional_state(self, new_emotional_state: str) -> None:
        """更新情感状态

        Args:
            new_emotional_state: 新情感状态
        """
        self.emotional_state = new_emotional_state
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state_id': self.state_id,
            'position': self.position,
            'orientation': self.orientation,
            'energy_level': self.energy_level,
            'emotional_state': self.emotional_state,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }

    @property
    def is_low_energy(self) -> bool:
        """检查是否低能量"""
        return self.energy_level < 0.2

    @property
    def is_high_energy(self) -> bool:
        """检查是否高能量"""
        return self.energy_level > 0.8

    @property
    def x(self) -> float:
        """获取X坐标"""
        return self.position.get('x', 0.0)

    @property
    def y(self) -> float:
        """获取Y坐标"""
        return self.position.get('y', 0.0)

    @property
    def z(self) -> float:
        """获取Z坐标"""
        return self.position.get('z', 0.0)
