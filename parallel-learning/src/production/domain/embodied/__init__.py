"""
具身感知限界上下文

实体：SensoryInput, MotorCommand
聚合根：EmbodiedState
值对象：PerceptionResult
"""

from .sensory_input import SensoryInput, SensoryType
from .motor_command import MotorCommand, MotorType
from .embodied_state import EmbodiedState
from .perception_result import PerceptionResult

__all__ = [
    "SensoryInput",
    "SensoryType",
    "MotorCommand",
    "MotorType",
    "EmbodiedState",
    "PerceptionResult",
]
