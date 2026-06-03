"""目标数据模型 — 学习目标的状态和数据结构"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class GoalStatus(Enum):
    """目标状态"""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    BLOCKED = 'blocked'
    ABANDONED = 'abandoned'


@dataclass
class Goal:
    """学习目标

    一个可分解、可追踪的学习目标。
    """
    id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    sub_goals: List[str] = field(default_factory=list)
    parent_goal: Optional[str] = None
    required_knowledge: List[str] = field(default_factory=list)
    priority: float = 0.5
    progress: float = 0.0
    created_step: int = 0

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'id': self.id,
            'description': self.description,
            'status': self.status.value,
            'sub_goals': self.sub_goals,
            'parent_goal': self.parent_goal,
            'required_knowledge': self.required_knowledge,
            'priority': self.priority,
            'progress': self.progress,
            'created_step': self.created_step,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Goal':
        """从字典反序列化"""
        status_val = data.get('status', 'pending')
        if isinstance(status_val, str):
            status_val = GoalStatus(status_val)
        return cls(
            id=data['id'],
            description=data['description'],
            status=status_val,
            sub_goals=data.get('sub_goals', []),
            parent_goal=data.get('parent_goal'),
            required_knowledge=data.get('required_knowledge', []),
            priority=data.get('priority', 0.5),
            progress=data.get('progress', 0.0),
            created_step=data.get('created_step', 0),
        )
