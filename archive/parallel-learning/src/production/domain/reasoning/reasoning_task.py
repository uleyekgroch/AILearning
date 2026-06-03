"""
ReasoningTask实体

推理任务实体，属于ReasoningSession聚合
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReasoningType(Enum):
    """推理类型枚举"""
    COMMONSENSE = "commonsense"
    CAUSAL = "causal"
    HYBRID = "hybrid"


@dataclass
class ReasoningTask:
    """推理任务实体

    推理任务是ReasoningSession聚合的内部实体，
    表示一个具体的推理任务。

    Attributes:
        task_id: 任务唯一标识
        query: 推理查询文本
        reasoning_type: 推理类型（commonsense/causal/hybrid）
        status: 任务状态
        created_at: 创建时间
        started_at: 开始时间
        completed_at: 完成时间
    """

    task_id: str
    query: str
    reasoning_type: str
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        """验证任务参数"""
        # 验证推理类型
        valid_types = [rt.value for rt in ReasoningType]
        if self.reasoning_type not in valid_types:
            raise ValueError(
                f"Invalid reasoning type: {self.reasoning_type}. "
                f"Must be one of: {valid_types}"
            )

        # 验证任务ID
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")

        # 验证查询
        if not self.query:
            raise ValueError("Query cannot be empty")

    def start(self):
        """开始执行任务"""
        if self.status != TaskStatus.PENDING.value:
            raise ValueError(
                f"Cannot start task in {self.status} status. "
                f"Task must be in 'pending' status."
            )

        self.status = TaskStatus.RUNNING.value
        self.started_at = datetime.now()

    def complete(self):
        """完成任务"""
        if self.status != TaskStatus.RUNNING.value:
            raise ValueError(
                f"Cannot complete task in {self.status} status. "
                f"Task must be in 'running' status."
            )

        self.status = TaskStatus.COMPLETED.value
        self.completed_at = datetime.now()

    def fail(self, error_message: str = ""):
        """任务失败"""
        if self.status != TaskStatus.RUNNING.value:
            raise ValueError(
                f"Cannot fail task in {self.status} status. "
                f"Task must be in 'running' status."
            )

        self.status = TaskStatus.FAILED.value
        self.completed_at = datetime.now()
        self.error_message = error_message

    @property
    def is_pending(self) -> bool:
        """检查任务是否待执行"""
        return self.status == TaskStatus.PENDING.value

    @property
    def is_running(self) -> bool:
        """检查任务是否运行中"""
        return self.status == TaskStatus.RUNNING.value

    @property
    def is_completed(self) -> bool:
        """检查任务是否已完成"""
        return self.status == TaskStatus.COMPLETED.value

    @property
    def is_failed(self) -> bool:
        """检查任务是否失败"""
        return self.status == TaskStatus.FAILED.value

    def __eq__(self, other):
        """任务相等性比较（基于ID）"""
        if not isinstance(other, ReasoningTask):
            return False
        return self.task_id == other.task_id

    def __hash__(self):
        """任务哈希值（基于ID）"""
        return hash(self.task_id)
