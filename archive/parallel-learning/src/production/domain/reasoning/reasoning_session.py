"""
ReasoningSession聚合根

推理会话聚合根，管理推理任务的生命周期
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from .reasoning_task import ReasoningTask
from .reasoning_result import ReasoningResult


# ============================================================================
# 领域事件
# ============================================================================

@dataclass(frozen=True)
class DomainEvent:
    """领域事件基类"""
    event_id: str
    occurred_on: datetime
    aggregate_id: str
    aggregate_version: int

    @property
    @abstractmethod
    def event_type(self) -> str:
        pass


@dataclass(frozen=True)
class TaskAddedEvent(DomainEvent):
    """任务添加事件"""
    task_id: str
    query: str
    reasoning_type: str

    @property
    def event_type(self) -> str:
        return "TaskAdded"


@dataclass(frozen=True)
class TaskCompletedEvent(DomainEvent):
    """任务完成事件"""
    task_id: str
    answer: str
    confidence: float

    @property
    def event_type(self) -> str:
        return "TaskCompleted"


# ============================================================================
# 推理链值对象
# ============================================================================

@dataclass(frozen=True)
class ReasoningChain:
    """推理链值对象"""
    task_id: str
    steps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: str) -> 'ReasoningChain':
        """添加步骤（返回新的不可变对象）"""
        return ReasoningChain(
            task_id=self.task_id,
            steps=self.steps + [step],
            created_at=self.created_at
        )


# ============================================================================
# ReasoningSession聚合根
# ============================================================================

@dataclass
class ReasoningSession:
    """推理会话聚合根

    推理会话是推理引擎限界上下文的核心聚合根，
    负责管理推理任务的生命周期。

    职责：
    - 管理推理任务的添加、执行、完成
    - 维护推理会话的版本和一致性
    - 发布领域事件

    Attributes:
        session_id: 会话唯一标识
        status: 会话状态
        version: 版本号
        created_at: 创建时间
        updated_at: 更新时间
        tasks: 任务列表
        results: 结果映射
        chains: 推理链映射
        _events: 领域事件列表
    """

    session_id: str
    status: str = "created"
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tasks: Dict[str, ReasoningTask] = field(default_factory=dict)
    results: Dict[str, ReasoningResult] = field(default_factory=dict)
    chains: Dict[str, ReasoningChain] = field(default_factory=dict)
    _events: List[DomainEvent] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """验证会话参数"""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")

    # ========================================================================
    # 任务管理
    # ========================================================================

    def add_task(self, task: ReasoningTask) -> None:
        """添加推理任务

        Args:
            task: 推理任务

        Raises:
            ValueError: 如果任务ID已存在
        """
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists")

        # 添加任务
        self.tasks[task.task_id] = task

        # 更新版本
        self.version += 1
        self.updated_at = datetime.now()

        # 发布领域事件
        event = TaskAddedEvent(
            event_id=f"event_{self.version}",
            occurred_on=datetime.now(),
            aggregate_id=self.session_id,
            aggregate_version=self.version,
            task_id=task.task_id,
            query=task.query,
            reasoning_type=task.reasoning_type
        )
        self._events.append(event)

    def has_task(self, task_id: str) -> bool:
        """检查任务是否存在"""
        return task_id in self.tasks

    @property
    def task_count(self) -> int:
        """获取任务数量"""
        return len(self.tasks)

    # ========================================================================
    # 任务执行
    # ========================================================================

    def execute_task(self, task_id: str) -> ReasoningResult:
        """执行推理任务

        Args:
            task_id: 任务ID

        Returns:
            推理结果

        Raises:
            ValueError: 如果任务不存在
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        # 开始任务
        task.start()

        # 模拟推理过程（实际实现会调用推理引擎）
        reasoning_chain = self._simulate_reasoning(task)

        # 创建结果
        result = ReasoningResult(
            task_id=task_id,
            answer=f"Answer for: {task.query}",
            confidence=0.95,
            reasoning_chain=reasoning_chain.steps,
            metadata={
                'reasoning_type': task.reasoning_type,
                'session_id': self.session_id,
            }
        )

        # 保存结果和推理链
        self.results[task_id] = result
        self.chains[task_id] = reasoning_chain

        # 完成任务
        task.complete()

        # 更新版本
        self.version += 1
        self.updated_at = datetime.now()

        # 发布领域事件
        event = TaskCompletedEvent(
            event_id=f"event_{self.version}",
            occurred_on=datetime.now(),
            aggregate_id=self.session_id,
            aggregate_version=self.version,
            task_id=task_id,
            answer=result.answer,
            confidence=result.confidence
        )
        self._events.append(event)

        return result

    def get_reasoning_chain(self, task_id: str) -> Optional[ReasoningChain]:
        """获取推理链"""
        return self.chains.get(task_id)

    def _simulate_reasoning(self, task: ReasoningTask) -> ReasoningChain:
        """模拟推理过程（占位实现）"""
        steps = [
            f"Received query: {task.query}",
            f"Selected reasoning type: {task.reasoning_type}",
            "Searching knowledge base...",
            "Found relevant facts",
            "Generated reasoning chain",
            "Returning result"
        ]

        chain = ReasoningChain(task_id=task.task_id)
        for step in steps:
            chain = chain.add_step(step)

        return chain

    # ========================================================================
    # 事件溯源
    # ========================================================================

    def get_events(self) -> List[DomainEvent]:
        """获取领域事件"""
        return self._events.copy()

    def clear_events(self):
        """清空已发布的事件"""
        self._events.clear()

    # ========================================================================
    # 统计
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """获取会话统计"""
        completed_tasks = sum(
            1 for t in self.tasks.values() if t.is_completed
        )
        failed_tasks = sum(
            1 for t in self.tasks.values() if t.is_failed
        )
        pending_tasks = sum(
            1 for t in self.tasks.values() if t.is_pending
        )

        avg_confidence = 0.0
        if self.results:
            avg_confidence = sum(
                r.confidence for r in self.results.values()
            ) / len(self.results)

        return {
            'session_id': self.session_id,
            'total_tasks': self.task_count,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'pending_tasks': pending_tasks,
            'avg_confidence': avg_confidence,
            'version': self.version,
        }
