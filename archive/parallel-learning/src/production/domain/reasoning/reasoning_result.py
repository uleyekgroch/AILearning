"""
ReasoningResult值对象

推理结果值对象，不可变
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


@dataclass(frozen=True)
class ReasoningResult:
    """推理结果值对象

    推理结果是不可变的值对象，表示推理任务的执行结果。

    Attributes:
        task_id: 关联的任务ID
        answer: 推理答案
        confidence: 置信度 [0, 1]
        status: 结果状态
        reasoning_chain: 推理链步骤
        metadata: 额外元数据
        created_at: 创建时间
    """

    task_id: str
    answer: str
    confidence: float
    status: str = "completed"
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证结果参数"""
        # 验证置信度
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        # 验证任务ID
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")

        # 验证答案
        if not self.answer:
            raise ValueError("Answer cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'answer': self.answer,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }

    @property
    def is_high_confidence(self) -> bool:
        """检查是否高置信度（>0.8）"""
        return self.confidence > 0.8

    @property
    def step_count(self) -> int:
        """获取推理步骤数量"""
        return len(self.reasoning_chain)
