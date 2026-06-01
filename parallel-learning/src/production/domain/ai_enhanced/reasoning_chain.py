"""
ReasoningChain值对象

推理链值对象，表示推理步骤序列
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


@dataclass
class ReasoningChain:
    """推理链值对象

    推理链表示推理步骤序列。

    Attributes:
        chain_id: 链唯一标识
        steps: 推理步骤列表
        confidence: 置信度 [0, 1]
        metadata: 额外元数据
        created_at: 创建时间
    """

    chain_id: str
    steps: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证推理链参数"""
        # 验证链ID
        if not self.chain_id:
            raise ValueError("Chain ID cannot be empty")

        # 验证置信度
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

    def add_step(self, step: Dict[str, Any]) -> None:
        """添加推理步骤

        Args:
            step: 推理步骤
        """
        # 添加步骤编号
        step['step'] = len(self.steps) + 1
        self.steps.append(step)

    def get_step(self, step_number: int) -> Dict[str, Any]:
        """获取指定步骤

        Args:
            step_number: 步骤编号

        Returns:
            推理步骤

        Raises:
            ValueError: 如果步骤不存在
        """
        if step_number < 1 or step_number > len(self.steps):
            raise ValueError(f"Step {step_number} not found")

        return self.steps[step_number - 1]

    @property
    def step_count(self) -> int:
        """获取步骤数量"""
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'chain_id': self.chain_id,
            'steps': self.steps,
            'step_count': self.step_count,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }

    def get_summary(self) -> str:
        """获取推理链摘要"""
        if not self.steps:
            return "Empty reasoning chain"

        summary_parts = []
        for step in self.steps:
            operation = step.get('operation', 'unknown')
            input_val = step.get('input', '')
            output_val = step.get('output', '')
            summary_parts.append(f"{operation}({input_val}) -> {output_val}")

        return " -> ".join(summary_parts)
