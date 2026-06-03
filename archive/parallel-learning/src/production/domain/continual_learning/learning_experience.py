"""
LearningExperience实体

学习经验实体，表示一次学习经历
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass
class LearningExperience:
    """学习经验实体

    学习经验是持续学习系统的基础实体，
    表示一次完整的学习经历。

    Attributes:
        experience_id: 经验唯一标识
        input_data: 输入数据
        output_data: 输出数据
        feedback: 反馈数据
        timestamp: 时间戳
        metadata: 额外元数据
    """

    experience_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    feedback: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证学习经验参数"""
        # 验证经验ID
        if not self.experience_id:
            raise ValueError("Experience ID cannot be empty")

        # 验证反馈中的置信度
        if "confidence" in self.feedback:
            confidence = self.feedback["confidence"]
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(
                    f"Confidence must be between 0 and 1, got {confidence}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'experience_id': self.experience_id,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'feedback': self.feedback,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }

    @property
    def is_correct(self) -> bool:
        """检查是否正确"""
        return self.feedback.get("correct", False)

    @property
    def confidence(self) -> float:
        """获取置信度"""
        return self.feedback.get("confidence", 0.0)

    def __eq__(self, other):
        """学习经验相等性比较（基于ID）"""
        if not isinstance(other, LearningExperience):
            return False
        return self.experience_id == other.experience_id

    def __hash__(self):
        """学习经验哈希值（基于ID）"""
        return hash(self.experience_id)
