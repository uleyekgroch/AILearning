"""
FusionResult值对象

融合结果值对象，表示多模态融合的结果
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


@dataclass(frozen=True)
class FusionResult:
    """融合结果值对象

    融合结果是不可变的值对象，表示多模态融合的结果。

    Attributes:
        fusion_id: 融合唯一标识
        fused_content: 融合后的内容
        confidence: 置信度 [0, 1]
        source_modalities: 来源模态ID列表
        metadata: 额外元数据
        created_at: 创建时间
    """

    fusion_id: str
    fused_content: str
    confidence: float
    source_modalities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证融合结果参数"""
        # 验证置信度
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        # 验证融合ID
        if not self.fusion_id:
            raise ValueError("Fusion ID cannot be empty")

        # 验证融合内容
        if not self.fused_content:
            raise ValueError("Fused content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'fusion_id': self.fusion_id,
            'fused_content': self.fused_content,
            'confidence': self.confidence,
            'source_modalities': self.source_modalities,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }

    @property
    def is_high_confidence(self) -> bool:
        """检查是否高置信度（>0.8）"""
        return self.confidence > 0.8

    @property
    def source_count(self) -> int:
        """获取来源模态数量"""
        return len(self.source_modalities)
