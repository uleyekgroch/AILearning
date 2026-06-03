"""
PerceptionResult值对象

感知结果值对象，表示感知处理的结果
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


@dataclass(frozen=True)
class PerceptionResult:
    """感知结果值对象

    感知结果是不可变的值对象，表示感知处理的结果。

    Attributes:
        perception_id: 感知唯一标识
        detected_objects: 检测到的物体列表
        scene_description: 场景描述
        confidence: 置信度 [0, 1]
        source_sensory: 来源感觉输入ID列表
        metadata: 额外元数据
        created_at: 创建时间
    """

    perception_id: str
    detected_objects: List[str]
    scene_description: str
    confidence: float
    source_sensory: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证感知结果参数"""
        # 验证置信度
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        # 验证感知ID
        if not self.perception_id:
            raise ValueError("Perception ID cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'perception_id': self.perception_id,
            'detected_objects': self.detected_objects,
            'scene_description': self.scene_description,
            'confidence': self.confidence,
            'source_sensory': self.source_sensory,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }

    @property
    def is_high_confidence(self) -> bool:
        """检查是否高置信度（>0.8）"""
        return self.confidence > 0.8

    @property
    def object_count(self) -> int:
        """获取检测到的物体数量"""
        return len(self.detected_objects)

    def has_object(self, object_name: str) -> bool:
        """检查是否检测到指定物体

        Args:
            object_name: 物体名称

        Returns:
            是否检测到
        """
        return object_name in self.detected_objects
