"""
KnowledgeUpdate实体

知识更新实体，表示对知识库的更新操作
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from enum import Enum


class UpdateType(Enum):
    """更新类型枚举"""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"


@dataclass
class KnowledgeUpdate:
    """知识更新实体

    知识更新是持续学习系统的核心实体，
    表示对知识库的一次更新操作。

    Attributes:
        update_id: 更新唯一标识
        update_type: 更新类型
        target_id: 目标知识ID
        new_data: 新数据
        confidence: 置信度 [0, 1]
        timestamp: 时间戳
        metadata: 额外元数据
    """

    update_id: str
    update_type: str
    target_id: str
    new_data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证知识更新参数"""
        # 验证更新类型
        valid_types = [ut.value for ut in UpdateType]
        if self.update_type not in valid_types:
            raise ValueError(
                f"Invalid update type: {self.update_type}. "
                f"Must be one of: {valid_types}"
            )

        # 验证更新ID
        if not self.update_id:
            raise ValueError("Update ID cannot be empty")

        # 验证目标ID
        if not self.target_id:
            raise ValueError("Target ID cannot be empty")

        # 验证置信度
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'update_id': self.update_id,
            'update_type': self.update_type,
            'target_id': self.target_id,
            'new_data': self.new_data,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }

    @property
    def is_add(self) -> bool:
        """检查是否为添加操作"""
        return self.update_type == UpdateType.ADD.value

    @property
    def is_update(self) -> bool:
        """检查是否为更新操作"""
        return self.update_type == UpdateType.UPDATE.value

    @property
    def is_delete(self) -> bool:
        """检查是否为删除操作"""
        return self.update_type == UpdateType.DELETE.value

    @property
    def is_merge(self) -> bool:
        """检查是否为合并操作"""
        return self.update_type == UpdateType.MERGE.value

    def __eq__(self, other):
        """知识更新相等性比较（基于ID）"""
        if not isinstance(other, KnowledgeUpdate):
            return False
        return self.update_id == other.update_id

    def __hash__(self):
        """知识更新哈希值（基于ID）"""
        return hash(self.update_id)
