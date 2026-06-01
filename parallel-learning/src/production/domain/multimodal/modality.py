"""
Modality实体

模态实体，表示一种输入模态
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from enum import Enum


class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class Modality:
    """模态实体

    模态是多模态输入的组成部分，
    表示一种输入模态（文本、图像、音频等）。

    Attributes:
        modality_id: 模态唯一标识
        modality_type: 模态类型
        content: 模态内容
        metadata: 额外元数据
        created_at: 创建时间
    """

    modality_id: str
    modality_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证模态参数"""
        # 验证模态类型
        valid_types = [mt.value for mt in ModalityType]
        if self.modality_type not in valid_types:
            raise ValueError(
                f"Invalid modality type: {self.modality_type}. "
                f"Must be one of: {valid_types}"
            )

        # 验证模态ID
        if not self.modality_id:
            raise ValueError("Modality ID cannot be empty")

        # 验证内容
        if not self.content:
            raise ValueError("Content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'modality_id': self.modality_id,
            'modality_type': self.modality_type,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }

    @property
    def is_text(self) -> bool:
        """检查是否为文本模态"""
        return self.modality_type == ModalityType.TEXT.value

    @property
    def is_image(self) -> bool:
        """检查是否为图像模态"""
        return self.modality_type == ModalityType.IMAGE.value

    @property
    def is_audio(self) -> bool:
        """检查是否为音频模态"""
        return self.modality_type == ModalityType.AUDIO.value

    def __eq__(self, other):
        """模态相等性比较（基于ID）"""
        if not isinstance(other, Modality):
            return False
        return self.modality_id == other.modality_id

    def __hash__(self):
        """模态哈希值（基于ID）"""
        return hash(self.modality_id)
