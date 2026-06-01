"""
MultimodalInput聚合根

多模态输入聚合根，管理多模态输入的生命周期
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from .modality import Modality, ModalityType


@dataclass
class MultimodalInput:
    """多模态输入聚合根

    多模态输入是多模态限界上下文的核心聚合根，
    负责管理多种输入模态的组合。

    Attributes:
        input_id: 输入唯一标识
        modalities: 模态列表
        metadata: 额外元数据
        created_at: 创建时间
    """

    input_id: str
    modalities: List[Modality] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证输入参数"""
        if not self.input_id:
            raise ValueError("Input ID cannot be empty")

    def add_modality(self, modality: Modality) -> None:
        """添加模态

        Args:
            modality: 模态实例

        Raises:
            ValueError: 如果模态ID已存在
        """
        # 检查是否已存在
        for existing in self.modalities:
            if existing.modality_id == modality.modality_id:
                raise ValueError(f"Modality {modality.modality_id} already exists")

        self.modalities.append(modality)

    def remove_modality(self, modality_id: str) -> bool:
        """移除模态

        Args:
            modality_id: 模态ID

        Returns:
            是否移除成功
        """
        for i, modality in enumerate(self.modalities):
            if modality.modality_id == modality_id:
                self.modalities.pop(i)
                return True
        return False

    def get_modality(self, modality_id: str) -> Optional[Modality]:
        """获取模态

        Args:
            modality_id: 模态ID

        Returns:
            模态实例，如果不存在返回None
        """
        for modality in self.modalities:
            if modality.modality_id == modality_id:
                return modality
        return None

    def get_modalities_by_type(self, modality_type: ModalityType) -> List[Modality]:
        """按类型获取模态

        Args:
            modality_type: 模态类型

        Returns:
            匹配的模态列表
        """
        return [
            modality for modality in self.modalities
            if modality.modality_type == modality_type.value
        ]

    @property
    def modality_count(self) -> int:
        """获取模态数量"""
        return len(self.modalities)

    @property
    def has_text(self) -> bool:
        """检查是否包含文本模态"""
        return any(m.is_text for m in self.modalities)

    @property
    def has_image(self) -> bool:
        """检查是否包含图像模态"""
        return any(m.is_image for m in self.modalities)

    @property
    def has_audio(self) -> bool:
        """检查是否包含音频模态"""
        return any(m.is_audio for m in self.modalities)

    def get_text_content(self) -> str:
        """获取文本内容

        Returns:
            文本内容，如果没有文本模态返回空字符串
        """
        text_modalities = self.get_modalities_by_type(ModalityType.TEXT)
        if text_modalities:
            return text_modalities[0].content
        return ""

    def get_all_content(self) -> List[str]:
        """获取所有内容

        Returns:
            所有模态的内容列表
        """
        return [modality.content for modality in self.modalities]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'input_id': self.input_id,
            'modalities': [m.to_dict() for m in self.modalities],
            'modality_count': self.modality_count,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }
