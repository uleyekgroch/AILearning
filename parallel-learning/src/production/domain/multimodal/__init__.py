"""
多模态输入限界上下文

实体：Modality
聚合根：MultimodalInput
值对象：FusionResult
"""

from .modality import Modality, ModalityType
from .multimodal_input import MultimodalInput
from .fusion_result import FusionResult

__all__ = [
    "Modality",
    "ModalityType",
    "MultimodalInput",
    "FusionResult",
]
