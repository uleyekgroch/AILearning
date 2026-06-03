"""
多模态应用服务

负责多模态输入的用例编排
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.production.domain.multimodal.modality import Modality, ModalityType
from src.production.domain.multimodal.multimodal_input import MultimodalInput
from src.production.domain.multimodal.fusion_result import FusionResult


class MultimodalApplicationService:
    """多模态应用服务

    职责：
    - 编排多模态输入的用例
    - 管理多模态融合
    - 提供多模态处理接口

    Attributes:
        inputs: 多模态输入映射
        fusion_results: 融合结果映射
    """

    def __init__(self):
        """初始化多模态应用服务"""
        self.inputs: Dict[str, MultimodalInput] = {}
        self.fusion_results: Dict[str, FusionResult] = {}

    def create_input(self, input_id: str, modalities: List[Modality]) -> MultimodalInput:
        """创建多模态输入

        Args:
            input_id: 输入ID
            modalities: 模态列表

        Returns:
            创建的多模态输入

        Raises:
            ValueError: 如果输入ID已存在
        """
        if input_id in self.inputs:
            raise ValueError(f"Input {input_id} already exists")

        multimodal_input = MultimodalInput(
            input_id=input_id,
            modalities=modalities,
            metadata={"created_by": "multimodal_service"}
        )

        self.inputs[input_id] = multimodal_input
        return multimodal_input

    def get_input(self, input_id: str) -> Optional[MultimodalInput]:
        """获取多模态输入

        Args:
            input_id: 输入ID

        Returns:
            多模态输入实例，如果不存在返回None
        """
        return self.inputs.get(input_id)

    def process_input(self, modalities: List[Modality]) -> FusionResult:
        """处理多模态输入

        Args:
            modalities: 模态列表

        Returns:
            融合结果
        """
        # 生成融合ID
        fusion_id = f"fusion_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # 提取内容
        contents = [modality.content for modality in modalities]

        # 简单融合策略：连接所有内容
        fused_content = " ".join(contents)

        # 计算置信度（简化：基于模态数量）
        confidence = min(0.9 + len(modalities) * 0.02, 1.0)

        # 获取来源模态ID
        source_modalities = [modality.modality_id for modality in modalities]

        # 创建融合结果
        result = FusionResult(
            fusion_id=fusion_id,
            fused_content=fused_content,
            confidence=confidence,
            source_modalities=source_modalities,
            metadata={
                "fusion_method": "simple_concatenation",
                "modality_count": len(modalities)
            }
        )

        # 保存结果
        self.fusion_results[fusion_id] = result

        return result

    def fuse_modalities(self, modalities: List[Modality]) -> FusionResult:
        """融合多模态

        Args:
            modalities: 模态列表

        Returns:
            融合结果
        """
        return self.process_input(modalities)

    def get_fusion_result(self, fusion_id: str) -> Optional[FusionResult]:
        """获取融合结果

        Args:
            fusion_id: 融合ID

        Returns:
            融合结果，如果不存在返回None
        """
        return self.fusion_results.get(fusion_id)

    def list_inputs(self) -> List[str]:
        """列出所有输入

        Returns:
            输入ID列表
        """
        return list(self.inputs.keys())

    def list_fusion_results(self) -> List[str]:
        """列出所有融合结果

        Returns:
            融合ID列表
        """
        return list(self.fusion_results.keys())

    def delete_input(self, input_id: str) -> bool:
        """删除输入

        Args:
            input_id: 输入ID

        Returns:
            是否删除成功
        """
        if input_id in self.inputs:
            del self.inputs[input_id]
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        total_modalities = sum(
            input.modality_count for input in self.inputs.values()
        )

        return {
            'total_inputs': len(self.inputs),
            'total_fusion_results': len(self.fusion_results),
            'total_modalities': total_modalities,
        }
