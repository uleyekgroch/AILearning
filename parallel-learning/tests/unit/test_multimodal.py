"""
多模态输入融合单元测试

TDD方法：先写测试，再写实现
测试驱动设计多模态输入
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestModality:
    """模态测试"""

    def test_create_text_modality(self):
        """测试创建文本模态"""
        # Arrange & Act
        from src.production.domain.multimodal.modality import Modality, ModalityType

        modality = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="水在100度沸腾",
            metadata={"language": "zh"}
        )

        # Assert
        assert modality.modality_id == "text_001"
        assert modality.modality_type == ModalityType.TEXT.value  # 使用字符串值
        assert modality.content == "水在100度沸腾"

    def test_create_image_modality(self):
        """测试创建图像模态"""
        # Arrange & Act
        from src.production.domain.multimodal.modality import Modality, ModalityType

        modality = Modality(
            modality_id="image_001",
            modality_type=ModalityType.IMAGE.value,  # 使用字符串值
            content="base64_encoded_image_data",
            metadata={"format": "png", "width": 100, "height": 100}
        )

        # Assert
        assert modality.modality_id == "image_001"
        assert modality.modality_type == ModalityType.IMAGE.value  # 使用字符串值

    def test_create_audio_modality(self):
        """测试创建音频模态"""
        # Arrange & Act
        from src.production.domain.multimodal.modality import Modality, ModalityType

        modality = Modality(
            modality_id="audio_001",
            modality_type=ModalityType.AUDIO.value,  # 使用字符串值
            content="base64_encoded_audio_data",
            metadata={"format": "wav", "duration": 5.0}
        )

        # Assert
        assert modality.modality_id == "audio_001"
        assert modality.modality_type == ModalityType.AUDIO.value  # 使用字符串值

    def test_modality_validation(self):
        """测试模态验证"""
        # Arrange & Act & Assert
        from src.production.domain.multimodal.modality import Modality, ModalityType

        # 测试无效模态类型
        with pytest.raises(ValueError, match="Invalid modality type"):
            Modality(
                modality_id="invalid_001",
                modality_type="invalid_type",
                content="test",
                metadata={}
            )

    def test_modality_equality(self):
        """测试模态相等性"""
        # Arrange
        from src.production.domain.multimodal.modality import Modality, ModalityType

        modality1 = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="test",
            metadata={}
        )

        modality2 = Modality(
            modality_id="text_001",  # 相同ID
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="different",
            metadata={}
        )

        modality3 = Modality(
            modality_id="text_002",  # 不同ID
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="test",
            metadata={}
        )

        # Act & Assert
        assert modality1 == modality2  # 相同ID = 相等
        assert modality1 != modality3  # 不同ID = 不等
        assert hash(modality1) == hash(modality2)


class TestMultimodalInput:
    """多模态输入测试"""

    def test_create_multimodal_input(self):
        """测试创建多模态输入"""
        # Arrange
        from src.production.domain.multimodal.multimodal_input import MultimodalInput
        from src.production.domain.multimodal.modality import Modality, ModalityType

        text_modality = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="水在100度沸腾",
            metadata={}
        )

        image_modality = Modality(
            modality_id="image_001",
            modality_type=ModalityType.IMAGE.value,  # 使用字符串值
            content="base64_image_data",
            metadata={}
        )

        # Act
        multimodal_input = MultimodalInput(
            input_id="input_001",
            modalities=[text_modality, image_modality],
            metadata={"source": "user"}
        )

        # Assert
        assert multimodal_input.input_id == "input_001"
        assert multimodal_input.modality_count == 2

    def test_get_modality_by_type(self):
        """测试按类型获取模态"""
        # Arrange
        from src.production.domain.multimodal.multimodal_input import MultimodalInput
        from src.production.domain.multimodal.modality import Modality, ModalityType

        text_modality = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="水在100度沸腾",
            metadata={}
        )

        image_modality = Modality(
            modality_id="image_001",
            modality_type=ModalityType.IMAGE.value,  # 使用字符串值
            content="base64_image_data",
            metadata={}
        )

        multimodal_input = MultimodalInput(
            input_id="input_001",
            modalities=[text_modality, image_modality],
            metadata={}
        )

        # Act
        text_modalities = multimodal_input.get_modalities_by_type(ModalityType.TEXT)
        image_modalities = multimodal_input.get_modalities_by_type(ModalityType.IMAGE)

        # Assert
        assert len(text_modalities) == 1
        assert len(image_modalities) == 1
        assert text_modalities[0].content == "水在100度沸腾"

    def test_multimodal_input_structure(self):
        """测试多模态输入结构"""
        # Arrange
        from src.production.domain.multimodal.multimodal_input import MultimodalInput
        from src.production.domain.multimodal.modality import Modality, ModalityType

        text_modality = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="test",
            metadata={}
        )

        multimodal_input = MultimodalInput(
            input_id="input_001",
            modalities=[text_modality],
            metadata={}
        )

        # Act & Assert - 验证结构
        assert multimodal_input.input_id == "input_001"
        assert multimodal_input.modality_count == 1
        assert multimodal_input.has_text == True
        assert multimodal_input.has_image == False


class TestMultimodalFusion:
    """多模态融合测试"""

    def test_create_fusion_result(self):
        """测试创建融合结果"""
        # Arrange & Act
        from src.production.domain.multimodal.fusion_result import FusionResult

        result = FusionResult(
            fusion_id="fusion_001",
            fused_content="水在100度沸腾，如图所示",
            confidence=0.95,
            source_modalities=["text_001", "image_001"],
            metadata={"fusion_method": "early_fusion"}
        )

        # Assert
        assert result.fusion_id == "fusion_001"
        assert result.confidence == 0.95
        assert len(result.source_modalities) == 2

    def test_fusion_result_validation(self):
        """测试融合结果验证"""
        # Arrange & Act & Assert
        from src.production.domain.multimodal.fusion_result import FusionResult

        # 测试无效置信度
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            FusionResult(
                fusion_id="fusion_001",
                fused_content="test",
                confidence=1.5,  # 无效
                source_modalities=[],
                metadata={}
            )


class TestMultimodalService:
    """多模态服务测试"""

    def test_process_multimodal_input(self):
        """测试处理多模态输入"""
        # Arrange
        from src.production.application.services.multimodal_service import MultimodalApplicationService
        from src.production.domain.multimodal.modality import Modality, ModalityType

        service = MultimodalApplicationService()

        text_modality = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="水在100度沸腾",
            metadata={}
        )

        # Act
        result = service.process_input([text_modality])

        # Assert
        assert result is not None
        assert result.confidence > 0

    def test_fuse_modalities(self):
        """测试融合多模态"""
        # Arrange
        from src.production.application.services.multimodal_service import MultimodalApplicationService
        from src.production.domain.multimodal.modality import Modality, ModalityType

        service = MultimodalApplicationService()

        text_modality = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,  # 使用字符串值
            content="水在100度沸腾",
            metadata={}
        )

        image_modality = Modality(
            modality_id="image_001",
            modality_type=ModalityType.IMAGE.value,  # 使用字符串值
            content="base64_image_data",
            metadata={}
        )

        # Act
        result = service.fuse_modalities([text_modality, image_modality])

        # Assert
        assert result is not None
        assert result.confidence > 0
        assert len(result.source_modalities) == 2
