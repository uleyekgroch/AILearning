"""
具身感知系统单元测试

TDD方法：先写测试，再写实现
测试驱动设计具身感知系统
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestSensoryInput:
    """感觉输入测试"""

    def test_create_visual_input(self):
        """测试创建视觉输入"""
        # Arrange & Act
        from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType

        sensory_input = SensoryInput(
            input_id="visual_001",
            sensory_type=SensoryType.VISUAL.value,
            data={"image": "base64_data", "width": 100, "height": 100},
            timestamp=datetime.now()
        )

        # Assert
        assert sensory_input.input_id == "visual_001"
        assert sensory_input.sensory_type == SensoryType.VISUAL.value

    def test_create_auditory_input(self):
        """测试创建听觉输入"""
        # Arrange & Act
        from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType

        sensory_input = SensoryInput(
            input_id="auditory_001",
            sensory_type=SensoryType.AUDITORY.value,
            data={"audio": "base64_data", "duration": 5.0},
            timestamp=datetime.now()
        )

        # Assert
        assert sensory_input.input_id == "auditory_001"
        assert sensory_input.sensory_type == SensoryType.AUDITORY.value

    def test_create_tactile_input(self):
        """测试创建触觉输入"""
        # Arrange & Act
        from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType

        sensory_input = SensoryInput(
            input_id="tactile_001",
            sensory_type=SensoryType.TACTILE.value,
            data={"pressure": 0.5, "temperature": 25.0},
            timestamp=datetime.now()
        )

        # Assert
        assert sensory_input.input_id == "tactile_001"
        assert sensory_input.sensory_type == SensoryType.TACTILE.value

    def test_sensory_input_validation(self):
        """测试感觉输入验证"""
        # Arrange & Act & Assert
        from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType

        # 测试无效感觉类型
        with pytest.raises(ValueError, match="Invalid sensory type"):
            SensoryInput(
                input_id="invalid_001",
                sensory_type="invalid_type",
                data={},
                timestamp=datetime.now()
            )


class TestMotorCommand:
    """运动命令测试"""

    def test_create_movement_command(self):
        """测试创建运动命令"""
        # Arrange & Act
        from src.production.domain.embodied.motor_command import MotorCommand, MotorType

        command = MotorCommand(
            command_id="movement_001",
            motor_type=MotorType.MOVEMENT.value,
            parameters={"direction": "forward", "speed": 1.0},
            priority=1
        )

        # Assert
        assert command.command_id == "movement_001"
        assert command.motor_type == MotorType.MOVEMENT.value

    def test_create_grasp_command(self):
        """测试创建抓取命令"""
        # Arrange & Act
        from src.production.domain.embodied.motor_command import MotorCommand, MotorType

        command = MotorCommand(
            command_id="grasp_001",
            motor_type=MotorType.GRASP.value,
            parameters={"object": "cup", "force": 0.5},
            priority=2
        )

        # Assert
        assert command.command_id == "grasp_001"
        assert command.motor_type == MotorType.GRASP.value

    def test_motor_command_validation(self):
        """测试运动命令验证"""
        # Arrange & Act & Assert
        from src.production.domain.embodied.motor_command import MotorCommand, MotorType

        # 测试无效运动类型
        with pytest.raises(ValueError, match="Invalid motor type"):
            MotorCommand(
                command_id="invalid_001",
                motor_type="invalid_type",
                parameters={},
                priority=1
            )


class TestEmbodiedState:
    """具身状态测试"""

    def test_create_embodied_state(self):
        """测试创建具身状态"""
        # Arrange & Act
        from src.production.domain.embodied.embodied_state import EmbodiedState

        state = EmbodiedState(
            state_id="state_001",
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            orientation={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            energy_level=1.0,
            emotional_state="neutral"
        )

        # Assert
        assert state.state_id == "state_001"
        assert state.energy_level == 1.0
        assert state.emotional_state == "neutral"

    def test_embodied_state_validation(self):
        """测试具身状态验证"""
        # Arrange & Act & Assert
        from src.production.domain.embodied.embodied_state import EmbodiedState

        # 测试无效能量水平
        with pytest.raises(ValueError, match="Energy level must be between 0 and 1"):
            EmbodiedState(
                state_id="invalid_001",
                position={"x": 0.0, "y": 0.0, "z": 0.0},
                orientation={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                energy_level=1.5,  # 无效
                emotional_state="neutral"
            )

    def test_update_position(self):
        """测试更新位置"""
        # Arrange
        from src.production.domain.embodied.embodied_state import EmbodiedState

        state = EmbodiedState(
            state_id="state_001",
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            orientation={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            energy_level=1.0,
            emotional_state="neutral"
        )

        # Act
        state.update_position({"x": 1.0, "y": 2.0, "z": 3.0})

        # Assert
        assert state.position["x"] == 1.0
        assert state.position["y"] == 2.0
        assert state.position["z"] == 3.0


class TestPerceptionResult:
    """感知结果测试"""

    def test_create_perception_result(self):
        """测试创建感知结果"""
        # Arrange & Act
        from src.production.domain.embodied.perception_result import PerceptionResult

        result = PerceptionResult(
            perception_id="perception_001",
            detected_objects=["cup", "table"],
            scene_description="A cup on a table",
            confidence=0.95,
            source_sensory=["visual_001"]
        )

        # Assert
        assert result.perception_id == "perception_001"
        assert len(result.detected_objects) == 2
        assert result.confidence == 0.95

    def test_perception_result_validation(self):
        """测试感知结果验证"""
        # Arrange & Act & Assert
        from src.production.domain.embodied.perception_result import PerceptionResult

        # 测试无效置信度
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            PerceptionResult(
                perception_id="invalid_001",
                detected_objects=[],
                scene_description="test",
                confidence=1.5,  # 无效
                source_sensory=[]
            )


class TestEmbodiedService:
    """具身服务测试"""

    def test_process_sensory_input(self):
        """测试处理感觉输入"""
        # Arrange
        from src.production.application.services.embodied_service import EmbodiedApplicationService
        from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType

        service = EmbodiedApplicationService()

        sensory_input = SensoryInput(
            input_id="visual_001",
            sensory_type=SensoryType.VISUAL.value,
            data={"image": "base64_data"},
            timestamp=datetime.now()
        )

        # Act
        result = service.process_sensory_input(sensory_input)

        # Assert
        assert result is not None
        assert result.confidence > 0

    def test_generate_motor_command(self):
        """测试生成运动命令"""
        # Arrange
        from src.production.application.services.embodied_service import EmbodiedApplicationService

        service = EmbodiedApplicationService()

        # Act
        command = service.generate_motor_command(
            motor_type="movement",
            parameters={"direction": "forward", "speed": 1.0}
        )

        # Assert
        assert command is not None
        assert command.motor_type == "movement"

    def test_update_embodied_state(self):
        """测试更新具身状态"""
        # Arrange
        from src.production.application.services.embodied_service import EmbodiedApplicationService

        service = EmbodiedApplicationService()

        # Act
        state = service.update_state(
            position={"x": 1.0, "y": 2.0, "z": 3.0},
            energy_level=0.8
        )

        # Assert
        assert state is not None
        assert state.position["x"] == 1.0
        assert state.energy_level == 0.8
