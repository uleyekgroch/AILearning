"""
具身应用服务

负责具身感知系统的用例编排
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType
from src.production.domain.embodied.motor_command import MotorCommand, MotorType
from src.production.domain.embodied.embodied_state import EmbodiedState
from src.production.domain.embodied.perception_result import PerceptionResult


class EmbodiedApplicationService:
    """具身应用服务

    职责：
    - 编排具身感知系统的用例
    - 管理感觉输入处理
    - 生成运动命令
    - 维护具身状态

    Attributes:
        current_state: 当前具身状态
        sensory_inputs: 感觉输入历史
        motor_commands: 运动命令历史
        perception_results: 感知结果历史
    """

    def __init__(self):
        """初始化具身应用服务"""
        # 初始化默认状态
        self.current_state = EmbodiedState(
            state_id="initial_state",
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            orientation={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            energy_level=1.0,
            emotional_state="neutral"
        )

        self.sensory_inputs: Dict[str, SensoryInput] = {}
        self.motor_commands: Dict[str, MotorCommand] = {}
        self.perception_results: Dict[str, PerceptionResult] = {}

    def process_sensory_input(self, sensory_input: SensoryInput) -> PerceptionResult:
        """处理感觉输入

        Args:
            sensory_input: 感觉输入

        Returns:
            感知结果
        """
        # 保存感觉输入
        self.sensory_inputs[sensory_input.input_id] = sensory_input

        # 生成感知ID
        perception_id = f"perception_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # 简化感知处理：根据感觉类型生成结果
        detected_objects = []
        scene_description = ""

        if sensory_input.is_visual:
            # 视觉处理：模拟物体检测
            detected_objects = ["object_1", "object_2"]
            scene_description = "Visual scene with detected objects"
        elif sensory_input.is_auditory:
            # 听觉处理：模拟声音识别
            detected_objects = ["sound_source"]
            scene_description = "Auditory input detected"
        elif sensory_input.is_tactile:
            # 触觉处理：模拟触觉反馈
            detected_objects = ["contact_point"]
            scene_description = "Tactile contact detected"

        # 计算置信度
        confidence = 0.9

        # 创建感知结果
        result = PerceptionResult(
            perception_id=perception_id,
            detected_objects=detected_objects,
            scene_description=scene_description,
            confidence=confidence,
            source_sensory=[sensory_input.input_id],
            metadata={
                "sensory_type": sensory_input.sensory_type,
                "processing_time": datetime.now().isoformat()
            }
        )

        # 保存结果
        self.perception_results[perception_id] = result

        return result

    def generate_motor_command(
        self,
        motor_type: str,
        parameters: Dict[str, Any],
        priority: int = 1
    ) -> MotorCommand:
        """生成运动命令

        Args:
            motor_type: 运动类型
            parameters: 命令参数
            priority: 优先级

        Returns:
            运动命令
        """
        # 生成命令ID
        command_id = f"command_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # 创建运动命令
        command = MotorCommand(
            command_id=command_id,
            motor_type=motor_type,
            parameters=parameters,
            priority=priority,
            metadata={
                "generated_by": "embodied_service",
                "current_state_id": self.current_state.state_id
            }
        )

        # 保存命令
        self.motor_commands[command_id] = command

        return command

    def update_state(
        self,
        position: Optional[Dict[str, float]] = None,
        orientation: Optional[Dict[str, float]] = None,
        energy_level: Optional[float] = None,
        emotional_state: Optional[str] = None
    ) -> EmbodiedState:
        """更新具身状态

        Args:
            position: 新位置（可选）
            orientation: 新姿态（可选）
            energy_level: 新能量水平（可选）
            emotional_state: 新情感状态（可选）

        Returns:
            更新后的具身状态
        """
        if position:
            self.current_state.update_position(position)

        if orientation:
            self.current_state.update_orientation(orientation)

        if energy_level is not None:
            self.current_state.update_energy_level(energy_level)

        if emotional_state:
            self.current_state.update_emotional_state(emotional_state)

        return self.current_state

    def get_current_state(self) -> EmbodiedState:
        """获取当前具身状态

        Returns:
            当前具身状态
        """
        return self.current_state

    def get_sensory_input(self, input_id: str) -> Optional[SensoryInput]:
        """获取感觉输入

        Args:
            input_id: 输入ID

        Returns:
            感觉输入，如果不存在返回None
        """
        return self.sensory_inputs.get(input_id)

    def get_motor_command(self, command_id: str) -> Optional[MotorCommand]:
        """获取运动命令

        Args:
            command_id: 命令ID

        Returns:
            运动命令，如果不存在返回None
        """
        return self.motor_commands.get(command_id)

    def get_perception_result(self, perception_id: str) -> Optional[PerceptionResult]:
        """获取感知结果

        Args:
            perception_id: 感知ID

        Returns:
            感知结果，如果不存在返回None
        """
        return self.perception_results.get(perception_id)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_sensory_inputs': len(self.sensory_inputs),
            'total_motor_commands': len(self.motor_commands),
            'total_perception_results': len(self.perception_results),
            'current_state': self.current_state.to_dict(),
        }
