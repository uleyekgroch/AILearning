"""
ContinualLearner聚合根

持续学习器聚合根，管理持续学习的生命周期
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LearningResult:
    """学习结果"""
    success: bool
    experience_id: str
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinualLearner:
    """持续学习器聚合根

    持续学习器是持续学习系统的核心聚合根，
    负责管理持续学习的生命周期。

    Attributes:
        learner_id: 学习器唯一标识
        learning_rate: 学习率 [0, 1]
        memory_size: 记忆容量
        experiences: 学习经验列表
        statistics: 学习统计
        created_at: 创建时间
        updated_at: 更新时间
    """

    learner_id: str
    learning_rate: float
    memory_size: int
    experiences: List[Any] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证持续学习器参数"""
        # 验证学习器ID
        if not self.learner_id:
            raise ValueError("Learner ID cannot be empty")

        # 验证学习率
        if not (0.0 <= self.learning_rate <= 1.0):
            raise ValueError(
                f"Learning rate must be between 0 and 1, got {self.learning_rate}"
            )

        # 验证记忆容量
        if self.memory_size <= 0:
            raise ValueError(
                f"Memory size must be positive, got {self.memory_size}"
            )

        # 初始化统计信息
        if not self.statistics:
            self.statistics = {
                "total_experiences": 0,
                "correct_experiences": 0,
                "accuracy": 0.0,
                "last_update": None,
            }

    def learn_from_experience(self, experience: Any) -> LearningResult:
        """从经验中学习

        Args:
            experience: 学习经验

        Returns:
            学习结果
        """
        # 保存经验
        self.experiences.append(experience)

        # 如果超过记忆容量，移除最旧的经验
        if len(self.experiences) > self.memory_size:
            self.experiences = self.experiences[-self.memory_size:]

        # 更新统计信息
        self.statistics["total_experiences"] += 1
        if experience.is_correct:
            self.statistics["correct_experiences"] += 1

        # 计算准确率
        total = self.statistics["total_experiences"]
        correct = self.statistics["correct_experiences"]
        self.statistics["accuracy"] = correct / total if total > 0 else 0.0
        self.statistics["last_update"] = datetime.now().isoformat()

        # 更新时间
        self.updated_at = datetime.now()

        return LearningResult(
            success=True,
            experience_id=experience.experience_id,
            message="Learning completed successfully",
            metadata={"accuracy": self.statistics["accuracy"]}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息

        Returns:
            统计信息字典
        """
        return {
            "learner_id": self.learner_id,
            "learning_rate": self.learning_rate,
            "memory_size": self.memory_size,
            "current_memory": len(self.experiences),
            **self.statistics,
        }

    def clear_memory(self) -> None:
        """清空记忆"""
        self.experiences.clear()
        self.statistics = {
            "total_experiences": 0,
            "correct_experiences": 0,
            "accuracy": 0.0,
            "last_update": None,
        }
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "learner_id": self.learner_id,
            "learning_rate": self.learning_rate,
            "memory_size": self.memory_size,
            "current_memory": len(self.experiences),
            "statistics": self.statistics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
