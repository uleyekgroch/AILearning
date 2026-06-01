"""
持续学习应用服务

负责持续学习的用例编排
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.production.domain.continual_learning.learning_experience import LearningExperience
from src.production.domain.continual_learning.knowledge_update import KnowledgeUpdate, UpdateType
from src.production.domain.continual_learning.learning_strategy import LearningStrategy, StrategyType
from src.production.domain.continual_learning.continual_learner import ContinualLearner, LearningResult


class ContinualLearningService:
    """持续学习应用服务

    职责：
    - 编排持续学习的用例
    - 管理学习经验
    - 处理知识更新
    - 提供学习进度接口

    Attributes:
        learner: 持续学习器
        updates: 知识更新历史
        strategies: 学习策略映射
    """

    def __init__(self, learner_id: str = "default_learner"):
        """初始化持续学习应用服务

        Args:
            learner_id: 学习器ID
        """
        self.learner = ContinualLearner(
            learner_id=learner_id,
            learning_rate=0.01,
            memory_size=1000
        )
        self.updates: Dict[str, KnowledgeUpdate] = {}
        self.strategies: Dict[str, LearningStrategy] = {}

    def process_learning_experience(self, experience: LearningExperience) -> LearningResult:
        """处理学习经验

        Args:
            experience: 学习经验

        Returns:
            学习结果
        """
        return self.learner.learn_from_experience(experience)

    def update_knowledge_base(self, update: KnowledgeUpdate) -> LearningResult:
        """更新知识库

        Args:
            update: 知识更新

        Returns:
            学习结果
        """
        # 保存更新记录
        self.updates[update.update_id] = update

        # 返回成功结果
        return LearningResult(
            success=True,
            experience_id=update.update_id,
            message=f"Knowledge {update.update_type} completed",
            metadata={
                "update_type": update.update_type,
                "target_id": update.target_id,
                "confidence": update.confidence,
            }
        )

    def add_strategy(self, strategy: LearningStrategy) -> None:
        """添加学习策略

        Args:
            strategy: 学习策略
        """
        self.strategies[strategy.strategy_id] = strategy

    def get_strategy(self, strategy_id: str) -> Optional[LearningStrategy]:
        """获取学习策略

        Args:
            strategy_id: 策略ID

        Returns:
            学习策略，如果不存在返回None
        """
        return self.strategies.get(strategy_id)

    def get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度

        Returns:
            学习进度字典
        """
        learner_stats = self.learner.get_statistics()

        return {
            "total_updates": len(self.updates),
            "total_strategies": len(self.strategies),
            "learner_statistics": learner_stats,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_updates": len(self.updates),
            "total_strategies": len(self.strategies),
            "learner": self.learner.get_statistics(),
        }

    def list_updates(self) -> List[str]:
        """列出所有更新

        Returns:
            更新ID列表
        """
        return list(self.updates.keys())

    def list_strategies(self) -> List[str]:
        """列出所有策略

        Returns:
            策略ID列表
        """
        return list(self.strategies.keys())
