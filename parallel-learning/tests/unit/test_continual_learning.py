"""
持续学习机制单元测试

TDD方法：先写测试，再写实现
测试驱动设计持续学习系统
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestLearningExperience:
    """学习经验测试"""

    def test_create_learning_experience(self):
        """测试创建学习经验"""
        # Arrange & Act
        from src.production.domain.continual_learning.learning_experience import LearningExperience

        experience = LearningExperience(
            experience_id="exp_001",
            input_data={"text": "水在100度沸腾"},
            output_data={"fact": "水沸点100度"},
            feedback={"correct": True, "confidence": 0.95},
            timestamp=datetime.now()
        )

        # Assert
        assert experience.experience_id == "exp_001"
        assert experience.feedback["correct"] == True

    def test_learning_experience_validation(self):
        """测试学习经验验证"""
        # Arrange & Act & Assert
        from src.production.domain.continual_learning.learning_experience import LearningExperience

        # 测试无效置信度
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            LearningExperience(
                experience_id="exp_001",
                input_data={"text": "test"},
                output_data={"fact": "test"},
                feedback={"correct": True, "confidence": 1.5},  # 无效
                timestamp=datetime.now()
            )


class TestKnowledgeUpdate:
    """知识更新测试"""

    def test_create_knowledge_update(self):
        """测试创建知识更新"""
        # Arrange & Act
        from src.production.domain.continual_learning.knowledge_update import KnowledgeUpdate, UpdateType

        update = KnowledgeUpdate(
            update_id="update_001",
            update_type=UpdateType.ADD.value,
            target_id="fact_001",
            new_data={"statement": "水在100度沸腾"},
            confidence=0.95,
            timestamp=datetime.now()
        )

        # Assert
        assert update.update_id == "update_001"
        assert update.update_type == UpdateType.ADD.value

    def test_knowledge_update_validation(self):
        """测试知识更新验证"""
        # Arrange & Act & Assert
        from src.production.domain.continual_learning.knowledge_update import KnowledgeUpdate, UpdateType

        # 测试无效更新类型
        with pytest.raises(ValueError, match="Invalid update type"):
            KnowledgeUpdate(
                update_id="update_001",
                update_type="invalid_type",
                target_id="fact_001",
                new_data={},
                confidence=0.95,
                timestamp=datetime.now()
            )


class TestLearningStrategy:
    """学习策略测试"""

    def test_create_learning_strategy(self):
        """测试创建学习策略"""
        # Arrange & Act
        from src.production.domain.continual_learning.learning_strategy import LearningStrategy, StrategyType

        strategy = LearningStrategy(
            strategy_id="strategy_001",
            strategy_type=StrategyType.INCREMENTAL.value,
            parameters={"learning_rate": 0.01, "batch_size": 32},
            priority=1
        )

        # Assert
        assert strategy.strategy_id == "strategy_001"
        assert strategy.strategy_type == StrategyType.INCREMENTAL.value

    def test_learning_strategy_validation(self):
        """测试学习策略验证"""
        # Arrange & Act & Assert
        from src.production.domain.continual_learning.learning_strategy import LearningStrategy, StrategyType

        # 测试无效策略类型
        with pytest.raises(ValueError, match="Invalid strategy type"):
            LearningStrategy(
                strategy_id="strategy_001",
                strategy_type="invalid_type",
                parameters={},
                priority=1
            )


class TestContinualLearner:
    """持续学习器测试"""

    def test_create_continual_learner(self):
        """测试创建持续学习器"""
        # Arrange & Act
        from src.production.domain.continual_learning.continual_learner import ContinualLearner

        learner = ContinualLearner(
            learner_id="learner_001",
            learning_rate=0.01,
            memory_size=1000
        )

        # Assert
        assert learner.learner_id == "learner_001"
        assert learner.learning_rate == 0.01

    def test_continual_learner_validation(self):
        """测试持续学习器验证"""
        # Arrange & Act & Assert
        from src.production.domain.continual_learning.continual_learner import ContinualLearner

        # 测试无效学习率
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1"):
            ContinualLearner(
                learner_id="learner_001",
                learning_rate=1.5,  # 无效
                memory_size=1000
            )

    def test_learn_from_experience(self):
        """测试从经验中学习"""
        # Arrange
        from src.production.domain.continual_learning.continual_learner import ContinualLearner
        from src.production.domain.continual_learning.learning_experience import LearningExperience

        learner = ContinualLearner(
            learner_id="learner_001",
            learning_rate=0.01,
            memory_size=1000
        )

        experience = LearningExperience(
            experience_id="exp_001",
            input_data={"text": "水在100度沸腾"},
            output_data={"fact": "水沸点100度"},
            feedback={"correct": True, "confidence": 0.95},
            timestamp=datetime.now()
        )

        # Act
        result = learner.learn_from_experience(experience)

        # Assert
        assert result is not None
        assert result.success == True

    def test_get_learning_statistics(self):
        """测试获取学习统计"""
        # Arrange
        from src.production.domain.continual_learning.continual_learner import ContinualLearner

        learner = ContinualLearner(
            learner_id="learner_001",
            learning_rate=0.01,
            memory_size=1000
        )

        # Act
        stats = learner.get_statistics()

        # Assert
        assert stats is not None
        assert stats["total_experiences"] == 0


class TestContinualLearningService:
    """持续学习服务测试"""

    def test_process_learning_experience(self):
        """测试处理学习经验"""
        # Arrange
        from src.production.application.services.continual_learning_service import ContinualLearningService
        from src.production.domain.continual_learning.learning_experience import LearningExperience

        service = ContinualLearningService()

        experience = LearningExperience(
            experience_id="exp_001",
            input_data={"text": "水在100度沸腾"},
            output_data={"fact": "水沸点100度"},
            feedback={"correct": True, "confidence": 0.95},
            timestamp=datetime.now()
        )

        # Act
        result = service.process_learning_experience(experience)

        # Assert
        assert result is not None
        assert result.success == True

    def test_update_knowledge_base(self):
        """测试更新知识库"""
        # Arrange
        from src.production.application.services.continual_learning_service import ContinualLearningService
        from src.production.domain.continual_learning.knowledge_update import KnowledgeUpdate, UpdateType

        service = ContinualLearningService()

        update = KnowledgeUpdate(
            update_id="update_001",
            update_type=UpdateType.ADD.value,
            target_id="fact_001",
            new_data={"statement": "水在100度沸腾"},
            confidence=0.95,
            timestamp=datetime.now()
        )

        # Act
        result = service.update_knowledge_base(update)

        # Assert
        assert result is not None
        assert result.success == True

    def test_get_learning_progress(self):
        """测试获取学习进度"""
        # Arrange
        from src.production.application.services.continual_learning_service import ContinualLearningService

        service = ContinualLearningService()

        # Act
        progress = service.get_learning_progress()

        # Assert
        assert progress is not None
        assert progress["total_updates"] >= 0
