"""
ReasoningSession聚合根单元测试

TDD方法：先写测试，再写实现
测试驱动设计领域模型
"""

import pytest
from datetime import datetime
from typing import List, Optional


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestReasoningSession:
    """ReasoningSession聚合根测试"""

    def test_create_reasoning_session(self):
        """测试创建推理会话"""
        # Arrange & Act
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        session = ReasoningSession(session_id="session_001")

        # Assert
        assert session.session_id == "session_001"
        assert session.status == "created"
        assert session.created_at is not None
        assert session.updated_at is not None

    def test_add_reasoning_task(self):
        """测试添加推理任务"""
        # Arrange
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        session = ReasoningSession(session_id="session_001")
        task = ReasoningTask(
            task_id="task_001",
            query="水在多少度沸腾",
            reasoning_type="commonsense"
        )

        # Act
        session.add_task(task)

        # Assert
        assert session.task_count == 1
        assert session.has_task("task_001")

    def test_execute_reasoning(self):
        """测试执行推理"""
        # Arrange
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        session = ReasoningSession(session_id="session_001")
        task = ReasoningTask(
            task_id="task_001",
            query="水在多少度沸腾",
            reasoning_type="commonsense"
        )
        session.add_task(task)

        # Act
        result = session.execute_task("task_001")

        # Assert
        assert result is not None
        assert result.task_id == "task_001"
        assert result.status == "completed"

    def test_get_reasoning_chain(self):
        """测试获取推理链"""
        # Arrange
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        session = ReasoningSession(session_id="session_001")
        task = ReasoningTask(
            task_id="task_001",
            query="水在多少度沸腾",
            reasoning_type="commonsense"
        )
        session.add_task(task)
        session.execute_task("task_001")

        # Act
        chain = session.get_reasoning_chain("task_001")

        # Assert
        assert chain is not None
        assert len(chain.steps) > 0

    def test_session_versioning(self):
        """测试会话版本控制"""
        # Arrange
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        session = ReasoningSession(session_id="session_001")
        initial_version = session.version

        # Act
        task = ReasoningTask(
            task_id="task_001",
            query="test",
            reasoning_type="commonsense"
        )
        session.add_task(task)

        # Assert
        assert session.version == initial_version + 1

    def test_session_immutability(self):
        """测试会话不可变性（通过事件溯源）"""
        # Arrange
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        session = ReasoningSession(session_id="session_001")

        task = ReasoningTask(
            task_id="task_001",
            query="test",
            reasoning_type="commonsense"
        )
        session.add_task(task)

        # Act - 获取事件历史
        events = session.get_events()

        # Assert
        assert len(events) == 1
        assert events[0].event_type == "TaskAdded"
        assert events[0].task_id == "task_001"


class TestReasoningTask:
    """ReasoningTask实体测试"""

    def test_create_reasoning_task(self):
        """测试创建推理任务"""
        # Arrange & Act
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        task = ReasoningTask(
            task_id="task_001",
            query="水在多少度沸腾",
            reasoning_type="commonsense"
        )

        # Assert
        assert task.task_id == "task_001"
        assert task.query == "水在多少度沸腾"
        assert task.reasoning_type == "commonsense"
        assert task.status == "pending"

    def test_task_validation(self):
        """测试任务验证"""
        # Arrange & Act & Assert
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        # 测试无效推理类型
        with pytest.raises(ValueError, match="Invalid reasoning type"):
            ReasoningTask(
                task_id="task_001",
                query="test",
                reasoning_type="invalid_type"  # 无效
            )

    def test_task_status_transition(self):
        """测试任务状态转换"""
        # Arrange
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        task = ReasoningTask(
            task_id="task_001",
            query="test",
            reasoning_type="commonsense"
        )

        # Act & Assert - 状态转换
        task.start()
        assert task.status == "running"

        task.complete()
        assert task.status == "completed"

    def test_task_equality(self):
        """测试任务相等性"""
        # Arrange
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        task1 = ReasoningTask(
            task_id="task_001",
            query="test1",
            reasoning_type="commonsense"
        )

        task2 = ReasoningTask(
            task_id="task_001",  # 相同ID
            query="test2",
            reasoning_type="causal"
        )

        task3 = ReasoningTask(
            task_id="task_002",  # 不同ID
            query="test1",
            reasoning_type="commonsense"
        )

        # Act & Assert
        assert task1 == task2  # 相同ID = 相等
        assert task1 != task3  # 不同ID = 不等
        assert hash(task1) == hash(task2)


class TestReasoningResult:
    """ReasoningResult值对象测试"""

    def test_create_reasoning_result(self):
        """测试创建推理结果"""
        # Arrange & Act
        from src.production.domain.reasoning.reasoning_result import ReasoningResult

        result = ReasoningResult(
            task_id="task_001",
            answer="水在100度沸腾",
            confidence=0.95,
            reasoning_chain=["查询常识库", "找到事实", "返回结果"]
        )

        # Assert
        assert result.task_id == "task_001"
        assert result.answer == "水在100度沸腾"
        assert result.confidence == 0.95
        assert len(result.reasoning_chain) == 3

    def test_result_validation(self):
        """测试结果验证"""
        # Arrange & Act & Assert
        from src.production.domain.reasoning.reasoning_result import ReasoningResult

        # 测试无效置信度
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            ReasoningResult(
                task_id="task_001",
                answer="test",
                confidence=1.5,  # 无效
                reasoning_chain=[]
            )

    def test_result_immutability(self):
        """测试结果不可变性"""
        # Arrange
        from src.production.domain.reasoning.reasoning_result import ReasoningResult

        result = ReasoningResult(
            task_id="task_001",
            answer="test",
            confidence=0.9,
            reasoning_chain=["step1", "step2"]
        )

        # Act & Assert - 尝试修改应该失败
        with pytest.raises(AttributeError):
            result.answer = "modified"

        with pytest.raises(AttributeError):
            result.confidence = 0.5
