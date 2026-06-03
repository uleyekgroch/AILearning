"""
基础设施层单元测试

TDD方法：先写测试，再写实现
测试驱动设计基础设施层
"""

import pytest
from datetime import datetime
from typing import List, Optional, Dict, Any


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestKnowledgeRepository:
    """知识库仓储测试"""

    def test_save_and_find_knowledge_base(self):
        """测试保存和查找知识库"""
        # Arrange
        from src.production.infrastructure.persistence.knowledge_repository import InMemoryKnowledgeRepository
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase

        repository = InMemoryKnowledgeRepository()
        kb = KnowledgeBase(name="test_kb")

        # Act
        repository.save(kb)
        found = repository.find_by_name("test_kb")

        # Assert
        assert found is not None
        assert found.name == "test_kb"

    def test_find_nonexistent_knowledge_base(self):
        """测试查找不存在的知识库"""
        # Arrange
        from src.production.infrastructure.persistence.knowledge_repository import InMemoryKnowledgeRepository

        repository = InMemoryKnowledgeRepository()

        # Act
        found = repository.find_by_name("nonexistent")

        # Assert
        assert found is None

    def test_save_and_find_fact(self):
        """测试保存和查找事实"""
        # Arrange
        from src.production.infrastructure.persistence.knowledge_repository import InMemoryKnowledgeRepository
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        repository = InMemoryKnowledgeRepository()
        kb = KnowledgeBase(name="test_kb")

        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )
        kb.add_fact(fact)

        # Act
        repository.save(kb)
        found_kb = repository.find_by_name("test_kb")
        found_fact = found_kb.facts.get("fact_001")

        # Assert
        assert found_fact is not None
        assert found_fact.statement == "水在100度沸腾"

    def test_delete_knowledge_base(self):
        """测试删除知识库"""
        # Arrange
        from src.production.infrastructure.persistence.knowledge_repository import InMemoryKnowledgeRepository
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase

        repository = InMemoryKnowledgeRepository()
        kb = KnowledgeBase(name="test_kb")
        repository.save(kb)

        # Act
        repository.delete("test_kb")
        found = repository.find_by_name("test_kb")

        # Assert
        assert found is None

    def test_list_all_knowledge_bases(self):
        """测试列出所有知识库"""
        # Arrange
        from src.production.infrastructure.persistence.knowledge_repository import InMemoryKnowledgeRepository
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase

        repository = InMemoryKnowledgeRepository()

        kb1 = KnowledgeBase(name="kb_001")
        kb2 = KnowledgeBase(name="kb_002")

        repository.save(kb1)
        repository.save(kb2)

        # Act
        all_kbs = repository.find_all()

        # Assert
        assert len(all_kbs) == 2


class TestReasoningRepository:
    """推理会话仓储测试"""

    def test_save_and_find_session(self):
        """测试保存和查找推理会话"""
        # Arrange
        from src.production.infrastructure.persistence.reasoning_repository import InMemoryReasoningRepository
        from src.production.domain.reasoning.reasoning_session import ReasoningSession

        repository = InMemoryReasoningRepository()
        session = ReasoningSession(session_id="session_001")

        # Act
        repository.save(session)
        found = repository.find_by_id("session_001")

        # Assert
        assert found is not None
        assert found.session_id == "session_001"

    def test_find_nonexistent_session(self):
        """测试查找不存在的推理会话"""
        # Arrange
        from src.production.infrastructure.persistence.reasoning_repository import InMemoryReasoningRepository

        repository = InMemoryReasoningRepository()

        # Act
        found = repository.find_by_id("nonexistent")

        # Assert
        assert found is None

    def test_save_session_with_tasks(self):
        """测试保存带有任务的推理会话"""
        # Arrange
        from src.production.infrastructure.persistence.reasoning_repository import InMemoryReasoningRepository
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        from src.production.domain.reasoning.reasoning_task import ReasoningTask

        repository = InMemoryReasoningRepository()
        session = ReasoningSession(session_id="session_001")

        task = ReasoningTask(
            task_id="task_001",
            query="水在多少度沸腾",
            reasoning_type="commonsense"
        )
        session.add_task(task)

        # Act
        repository.save(session)
        found = repository.find_by_id("session_001")

        # Assert
        assert found is not None
        assert found.task_count == 1
        assert found.has_task("task_001")

    def test_delete_session(self):
        """测试删除推理会话"""
        # Arrange
        from src.production.infrastructure.persistence.reasoning_repository import InMemoryReasoningRepository
        from src.production.domain.reasoning.reasoning_session import ReasoningSession

        repository = InMemoryReasoningRepository()
        session = ReasoningSession(session_id="session_001")
        repository.save(session)

        # Act
        repository.delete("session_001")
        found = repository.find_by_id("session_001")

        # Assert
        assert found is None

    def test_list_all_sessions(self):
        """测试列出所有推理会话"""
        # Arrange
        from src.production.infrastructure.persistence.reasoning_repository import InMemoryReasoningRepository
        from src.production.domain.reasoning.reasoning_session import ReasoningSession

        repository = InMemoryReasoningRepository()

        session1 = ReasoningSession(session_id="session_001")
        session2 = ReasoningSession(session_id="session_002")

        repository.save(session1)
        repository.save(session2)

        # Act
        all_sessions = repository.find_all()

        # Assert
        assert len(all_sessions) == 2


class TestFactIndex:
    """事实索引测试"""

    def test_index_and_search(self):
        """测试索引和搜索"""
        # Arrange
        from src.production.infrastructure.indexing.fact_index import InMemoryFactIndex

        index = InMemoryFactIndex()

        # Act
        index.add("fact_001", {"subject": "水", "predicate": "沸点", "object": "100度"})
        index.add("fact_002", {"subject": "人", "predicate": "需要", "object": "水"})

        results = index.search_by_subject("水")

        # Assert
        assert len(results) == 1
        assert "fact_001" in results

    def test_prefix_search(self):
        """测试前缀搜索"""
        # Arrange
        from src.production.infrastructure.indexing.fact_index import InMemoryFactIndex

        index = InMemoryFactIndex()

        index.add("fact_001", {"subject": "水", "predicate": "沸点", "object": "100度"})
        index.add("fact_002", {"subject": "水蒸气", "predicate": "是", "object": "气体"})
        index.add("fact_003", {"subject": "火", "predicate": "温度", "object": "高"})

        # Act
        results = index.search_by_prefix("水")

        # Assert
        assert len(results) == 2  # 水、水蒸气

    def test_remove_from_index(self):
        """测试从索引中移除"""
        # Arrange
        from src.production.infrastructure.indexing.fact_index import InMemoryFactIndex

        index = InMemoryFactIndex()

        index.add("fact_001", {"subject": "水", "predicate": "沸点", "object": "100度"})
        index.add("fact_002", {"subject": "人", "predicate": "需要", "object": "水"})

        # Act
        index.remove("fact_001")
        results = index.search_by_subject("水")

        # Assert
        assert len(results) == 0

    def test_get_statistics(self):
        """测试获取索引统计"""
        # Arrange
        from src.production.infrastructure.indexing.fact_index import InMemoryFactIndex

        index = InMemoryFactIndex()

        index.add("fact_001", {"subject": "水", "predicate": "沸点", "object": "100度"})
        index.add("fact_002", {"subject": "人", "predicate": "需要", "object": "水"})

        # Act
        stats = index.get_statistics()

        # Assert
        assert stats['total_entries'] == 2
        assert stats['total_concepts'] > 0
