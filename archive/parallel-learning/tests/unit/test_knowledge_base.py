"""
KnowledgeBase聚合根单元测试

TDD方法：先写测试，再写实现
测试驱动设计领域模型
"""

import pytest
from datetime import datetime
from typing import List, Set, Optional


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestKnowledgeBase:
    """KnowledgeBase聚合根测试"""

    def test_create_knowledge_base(self):
        """测试创建知识库"""
        # Arrange & Act
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(name="test_kb")

        # Assert
        assert kb.name == "test_kb"
        assert kb.version == 1
        assert kb.created_at is not None
        assert kb.updated_at is not None

    def test_add_commonsense_fact(self):
        """测试添加常识事实"""
        # Arrange
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        kb = KnowledgeBase(name="test_kb")
        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )

        # Act
        kb.add_fact(fact)

        # Assert
        assert kb.fact_count == 1
        assert kb.has_fact("fact_001")

    def test_query_facts_by_subject(self):
        """测试按主题查询事实"""
        # Arrange
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        kb = KnowledgeBase(name="test_kb")

        fact1 = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )

        fact2 = CommonsenseFact(
            fact_id="fact_002",
            statement="人需要水",
            subject="人",
            predicate="需要",
            object="水",
            confidence=1.0
        )

        kb.add_fact(fact1)
        kb.add_fact(fact2)

        # Act
        results = kb.query_by_subject("水")

        # Assert
        assert len(results) == 1
        assert results[0].fact_id == "fact_001"

    def test_knowledge_base_versioning(self):
        """测试知识库版本控制"""
        # Arrange
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        kb = KnowledgeBase(name="test_kb")
        initial_version = kb.version

        # Act
        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="test",
            subject="test",
            predicate="test",
            object="test",
            confidence=1.0
        )
        kb.add_fact(fact)

        # Assert
        assert kb.version == initial_version + 1

    def test_knowledge_base_immutability(self):
        """测试知识库不可变性（通过事件溯源）"""
        # Arrange
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        kb = KnowledgeBase(name="test_kb")

        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="test",
            subject="test",
            predicate="test",
            object="test",
            confidence=1.0
        )
        kb.add_fact(fact)

        # Act - 获取事件历史
        events = kb.get_events()

        # Assert
        assert len(events) == 1
        assert events[0].event_type == "FactAdded"
        assert events[0].fact_id == "fact_001"


class TestCommonsenseFact:
    """CommonsenseFact实体测试"""

    def test_create_fact_with_triple(self):
        """测试创建三元组事实"""
        # Arrange & Act
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )

        # Assert
        assert fact.fact_id == "fact_001"
        assert fact.subject == "水"
        assert fact.predicate == "沸点"
        assert fact.object == "100度"
        assert fact.confidence == 1.0

    def test_fact_validation(self):
        """测试事实验证"""
        # Arrange & Act & Assert
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        # 测试无效置信度
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CommonsenseFact(
                fact_id="fact_001",
                statement="test",
                subject="test",
                predicate="test",
                object="test",
                confidence=1.5  # 无效
            )

    def test_fact_to_triple(self):
        """测试转换为三元组"""
        # Arrange
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        fact = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )

        # Act
        triple = fact.to_triple()

        # Assert
        assert triple == ("水", "沸点", "100度")

    def test_fact_equality(self):
        """测试事实相等性"""
        # Arrange
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        fact1 = CommonsenseFact(
            fact_id="fact_001",
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )

        fact2 = CommonsenseFact(
            fact_id="fact_001",  # 相同ID
            statement="different statement",
            subject="different",
            predicate="different",
            object="different",
            confidence=0.5
        )

        fact3 = CommonsenseFact(
            fact_id="fact_002",  # 不同ID
            statement="水在100度沸腾",
            subject="水",
            predicate="沸点",
            object="100度",
            confidence=1.0
        )

        # Act & Assert
        assert fact1 == fact2  # 相同ID = 相等
        assert fact1 != fact3  # 不同ID = 不等
        assert hash(fact1) == hash(fact2)


class TestKnowledgeIndex:
    """KnowledgeIndex值对象测试"""

    def test_index_concept(self):
        """测试索引概念"""
        # Arrange
        from src.production.domain.knowledge.knowledge_index import KnowledgeIndex

        index = KnowledgeIndex()

        # Act
        index.add_mapping("水", "fact_001")
        index.add_mapping("水", "fact_002")
        index.add_mapping("火", "fact_003")

        # Assert
        water_facts = index.get_by_concept("水")
        assert len(water_facts) == 2
        assert "fact_001" in water_facts
        assert "fact_002" in water_facts

    def test_index_prefix_search(self):
        """测试前缀搜索"""
        # Arrange
        from src.production.domain.knowledge.knowledge_index import KnowledgeIndex

        index = KnowledgeIndex()

        index.add_mapping("水", "fact_001")
        index.add_mapping("水蒸气", "fact_002")
        index.add_mapping("水滴", "fact_003")
        index.add_mapping("火", "fact_004")

        # Act
        results = index.search_by_prefix("水")

        # Assert
        assert len(results) == 3  # 水、水蒸气、水滴

    def test_index_statistics(self):
        """测试索引统计"""
        # Arrange
        from src.production.domain.knowledge.knowledge_index import KnowledgeIndex

        index = KnowledgeIndex()

        index.add_mapping("水", "fact_001")
        index.add_mapping("火", "fact_002")

        # Act
        stats = index.get_statistics()

        # Assert
        assert stats['total_concepts'] == 2
        assert stats['total_mappings'] == 2
