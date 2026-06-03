"""
应用层服务单元测试

TDD方法：先写测试，再写实现
测试驱动设计应用层
"""

import pytest
from datetime import datetime
from typing import List, Optional


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestKnowledgeApplicationService:
    """知识管理应用服务测试"""

    def test_add_fact_to_knowledge_base(self):
        """测试添加事实到知识库"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

        service = KnowledgeApplicationService()

        fact_data = {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        }

        # Act
        result = service.add_fact('kb_001', fact_data)

        # Assert
        assert result is not None
        assert result.fact_id == 'fact_001'
        assert result.statement == '水在100度沸腾'

    def test_query_facts_by_subject(self):
        """测试按主题查询事实"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        service = KnowledgeApplicationService()

        # 添加测试数据
        service.add_fact('kb_001', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        service.add_fact('kb_001', {
            'fact_id': 'fact_002',
            'statement': '人需要水',
            'subject': '人',
            'predicate': '需要',
            'object': '水',
            'confidence': 1.0
        })

        # Act
        results = service.query_by_subject('kb_001', '水')

        # Assert
        assert len(results) == 1
        assert results[0].fact_id == 'fact_001'

    def test_get_knowledge_base_statistics(self):
        """测试获取知识库统计信息"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        service = KnowledgeApplicationService()

        # 添加测试数据
        service.add_fact('kb_001', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        # Act
        stats = service.get_statistics('kb_001')

        # Assert
        assert stats is not None
        assert stats['total_facts'] == 1
        assert stats['name'] == 'kb_001'


class TestReasoningApplicationService:
    """推理应用服务测试"""

    def test_create_reasoning_session(self):
        """测试创建推理会话"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        service = ReasoningApplicationService()

        # Act
        session = service.create_session('session_001')

        # Assert
        assert session is not None
        assert session.session_id == 'session_001'
        assert session.status == 'created'

    def test_add_reasoning_task(self):
        """测试添加推理任务"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        service = ReasoningApplicationService()
        service.create_session('session_001')

        task_data = {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        }

        # Act
        task = service.add_task('session_001', task_data)

        # Assert
        assert task is not None
        assert task.task_id == 'task_001'
        assert task.query == '水在多少度沸腾'

    def test_execute_reasoning_task(self):
        """测试执行推理任务"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        service = ReasoningApplicationService()
        service.create_session('session_001')

        service.add_task('session_001', {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        })

        # Act
        result = service.execute_task('session_001', 'task_001')

        # Assert
        assert result is not None
        assert result.task_id == 'task_001'
        assert result.status == 'completed'
        assert result.confidence > 0

    def test_get_reasoning_chain(self):
        """测试获取推理链"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        service = ReasoningApplicationService()
        service.create_session('session_001')

        service.add_task('session_001', {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        })

        service.execute_task('session_001', 'task_001')

        # Act
        chain = service.get_reasoning_chain('session_001', 'task_001')

        # Assert
        assert chain is not None
        assert chain.step_count > 0


class TestQueryApplicationService:
    """查询应用服务测试"""

    def test_query_commonsense(self):
        """测试常识查询"""
        # Arrange
        from src.production.application.services.query_service import QueryApplicationService
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        # 先添加知识
        kb_service = KnowledgeApplicationService()
        kb_service.add_fact('kb_001', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        query_service = QueryApplicationService(kb_service)

        # Act - 使用主题查询
        results = query_service.query_by_triple_pattern(
            subject='水', kb_name='kb_001'
        )

        # Assert
        assert len(results) > 0
        assert results[0].confidence > 0

    def test_query_with_context(self):
        """测试带上下文的查询"""
        # Arrange
        from src.production.application.services.query_service import QueryApplicationService
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        kb_service = KnowledgeApplicationService()
        kb_service.add_fact('kb_001', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        query_service = QueryApplicationService(kb_service)

        context = {
            'domain': 'physics',
            'precision': 'high'
        }

        # Act - 使用主题查询
        results = query_service.query_with_context('水', context, kb_name='kb_001')

        # Assert
        assert len(results) > 0
