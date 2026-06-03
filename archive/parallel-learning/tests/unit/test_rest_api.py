"""
REST API接口层单元测试

TDD方法：先写测试，再写实现
测试驱动设计接口层
"""

import pytest
from datetime import datetime
from typing import Dict, Any


# ============================================================================
# 测试用例 - 先定义接口契约
# ============================================================================

class TestKnowledgeAPI:
    """知识管理API测试"""

    def test_create_knowledge_base(self):
        """测试创建知识库API"""
        # Arrange
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI

        api = KnowledgeAPI()

        request_data = {
            'name': 'test_kb'
        }

        # Act
        response = api.create_knowledge_base(request_data)

        # Assert
        assert response['status'] == 'success'
        assert response['data']['name'] == 'test_kb'

    def test_add_fact(self):
        """测试添加事实API"""
        # Arrange
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI

        api = KnowledgeAPI()

        # 先创建知识库
        api.create_knowledge_base({'name': 'test_kb'})

        request_data = {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        }

        # Act
        response = api.add_fact('test_kb', request_data)

        # Assert
        assert response['status'] == 'success'
        assert response['data']['fact_id'] == 'fact_001'

    def test_query_facts(self):
        """测试查询事实API"""
        # Arrange
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI

        api = KnowledgeAPI()

        # 先创建知识库和添加事实
        api.create_knowledge_base({'name': 'test_kb'})
        api.add_fact('test_kb', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        # Act
        response = api.query_facts('test_kb', {'subject': '水'})

        # Assert
        assert response['status'] == 'success'
        assert len(response['data']) > 0

    def test_get_statistics(self):
        """测试获取统计信息API"""
        # Arrange
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI

        api = KnowledgeAPI()

        # 先创建知识库
        api.create_knowledge_base({'name': 'test_kb'})

        # Act
        response = api.get_statistics('test_kb')

        # Assert
        assert response['status'] == 'success'
        assert 'total_facts' in response['data']


class TestReasoningAPI:
    """推理API测试"""

    def test_create_session(self):
        """测试创建推理会话API"""
        # Arrange
        from src.production.interfaces.rest.reasoning_api import ReasoningAPI

        api = ReasoningAPI()

        request_data = {
            'session_id': 'session_001'
        }

        # Act
        response = api.create_session(request_data)

        # Assert
        assert response['status'] == 'success'
        assert response['data']['session_id'] == 'session_001'

    def test_add_task(self):
        """测试添加推理任务API"""
        # Arrange
        from src.production.interfaces.rest.reasoning_api import ReasoningAPI

        api = ReasoningAPI()

        # 先创建会话
        api.create_session({'session_id': 'session_001'})

        request_data = {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        }

        # Act
        response = api.add_task('session_001', request_data)

        # Assert
        assert response['status'] == 'success'
        assert response['data']['task_id'] == 'task_001'

    def test_execute_task(self):
        """测试执行推理任务API"""
        # Arrange
        from src.production.interfaces.rest.reasoning_api import ReasoningAPI

        api = ReasoningAPI()

        # 先创建会话和添加任务
        api.create_session({'session_id': 'session_001'})
        api.add_task('session_001', {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        })

        # Act
        response = api.execute_task('session_001', 'task_001')

        # Assert
        assert response['status'] == 'success'
        assert response['data']['status'] == 'completed'
        assert response['data']['confidence'] > 0

    def test_get_reasoning_chain(self):
        """测试获取推理链API"""
        # Arrange
        from src.production.interfaces.rest.reasoning_api import ReasoningAPI

        api = ReasoningAPI()

        # 先创建会话、添加任务、执行任务
        api.create_session({'session_id': 'session_001'})
        api.add_task('session_001', {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        })
        api.execute_task('session_001', 'task_001')

        # Act
        response = api.get_reasoning_chain('session_001', 'task_001')

        # Assert
        assert response['status'] == 'success'
        assert len(response['data']['steps']) > 0


class TestQueryAPI:
    """查询API测试"""

    def test_query_commonsense(self):
        """测试常识查询API"""
        # Arrange
        from src.production.interfaces.rest.query_api import QueryAPI
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        # 先添加知识
        kb_service = KnowledgeApplicationService()
        kb_service.add_fact('test_kb', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        query_api = QueryAPI(kb_service)

        # Act
        response = query_api.query_commonsense('水', 'test_kb')

        # Assert
        assert response['status'] == 'success'
        assert len(response['data']) > 0

    def test_query_with_context(self):
        """测试带上下文的查询API"""
        # Arrange
        from src.production.interfaces.rest.query_api import QueryAPI
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        kb_service = KnowledgeApplicationService()
        kb_service.add_fact('test_kb', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        query_api = QueryAPI(kb_service)

        context = {
            'domain': 'physics',
            'precision': 'high'
        }

        # Act
        response = query_api.query_with_context('水', context, 'test_kb')

        # Assert
        assert response['status'] == 'success'


class TestErrorHandler:
    """错误处理测试"""

    def test_handle_not_found(self):
        """测试处理不存在的资源"""
        # Arrange
        from src.production.interfaces.rest.error_handler import ErrorHandler

        handler = ErrorHandler()

        # Act
        response = handler.handle_not_found('Knowledge base', 'nonexistent_kb')

        # Assert
        assert response['status'] == 'error'
        assert response['error_code'] == 'NOT_FOUND'
        assert 'nonexistent_kb' in response['message']

    def test_handle_validation_error(self):
        """测试处理验证错误"""
        # Arrange
        from src.production.interfaces.rest.error_handler import ErrorHandler

        handler = ErrorHandler()

        # Act
        response = handler.handle_validation_error('Confidence must be between 0 and 1')

        # Assert
        assert response['status'] == 'error'
        assert response['error_code'] == 'VALIDATION_ERROR'

    def test_handle_internal_error(self):
        """测试处理内部错误"""
        # Arrange
        from src.production.interfaces.rest.error_handler import ErrorHandler

        handler = ErrorHandler()

        # Act
        response = handler.handle_internal_error('Database connection failed')

        # Assert
        assert response['status'] == 'error'
        assert response['error_code'] == 'INTERNAL_ERROR'
