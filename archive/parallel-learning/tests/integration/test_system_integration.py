"""
系统集成测试

测试系统各模块之间的集成
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any


# ============================================================================
# 集成测试用例
# ============================================================================

class TestDomainToApplicationIntegration:
    """领域层与应用层集成测试"""

    def test_knowledge_domain_to_application(self):
        """测试知识领域与应用服务集成"""
        # Arrange
        from src.production.domain.knowledge.knowledge_base import KnowledgeBase
        from src.production.domain.knowledge.commonsense_fact import CommonsenseFact
        from src.production.application.services.knowledge_service import KnowledgeApplicationService

        service = KnowledgeApplicationService()

        # 创建知识库
        kb = service.create_knowledge_base("test_kb")

        # 添加事实
        fact_data = {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        }

        # Act
        result = service.add_fact("test_kb", fact_data)

        # Assert
        assert result is not None
        assert result.fact_id == 'fact_001'

        # 查询事实
        facts = service.query_by_subject("test_kb", "水")
        assert len(facts) == 1
        assert facts[0].statement == '水在100度沸腾'

    def test_reasoning_domain_to_application(self):
        """测试推理领域与应用服务集成"""
        # Arrange
        from src.production.domain.reasoning.reasoning_session import ReasoningSession
        from src.production.domain.reasoning.reasoning_task import ReasoningTask
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        service = ReasoningApplicationService()

        # 创建会话
        session = service.create_session("session_001")

        # 添加任务
        task_data = {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        }

        # Act
        task = service.add_task("session_001", task_data)

        # Assert
        assert task is not None
        assert task.task_id == 'task_001'

        # 执行任务
        result = service.execute_task("session_001", "task_001")
        assert result is not None
        assert result.status == 'completed'

    def test_multimodal_domain_to_application(self):
        """测试多模态领域与应用服务集成"""
        # Arrange
        from src.production.domain.multimodal.modality import Modality, ModalityType
        from src.production.domain.multimodal.multimodal_input import MultimodalInput
        from src.production.application.services.multimodal_service import MultimodalApplicationService

        service = MultimodalApplicationService()

        # 创建模态
        text_modality = Modality(
            modality_id="text_001",
            modality_type=ModalityType.TEXT.value,
            content="水在100度沸腾",
            metadata={}
        )

        # Act
        result = service.process_input([text_modality])

        # Assert
        assert result is not None
        assert result.confidence > 0

    def test_embodied_domain_to_application(self):
        """测试具身领域与应用服务集成"""
        # Arrange
        from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType
        from src.production.domain.embodied.embodied_state import EmbodiedState
        from src.production.application.services.embodied_service import EmbodiedApplicationService

        service = EmbodiedApplicationService()

        # 创建感觉输入
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

    def test_continual_learning_domain_to_application(self):
        """测试持续学习领域与应用服务集成"""
        # Arrange
        from src.production.domain.continual_learning.learning_experience import LearningExperience
        from src.production.domain.continual_learning.knowledge_update import KnowledgeUpdate, UpdateType
        from src.production.application.services.continual_learning_service import ContinualLearningService

        service = ContinualLearningService()

        # 创建学习经验
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


class TestApplicationToInterfaceIntegration:
    """应用层与接口层集成测试"""

    def test_knowledge_application_to_api(self):
        """测试知识应用服务与API集成"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI

        service = KnowledgeApplicationService()
        api = KnowledgeAPI(service)

        # 创建知识库
        api.create_knowledge_base({'name': 'test_kb'})

        # 添加事实
        fact_data = {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        }

        # Act
        response = api.add_fact('test_kb', fact_data)

        # Assert
        assert response['status'] == 'success'
        assert response['data']['fact_id'] == 'fact_001'

        # 查询事实
        query_response = api.query_facts('test_kb', {'subject': '水'})
        assert query_response['status'] == 'success'
        assert len(query_response['data']) == 1

    def test_reasoning_application_to_api(self):
        """测试推理应用服务与API集成"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService
        from src.production.interfaces.rest.reasoning_api import ReasoningAPI

        service = ReasoningApplicationService()
        api = ReasoningAPI(service)

        # 创建会话
        api.create_session({'session_id': 'session_001'})

        # 添加任务
        task_data = {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        }

        # Act
        response = api.add_task('session_001', task_data)

        # Assert
        assert response['status'] == 'success'
        assert response['data']['task_id'] == 'task_001'

        # 执行任务
        execute_response = api.execute_task('session_001', 'task_001')
        assert execute_response['status'] == 'success'
        assert execute_response['data']['status'] == 'completed'

    def test_query_application_to_api(self):
        """测试查询应用服务与API集成"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.application.services.query_service import QueryApplicationService
        from src.production.interfaces.rest.query_api import QueryAPI

        kb_service = KnowledgeApplicationService()
        query_service = QueryApplicationService(kb_service)
        api = QueryAPI(kb_service)

        # 添加知识
        kb_service.add_fact('test_kb', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        # Act
        response = api.query_commonsense('水', 'test_kb')

        # Assert
        assert response['status'] == 'success'
        assert len(response['data']) > 0


class TestCrossModuleIntegration:
    """跨模块集成测试"""

    def test_knowledge_to_reasoning_integration(self):
        """测试知识库与推理集成"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.application.services.reasoning_service import ReasoningApplicationService

        kb_service = KnowledgeApplicationService()
        reasoning_service = ReasoningApplicationService()

        # 添加知识
        kb_service.add_fact('kb_001', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        # 创建推理会话
        reasoning_service.create_session('session_001')
        reasoning_service.add_task('session_001', {
            'task_id': 'task_001',
            'query': '水的沸点',
            'reasoning_type': 'commonsense'
        })

        # Act
        result = reasoning_service.execute_task('session_001', 'task_001')

        # Assert
        assert result is not None
        assert result.status == 'completed'

    def test_knowledge_to_query_integration(self):
        """测试知识库与查询集成"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.application.services.query_service import QueryApplicationService

        kb_service = KnowledgeApplicationService()
        query_service = QueryApplicationService(kb_service)

        # 添加知识
        kb_service.add_fact('kb_001', {
            'fact_id': 'fact_001',
            'statement': '水在100度沸腾',
            'subject': '水',
            'predicate': '沸点',
            'object': '100度',
            'confidence': 1.0
        })

        kb_service.add_fact('kb_001', {
            'fact_id': 'fact_002',
            'statement': '人需要水',
            'subject': '人',
            'predicate': '需要',
            'object': '水',
            'confidence': 1.0
        })

        # Act
        results = query_service.query_commonsense('水', 'kb_001')

        # Assert
        assert len(results) > 0

    def test_reasoning_to_continual_learning_integration(self):
        """测试推理与持续学习集成"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService
        from src.production.application.services.continual_learning_service import ContinualLearningService
        from src.production.domain.continual_learning.learning_experience import LearningExperience

        reasoning_service = ReasoningApplicationService()
        learning_service = ContinualLearningService()

        # 创建推理会话
        reasoning_service.create_session('session_001')
        reasoning_service.add_task('session_001', {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        })

        # 执行推理
        result = reasoning_service.execute_task('session_001', 'task_001')

        # 创建学习经验
        experience = LearningExperience(
            experience_id="exp_001",
            input_data={"query": "水在多少度沸腾"},
            output_data={"answer": result.answer},
            feedback={"correct": True, "confidence": result.confidence},
            timestamp=datetime.now()
        )

        # Act
        learning_result = learning_service.process_learning_experience(experience)

        # Assert
        assert learning_result is not None
        assert learning_result.success == True


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_complete_knowledge_workflow(self):
        """测试完整知识管理工作流"""
        # Arrange
        from src.production.application.services.knowledge_service import KnowledgeApplicationService
        from src.production.application.services.query_service import QueryApplicationService
        from src.production.interfaces.rest.knowledge_api import KnowledgeAPI
        from src.production.interfaces.rest.query_api import QueryAPI

        # 创建服务
        kb_service = KnowledgeApplicationService()
        query_service = QueryApplicationService(kb_service)
        kb_api = KnowledgeAPI(kb_service)
        query_api = QueryAPI(kb_service)

        # 1. 创建知识库
        kb_api.create_knowledge_base({'name': 'physics_kb'})

        # 2. 添加知识
        facts = [
            {
                'fact_id': 'fact_001',
                'statement': '水在100度沸腾',
                'subject': '水',
                'predicate': '沸点',
                'object': '100度',
                'confidence': 1.0
            },
            {
                'fact_id': 'fact_002',
                'statement': '冰在0度融化',
                'subject': '冰',
                'predicate': '融化点',
                'object': '0度',
                'confidence': 1.0
            },
        ]

        for fact in facts:
            kb_api.add_fact('physics_kb', fact)

        # 3. 查询知识
        query_response = query_api.query_commonsense('水', 'physics_kb')

        # 4. 验证结果
        assert query_response['status'] == 'success'
        assert len(query_response['data']) > 0

        # 5. 获取统计信息
        stats_response = kb_api.get_statistics('physics_kb')
        assert stats_response['status'] == 'success'
        assert stats_response['data']['total_facts'] == 2

    def test_complete_reasoning_workflow(self):
        """测试完整推理工作流"""
        # Arrange
        from src.production.application.services.reasoning_service import ReasoningApplicationService
        from src.production.interfaces.rest.reasoning_api import ReasoningAPI

        # 创建服务
        reasoning_service = ReasoningApplicationService()
        reasoning_api = ReasoningAPI(reasoning_service)

        # 1. 创建会话
        reasoning_api.create_session({'session_id': 'session_001'})

        # 2. 添加任务
        reasoning_api.add_task('session_001', {
            'task_id': 'task_001',
            'query': '水在多少度沸腾',
            'reasoning_type': 'commonsense'
        })

        # 3. 执行任务
        execute_response = reasoning_api.execute_task('session_001', 'task_001')

        # 4. 验证结果
        assert execute_response['status'] == 'success'
        assert execute_response['data']['status'] == 'completed'

        # 5. 获取推理链
        chain_response = reasoning_api.get_reasoning_chain('session_001', 'task_001')
        assert chain_response['status'] == 'success'
        assert len(chain_response['data']['steps']) > 0

        # 6. 获取统计信息
        stats_response = reasoning_api.get_session_statistics('session_001')
        assert stats_response['status'] == 'success'
