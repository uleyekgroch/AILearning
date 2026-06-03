"""
查询REST API

提供查询的REST接口
"""

from typing import Dict, List, Any, Optional

from src.production.application.services.query_service import QueryApplicationService
from src.production.application.services.knowledge_service import KnowledgeApplicationService
from .error_handler import ErrorHandler


class QueryAPI:
    """查询REST API

    职责：
    - 处理HTTP请求
    - 输入验证
    - 调用应用服务
    - 格式化响应

    Attributes:
        service: 查询应用服务
        error_handler: 错误处理器
    """

    def __init__(self, kb_service: KnowledgeApplicationService = None):
        """初始化查询API

        Args:
            kb_service: 知识管理应用服务（可选）
        """
        self.kb_service = kb_service or KnowledgeApplicationService()
        self.service = QueryApplicationService(self.kb_service)
        self.error_handler = ErrorHandler()

    def query_commonsense(self, query: str, kb_name: str = 'default') -> Dict[str, Any]:
        """常识查询

        Args:
            query: 查询文本
            kb_name: 知识库名称

        Returns:
            响应数据
        """
        try:
            # 输入验证
            if not query:
                return self.error_handler.handle_validation_error(
                    'Query is required'
                )

            # 调用应用服务
            facts = self.service.query_commonsense(query, kb_name)

            # 格式化响应
            facts_data = []
            for fact in facts:
                facts_data.append({
                    'fact_id': fact.fact_id,
                    'statement': fact.statement,
                    'subject': fact.subject,
                    'predicate': fact.predicate,
                    'object': fact.object,
                    'confidence': fact.confidence
                })

            return {
                'status': 'success',
                'data': facts_data,
                'total': len(facts_data)
            }

        except ValueError as e:
            return self.error_handler.handle_not_found('Knowledge base', kb_name)
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def query_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        kb_name: str = 'default'
    ) -> Dict[str, Any]:
        """带上下文的查询

        Args:
            query: 查询文本
            context: 上下文信息
            kb_name: 知识库名称

        Returns:
            响应数据
        """
        try:
            # 输入验证
            if not query:
                return self.error_handler.handle_validation_error(
                    'Query is required'
                )

            # 调用应用服务
            facts = self.service.query_with_context(query, context, kb_name)

            # 格式化响应
            facts_data = []
            for fact in facts:
                facts_data.append({
                    'fact_id': fact.fact_id,
                    'statement': fact.statement,
                    'subject': fact.subject,
                    'predicate': fact.predicate,
                    'object': fact.object,
                    'confidence': fact.confidence
                })

            return {
                'status': 'success',
                'data': facts_data,
                'total': len(facts_data)
            }

        except ValueError as e:
            return self.error_handler.handle_not_found('Knowledge base', kb_name)
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def query_by_triple_pattern(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        kb_name: str = 'default'
    ) -> Dict[str, Any]:
        """按三元组模式查询

        Args:
            subject: 主题（可选）
            predicate: 谓语（可选）
            object_: 宾语（可选）
            kb_name: 知识库名称

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            facts = self.service.query_by_triple_pattern(
                subject, predicate, object_, kb_name
            )

            # 格式化响应
            facts_data = []
            for fact in facts:
                facts_data.append({
                    'fact_id': fact.fact_id,
                    'statement': fact.statement,
                    'subject': fact.subject,
                    'predicate': fact.predicate,
                    'object': fact.object,
                    'confidence': fact.confidence
                })

            return {
                'status': 'success',
                'data': facts_data,
                'total': len(facts_data)
            }

        except ValueError as e:
            return self.error_handler.handle_not_found('Knowledge base', kb_name)
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def get_statistics(self, kb_name: str = 'default') -> Dict[str, Any]:
        """获取查询统计信息

        Args:
            kb_name: 知识库名称

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            stats = self.service.get_statistics(kb_name)

            return {
                'status': 'success',
                'data': stats
            }

        except ValueError as e:
            return self.error_handler.handle_not_found('Knowledge base', kb_name)
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))
