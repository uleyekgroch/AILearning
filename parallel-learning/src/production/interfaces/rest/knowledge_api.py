"""
知识管理REST API

提供知识库的REST接口
"""

from typing import Dict, List, Any, Optional

from src.production.application.services.knowledge_service import KnowledgeApplicationService
from .error_handler import ErrorHandler


class KnowledgeAPI:
    """知识管理REST API

    职责：
    - 处理HTTP请求
    - 输入验证
    - 调用应用服务
    - 格式化响应

    Attributes:
        service: 知识管理应用服务
        error_handler: 错误处理器
    """

    def __init__(self, service: KnowledgeApplicationService = None):
        """初始化知识管理API

        Args:
            service: 知识管理应用服务（可选）
        """
        self.service = service or KnowledgeApplicationService()
        self.error_handler = ErrorHandler()

    def create_knowledge_base(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建知识库

        Args:
            request_data: 请求数据，包含name字段

        Returns:
            响应数据
        """
        try:
            # 输入验证
            name = request_data.get('name')
            if not name:
                return self.error_handler.handle_validation_error(
                    'Knowledge base name is required'
                )

            # 调用应用服务
            kb = self.service.create_knowledge_base(name)

            # 返回成功响应
            return {
                'status': 'success',
                'data': {
                    'name': kb.name,
                    'version': kb.version,
                    'created_at': kb.created_at.isoformat()
                }
            }

        except ValueError as e:
            return self.error_handler.handle_validation_error(str(e))
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def add_fact(self, kb_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加事实

        Args:
            kb_name: 知识库名称
            request_data: 请求数据

        Returns:
            响应数据
        """
        try:
            # 输入验证
            required_fields = ['fact_id', 'statement', 'subject', 'predicate', 'object', 'confidence']
            for field in required_fields:
                if field not in request_data:
                    return self.error_handler.handle_validation_error(
                        f'{field} is required'
                    )

            # 调用应用服务
            fact = self.service.add_fact(kb_name, request_data)

            # 返回成功响应
            return {
                'status': 'success',
                'data': {
                    'fact_id': fact.fact_id,
                    'statement': fact.statement,
                    'subject': fact.subject,
                    'predicate': fact.predicate,
                    'object': fact.object,
                    'confidence': fact.confidence,
                    'created_at': fact.created_at.isoformat()
                }
            }

        except ValueError as e:
            return self.error_handler.handle_validation_error(str(e))
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def query_facts(self, kb_name: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """查询事实

        Args:
            kb_name: 知识库名称
            query_params: 查询参数

        Returns:
            响应数据
        """
        try:
            # 获取查询参数
            subject = query_params.get('subject')
            predicate = query_params.get('predicate')
            object_ = query_params.get('object')

            # 调用应用服务
            facts = self.service.query_by_triple(kb_name, subject, predicate, object_)

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

    def get_statistics(self, kb_name: str) -> Dict[str, Any]:
        """获取统计信息

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

    def list_knowledge_bases(self) -> Dict[str, Any]:
        """列出所有知识库

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            kb_names = self.service.list_knowledge_bases()

            return {
                'status': 'success',
                'data': kb_names,
                'total': len(kb_names)
            }

        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def delete_knowledge_base(self, kb_name: str) -> Dict[str, Any]:
        """删除知识库

        Args:
            kb_name: 知识库名称

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            success = self.service.delete_knowledge_base(kb_name)

            if success:
                return {
                    'status': 'success',
                    'message': f'Knowledge base {kb_name} deleted successfully'
                }
            else:
                return self.error_handler.handle_not_found('Knowledge base', kb_name)

        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))
