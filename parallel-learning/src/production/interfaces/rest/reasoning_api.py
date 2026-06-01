"""
推理REST API

提供推理会话的REST接口
"""

from typing import Dict, List, Any, Optional

from src.production.application.services.reasoning_service import ReasoningApplicationService
from .error_handler import ErrorHandler


class ReasoningAPI:
    """推理REST API

    职责：
    - 处理HTTP请求
    - 输入验证
    - 调用应用服务
    - 格式化响应

    Attributes:
        service: 推理应用服务
        error_handler: 错误处理器
    """

    def __init__(self, service: ReasoningApplicationService = None):
        """初始化推理API

        Args:
            service: 推理应用服务（可选）
        """
        self.service = service or ReasoningApplicationService()
        self.error_handler = ErrorHandler()

    def create_session(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建推理会话

        Args:
            request_data: 请求数据，包含session_id字段

        Returns:
            响应数据
        """
        try:
            # 输入验证
            session_id = request_data.get('session_id')
            if not session_id:
                return self.error_handler.handle_validation_error(
                    'Session ID is required'
                )

            # 调用应用服务
            session = self.service.create_session(session_id)

            # 返回成功响应
            return {
                'status': 'success',
                'data': {
                    'session_id': session.session_id,
                    'status': session.status,
                    'created_at': session.created_at.isoformat()
                }
            }

        except ValueError as e:
            return self.error_handler.handle_validation_error(str(e))
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def add_task(self, session_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加推理任务

        Args:
            session_id: 会话ID
            request_data: 请求数据

        Returns:
            响应数据
        """
        try:
            # 输入验证
            required_fields = ['task_id', 'query', 'reasoning_type']
            for field in required_fields:
                if field not in request_data:
                    return self.error_handler.handle_validation_error(
                        f'{field} is required'
                    )

            # 调用应用服务
            task = self.service.add_task(session_id, request_data)

            # 返回成功响应
            return {
                'status': 'success',
                'data': {
                    'task_id': task.task_id,
                    'query': task.query,
                    'reasoning_type': task.reasoning_type,
                    'status': task.status,
                    'created_at': task.created_at.isoformat()
                }
            }

        except ValueError as e:
            return self.error_handler.handle_validation_error(str(e))
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def execute_task(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """执行推理任务

        Args:
            session_id: 会话ID
            task_id: 任务ID

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            result = self.service.execute_task(session_id, task_id)

            # 返回成功响应
            return {
                'status': 'success',
                'data': {
                    'task_id': result.task_id,
                    'answer': result.answer,
                    'confidence': result.confidence,
                    'status': result.status,
                    'reasoning_chain': result.reasoning_chain,
                    'created_at': result.created_at.isoformat()
                }
            }

        except ValueError as e:
            return self.error_handler.handle_not_found('Task', task_id)
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def get_reasoning_chain(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """获取推理链

        Args:
            session_id: 会话ID
            task_id: 任务ID

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            chain = self.service.get_reasoning_chain(session_id, task_id)

            if chain is None:
                return self.error_handler.handle_not_found('Reasoning chain', task_id)

            # 返回成功响应
            return {
                'status': 'success',
                'data': {
                    'task_id': chain.task_id,
                    'steps': chain.steps,
                    'step_count': chain.step_count,
                    'created_at': chain.created_at.isoformat()
                }
            }

        except ValueError as e:
            return self.error_handler.handle_not_found('Session', session_id)
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计信息

        Args:
            session_id: 会话ID

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            stats = self.service.get_session_statistics(session_id)

            return {
                'status': 'success',
                'data': stats
            }

        except ValueError as e:
            return self.error_handler.handle_not_found('Session', session_id)
        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def list_sessions(self) -> Dict[str, Any]:
        """列出所有会话

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            session_ids = self.service.list_sessions()

            return {
                'status': 'success',
                'data': session_ids,
                'total': len(session_ids)
            }

        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """删除会话

        Args:
            session_id: 会话ID

        Returns:
            响应数据
        """
        try:
            # 调用应用服务
            success = self.service.delete_session(session_id)

            if success:
                return {
                    'status': 'success',
                    'message': f'Session {session_id} deleted successfully'
                }
            else:
                return self.error_handler.handle_not_found('Session', session_id)

        except Exception as e:
            return self.error_handler.handle_internal_error(str(e))
