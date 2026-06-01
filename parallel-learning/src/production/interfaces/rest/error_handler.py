"""
错误处理器

提供统一的错误处理
"""

from typing import Dict, Any


class ErrorHandler:
    """错误处理器

    职责：
    - 处理各种错误情况
    - 格式化错误响应
    - 提供统一的错误接口
    """

    def handle_not_found(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """处理资源不存在错误

        Args:
            resource_type: 资源类型
            resource_id: 资源ID

        Returns:
            错误响应
        """
        return {
            'status': 'error',
            'error_code': 'NOT_FOUND',
            'message': f'{resource_type} with id {resource_id} not found',
            'details': {
                'resource_type': resource_type,
                'resource_id': resource_id
            }
        }

    def handle_validation_error(self, message: str) -> Dict[str, Any]:
        """处理验证错误

        Args:
            message: 错误消息

        Returns:
            错误响应
        """
        return {
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': message,
            'details': {}
        }

    def handle_internal_error(self, message: str) -> Dict[str, Any]:
        """处理内部错误

        Args:
            message: 错误消息

        Returns:
            错误响应
        """
        return {
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': message,
            'details': {}
        }

    def handle_conflict(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """处理资源冲突错误

        Args:
            resource_type: 资源类型
            resource_id: 资源ID

        Returns:
            错误响应
        """
        return {
            'status': 'error',
            'error_code': 'CONFLICT',
            'message': f'{resource_type} with id {resource_id} already exists',
            'details': {
                'resource_type': resource_type,
                'resource_id': resource_id
            }
        }

    def handle_unauthorized(self, message: str = "Unauthorized") -> Dict[str, Any]:
        """处理未授权错误

        Args:
            message: 错误消息

        Returns:
            错误响应
        """
        return {
            'status': 'error',
            'error_code': 'UNAUTHORIZED',
            'message': message,
            'details': {}
        }

    def handle_forbidden(self, message: str = "Forbidden") -> Dict[str, Any]:
        """处理禁止访问错误

        Args:
            message: 错误消息

        Returns:
            错误响应
        """
        return {
            'status': 'error',
            'error_code': 'FORBIDDEN',
            'message': message,
            'details': {}
        }
