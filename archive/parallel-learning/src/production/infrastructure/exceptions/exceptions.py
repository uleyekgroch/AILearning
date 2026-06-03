"""
异常处理系统 - 生产级实现

定义自定义异常类，提供统一的错误处理
"""

from typing import Optional, Dict, Any


class BaseException(Exception):
    """基础异常类"""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化异常

        Args:
            message: 错误消息
            error_code: 错误代码
            status_code: HTTP状态码
            details: 错误详情
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status': 'error',
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
        }


class NotFoundException(BaseException):
    """资源不存在异常"""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        message: Optional[str] = None
    ):
        """
        初始化异常

        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            message: 错误消息
        """
        if message is None:
            message = f"{resource_type} with id '{resource_id}' not found"

        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details={
                'resource_type': resource_type,
                'resource_id': resource_id,
            }
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ValidationException(BaseException):
    """验证异常"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None
    ):
        """
        初始化异常

        Args:
            message: 错误消息
            field: 字段名
            value: 字段值
        """
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={
                'field': field,
                'value': value,
            }
        )
        self.field = field
        self.value = value


class DuplicateException(BaseException):
    """重复资源异常"""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        message: Optional[str] = None
    ):
        """
        初始化异常

        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            message: 错误消息
        """
        if message is None:
            message = f"{resource_type} with id '{resource_id}' already exists"

        super().__init__(
            message=message,
            error_code="DUPLICATE",
            status_code=409,
            details={
                'resource_type': resource_type,
                'resource_id': resource_id,
            }
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class UnauthorizedException(BaseException):
    """未授权异常"""

    def __init__(self, message: str = "Unauthorized"):
        """
        初始化异常

        Args:
            message: 错误消息
        """
        super().__init__(
            message=message,
            error_code="UNAUTHORIZED",
            status_code=401
        )


class ForbiddenException(BaseException):
    """禁止访问异常"""

    def __init__(self, message: str = "Forbidden"):
        """
        初始化异常

        Args:
            message: 错误消息
        """
        super().__init__(
            message=message,
            error_code="FORBIDDEN",
            status_code=403
        )


class InternalException(BaseException):
    """内部错误异常"""

    def __init__(self, message: str = "Internal server error"):
        """
        初始化异常

        Args:
            message: 错误消息
        """
        super().__init__(
            message=message,
            error_code="INTERNAL_ERROR",
            status_code=500
        )


class DatabaseException(BaseException):
    """数据库异常"""

    def __init__(self, message: str, operation: Optional[str] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            operation: 操作类型
        """
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details={
                'operation': operation,
            }
        )
        self.operation = operation


class CacheException(BaseException):
    """缓存异常"""

    def __init__(self, message: str, key: Optional[str] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            key: 缓存键
        """
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details={
                'key': key,
            }
        )
        self.key = key


class ServiceException(BaseException):
    """服务异常"""

    def __init__(self, service_name: str, message: str):
        """
        初始化异常

        Args:
            service_name: 服务名称
            message: 错误消息
        """
        super().__init__(
            message=f"Service '{service_name}' error: {message}",
            error_code="SERVICE_ERROR",
            status_code=503,
            details={
                'service_name': service_name,
            }
        )
        self.service_name = service_name


class TimeoutException(BaseException):
    """超时异常"""

    def __init__(self, operation: str, timeout: int):
        """
        初始化异常

        Args:
            operation: 操作类型
            timeout: 超时时间（秒）
        """
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout} seconds",
            error_code="TIMEOUT",
            status_code=504,
            details={
                'operation': operation,
                'timeout': timeout,
            }
        )
        self.operation = operation
        self.timeout = timeout


class RateLimitException(BaseException):
    """速率限制异常"""

    def __init__(self, limit: int, window: int):
        """
        初始化异常

        Args:
            limit: 限制次数
            window: 时间窗口（秒）
        """
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window} seconds",
            error_code="RATE_LIMIT",
            status_code=429,
            details={
                'limit': limit,
                'window': window,
            }
        )
        self.limit = limit
        self.window = window
