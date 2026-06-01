"""
CircuitBreaker熔断器

负责服务熔断和降级
"""

from typing import Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """熔断器

    职责：
    - 故障检测
    - 熔断保护
    - 自动恢复

    Attributes:
        service_name: 服务名称
        failure_threshold: 失败阈值
        recovery_timeout: 恢复超时（秒）
        state: 当前状态
        failure_count: 失败计数
        last_failure_time: 最后失败时间
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30
    ):
        """初始化熔断器

        Args:
            service_name: 服务名称
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时（秒）
        """
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """调用函数

        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数返回值

        Raises:
            Exception: 如果熔断器打开或调用失败
        """
        # 检查熔断器状态
        if self.state == CircuitState.OPEN:
            # 检查是否可以半开
            if self._should_try_recovery():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is open")

        try:
            # 调用函数
            result = func(*args, **kwargs)

            # 调用成功
            self._on_success()

            return result

        except Exception as e:
            # 调用失败
            self._on_failure()

            raise

    def _on_success(self) -> None:
        """调用成功处理"""
        if self.state == CircuitState.HALF_OPEN:
            # 半开状态成功，关闭熔断器
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            # 关闭状态成功，重置计数
            self.failure_count = 0

    def _on_failure(self) -> None:
        """调用失败处理"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # 半开状态失败，打开熔断器
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED:
            # 关闭状态失败，检查是否达到阈值
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_try_recovery(self) -> bool:
        """检查是否应该尝试恢复

        Returns:
            是否应该尝试恢复
        """
        if self.last_failure_time is None:
            return True

        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.recovery_timeout)

    def reset(self) -> None:
        """重置熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }
