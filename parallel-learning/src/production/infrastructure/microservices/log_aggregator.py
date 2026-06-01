"""
LogAggregator日志聚合器

负责日志收集和聚合
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    """日志条目"""
    service_name: str
    level: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class LogAggregator:
    """日志聚合器

    职责：
    - 日志收集
    - 日志聚合
    - 日志查询
    - 日志统计

    Attributes:
        logs: 日志列表
    """

    def __init__(self):
        """初始化日志聚合器"""
        self.logs: List[LogEntry] = []

    def add_log(
        self,
        service_name: str,
        level: str,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """添加日志

        Args:
            service_name: 服务名称
            level: 日志级别
            message: 日志消息
            metadata: 额外元数据
        """
        log_entry = LogEntry(
            service_name=service_name,
            level=level,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.logs.append(log_entry)

    def get_logs(
        self,
        service_name: str = None,
        level: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取日志

        Args:
            service_name: 服务名称（可选）
            level: 日志级别（可选）
            limit: 返回数量限制

        Returns:
            日志列表
        """
        filtered = self.logs

        # 按服务名称过滤
        if service_name:
            filtered = [log for log in filtered if log.service_name == service_name]

        # 按级别过滤
        if level:
            filtered = [log for log in filtered if log.level == level]

        # 限制数量
        filtered = filtered[-limit:]

        # 转换为字典
        return [
            {
                "service_name": log.service_name,
                "level": log.level,
                "message": log.message,
                "timestamp": log.timestamp.isoformat(),
                "metadata": log.metadata,
            }
            for log in filtered
        ]

    def clear(self) -> None:
        """清空日志"""
        self.logs.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        total_logs = len(self.logs)

        # 按级别统计
        by_level = {}
        for log in self.logs:
            level = log.level
            by_level[level] = by_level.get(level, 0) + 1

        # 按服务统计
        by_service = {}
        for log in self.logs:
            service = log.service_name
            by_service[service] = by_service.get(service, 0) + 1

        return {
            "total_logs": total_logs,
            "by_level": by_level,
            "by_service": by_service,
        }
