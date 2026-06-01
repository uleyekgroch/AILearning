"""
推理会话仓储实现

内存实现的推理会话仓储
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from src.production.domain.reasoning.reasoning_session import ReasoningSession


class InMemoryReasoningRepository:
    """内存推理会话仓储

    使用内存存储推理会话的实现。
    适用于测试和开发环境。

    Attributes:
        storage: 存储映射
    """

    def __init__(self):
        """初始化内存推理会话仓储"""
        self.storage: Dict[str, ReasoningSession] = {}

    def save(self, session: ReasoningSession) -> None:
        """保存推理会话

        Args:
            session: 推理会话实例
        """
        self.storage[session.session_id] = session

    def find_by_id(self, session_id: str) -> Optional[ReasoningSession]:
        """根据ID查找推理会话

        Args:
            session_id: 会话ID

        Returns:
            推理会话实例，如果不存在返回None
        """
        return self.storage.get(session_id)

    def find_all(self) -> List[ReasoningSession]:
        """查找所有推理会话

        Returns:
            推理会话列表
        """
        return list(self.storage.values())

    def delete(self, session_id: str) -> bool:
        """删除推理会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        if session_id in self.storage:
            del self.storage[session_id]
            return True
        return False

    def exists(self, session_id: str) -> bool:
        """检查推理会话是否存在

        Args:
            session_id: 会话ID

        Returns:
            是否存在
        """
        return session_id in self.storage

    def count(self) -> int:
        """获取推理会话数量

        Returns:
            推理会话数量
        """
        return len(self.storage)

    def clear(self) -> None:
        """清空所有推理会话"""
        self.storage.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取仓储统计信息

        Returns:
            统计信息字典
        """
        total_tasks = sum(s.task_count for s in self.storage.values())

        return {
            'total_sessions': self.count(),
            'total_tasks': total_tasks,
            'sessions': list(self.storage.keys()),
        }
