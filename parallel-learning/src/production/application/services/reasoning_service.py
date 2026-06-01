"""
推理应用服务

负责推理会话的用例编排
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.production.domain.reasoning.reasoning_session import ReasoningSession
from src.production.domain.reasoning.reasoning_task import ReasoningTask
from src.production.domain.reasoning.reasoning_result import ReasoningResult


class ReasoningApplicationService:
    """推理应用服务

    职责：
    - 编排推理会话的用例
    - 管理推理会话的生命周期
    - 提供推理任务接口

    Attributes:
        sessions: 推理会话映射
    """

    def __init__(self):
        """初始化推理应用服务"""
        self.sessions: Dict[str, ReasoningSession] = {}

    def create_session(self, session_id: str) -> ReasoningSession:
        """创建推理会话

        Args:
            session_id: 会话ID

        Returns:
            创建的推理会话

        Raises:
            ValueError: 如果会话ID已存在
        """
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        session = ReasoningSession(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ReasoningSession]:
        """获取推理会话

        Args:
            session_id: 会话ID

        Returns:
            推理会话实例，如果不存在返回None
        """
        return self.sessions.get(session_id)

    def add_task(self, session_id: str, task_data: Dict[str, Any]) -> ReasoningTask:
        """添加推理任务

        Args:
            session_id: 会话ID
            task_data: 任务数据字典

        Returns:
            添加的任务

        Raises:
            ValueError: 如果会话不存在
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # 创建任务对象
        task = ReasoningTask(
            task_id=task_data['task_id'],
            query=task_data['query'],
            reasoning_type=task_data['reasoning_type']
        )

        # 添加到会话
        session.add_task(task)

        return task

    def execute_task(self, session_id: str, task_id: str) -> ReasoningResult:
        """执行推理任务

        Args:
            session_id: 会话ID
            task_id: 任务ID

        Returns:
            推理结果

        Raises:
            ValueError: 如果会话或任务不存在
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # 执行任务
        result = session.execute_task(task_id)

        return result

    def get_reasoning_chain(self, session_id: str, task_id: str) -> Optional[Any]:
        """获取推理链

        Args:
            session_id: 会话ID
            task_id: 任务ID

        Returns:
            推理链，如果不存在返回None

        Raises:
            ValueError: 如果会话不存在
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        return session.get_reasoning_chain(task_id)

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计信息

        Args:
            session_id: 会话ID

        Returns:
            统计信息字典

        Raises:
            ValueError: 如果会话不存在
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        return session.get_statistics()

    def list_sessions(self) -> List[str]:
        """列出所有会话

        Returns:
            会话ID列表
        """
        return list(self.sessions.keys())

    def delete_session(self, session_id: str) -> bool:
        """删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
