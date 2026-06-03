"""
SQL推理仓储 - 生产级实现

使用真正的数据库存储，而不是内存
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

from .database import DatabaseConnection
from src.production.domain.reasoning.reasoning_session import ReasoningSession
from src.production.domain.reasoning.reasoning_task import ReasoningTask
from src.production.domain.reasoning.reasoning_result import ReasoningResult

logger = logging.getLogger(__name__)


class SQLReasoningRepository:
    """SQL推理仓储

    使用真正的数据库存储推理会话和任务。

    Attributes:
        db: 数据库连接
    """

    def __init__(self, db: DatabaseConnection):
        """
        初始化SQL推理仓储

        Args:
            db: 数据库连接
        """
        self.db = db

    def save(self, session: ReasoningSession) -> None:
        """
        保存推理会话

        Args:
            session: 推理会话实例
        """
        try:
            # 保存会话
            self.db.execute(
                """
                INSERT OR REPLACE INTO reasoning_sessions (id, status, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                {
                    'id': session.session_id,
                    'status': session.status,
                    'version': session.version,
                    'created_at': session.created_at.isoformat(),
                    'updated_at': session.updated_at.isoformat()
                }
            )

            # 保存任务
            for task in session.tasks.values():
                self._save_task(session.session_id, task)

            # 保存结果
            for task_id, result in session.results.items():
                self._save_result(task_id, result)

            logger.info(f"Saved reasoning session: {session.session_id}")

        except Exception as e:
            logger.error(f"Failed to save reasoning session: {e}")
            raise

    def _save_task(self, session_id: str, task: ReasoningTask) -> None:
        """
        保存推理任务

        Args:
            session_id: 会话ID
            task: 推理任务实例
        """
        self.db.execute(
            """
            INSERT OR REPLACE INTO reasoning_tasks
            (id, session_id, query, reasoning_type, status, created_at, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            {
                'id': task.task_id,
                'session_id': session_id,
                'query': task.query,
                'reasoning_type': task.reasoning_type,
                'status': task.status,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None
            }
        )

    def _save_result(self, task_id: str, result: ReasoningResult) -> None:
        """
        保存推理结果

        Args:
            task_id: 任务ID
            result: 推理结果实例
        """
        self.db.execute(
            """
            INSERT OR REPLACE INTO reasoning_results
            (id, task_id, answer, confidence, status, reasoning_chain, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            {
                'id': f"result_{task_id}",
                'task_id': task_id,
                'answer': result.answer,
                'confidence': result.confidence,
                'status': result.status,
                'reasoning_chain': json.dumps(result.reasoning_chain),
                'metadata': json.dumps(result.metadata),
                'created_at': result.created_at.isoformat()
            }
        )

    def find_by_id(self, session_id: str) -> Optional[ReasoningSession]:
        """
        根据ID查找推理会话

        Args:
            session_id: 会话ID

        Returns:
            推理会话实例，如果不存在返回None
        """
        try:
            # 查询会话
            row = self.db.fetch_one(
                "SELECT * FROM reasoning_sessions WHERE id = ?",
                {'id': session_id}
            )

            if not row:
                return None

            # 创建会话实例
            session = ReasoningSession(session_id=row['id'])

            # 查询任务
            tasks = self.db.fetch_all(
                "SELECT * FROM reasoning_tasks WHERE session_id = ?",
                {'session_id': session_id}
            )

            # 添加任务
            for task_row in tasks:
                task = ReasoningTask(
                    task_id=task_row['id'],
                    query=task_row['query'],
                    reasoning_type=task_row['reasoning_type'],
                    status=task_row['status'],
                    created_at=datetime.fromisoformat(task_row['created_at'])
                )
                session.add_task(task)

                # 查询结果
                result_row = self.db.fetch_one(
                    "SELECT * FROM reasoning_results WHERE task_id = ?",
                    {'task_id': task_row['id']}
                )

                if result_row:
                    result = ReasoningResult(
                        task_id=result_row['task_id'],
                        answer=result_row['answer'],
                        confidence=result_row['confidence'],
                        status=result_row['status'],
                        reasoning_chain=json.loads(result_row['reasoning_chain']),
                        metadata=json.loads(result_row['metadata']),
                        created_at=datetime.fromisoformat(result_row['created_at'])
                    )
                    session.results[task_row['id']] = result

            return session

        except Exception as e:
            logger.error(f"Failed to find reasoning session: {e}")
            raise

    def find_all(self) -> List[ReasoningSession]:
        """
        查找所有推理会话

        Returns:
            推理会话列表
        """
        try:
            rows = self.db.fetch_all("SELECT * FROM reasoning_sessions")
            return [self.find_by_id(row['id']) for row in rows]
        except Exception as e:
            logger.error(f"Failed to find all reasoning sessions: {e}")
            raise

    def delete(self, session_id: str) -> bool:
        """
        删除推理会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        try:
            # 删除结果
            self.db.execute(
                "DELETE FROM reasoning_results WHERE task_id IN (SELECT id FROM reasoning_tasks WHERE session_id = ?)",
                {'session_id': session_id}
            )

            # 删除任务
            self.db.execute(
                "DELETE FROM reasoning_tasks WHERE session_id = ?",
                {'session_id': session_id}
            )

            # 删除会话
            self.db.execute(
                "DELETE FROM reasoning_sessions WHERE id = ?",
                {'id': session_id}
            )

            logger.info(f"Deleted reasoning session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete reasoning session: {e}")
            raise

    def exists(self, session_id: str) -> bool:
        """
        检查推理会话是否存在

        Args:
            session_id: 会话ID

        Returns:
            是否存在
        """
        try:
            row = self.db.fetch_one(
                "SELECT 1 FROM reasoning_sessions WHERE id = ?",
                {'id': session_id}
            )
            return row is not None
        except Exception as e:
            logger.error(f"Failed to check reasoning session existence: {e}")
            raise

    def count(self) -> int:
        """
        获取推理会话数量

        Returns:
            推理会话数量
        """
        try:
            row = self.db.fetch_one("SELECT COUNT(*) as count FROM reasoning_sessions")
            return row['count'] if row else 0
        except Exception as e:
            logger.error(f"Failed to count reasoning sessions: {e}")
            raise

    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        获取推理会话统计信息

        Args:
            session_id: 会话ID

        Returns:
            统计信息字典
        """
        try:
            # 总任务数
            total_row = self.db.fetch_one(
                "SELECT COUNT(*) as count FROM reasoning_tasks WHERE session_id = ?",
                {'session_id': session_id}
            )
            total_tasks = total_row['count'] if total_row else 0

            # 按状态统计
            status_rows = self.db.fetch_all(
                "SELECT status, COUNT(*) as count FROM reasoning_tasks WHERE session_id = ? GROUP BY status",
                {'session_id': session_id}
            )
            by_status = {row['status']: row['count'] for row in status_rows}

            # 平均置信度
            conf_row = self.db.fetch_one(
                """
                SELECT AVG(r.confidence) as avg_conf
                FROM reasoning_results r
                JOIN reasoning_tasks t ON r.task_id = t.id
                WHERE t.session_id = ?
                """,
                {'session_id': session_id}
            )
            avg_confidence = conf_row['avg_conf'] if conf_row and conf_row['avg_conf'] else 0.0

            return {
                'session_id': session_id,
                'total_tasks': total_tasks,
                'by_status': by_status,
                'avg_confidence': avg_confidence,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
