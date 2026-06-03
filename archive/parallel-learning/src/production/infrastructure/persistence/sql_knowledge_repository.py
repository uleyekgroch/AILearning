"""
SQL知识库仓储 - 生产级实现

使用真正的数据库存储，而不是内存
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

from .database import DatabaseConnection
from src.production.domain.knowledge.knowledge_base import KnowledgeBase
from src.production.domain.knowledge.commonsense_fact import CommonsenseFact

logger = logging.getLogger(__name__)


class SQLKnowledgeRepository:
    """SQL知识库仓储

    使用真正的数据库存储知识库和事实。

    Attributes:
        db: 数据库连接
    """

    def __init__(self, db: DatabaseConnection):
        """
        初始化SQL知识库仓储

        Args:
            db: 数据库连接
        """
        self.db = db

    def save(self, knowledge_base: KnowledgeBase) -> None:
        """
        保存知识库

        Args:
            knowledge_base: 知识库实例
        """
        try:
            # 保存知识库
            self.db.execute(
                """
                INSERT OR REPLACE INTO knowledge_bases (id, name, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                {
                    'id': knowledge_base.name,
                    'name': knowledge_base.name,
                    'version': knowledge_base.version,
                    'created_at': knowledge_base.created_at.isoformat(),
                    'updated_at': knowledge_base.updated_at.isoformat()
                }
            )

            # 保存事实
            for fact in knowledge_base.facts.values():
                self._save_fact(knowledge_base.name, fact)

            logger.info(f"Saved knowledge base: {knowledge_base.name}")

        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
            raise

    def _save_fact(self, kb_name: str, fact: CommonsenseFact) -> None:
        """
        保存事实

        Args:
            kb_name: 知识库名称
            fact: 事实实例
        """
        self.db.execute(
            """
            INSERT OR REPLACE INTO facts
            (id, knowledge_base_id, statement, subject, predicate, object,
             confidence, fact_type, source, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            {
                'id': fact.fact_id,
                'knowledge_base_id': kb_name,
                'statement': fact.statement,
                'subject': fact.subject,
                'predicate': fact.predicate,
                'object': fact.object,
                'confidence': fact.confidence,
                'fact_type': fact.fact_type,
                'source': fact.source,
                'metadata': json.dumps(fact.metadata),
                'created_at': fact.created_at.isoformat()
            }
        )

    def find_by_name(self, name: str) -> Optional[KnowledgeBase]:
        """
        根据名称查找知识库

        Args:
            name: 知识库名称

        Returns:
            知识库实例，如果不存在返回None
        """
        try:
            # 查询知识库
            row = self.db.fetch_one(
                "SELECT * FROM knowledge_bases WHERE name = ?",
                {'name': name}
            )

            if not row:
                return None

            # 创建知识库实例
            kb = KnowledgeBase(name=row['name'])

            # 查询事实
            facts = self.db.fetch_all(
                "SELECT * FROM facts WHERE knowledge_base_id = ?",
                {'knowledge_base_id': name}
            )

            # 添加事实
            for fact_row in facts:
                fact = CommonsenseFact(
                    fact_id=fact_row['id'],
                    statement=fact_row['statement'],
                    subject=fact_row['subject'],
                    predicate=fact_row['predicate'],
                    object=fact_row['object'],
                    confidence=fact_row['confidence'],
                    fact_type=fact_row['fact_type'],
                    source=fact_row['source'],
                    metadata=json.loads(fact_row['metadata']),
                    created_at=datetime.fromisoformat(fact_row['created_at'])
                )
                kb.add_fact(fact)

            return kb

        except Exception as e:
            logger.error(f"Failed to find knowledge base: {e}")
            raise

    def find_all(self) -> List[KnowledgeBase]:
        """
        查找所有知识库

        Returns:
            知识库列表
        """
        try:
            rows = self.db.fetch_all("SELECT * FROM knowledge_bases")
            return [self.find_by_name(row['name']) for row in rows]
        except Exception as e:
            logger.error(f"Failed to find all knowledge bases: {e}")
            raise

    def delete(self, name: str) -> bool:
        """
        删除知识库

        Args:
            name: 知识库名称

        Returns:
            是否删除成功
        """
        try:
            # 删除事实
            self.db.execute(
                "DELETE FROM facts WHERE knowledge_base_id = ?",
                {'knowledge_base_id': name}
            )

            # 删除知识库
            result = self.db.execute(
                "DELETE FROM knowledge_bases WHERE name = ?",
                {'name': name}
            )

            logger.info(f"Deleted knowledge base: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete knowledge base: {e}")
            raise

    def exists(self, name: str) -> bool:
        """
        检查知识库是否存在

        Args:
            name: 知识库名称

        Returns:
            是否存在
        """
        try:
            row = self.db.fetch_one(
                "SELECT 1 FROM knowledge_bases WHERE name = ?",
                {'name': name}
            )
            return row is not None
        except Exception as e:
            logger.error(f"Failed to check knowledge base existence: {e}")
            raise

    def count(self) -> int:
        """
        获取知识库数量

        Returns:
            知识库数量
        """
        try:
            row = self.db.fetch_one("SELECT COUNT(*) as count FROM knowledge_bases")
            return row['count'] if row else 0
        except Exception as e:
            logger.error(f"Failed to count knowledge bases: {e}")
            raise

    def query_facts(
        self,
        kb_name: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        limit: int = 100
    ) -> List[CommonsenseFact]:
        """
        查询事实

        Args:
            kb_name: 知识库名称
            subject: 主语过滤
            predicate: 谓语过滤
            object_: 宾语过滤
            limit: 返回数量限制

        Returns:
            事实列表
        """
        try:
            query = "SELECT * FROM facts WHERE knowledge_base_id = ?"
            params = {'knowledge_base_id': kb_name}

            if subject:
                query += " AND subject = ?"
                params['subject'] = subject

            if predicate:
                query += " AND predicate = ?"
                params['predicate'] = predicate

            if object_:
                query += " AND object = ?"
                params['object'] = object_

            query += f" LIMIT {limit}"

            rows = self.db.fetch_all(query, params)

            return [
                CommonsenseFact(
                    fact_id=row['id'],
                    statement=row['statement'],
                    subject=row['subject'],
                    predicate=row['predicate'],
                    object=row['object'],
                    confidence=row['confidence'],
                    fact_type=row['fact_type'],
                    source=row['source'],
                    metadata=json.loads(row['metadata']),
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to query facts: {e}")
            raise

    def get_statistics(self, kb_name: str) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Args:
            kb_name: 知识库名称

        Returns:
            统计信息字典
        """
        try:
            # 总事实数
            total_row = self.db.fetch_one(
                "SELECT COUNT(*) as count FROM facts WHERE knowledge_base_id = ?",
                {'knowledge_base_id': kb_name}
            )
            total_facts = total_row['count'] if total_row else 0

            # 按类型统计
            type_rows = self.db.fetch_all(
                "SELECT fact_type, COUNT(*) as count FROM facts WHERE knowledge_base_id = ? GROUP BY fact_type",
                {'knowledge_base_id': kb_name}
            )
            by_type = {row['fact_type']: row['count'] for row in type_rows}

            # 平均置信度
            conf_row = self.db.fetch_one(
                "SELECT AVG(confidence) as avg_conf FROM facts WHERE knowledge_base_id = ?",
                {'knowledge_base_id': kb_name}
            )
            avg_confidence = conf_row['avg_conf'] if conf_row and conf_row['avg_conf'] else 0.0

            return {
                'name': kb_name,
                'total_facts': total_facts,
                'by_type': by_type,
                'avg_confidence': avg_confidence,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
