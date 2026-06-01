"""
知识库仓储实现

内存实现的知识库仓储
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from src.production.domain.knowledge.knowledge_base import KnowledgeBase


class InMemoryKnowledgeRepository:
    """内存知识库仓储

    使用内存存储知识库的实现。
    适用于测试和开发环境。

    Attributes:
        storage: 存储映射
    """

    def __init__(self):
        """初始化内存知识库仓储"""
        self.storage: Dict[str, KnowledgeBase] = {}

    def save(self, knowledge_base: KnowledgeBase) -> None:
        """保存知识库

        Args:
            knowledge_base: 知识库实例
        """
        self.storage[knowledge_base.name] = knowledge_base

    def find_by_name(self, name: str) -> Optional[KnowledgeBase]:
        """根据名称查找知识库

        Args:
            name: 知识库名称

        Returns:
            知识库实例，如果不存在返回None
        """
        return self.storage.get(name)

    def find_all(self) -> List[KnowledgeBase]:
        """查找所有知识库

        Returns:
            知识库列表
        """
        return list(self.storage.values())

    def delete(self, name: str) -> bool:
        """删除知识库

        Args:
            name: 知识库名称

        Returns:
            是否删除成功
        """
        if name in self.storage:
            del self.storage[name]
            return True
        return False

    def exists(self, name: str) -> bool:
        """检查知识库是否存在

        Args:
            name: 知识库名称

        Returns:
            是否存在
        """
        return name in self.storage

    def count(self) -> int:
        """获取知识库数量

        Returns:
            知识库数量
        """
        return len(self.storage)

    def clear(self) -> None:
        """清空所有知识库"""
        self.storage.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取仓储统计信息

        Returns:
            统计信息字典
        """
        total_facts = sum(kb.fact_count for kb in self.storage.values())

        return {
            'total_knowledge_bases': self.count(),
            'total_facts': total_facts,
            'knowledge_bases': list(self.storage.keys()),
        }
