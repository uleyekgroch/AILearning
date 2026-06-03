"""
事实索引实现

内存实现的事实索引
"""

from typing import Dict, Set, List, Any
from collections import defaultdict


class InMemoryFactIndex:
    """内存事实索引

    使用内存存储事实索引的实现。
    支持按主题、谓语、宾语搜索。

    Attributes:
        subject_index: 主题索引
        predicate_index: 谓语索引
        object_index: 宾语索引
        prefix_index: 前缀索引
        metadata: 事实元数据
    """

    def __init__(self):
        """初始化内存事实索引"""
        self.subject_index: Dict[str, Set[str]] = defaultdict(set)
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)
        self.object_index: Dict[str, Set[str]] = defaultdict(set)
        self.prefix_index: Dict[str, Set[str]] = defaultdict(set)
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, fact_id: str, data: Dict[str, Any]) -> None:
        """添加事实到索引

        Args:
            fact_id: 事实ID
            data: 事实数据（包含subject, predicate, object）
        """
        subject = data.get('subject', '')
        predicate = data.get('predicate', '')
        object_ = data.get('object', '')

        # 添加到各索引
        if subject:
            self.subject_index[subject].add(fact_id)
            self._add_prefix(subject, fact_id)

        if predicate:
            self.predicate_index[predicate].add(fact_id)
            self._add_prefix(predicate, fact_id)

        if object_:
            self.object_index[object_].add(fact_id)
            self._add_prefix(object_, fact_id)

        # 保存元数据
        self.metadata[fact_id] = data

    def remove(self, fact_id: str) -> None:
        """从事实索引中移除

        Args:
            fact_id: 事实ID
        """
        if fact_id not in self.metadata:
            return

        data = self.metadata[fact_id]
        subject = data.get('subject', '')
        predicate = data.get('predicate', '')
        object_ = data.get('object', '')

        # 从各索引中移除
        if subject and subject in self.subject_index:
            self.subject_index[subject].discard(fact_id)
            if not self.subject_index[subject]:
                del self.subject_index[subject]

        if predicate and predicate in self.predicate_index:
            self.predicate_index[predicate].discard(fact_id)
            if not self.predicate_index[predicate]:
                del self.predicate_index[predicate]

        if object_ and object_ in self.object_index:
            self.object_index[object_].discard(fact_id)
            if not self.object_index[object_]:
                del self.object_index[object_]

        # 清空前缀索引
        self._remove_prefix(subject, fact_id)
        self._remove_prefix(predicate, fact_id)
        self._remove_prefix(object_, fact_id)

        # 删除元数据
        del self.metadata[fact_id]

    def search_by_subject(self, subject: str) -> Set[str]:
        """按主题搜索

        Args:
            subject: 主题

        Returns:
            事实ID集合
        """
        return self.subject_index.get(subject, set())

    def search_by_predicate(self, predicate: str) -> Set[str]:
        """按谓语搜索

        Args:
            predicate: 谓语

        Returns:
            事实ID集合
        """
        return self.predicate_index.get(predicate, set())

    def search_by_object(self, object_: str) -> Set[str]:
        """按宾语搜索

        Args:
            object_: 宾语

        Returns:
            事实ID集合
        """
        return self.object_index.get(object_, set())

    def search_by_prefix(self, prefix: str, limit: int = 10) -> List[str]:
        """按前缀搜索

        Args:
            prefix: 前缀
            limit: 返回数量限制

        Returns:
            匹配的概念列表
        """
        concepts = self.prefix_index.get(prefix, set())
        return list(concepts)[:limit]

    def _add_prefix(self, concept: str, fact_id: str) -> None:
        """添加前缀索引

        Args:
            concept: 概念
            fact_id: 事实ID
        """
        for i in range(1, len(concept) + 1):
            prefix = concept[:i]
            self.prefix_index[prefix].add(concept)

    def _remove_prefix(self, concept: str, fact_id: str) -> None:
        """移除前缀索引

        Args:
            concept: 概念
            fact_id: 事实ID
        """
        if not concept:
            return

        for i in range(1, len(concept) + 1):
            prefix = concept[:i]
            if prefix in self.prefix_index:
                self.prefix_index[prefix].discard(concept)
                if not self.prefix_index[prefix]:
                    del self.prefix_index[prefix]

    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息

        Returns:
            统计信息字典
        """
        total_concepts = (
            len(self.subject_index) +
            len(self.predicate_index) +
            len(self.object_index)
        )

        return {
            'total_entries': len(self.metadata),
            'total_concepts': total_concepts,
            'subject_count': len(self.subject_index),
            'predicate_count': len(self.predicate_index),
            'object_count': len(self.object_index),
            'prefix_count': len(self.prefix_index),
        }

    def clear(self) -> None:
        """清空索引"""
        self.subject_index.clear()
        self.predicate_index.clear()
        self.object_index.clear()
        self.prefix_index.clear()
        self.metadata.clear()
