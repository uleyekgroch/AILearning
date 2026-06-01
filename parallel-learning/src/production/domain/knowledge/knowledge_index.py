"""
KnowledgeIndex值对象

知识索引值对象，用于高效查询
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Any
from collections import defaultdict


@dataclass
class KnowledgeIndex:
    """知识索引值对象

    知识索引用于高效查询常识事实。
    支持概念索引和前缀搜索。

    Attributes:
        concept_to_facts: 概念到事实ID的映射
        prefix_index: 前缀索引
    """

    concept_to_facts: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    prefix_index: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def add_mapping(self, concept: str, fact_id: str) -> None:
        """添加概念到事实的映射

        Args:
            concept: 概念（通常是三元组的元素）
            fact_id: 事实ID
        """
        # 添加概念映射
        self.concept_to_facts[concept].add(fact_id)

        # 添加前缀索引
        for i in range(1, len(concept) + 1):
            prefix = concept[:i]
            self.prefix_index[prefix].add(concept)

    def get_by_concept(self, concept: str) -> Set[str]:
        """根据概念获取事实ID集合

        Args:
            concept: 概念

        Returns:
            事实ID集合
        """
        return self.concept_to_facts.get(concept, set())

    def search_by_prefix(self, prefix: str, limit: int = 10) -> List[str]:
        """根据前缀搜索概念

        Args:
            prefix: 前缀
            limit: 返回数量限制

        Returns:
            匹配的概念列表
        """
        concepts = self.prefix_index.get(prefix, set())
        return list(concepts)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息

        Returns:
            统计信息字典
        """
        total_concepts = len(self.concept_to_facts)
        total_mappings = sum(
            len(facts) for facts in self.concept_to_facts.values()
        )

        return {
            'total_concepts': total_concepts,
            'total_mappings': total_mappings,
            'prefix_index_size': len(self.prefix_index),
        }

    def remove_fact(self, fact_id: str) -> None:
        """移除事实的所有索引

        Args:
            fact_id: 事实ID
        """
        # 从概念映射中移除
        concepts_to_remove = []
        for concept, facts in self.concept_to_facts.items():
            if fact_id in facts:
                facts.discard(fact_id)
                if not facts:
                    concepts_to_remove.append(concept)

        # 清理空概念
        for concept in concepts_to_remove:
            del self.concept_to_facts[concept]
            # 清理前缀索引
            for i in range(1, len(concept) + 1):
                prefix = concept[:i]
                if prefix in self.prefix_index:
                    self.prefix_index[prefix].discard(concept)
                    if not self.prefix_index[prefix]:
                        del self.prefix_index[prefix]

    def clear(self) -> None:
        """清空索引"""
        self.concept_to_facts.clear()
        self.prefix_index.clear()
