"""
知识管理限界上下文

聚合根：KnowledgeBase
实体：CommonsenseFact, CausalRule
值对象：Triple, Confidence, FactType, KnowledgeIndex
"""

from .knowledge_base import KnowledgeBase
from .commonsense_fact import CommonsenseFact
from .knowledge_index import KnowledgeIndex

__all__ = [
    "KnowledgeBase",
    "CommonsenseFact",
    "KnowledgeIndex",
]
