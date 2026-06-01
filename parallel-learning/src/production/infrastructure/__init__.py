"""
基础设施层

负责持久化、外部服务集成、技术细节实现
"""

from .persistence.knowledge_repository import InMemoryKnowledgeRepository
from .persistence.reasoning_repository import InMemoryReasoningRepository
from .indexing.fact_index import InMemoryFactIndex

__all__ = [
    "InMemoryKnowledgeRepository",
    "InMemoryReasoningRepository",
    "InMemoryFactIndex",
]
