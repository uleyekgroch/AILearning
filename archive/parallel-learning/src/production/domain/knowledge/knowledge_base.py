"""
KnowledgeBase聚合根

知识库聚合根，管理常识事实的生命周期
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Optional, Any

from .commonsense_fact import CommonsenseFact
from .knowledge_index import KnowledgeIndex


# ============================================================================
# 领域事件
# ============================================================================

@dataclass(frozen=True)
class DomainEvent:
    """领域事件基类"""
    event_id: str
    occurred_on: datetime
    aggregate_id: str
    aggregate_version: int
    event_type: str


@dataclass(frozen=True)
class FactAddedEvent(DomainEvent):
    """事实添加事件"""
    fact_id: str
    statement: str
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass(frozen=True)
class FactRemovedEvent(DomainEvent):
    """事实移除事件"""
    fact_id: str


# ============================================================================
# KnowledgeBase聚合根
# ============================================================================

@dataclass
class KnowledgeBase:
    """知识库聚合根

    知识库是知识管理限界上下文的核心聚合根，
    负责管理常识事实的生命周期。

    职责：
    - 管理常识事实的添加、更新、删除
    - 提供事实查询和检索功能
    - 维护知识库的版本和一致性

    Attributes:
        name: 知识库名称
        version: 版本号
        created_at: 创建时间
        updated_at: 更新时间
        facts: 事实映射
        index: 知识索引
        _events: 领域事件列表
    """

    name: str
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    facts: Dict[str, CommonsenseFact] = field(default_factory=dict)
    index: KnowledgeIndex = field(default_factory=KnowledgeIndex)
    _events: List[DomainEvent] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """验证知识库参数"""
        if not self.name:
            raise ValueError("Knowledge base name cannot be empty")

    # ========================================================================
    # 事实管理
    # ========================================================================

    def add_fact(self, fact: CommonsenseFact) -> None:
        """添加常识事实

        Args:
            fact: 常识事实

        Raises:
            ValueError: 如果事实ID已存在
        """
        if fact.fact_id in self.facts:
            raise ValueError(f"Fact {fact.fact_id} already exists")

        # 添加事实
        self.facts[fact.fact_id] = fact

        # 更新索引
        self._index_fact(fact)

        # 更新版本
        self.version += 1
        self.updated_at = datetime.now()

        # 发布领域事件
        event = FactAddedEvent(
            event_id=f"event_{self.version}",
            occurred_on=datetime.now(),
            aggregate_id=self.name,
            aggregate_version=self.version,
            event_type="FactAdded",
            fact_id=fact.fact_id,
            statement=fact.statement,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            confidence=fact.confidence
        )
        self._events.append(event)

    def remove_fact(self, fact_id: str) -> None:
        """移除常识事实

        Args:
            fact_id: 事实ID

        Raises:
            ValueError: 如果事实不存在
        """
        if fact_id not in self.facts:
            raise ValueError(f"Fact {fact_id} not found")

        # 移除事实
        del self.facts[fact_id]

        # 更新索引
        self.index.remove_fact(fact_id)

        # 更新版本
        self.version += 1
        self.updated_at = datetime.now()

        # 发布领域事件
        event = FactRemovedEvent(
            event_id=f"event_{self.version}",
            occurred_on=datetime.now(),
            aggregate_id=self.name,
            aggregate_version=self.version,
            event_type="FactRemoved",
            fact_id=fact_id
        )
        self._events.append(event)

    def has_fact(self, fact_id: str) -> bool:
        """检查事实是否存在

        Args:
            fact_id: 事实ID

        Returns:
            是否存在
        """
        return fact_id in self.facts

    @property
    def fact_count(self) -> int:
        """获取事实数量"""
        return len(self.facts)

    # ========================================================================
    # 查询功能
    # ========================================================================

    def query_by_subject(self, subject: str) -> List[CommonsenseFact]:
        """按主题查询事实

        Args:
            subject: 主题

        Returns:
            匹配的事实列表
        """
        # 严格按subject查询，不使用概念索引
        return [
            fact for fact in self.facts.values()
            if fact.subject == subject
        ]

    def query_by_object(self, object_: str) -> List[CommonsenseFact]:
        """按宾语查询事实

        Args:
            object_: 宾语

        Returns:
            匹配的事实列表
        """
        return [
            fact for fact in self.facts.values()
            if fact.object == object_
        ]

    def query_by_predicate(self, predicate: str) -> List[CommonsenseFact]:
        """按谓语查询事实

        Args:
            predicate: 谓语

        Returns:
            匹配的事实列表
        """
        return [
            fact for fact in self.facts.values()
            if fact.predicate == predicate
        ]

    def query_by_triple(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None
    ) -> List[CommonsenseFact]:
        """按三元组查询事实

        Args:
            subject: 主题（可选）
            predicate: 谓语（可选）
            object_: 宾语（可选）

        Returns:
            匹配的事实列表
        """
        results = []

        for fact in self.facts.values():
            match = True

            if subject and fact.subject != subject:
                match = False
            if predicate and fact.predicate != predicate:
                match = False
            if object_ and fact.object != object_:
                match = False

            if match:
                results.append(fact)

        return results

    def search_by_statement(self, query: str, limit: int = 10) -> List[CommonsenseFact]:
        """按陈述搜索事实

        Args:
            query: 搜索查询
            limit: 返回数量限制

        Returns:
            匹配的事实列表
        """
        results = []
        query_lower = query.lower()

        for fact in self.facts.values():
            if query_lower in fact.statement.lower():
                results.append(fact)
                if len(results) >= limit:
                    break

        return results

    # ========================================================================
    # 索引管理
    # ========================================================================

    def _index_fact(self, fact: CommonsenseFact) -> None:
        """索引事实

        Args:
            fact: 事实
        """
        # 索引三元组元素
        self.index.add_mapping(fact.subject, fact.fact_id)
        self.index.add_mapping(fact.predicate, fact.fact_id)
        self.index.add_mapping(fact.object, fact.fact_id)

    # ========================================================================
    # 事件溯源
    # ========================================================================

    def get_events(self) -> List[DomainEvent]:
        """获取领域事件"""
        return self._events.copy()

    def clear_events(self):
        """清空已发布的事件"""
        self._events.clear()

    # ========================================================================
    # 统计
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计"""
        # 按类型统计
        by_type = {}
        for fact in self.facts.values():
            fact_type = fact.fact_type
            by_type[fact_type] = by_type.get(fact_type, 0) + 1

        # 按来源统计
        by_source = {}
        for fact in self.facts.values():
            source = fact.source
            by_source[source] = by_source.get(source, 0) + 1

        # 置信度统计
        confidences = [f.confidence for f in self.facts.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            'name': self.name,
            'total_facts': self.fact_count,
            'by_type': by_type,
            'by_source': by_source,
            'avg_confidence': avg_confidence,
            'version': self.version,
            'index_stats': self.index.get_statistics(),
        }
