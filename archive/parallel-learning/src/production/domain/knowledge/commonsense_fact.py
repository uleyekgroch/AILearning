"""
CommonsenseFact实体

常识事实实体，属于KnowledgeBase聚合
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from enum import Enum


class FactType(Enum):
    """事实类型枚举"""
    PHYSICAL = "physical"
    BIOLOGICAL = "biological"
    SOCIAL = "social"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    FUNCTIONAL = "functional"


@dataclass
class CommonsenseFact:
    """常识事实实体

    常识事实是KnowledgeBase聚合的内部实体，
    表示一个常识性的知识条目。

    Attributes:
        fact_id: 事实唯一标识
        statement: 自然语言表达
        subject: 主语（三元组第一元素）
        predicate: 谓语（关系）
        object: 宾语（三元组第三元素）
        confidence: 置信度 [0, 1]
        fact_type: 事实类型
        source: 来源
        metadata: 额外元数据
        created_at: 创建时间
    """

    fact_id: str
    statement: str
    subject: str
    predicate: str
    object: str
    confidence: float
    fact_type: str = "physical"
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证事实参数"""
        # 验证置信度
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        # 验证事实ID
        if not self.fact_id:
            raise ValueError("Fact ID cannot be empty")

        # 验证三元组
        if not self.subject:
            raise ValueError("Subject cannot be empty")
        if not self.predicate:
            raise ValueError("Predicate cannot be empty")
        if not self.object:
            raise ValueError("Object cannot be empty")

        # 验证事实类型
        valid_types = [ft.value for ft in FactType]
        if self.fact_type not in valid_types:
            raise ValueError(
                f"Invalid fact type: {self.fact_type}. "
                f"Must be one of: {valid_types}"
            )

    def to_triple(self) -> Tuple[str, str, str]:
        """转换为三元组"""
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'fact_id': self.fact_id,
            'statement': self.statement,
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'fact_type': self.fact_type,
            'source': self.source,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }

    @property
    def is_high_confidence(self) -> bool:
        """检查是否高置信度（>0.8）"""
        return self.confidence > 0.8

    def __eq__(self, other):
        """事实相等性比较（基于ID）"""
        if not isinstance(other, CommonsenseFact):
            return False
        return self.fact_id == other.fact_id

    def __hash__(self):
        """事实哈希值（基于ID）"""
        return hash(self.fact_id)
