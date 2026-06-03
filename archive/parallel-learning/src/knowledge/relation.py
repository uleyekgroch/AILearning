"""知识关系 — 知识图谱中的边"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


# 关系类型常量
REL_IS_A = 'is_a'
REL_HAS_PROPERTY = 'has_property'
REL_PART_OF = 'part_of'
REL_CAUSES = 'causes'
REL_SIMILAR_TO = 'similar_to'
REL_COLLOCATES = 'collocates_with'
REL_RELATED = 'related_to'


@dataclass
class Relation:
    """知识图谱中的关系（边）

    连接两个实体，表达它们之间的语义关系。
    """
    source_id: str
    target_id: str
    type: str
    confidence: float = 1.0
    evidence_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def strengthen(self, delta: float = 0.1) -> None:
        """加强关系置信度"""
        self.confidence = min(1.0, self.confidence + delta)
        self.evidence_count += 1

    def weaken(self, delta: float = 0.1) -> None:
        """削弱关系置信度"""
        self.confidence = max(0.0, self.confidence - delta)

    def to_dict(self) -> dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type,
            'confidence': self.confidence,
            'evidence_count': self.evidence_count,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Relation':
        return cls(
            source_id=d['source_id'],
            target_id=d['target_id'],
            type=d['type'],
            confidence=d.get('confidence', 1.0),
            evidence_count=d.get('evidence_count', 0),
            metadata=d.get('metadata', {}),
        )
