"""知识实体 — 知识图谱中的节点"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Entity:
    """知识图谱中的实体（节点）

    代表一个具体的或抽象的概念：物体、属性、动作、事件等。
    """
    id: str
    type: str  # object / concept / property / action / event / symbol
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None
    confidence: float = 1.0
    source: str = ''  # vocabulary / perception / reasoning / transfer / hypothesis
    tags: List[str] = field(default_factory=list)

    def matches(self, **criteria) -> bool:
        """检查实体是否匹配给定条件"""
        for key, value in criteria.items():
            if key == 'type' and self.type != value:
                return False
            if key == 'tag' and value not in self.tags:
                return False
            if key in self.properties and self.properties[key] != value:
                return False
        return True

    def to_dict(self) -> dict:
        d = {
            'id': self.id, 'type': self.type,
            'properties': self.properties,
            'confidence': self.confidence,
            'source': self.source, 'tags': self.tags,
        }
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Entity':
        emb = d.get('embedding')
        if emb is not None:
            emb = torch.tensor(emb, dtype=torch.float32)
        return cls(
            id=d['id'], type=d['type'],
            properties=d.get('properties', {}),
            embedding=emb,
            confidence=d.get('confidence', 1.0),
            source=d.get('source', ''),
            tags=d.get('tags', []),
        )
