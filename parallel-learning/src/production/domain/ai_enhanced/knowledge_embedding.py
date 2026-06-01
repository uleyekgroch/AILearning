"""
KnowledgeEmbedding值对象

知识嵌入值对象，表示实体的向量表示
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import numpy as np


@dataclass(frozen=True)
class KnowledgeEmbedding:
    """知识嵌入值对象

    知识嵌入是不可变的值对象，表示实体的向量表示。

    Attributes:
        entity_id: 实体唯一标识
        entity_name: 实体名称
        embedding_vector: 嵌入向量
        metadata: 额外元数据
        created_at: 创建时间
    """

    entity_id: str
    entity_name: str
    embedding_vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """验证知识嵌入参数"""
        # 验证实体ID
        if not self.entity_id:
            raise ValueError("Entity ID cannot be empty")

        # 验证嵌入向量
        if not self.embedding_vector:
            raise ValueError("Embedding vector cannot be empty")

    def compute_similarity(self, other: 'KnowledgeEmbedding') -> float:
        """计算与其他嵌入的余弦相似度

        Args:
            other: 其他知识嵌入

        Returns:
            余弦相似度 [0, 1]
        """
        # 转换为numpy数组
        vec1 = np.array(self.embedding_vector)
        vec2 = np.array(other.embedding_vector)

        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # 确保在[0, 1]范围内
        return max(0.0, min(1.0, float(similarity)))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'embedding_vector': self.embedding_vector,
            'embedding_dim': len(self.embedding_vector),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }

    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度"""
        return len(self.embedding_vector)
