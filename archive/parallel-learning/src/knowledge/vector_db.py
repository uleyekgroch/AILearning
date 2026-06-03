"""
向量数据库抽象接口 - Vector Database Abstraction

支持：
- 向量存储与检索
- 相似度搜索（余弦、欧氏距离）
- 高性能ANN查询
- 可替换实现（内存/Milvus/Pinecone）

设计原则：
- 接口抽象，便于替换
- 支持大规模向量（百万级）
- 高性能相似度搜索
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch


@dataclass
class VectorEmbedding:
    """向量嵌入"""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    dimension: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.vector, torch.Tensor):
            self.vector = self.vector.cpu().numpy()
        self.dimension = self.vector.shape[0]


@dataclass
class SearchResult:
    """相似度搜索结果"""
    id: str
    score: float  # 相似度分数 [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorDatabase(ABC):
    """
    向量数据库抽象接口

    定义所有向量数据库必须实现的核心操作
    """

    @abstractmethod
    def insert(self, embedding: VectorEmbedding) -> bool:
        """插入向量

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray,
               top_k: int = 10,
               metric: str = 'cosine') -> List[SearchResult]:
        """相似度搜索

        Args:
            query_vector: 查询向量
            top_k: 返回top-k个结果
            metric: 相似度度量 ('cosine', 'euclidean')

        Returns:
            搜索结果列表（按相似度降序排序）
        """
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """删除向量

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def get(self, vector_id: str) -> Optional[VectorEmbedding]:
        """获取向量"""
        pass

    @abstractmethod
    def update(self, embedding: VectorEmbedding) -> bool:
        """更新向量

        Returns:
            是否成功
        """
        pass


class InMemoryVectorDB(VectorDatabase):
    """
    内存向量数据库实现

    用于开发和小规模部署
    后续可替换为Milvus/Pinecone实现
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors: Dict[str, VectorEmbedding] = {}

        # 向量索引（用于快速搜索）
        self.vector_matrix = np.zeros((0, dimension))
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}

        # 统计信息
        self.stats = {
            'insert_count': 0,
            'search_count': 0,
            'delete_count': 0
        }

    def insert(self, embedding: VectorEmbedding) -> bool:
        """插入向量"""
        if embedding.id in self.vectors:
            return False

        # 检查维度
        if embedding.vector.shape[0] != self.dimension:
            # 尝试调整维度
            if embedding.vector.shape[0] < self.dimension:
                # 填充
                padded = np.zeros(self.dimension)
                padded[:embedding.vector.shape[0]] = embedding.vector
                embedding.vector = padded
            else:
                # 截断
                embedding.vector = embedding.vector[:self.dimension]

        # 存储向量
        self.vectors[embedding.id] = embedding

        # 更新索引矩阵
        new_index = self.vector_matrix.shape[0]
        self.id_to_index[embedding.id] = new_index
        self.index_to_id[new_index] = embedding.id

        # 扩展矩阵
        new_matrix = np.zeros((new_index + 1, self.dimension))
        new_matrix[:new_index] = self.vector_matrix
        new_matrix[new_index] = embedding.vector
        self.vector_matrix = new_matrix

        self.stats['insert_count'] += 1
        return True

    def search(self, query_vector: np.ndarray,
               top_k: int = 10,
               metric: str = 'cosine') -> List[SearchResult]:
        """相似度搜索"""
        if self.vector_matrix.shape[0] == 0:
            return []

        # 标准化查询向量
        if metric == 'cosine':
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            matrix_norm = self.vector_matrix / (np.linalg.norm(self.vector_matrix, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(matrix_norm, query_norm)
        elif metric == 'euclidean':
            # 欧氏距离（转换为相似度）
            distances = np.linalg.norm(self.vector_matrix - query_vector, axis=1)
            max_dist = distances.max() if distances.size > 0 else 1.0
            similarities = 1.0 - (distances / (max_dist + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # 获取top-k索引
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        # 构建结果
        results = []
        for idx in top_k_indices:
            if idx < len(self.index_to_id):
                vector_id = self.index_to_id[idx]
                embedding = self.vectors[vector_id]
                results.append(SearchResult(
                    id=vector_id,
                    score=float(similarities[idx]),
                    metadata=embedding.metadata
                ))

        self.stats['search_count'] += 1
        return results

    def delete(self, vector_id: str) -> bool:
        """删除向量"""
        if vector_id not in self.vectors:
            return False

        # 从索引中删除
        index = self.id_to_index[vector_id]

        # 重建矩阵（简化实现）
        self.vector_matrix = np.delete(self.vector_matrix, index, axis=0)

        # 重建索引映射
        self.id_to_index = {}
        self.index_to_id = {}
        for i, (vid, vec) in enumerate(self.vectors.items()):
            self.id_to_index[vid] = i
            self.index_to_id[i] = vid

        del self.vectors[vector_id]
        self.stats['delete_count'] += 1
        return True

    def get(self, vector_id: str) -> Optional[VectorEmbedding]:
        """获取向量"""
        return self.vectors.get(vector_id)

    def update(self, embedding: VectorEmbedding) -> bool:
        """更新向量"""
        if embedding.id not in self.vectors:
            return False

        # 先删除旧的
        self.delete(embedding.id)
        # 再插入新的
        return self.insert(embedding)

    def batch_insert(self, embeddings: List[VectorEmbedding]) -> int:
        """批量插入

        Args:
            embeddings: 向量列表

        Returns:
            成功插入的数量
        """
        count = 0
        for emb in embeddings:
            if self.insert(emb):
                count += 1
        return count

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'vector_count': len(self.vectors),
            'dimension': self.dimension,
            'insert_count': self.stats['insert_count'],
            'search_count': self.stats['search_count'],
            'delete_count': self.stats['delete_count'],
        }


class VectorDBFactory:
    """向量数据库工厂"""

    _instance = None
    _backends = {}

    @classmethod
    def register_backend(cls, name: str, backend_class):
        """注册后端"""
        cls._backends[name] = backend_class

    @classmethod
    def get_instance(cls, backend: str = 'memory',
                     dimension: int = 128, **kwargs) -> VectorDatabase:
        """获取向量数据库实例

        Args:
            backend: 后端类型 ('memory', 'milvus', 'pinecone')
            dimension: 向量维度
            **kwargs: 其他参数

        Returns:
            向量数据库实例
        """
        if backend == 'memory':
            if cls._instance is None or cls._instance.dimension != dimension:
                cls._instance = InMemoryVectorDB(dimension)
        elif backend in cls._backends:
            cls._instance = cls._backends[backend](dimension, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return cls._instance

    @classmethod
    def reset(cls):
        """重置实例（用于测试）"""
        cls._instance = None


# 便捷函数
def get_vector_db(dimension: int = 128) -> VectorDatabase:
    """获取向量数据库实例"""
    return VectorDBFactory.get_instance(dimension=dimension)


if __name__ == '__main__':
    print("=== 向量数据库抽象接口 ===")
    print()
    print("功能:")
    print("- 向量存储与检索")
    print("- 相似度搜索（余弦/欧氏）")
    print("- 批量操作")
    print("- 可扩展架构")
    print()
    print("实现:")
    print("- InMemoryVectorDB (当前)")
    print("- MilvusVectorDB (后续)")
    print("- PineconeVectorDB (后续)")
