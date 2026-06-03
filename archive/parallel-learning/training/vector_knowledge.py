"""向量化知识理解系统 — 从正则到向量空间

核心思想：
- 文本 → TF-IDF 向量（稀疏高维表示）
- 聚类 → 自动发现概念（KMeans/HDBSCAN）
- 近邻搜索 → 知识检索（KNN）
- 降维 → 可视化和理解（PCA）

与正则方案的对比：
- 正则：O(n*m) 字符串匹配，CPU 单线程
- 向量：O(n*log(n)) 矩阵运算，可并行/GPU加速

运行方式：
    python training/vector_knowledge.py
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity


class VectorKnowledgeSystem:
    """向量化知识系统

    核心组件：
    1. TF-IDF 向量化器：文本 → 高维稀疏向量
    2. KMeans 聚类器：自动发现概念簇
    3. KNN 检索器：快速查找相关知识
    4. SVD/PCA 降维器：可视化和压缩
    """

    def __init__(self, max_features=50000, n_clusters=1000, n_components=None):
        """
        Args:
            max_features: TF-IDF 最大特征数
            n_clusters: 聚类数（概念数）
            n_components: SVD 降维后的维度
        """
        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer='char_wb',  # 字符级 n-gram，适合中文
            ngram_range=(2, 4),  # 2-4字符的 n-gram
            sublinear_tf=True,   # 对 TF 取对数
            max_df=0.95,         # 忽略出现超过95%的词
            min_df=2,            # 忽略出现少于2次的词
        )

        # 聚类
        self.n_clusters = n_clusters
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            max_iter=100,
            random_state=42,
        )

        # 近邻搜索
        self.nn_model = NearestNeighbors(
            n_neighbors=10,
            metric='cosine',
            algorithm='auto',
        )

        # 降维（自动选择组件数）
        self._n_components = n_components
        self.svd = None  # 延迟初始化，需要知道特征数

        # 数据存储
        self.documents: List[Dict] = []  # 原始文档
        self.vectors = None              # TF-IDF 向量矩阵
        self.reduced_vectors = None      # 降维后的向量
        self.cluster_labels = None       # 聚类标签
        self.cluster_centers = None      # 聚类中心

        # 知识索引
        self.cluster_docs: Dict[int, List[int]] = defaultdict(list)  # 簇 → 文档索引
        self.entity_index: Dict[str, List[int]] = defaultdict(list)  # 实体 → 文档索引

        # 状态
        self.fitted = False
        self.total_processed = 0

    def add_document(self, title: str, text: str, metadata: Dict = None):
        """添加文档"""
        doc = {
            'id': len(self.documents),
            'title': title,
            'text': text[:1000],  # 限制长度
            'metadata': metadata or {},
        }
        self.documents.append(doc)

        # 提取实体（简单分词）
        import re
        words = set(re.findall(r'[一-鿿]{2,4}', text))
        for word in words:
            self.entity_index[word].append(doc['id'])

        self.total_processed += 1

    def fit(self, batch_size=10000):
        """训练模型

        步骤：
        1. TF-IDF 向量化
        2. KMeans 聚类
        3. KNN 索引
        4. SVD 降维
        """
        if not self.documents:
            print("  没有文档可训练")
            return

        print(f"  文档数: {len(self.documents):,}")

        # 1. TF-IDF 向量化
        print("  [1/4] TF-IDF 向量化...")
        texts = [doc['text'] for doc in self.documents]
        self.vectors = self.vectorizer.fit_transform(texts)
        print(f"    向量维度: {self.vectors.shape}")

        # 2. KMeans 聚类
        print("  [2/4] KMeans 聚类...")
        self.cluster_labels = self.clusterer.fit_predict(self.vectors)
        self.cluster_centers = self.clusterer.cluster_centers_

        # 构建簇索引
        for i, label in enumerate(self.cluster_labels):
            self.cluster_docs[label].append(i)
        print(f"    聚类数: {self.n_clusters}")

        # 3. KNN 索引
        print("  [3/4] KNN 索引构建...")
        self.nn_model.fit(self.vectors)
        print(f"    索引构建完成")

        # 4. SVD 降维
        print("  [4/4] SVD 降维...")
        n_features = self.vectors.shape[1]
        # 自动确定 n_components：不超过特征数-1，不超过200，默认100
        if self._n_components is None:
            n_comp = min(100, n_features - 1)
        else:
            n_comp = min(self._n_components, n_features - 1)
        n_comp = max(2, n_comp)  # 至少2维
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.reduced_vectors = self.svd.fit_transform(self.vectors)
        print(f"    降维后维度: {self.reduced_vectors.shape}")

        self.fitted = True
        print("  训练完成!")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关知识

        使用 KNN 快速查找最相似的文档
        """
        if not self.fitted:
            return []

        # 向量化查询
        query_vec = self.vectorizer.transform([query])

        # KNN 搜索
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc = self.documents[idx]
            results.append({
                'title': doc['title'],
                'text': doc['text'][:200],
                'similarity': 1 - dist,  # 转换为相似度
                'cluster': int(self.cluster_labels[idx]),
                'metadata': doc['metadata'],
            })

        return results

    def get_cluster_info(self, cluster_id: int, top_n: int = 5) -> Dict:
        """获取聚类信息"""
        if not self.fitted:
            return {}

        doc_indices = self.cluster_docs.get(cluster_id, [])
        if not doc_indices:
            return {}

        # 获取簇内的文档
        docs = [self.documents[i] for i in doc_indices[:top_n]]

        # 获取簇中心的 top 特征词
        center = self.cluster_centers[cluster_id]
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = np.argsort(center)[-10:][::-1]
        top_words = [feature_names[i] for i in top_features]

        return {
            'cluster_id': cluster_id,
            'size': len(doc_indices),
            'top_words': top_words,
            'sample_docs': [{'title': d['title'], 'text': d['text'][:100]} for d in docs],
        }

    def find_related_entities(self, entity: str, top_k: int = 5) -> List[Dict]:
        """查找相关实体"""
        if not self.fitted:
            return []

        # 获取包含该实体的文档
        doc_indices = self.entity_index.get(entity, [])
        if not doc_indices:
            return []

        # 取第一个文档的向量作为实体表示
        entity_vec = self.vectors[doc_indices[0]]

        # 计算与所有文档的相似度
        similarities = cosine_similarity(entity_vec, self.vectors).flatten()

        # 排除自身
        for idx in doc_indices:
            similarities[idx] = -1

        # 获取最相似的文档
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            # 提取该文档中的实体
            import re
            words = set(re.findall(r'[一-鿿]{2,4}', doc['text']))
            results.append({
                'title': doc['title'],
                'similarity': float(similarities[idx]),
                'entities': list(words)[:10],
            })

        return results

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_documents': len(self.documents),
            'total_entities': len(self.entity_index),
            'fitted': self.fitted,
        }

        if self.fitted:
            stats['vector_dimensions'] = self.vectors.shape[1]
            stats['reduced_dimensions'] = self.reduced_vectors.shape[1] if self.reduced_vectors is not None else 0
            stats['n_clusters'] = self.n_clusters
            stats['avg_cluster_size'] = len(self.documents) / self.n_clusters

        return stats

    def save(self, path: str):
        """保存模型"""
        import pickle
        data = {
            'documents': self.documents,
            'entity_index': dict(self.entity_index),
            'cluster_docs': dict(self.cluster_docs),
            'stats': self.get_stats(),
            'fitted': self.fitted,
        }
        if self.fitted:
            data['vectorizer'] = self.vectorizer
            data['clusterer'] = self.clusterer
            data['nn_model'] = self.nn_model
            data['svd'] = self.svd
            data['cluster_labels'] = self.cluster_labels
            data['cluster_centers'] = self.cluster_centers
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  保存到: {path}")

    def load(self, path: str):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.entity_index = defaultdict(list, data['entity_index'])
        self.cluster_docs = defaultdict(list, data['cluster_docs'])
        if data.get('fitted', False):
            self.vectorizer = data['vectorizer']
            self.clusterer = data['clusterer']
            self.nn_model = data['nn_model']
            self.svd = data['svd']
            self.cluster_labels = data['cluster_labels']
            self.cluster_centers = data['cluster_centers']
            self.fitted = True
            # 重建向量矩阵用于搜索
            texts = [doc['text'] for doc in self.documents]
            self.vectors = self.vectorizer.transform(texts)
            self.reduced_vectors = self.svd.transform(self.vectors)
        print(f"  加载: {len(self.documents)} 文档")


def demo():
    """演示向量化知识系统"""
    print("=" * 70)
    print("向量化知识系统演示")
    print("=" * 70)

    system = VectorKnowledgeSystem(max_features=10000, n_clusters=5)

    # 添加测试文档
    test_docs = [
        ("重力", "重力是地球对物体的吸引力。重力使物体落向地面。牛顿发现了万有引力定律。"),
        ("Python", "Python是一种编程语言。Python由吉多·范罗苏姆发明。Python用于Web开发和人工智能。"),
        ("太阳", "太阳是太阳系的中心天体。太阳是一颗恒星。太阳的表面温度约为5500摄氏度。"),
        ("机器学习", "机器学习是人工智能的一个分支。机器学习通过数据来学习规律。深度学习是机器学习的一种。"),
        ("中国", "中国位于亚洲东部。中国的首都是北京。中国是世界上人口最多的国家之一。"),
        ("量子力学", "量子力学是物理学的一个分支。量子力学描述微观粒子的行为。薛定谔提出了波动方程。"),
        ("DNA", "DNA是脱氧核糖核酸的简称。DNA携带遗传信息。DNA的双螺旋结构由沃森和克里克发现。"),
        ("相对论", "相对论是爱因斯坦提出的物理理论。狭义相对论描述了光速不变原理。广义相对论描述了引力。"),
        ("互联网", "互联网是全球计算机网络。互联网由TCP/IP协议连接。万维网是互联网的一种服务。"),
        ("进化论", "进化论是达尔文提出的生物理论。自然选择是进化的主要机制。物种通过适应环境而进化。"),
    ]

    for title, text in test_docs:
        system.add_document(title, text)

    print(f"\n添加了 {len(test_docs)} 个文档")

    # 训练
    print("\n训练模型...")
    system.fit()

    # 测试搜索
    print("\n测试搜索:")
    queries = [
        "什么是重力",
        "Python编程",
        "人工智能",
        "物理理论",
    ]

    for query in queries:
        print(f"\n  查询: {query}")
        results = system.search(query, top_k=3)
        for r in results:
            print(f"    → {r['title']} (相似度: {r['similarity']:.3f})")

    # 测试聚类
    print("\n聚类信息:")
    for i in range(min(5, system.n_clusters)):
        info = system.get_cluster_info(i)
        if info:
            print(f"  簇 {i}: {info['size']} 文档, 关键词: {info['top_words'][:5]}")

    # 统计
    print("\n统计:")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    demo()
