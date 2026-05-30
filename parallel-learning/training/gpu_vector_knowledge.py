"""GPU加速向量知识系统

使用PyTorch GPU加速：
- TF-IDF向量化
- KMeans聚类
- KNN检索

运行方式：
    python training/gpu_vector_knowledge.py
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


class GPUVectorKnowledgeSystem:
    """GPU加速向量知识系统

    核心组件：
    1. TF-IDF向量化（CPU，因为sklearn不支持GPU）
    2. KMeans聚类（GPU，使用PyTorch）
    3. KNN检索（GPU，使用PyTorch）
    """

    def __init__(self, max_features=50000, n_clusters=1000):
        """
        Args:
            max_features: TF-IDF最大特征数
            n_clusters: 聚类数
        """
        # 检查CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # TF-IDF向量化（CPU）
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer='char_wb',
            ngram_range=(2, 4),
            sublinear_tf=True,
            max_df=0.95,
            min_df=2,
        )

        # 聚类参数
        self.n_clusters = n_clusters

        # 数据存储
        self.documents: List[Dict] = []
        self.vectors = None  # CPU sparse matrix
        self.vectors_gpu = None  # GPU dense tensor
        self.cluster_labels = None
        self.cluster_centers = None  # GPU tensor

        # 索引
        self.cluster_docs: Dict[int, List[int]] = defaultdict(list)
        self.entity_index: Dict[str, List[int]] = defaultdict(list)

        # 状态
        self.fitted = False
        self.total_processed = 0

    def add_document(self, title: str, text: str, metadata: Dict = None):
        """添加文档"""
        import re
        doc = {
            'id': len(self.documents),
            'title': title,
            'text': text[:1000],
            'metadata': metadata or {},
        }
        self.documents.append(doc)

        # 提取实体
        words = set(re.findall(r'[一-鿿]{2,4}', text))
        for word in words:
            self.entity_index[word].append(doc['id'])

        self.total_processed += 1

    def fit(self):
        """训练模型（GPU加速）"""
        if not self.documents:
            print("  没有文档可训练")
            return

        print(f"  文档数: {len(self.documents):,}")
        print(f"  设备: {self.device}")

        # 1. TF-IDF向量化（CPU）
        print("  [1/3] TF-IDF向量化 (CPU)...")
        start = time.time()
        texts = [doc['text'] for doc in self.documents]
        self.vectors = self.vectorizer.fit_transform(texts)
        print(f"    向量维度: {self.vectors.shape}")
        print(f"    耗时: {time.time()-start:.3f}s")

        # 2. 转换为GPU tensor
        print("  [2/3] 转换到GPU...")
        start = time.time()
        # 转换为dense tensor
        vectors_dense = self.vectors.toarray()
        self.vectors_gpu = torch.from_numpy(vectors_dense).to(self.device)
        print(f"    耗时: {time.time()-start:.3f}s")

        # 3. KMeans聚类（GPU）
        print("  [3/3] KMeans聚类 (GPU)...")
        start = time.time()
        self._gpu_kmeans()
        print(f"    聚类数: {self.n_clusters}")
        print(f"    耗时: {time.time()-start:.3f}s")

        self.fitted = True
        print("  训练完成!")

    def _gpu_kmeans(self):
        """GPU加速KMeans"""
        n_samples = self.vectors_gpu.shape[0]
        n_features = self.vectors_gpu.shape[1]

        # 随机初始化聚类中心
        indices = torch.randperm(n_samples)[:self.n_clusters]
        centroids = self.vectors_gpu[indices].clone()

        # KMeans迭代
        for iteration in range(20):
            # 计算距离（GPU）
            distances = torch.cdist(self.vectors_gpu, centroids)

            # 分配聚类
            labels = distances.argmin(dim=1)

            # 更新聚类中心
            new_centroids = torch.zeros_like(centroids)
            for i in range(self.n_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    new_centroids[i] = self.vectors_gpu[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]

            # 检查收敛
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                print(f"    迭代 {iteration+1} 次后收敛")
                break

            centroids = new_centroids

        # 保存结果
        self.cluster_labels = labels.cpu().numpy()
        self.cluster_centers = centroids

        # 构建簇索引
        for i, label in enumerate(self.cluster_labels):
            self.cluster_docs[int(label)].append(i)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关知识（GPU加速）"""
        if not self.fitted:
            return []

        # 向量化查询
        query_vec = self.vectorizer.transform([query])
        query_gpu = torch.from_numpy(query_vec.toarray()).to(self.device)

        # GPU计算相似度
        similarities = torch.mm(query_gpu, self.vectors_gpu.T).squeeze()

        # 获取top_k
        top_values, top_indices = similarities.topk(min(top_k, len(self.documents)))

        results = []
        for sim, idx in zip(top_values.cpu().numpy(), top_indices.cpu().numpy()):
            doc = self.documents[idx]
            results.append({
                'title': doc['title'],
                'text': doc['text'][:200],
                'similarity': float(sim),
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

        docs = [self.documents[i] for i in doc_indices[:top_n]]

        # 获取簇中心的top特征词
        center = self.cluster_centers[cluster_id].cpu().numpy()
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = np.argsort(center)[-10:][::-1]
        top_words = [feature_names[i] for i in top_features]

        return {
            'cluster_id': cluster_id,
            'size': len(doc_indices),
            'top_words': top_words,
            'sample_docs': [{'title': d['title'], 'text': d['text'][:100]} for d in docs],
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_documents': len(self.documents),
            'total_entities': len(self.entity_index),
            'fitted': self.fitted,
            'device': str(self.device),
        }

        if self.fitted:
            stats['vector_dimensions'] = self.vectors.shape[1]
            stats['n_clusters'] = self.n_clusters
            stats['avg_cluster_size'] = len(self.documents) / self.n_clusters

        return stats


def test_gpu_vector_knowledge():
    """测试GPU加速向量知识系统"""
    print("=" * 70)
    print("GPU加速向量知识系统测试")
    print("=" * 70)

    system = GPUVectorKnowledgeSystem(max_features=10000, n_clusters=10)

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

    # 统计
    print("\n统计:")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_gpu_vector_knowledge()
