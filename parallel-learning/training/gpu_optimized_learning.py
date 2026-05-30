"""GPU优化版文本处理

利用RTX 4060特性：
1. 批量处理 — 一次处理多个文本
2. 预编译正则 — 避免重复编译
3. 向量化操作 — 使用PyTorch加速
4. 异步处理 — CPU/GPU并行

运行方式：
    python training/gpu_optimized_learning.py
"""

import json
import os
import sys
import time
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PrecompiledPatterns:
    """预编译正则表达式模式"""

    def __init__(self):
        # 语义模式
        self.semantic_patterns = [
            re.compile(r'(.{2,10}?)是(.{2,30}?)$'),
            re.compile(r'(.{2,10}?)属于(.{2,20}?)$'),
            re.compile(r'(.{2,10}?)位于(.{2,20}?)$'),
            re.compile(r'(.{2,10}?)称为(.{2,15}?)$'),
            re.compile(r'(.{2,10}?)叫做(.{2,15}?)$'),
            re.compile(r'(.{2,10}?)指的是(.{2,30}?)$'),
        ]

        # 因果模式
        self.causal_patterns = [
            re.compile(r'因为(.+?)，所以(.+)'),
            re.compile(r'由于(.+?)，(.+)'),
            re.compile(r'(.+)导致(.+)'),
            re.compile(r'(.+)引起(.+)'),
        ]

        # 事件模式
        self.event_patterns = [
            re.compile(r'(.{2,10}?)发明了?(.{2,20}?)$'),
            re.compile(r'(.{2,10}?)发现了?(.{2,20}?)$'),
            re.compile(r'(.{2,10}?)创造了?(.{2,20}?)$'),
            re.compile(r'(.{2,10}?)提出了?(.{2,20}?)$'),
        ]

        # 实体提取模式
        self.entity_pattern = re.compile(r'[一-鿿]{2,6}')
        self.en_entity_pattern = re.compile(r'[A-Z][a-zA-Z]+')

        # 分句模式
        self.sentence_pattern = re.compile(r'[。！？；\n]')

        # 分隔符
        self.separator_pattern = re.compile(r'[，。！？；：、\s的了是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]')


class GPUBatchProcessor:
    """GPU批量处理器"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patterns = PrecompiledPatterns()

    def batch_extract_entities(self, texts: List[str]) -> List[List[str]]:
        """批量提取实体"""
        results = []
        for text in texts:
            entities = self._extract_entities_fast(text)
            results.append(entities)
        return results

    def _extract_entities_fast(self, text: str) -> List[str]:
        """快速实体提取"""
        # 分割
        parts = self.patterns.separator_pattern.split(text)

        entities = []
        for part in parts:
            part = part.strip()
            if not part or len(part) < 2:
                continue

            # 中文实体
            zh_matches = self.patterns.entity_pattern.findall(part)
            entities.extend([e for e in zh_matches if len(e) >= 2])

            # 英文实体
            en_matches = self.patterns.en_entity_pattern.findall(part)
            entities.extend(en_matches)

        return list(set(entities))

    def batch_extract_triples(self, texts: List[str]) -> List[List[Tuple[str, str, str]]]:
        """批量提取三元组"""
        results = []
        for text in texts:
            triples = self._extract_triples_fast(text)
            results.append(triples)
        return results

    def _extract_triples_fast(self, text: str) -> List[Tuple[str, str, str]]:
        """快速三元组提取"""
        triples = []

        # 分句
        sentences = self.patterns.sentence_pattern.split(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # 语义模式
            for pattern in self.patterns.semantic_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    subject = match[0].strip()
                    obj = match[1].strip()
                    if 2 <= len(subject) <= 15 and 2 <= len(obj) <= 30:
                        triples.append((subject, '是', obj))

            # 因果模式
            for pattern in self.patterns.causal_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    cause = match[0].strip()
                    effect = match[1].strip()
                    if 2 <= len(cause) <= 20 and 2 <= len(effect) <= 30:
                        triples.append((cause, '导致', effect))

            # 事件模式
            for pattern in self.patterns.event_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    agent = match[0].strip()
                    obj = match[1].strip()
                    if 2 <= len(agent) <= 15 and 2 <= len(obj) <= 30:
                        triples.append((agent, '执行', obj))

        return triples

    def batch_vectorize(self, texts: List[str], max_features: int = 10000) -> torch.Tensor:
        """批量向量化（GPU加速）"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # CPU向量化
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer='char_wb',
            ngram_range=(2, 4),
            sublinear_tf=True,
            max_df=0.95,
            min_df=2,
        )

        vectors = vectorizer.fit_transform(texts)

        # 转换为GPU tensor
        vectors_dense = vectors.toarray()
        vectors_gpu = torch.from_numpy(vectors_dense).to(self.device)

        return vectors_gpu, vectorizer

    def gpu_kmeans(self, vectors: torch.Tensor, n_clusters: int, max_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU加速KMeans"""
        n_samples = vectors.shape[0]

        # 确保聚类数不超过样本数且大于0
        n_clusters = min(n_clusters, n_samples)
        if n_clusters <= 0:
            n_clusters = 1

        # 随机初始化
        indices = torch.randperm(n_samples)[:n_clusters]
        centroids = vectors[indices].clone()
        new_centroids = torch.zeros_like(centroids)  # 预分配

        for _ in range(max_iter):
            # 计算距离
            distances = torch.cdist(vectors, centroids)

            # 分配聚类
            labels = distances.argmin(dim=1)

            # 更新中心（复用预分配的张量）
            new_centroids.zero_()
            for i in range(n_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    new_centroids[i] = vectors[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]

            # 检查收敛
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break

            # 交换引用而不是复制数据
            centroids, new_centroids = new_centroids, centroids

        return labels, centroids

    def gpu_similarity_search(self, query: torch.Tensor, vectors: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU加速相似度搜索"""
        # 计算余弦相似度（使用F.normalize避免除零）
        query_norm = torch.nn.functional.normalize(query, p=2, dim=-1)
        vectors_norm = torch.nn.functional.normalize(vectors, p=2, dim=-1)

        similarities = torch.mm(query_norm, vectors_norm.T)

        # 获取top_k
        top_values, top_indices = similarities.topk(min(top_k, vectors.shape[0]))

        return top_values, top_indices


class OptimizedLearningSystem:
    """优化版学习系统"""

    def __init__(self):
        self.processor = GPUBatchProcessor()

        # 知识存储
        self.triples: List[Tuple[str, str, str]] = []
        self.entities: Dict[str, Dict] = {}
        self.documents: List[Dict] = []

        # 向量索引
        self.vectors = None
        self.vectorizer = None
        self.cluster_labels = None
        self.cluster_centers = None

        # 统计
        self.stats = {
            'documents_processed': 0,
            'triples_extracted': 0,
            'entities_found': 0,
            'processing_time': 0,
        }

    def learn_batch(self, texts: List[str], sources: List[str] = None):
        """批量学习"""
        start = time.time()

        # 批量提取实体
        entities_batch = self.processor.batch_extract_entities(texts)

        # 批量提取三元组
        triples_batch = self.processor.batch_extract_triples(texts)

        # 存储结果
        for i, (text, entities, triples) in enumerate(zip(texts, entities_batch, triples_batch)):
            doc = {
                'id': len(self.documents),
                'text': text[:500],
                'source': sources[i] if sources else '',
            }
            self.documents.append(doc)

            # 存储三元组
            self.triples.extend(triples)
            self.stats['triples_extracted'] += len(triples)

            # 存储实体
            for entity in entities:
                if entity not in self.entities:
                    self.entities[entity] = {'count': 0, 'contexts': []}
                self.entities[entity]['count'] += 1
                self.entities[entity]['contexts'].append(text[:100])

            self.stats['entities_found'] += len(entities)

        self.stats['documents_processed'] += len(texts)
        self.stats['processing_time'] += time.time() - start

    def build_index(self):
        """构建向量索引（GPU加速）"""
        if not self.documents or len(self.documents) < 2:
            print("文档数不足，跳过索引构建")
            return

        print("构建向量索引...")
        texts = [doc['text'] for doc in self.documents]

        # GPU向量化
        self.vectors, self.vectorizer = self.processor.batch_vectorize(texts)

        # GPU KMeans聚类
        n_clusters = min(1000, len(self.documents) // 10)
        self.cluster_labels, self.cluster_centers = self.processor.gpu_kmeans(self.vectors, n_clusters)

        print(f"  向量维度: {self.vectors.shape}")
        print(f"  聚类数: {n_clusters}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索（GPU加速）"""
        if self.vectors is None or self.vectorizer is None:
            return []

        # 向量化查询
        query_vec = self.vectorizer.transform([query])
        query_gpu = torch.from_numpy(query_vec.toarray()).to(self.processor.device)

        # GPU相似度搜索
        similarities, indices = self.processor.gpu_similarity_search(query_gpu, self.vectors, top_k)

        results = []
        for sim, idx in zip(similarities[0].cpu().numpy(), indices[0].cpu().numpy()):
            doc = self.documents[idx]
            results.append({
                'text': doc['text'][:200],
                'similarity': float(sim),
                'source': doc.get('source', ''),
            })

        return results

    def query(self, question: str) -> Dict:
        """查询"""
        # 搜索相关文档
        search_results = self.search(question, top_k=5)

        # 搜索三元组
        triple_results = []
        keywords = self._extract_keywords(question)

        for keyword in keywords:
            for s, r, o in self.triples:
                if keyword in s or keyword in o:
                    triple_results.append({'subject': s, 'relation': r, 'object': o})

        return {
            'search_results': search_results,
            'triple_results': triple_results[:10],
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        separators = r'[为什么怎么如何的是有在位于属于包括使用产生导致引起因所如果那但而且或者而]'
        parts = re.split(separators, text)

        keywords = []
        for part in parts:
            part = part.strip()
            if part and len(part) >= 2:
                keywords.extend(re.findall(r'[一-鿿]{2,6}', part))

        return list(set(keywords))

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'avg_time_per_doc': self.stats['processing_time'] / max(1, self.stats['documents_processed']),
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        }


def test_optimized_system():
    """测试优化系统"""
    print("=" * 70)
    print("GPU优化学习系统测试")
    print("=" * 70)

    system = OptimizedLearningSystem()

    # 测试数据
    test_texts = [
        '人工智能是计算机科学的一个分支。机器学习是人工智能的一个子领域。',
        'Python是一种编程语言。Python由吉多·范罗苏姆发明。Python用于Web开发和人工智能。',
        '太阳是太阳系的中心天体。太阳是一颗恒星。太阳的表面温度约为5500摄氏度。',
        '牛顿发现了万有引力定律。牛顿是英国物理学家。牛顿出生于1643年。',
        '因为下雨，所以地面湿了。由于全球变暖，冰川开始融化。',
    ]

    # 批量学习
    start = time.time()
    system.learn_batch(test_texts)
    elapsed = time.time() - start

    print(f"\n学习完成:")
    print(f"  文档数: {system.stats['documents_processed']}")
    print(f"  三元组: {system.stats['triples_extracted']}")
    print(f"  实体数: {system.stats['entities_found']}")
    print(f"  耗时: {elapsed:.3f}秒")

    # 构建索引
    system.build_index()

    # 测试查询
    print("\n查询测试:")
    test_queries = [
        "什么是人工智能",
        "Python是什么",
        "牛顿发现了什么",
    ]

    for q in test_queries:
        print(f"\n  问: {q}")
        result = system.query(q)
        print(f"  三元组: {len(result['triple_results'])}个")
        print(f"  搜索结果: {len(result['search_results'])}个")

    # 统计
    print("\n统计:")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_optimized_system()
