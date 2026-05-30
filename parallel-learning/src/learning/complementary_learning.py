#!/usr/bin/env python3
"""互补学习系统 (Complementary Learning Systems)

基于论文:
- McClelland, McNaughton & O'Reilly 1995: "Why there are complementary learning systems
  in the hippocampus and neocortex"
- Nature Neuroscience 2023: "Organizing memories for generalization in complementary
  learning systems"
- arXiv 2025: "A Neural Network Model of Complementary Learning Systems"
- Kumaran & McClelland 2012: "Generalization through the recurrent interaction
  of episodic memories"

核心思想:
  大脑有两个互补的学习系统：
  1. 海马体(Hippocampus): 快速、稀疏、模式分离
     - 一次学习就能记住具体事件
     - 不同事件的表征高度分离（防止干扰）
     - 容量有限，需要定期"清空"

  2. 新皮层(Neocortex): 慢速、分布式、模式完成
     - 多次暴露才能提取统计规律
     - 共享表征，支持泛化
     - 容量巨大，长期稳定

  两者通过回放(Replay)桥接：
  - 海马体存储的新记忆在睡眠时回放
  - 新皮层从回放中慢慢提取共性
  - 最终海马体的具体记忆被整合到皮层的抽象知识中

  对学习系统的意义:
  - 快速记忆区（情节记忆）：记住每个具体实例
  - 慢速抽象区（语义记忆）：提取跨实例的统计规律
  - 回放整合：从具体到抽象的自动桥梁
  - 解决稳定性-可塑性困境（stability-plasticity dilemma）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import math
import hashlib


@dataclass
class EpisodicTrace:
    """海马体情节记忆痕迹

    特征: 稀疏、分离、一次形成
    """
    trace_id: str                   # 唯一标识
    content: str                    # 原始内容
    embedding: torch.Tensor         # 稀疏编码的嵌入
    entities: List[str]             # 涉及的实体
    context: str                    # 上下文
    timestamp: int                  # 学习时间
    replay_count: int = 0           # 回放次数
    consolidation_score: float = 0.0  # 巩固进度(0-1)
    active: bool = True             # 是否仍在海马体中


@dataclass
class SemanticPattern:
    """皮层语义记忆模式

    特征: 分布式、共享表征、多次形成
    """
    pattern_id: str                 # 唯一标识
    concept: str                    # 概念名称
    embedding: torch.Tensor         # 分布式嵌入
    exemplars: List[str]            # 来源实例ID列表
    frequency: int                  # 遇到次数
    abstraction_level: float        # 抽象程度(0-1)
    confidence: float               # 置信度
    related_patterns: Set[str] = field(default_factory=set)  # 关联模式


class HippocampalMemory:
    """海马体记忆系统

    特征:
    - 快速编码: 一次学习即可存储
    - 模式分离: 不同记忆的正交化表征
    - 临时存储: 定期转移到皮层
    - 稀疏激活: 只有少数神经元活跃
    """

    def __init__(self, d_model: int, capacity: int = 5000, sparsity: float = 0.1,
                 device: str = 'cpu'):
        self.d_model = d_model
        self.capacity = capacity
        self.sparsity = sparsity  # 稀疏度：活跃神经元的比例
        self.device = torch.device(device)

        # 情节记忆存储
        self.traces: Dict[str, EpisodicTrace] = {}

        # 模式分离矩阵（使不同输入正交化）
        self.separator = nn.Linear(d_model, d_model, bias=False).to(self.device)
        # 初始化为接近正交的矩阵
        nn.init.orthogonal_(self.separator.weight)

        # 稀疏化阈值
        self.sparse_threshold = 1.0 - sparsity

    def encode(self, embedding: torch.Tensor) -> torch.Tensor:
        """稀疏编码 — 模式分离

        使不同输入的表征尽可能正交，防止干扰。
        这是海马体CA3区域的核心功能。
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # 确保在正确设备上
        embedding = embedding.to(self.device)

        # 通过分离矩阵
        separated = self.separator(embedding)

        # 稀疏化：只保留最强的激活
        k = max(1, int(self.d_model * self.sparsity))
        topk_values, topk_indices = torch.topk(separated, k, dim=-1)

        sparse = torch.zeros_like(separated)
        sparse.scatter_(-1, topk_indices, topk_values)

        return sparse.squeeze(0)

    def store(self, content: str, embedding: torch.Tensor,
              entities: List[str], context: str, timestamp: int) -> str:
        """快速存储一个情节记忆

        海马体的核心能力：一次编码即可记住。
        """
        trace_id = hashlib.md5(content.encode()).hexdigest()[:12]

        # 模式分离编码
        sparse_emb = self.encode(embedding.detach())

        trace = EpisodicTrace(
            trace_id=trace_id,
            content=content,
            embedding=sparse_emb,
            entities=entities,
            context=context,
            timestamp=timestamp,
        )

        self.traces[trace_id] = trace

        # 超容量时标记最旧的为非活跃
        if len(self.traces) > self.capacity:
            self._evict_oldest()

        return trace_id

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[EpisodicTrace]:
        """检索最相关的情节记忆

        使用模式完成（pattern completion）——
        即使查询只匹配部分特征，也能恢复完整记忆。
        """
        query = self.encode(query_embedding)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        scored = []
        for trace_id, trace in self.traces.items():
            if not trace.active:
                continue

            # 余弦相似度
            trace_emb = trace.embedding.unsqueeze(0) if trace.embedding.dim() == 1 else trace.embedding
            sim = F.cosine_similarity(query, trace_emb).item()
            scored.append((sim, trace))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [trace for _, trace in scored[:top_k]]

    def _evict_oldest(self):
        """淘汰最旧且已巩固的记忆"""
        candidates = [
            (t.timestamp, tid) for tid, t in self.traces.items()
            if t.consolidation_score > 0.5
        ]
        if candidates:
            candidates.sort()
            oldest_id = candidates[0][1]
            self.traces[oldest_id].active = False

    def get_active_count(self) -> int:
        return sum(1 for t in self.traces.values() if t.active)


class NeocorticalMemory:
    """新皮层记忆系统

    特征:
    - 慢速学习: 多次暴露才能形成
    - 分布式表征: 概念在大量神经元上分布
    - 模式完成: 部分输入即可恢复完整模式
    - 泛化能力: 从具体实例提取抽象规律
    """

    def __init__(self, d_model: int):
        self.d_model = d_model

        # 语义模式存储
        self.patterns: Dict[str, SemanticPattern] = {}

        # 概念到模式的映射
        self.concept_index: Dict[str, str] = {}

    def integrate(self, concept: str, embedding: torch.Tensor,
                  exemplar_id: str) -> SemanticPattern:
        """从情节记忆整合到语义记忆

        皮层的慢速学习：每次只微调一点。
        需要多次暴露才能形成稳定的语义表征。
        """
        if concept in self.concept_index:
            # 已有模式：缓慢更新
            pattern_id = self.concept_index[concept]
            pattern = self.patterns[pattern_id]

            # EMA更新嵌入（慢速学习）
            alpha = 1.0 / (1 + pattern.frequency)  # 频率越高，更新越慢
            with torch.no_grad():
                pattern.embedding = (
                    (1 - alpha) * pattern.embedding + alpha * embedding.detach()
                )

            pattern.frequency += 1
            pattern.exemplars.append(exemplar_id)

            # 抽象程度随频率增加
            pattern.abstraction_level = min(1.0, pattern.frequency / 10.0)

            # 置信度也随频率增加
            pattern.confidence = min(1.0, 0.3 + pattern.frequency * 0.05)

            return pattern
        else:
            # 新概念：创建初始模式
            pattern_id = hashlib.md5(concept.encode()).hexdigest()[:12]
            pattern = SemanticPattern(
                pattern_id=pattern_id,
                concept=concept,
                embedding=embedding.detach().clone(),
                exemplars=[exemplar_id],
                frequency=1,
                abstraction_level=0.1,
                confidence=0.3,
            )
            self.patterns[pattern_id] = pattern
            self.concept_index[concept] = pattern_id
            return pattern

    def retrieve(self, concept: str) -> Optional[SemanticPattern]:
        """检索语义模式"""
        if concept in self.concept_index:
            return self.patterns[self.concept_index[concept]]
        return None

    def find_related(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """查找相关概念 — 泛化的核心

        通过嵌入相似度发现概念间的关系。
        这是皮层泛化能力的体现：
        没有直接学过"A和B相关"，但通过共享特征可以发现。
        """
        if concept not in self.concept_index:
            return []

        pattern = self.patterns[self.concept_index[concept]]
        query = pattern.embedding
        if query.dim() == 1:
            query = query.unsqueeze(0)

        results = []
        for pid, p in self.patterns.items():
            if pid == self.concept_index[concept]:
                continue
            p_emb = p.embedding.unsqueeze(0) if p.embedding.dim() == 1 else p.embedding
            sim = F.cosine_similarity(query, p_emb).item()
            if sim > 0.3:
                results.append((p.concept, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def generalize(self, concept: str, target_concept: str) -> Optional[float]:
        """泛化推理：从已知概念推断未知概念的属性

        如果A和B在语义空间中很接近，且A有属性P，
        那么B可能也有属性P（泛化）。
        """
        if concept not in self.concept_index or target_concept not in self.concept_index:
            return None

        p_a = self.patterns[self.concept_index[concept]]
        p_b = self.patterns[self.concept_index[target_concept]]

        emb_a = p_a.embedding.unsqueeze(0) if p_a.embedding.dim() == 1 else p_a.embedding
        emb_b = p_b.embedding.unsqueeze(0) if p_b.embedding.dim() == 1 else p_b.embedding

        similarity = F.cosine_similarity(emb_a, emb_b).item()

        # 泛化强度 = 相似度 × 两个模式的置信度
        generalization = similarity * p_a.confidence * p_b.confidence
        return max(0.0, generalization)


class ComplementaryLearningSystem:
    """互补学习系统 — 完整的海马体-皮层双系统

    使用方法:
    1. 学习新知识: store_episode() — 快速存储到海马体
    2. 回放整合: replay_consolidate() — 从海马体转移到皮层
    3. 检索: retrieve() — 先查皮层（语义），再查海马体（情节）
    4. 泛化: generalize() — 通过皮层共享表征推断新知识
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 双系统
        self.hippocampus = HippocampalMemory(d_model, device=device)
        self.neocortex = NeocorticalMemory(d_model)

        # 时间计数器
        self._time = 0

        # 统计
        self.stats = {
            'episodes_stored': 0,
            'replays_executed': 0,
            'patterns_consolidated': 0,
            'generalizations_made': 0,
        }

    def store_episode(self, content: str, embedding: torch.Tensor,
                      entities: List[str], context: str = '') -> str:
        """存储情节记忆（海马体快速编码）

        一次学习即可记住！这是海马体的核心能力。
        """
        self._time += 1
        trace_id = self.hippocampus.store(
            content=content,
            embedding=embedding,
            entities=entities,
            context=context,
            timestamp=self._time,
        )
        self.stats['episodes_stored'] += 1
        return trace_id

    def replay_consolidate(self, batch_size: int = 50) -> Dict:
        """回放巩固 — 从海马体转移到皮层

        模拟睡眠回放：
        1. 从海马体选择记忆
        2. 在皮层中整合（慢速学习）
        3. 标记已巩固的记忆

        Args:
            batch_size: 每次回放多少条

        Returns:
            巩固统计
        """
        # 选择候选：未巩固或巩固度低的
        candidates = [
            t for t in self.hippocampus.traces.values()
            if t.active and t.consolidation_score < 1.0
        ]

        if not candidates:
            return {'consolidated': 0}

        # 优先选择: 高重要性(实体多) + 低巩固度
        candidates.sort(key=lambda t: (
            len(t.entities) * (1 - t.consolidation_score)
        ), reverse=True)

        consolidated = 0
        for trace in candidates[:batch_size]:
            # 在皮层中整合每个实体
            for entity in trace.entities:
                self.neocortex.integrate(
                    concept=entity,
                    embedding=trace.embedding,
                    exemplar_id=trace.trace_id,
                )

            # 更新巩固进度
            trace.replay_count += 1
            trace.consolidation_score = min(1.0, trace.consolidation_score + 0.2)
            consolidated += 1

        self.stats['replays_executed'] += 1
        self.stats['patterns_consolidated'] += consolidated

        return {
            'consolidated': consolidated,
            'hippocampus_active': self.hippocampus.get_active_count(),
            'neocortex_patterns': len(self.neocortex.patterns),
        }

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5
                 ) -> List[Tuple[str, float, str]]:
        """双系统检索

        策略：
        1. 先查皮层（语义记忆）— 快速、泛化
        2. 再查海马体（情节记忆）— 具体、详细
        3. 合并结果

        Returns:
            [(content, score, source), ...] source = 'semantic' or 'episodic'
        """
        results = []

        # 皮层检索：查找最相关的概念
        query = query_embedding
        if query.dim() == 1:
            query = query.unsqueeze(0)

        for pid, pattern in self.neocortex.patterns.items():
            p_emb = pattern.embedding.unsqueeze(0) if pattern.embedding.dim() == 1 else pattern.embedding
            sim = F.cosine_similarity(query, p_emb).item()
            if sim > 0.3:
                results.append((pattern.concept, sim * pattern.confidence, 'semantic'))

        # 海马体检索：查找具体的情节
        episodic = self.hippocampus.retrieve(query_embedding, top_k=top_k)
        for trace in episodic:
            trace_emb = trace.embedding.unsqueeze(0) if trace.embedding.dim() == 1 else trace.embedding
            sim = F.cosine_similarity(query, trace_emb).item()
            if sim > 0.2:
                # 情节记忆的分数受巩固度影响
                results.append((trace.content[:80], sim * (1 - trace.consolidation_score * 0.5), 'episodic'))

        # 合并排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def generalize(self, from_concept: str, to_concept: str) -> Optional[float]:
        """泛化推理

        通过皮层语义空间的相似性进行推理。
        """
        result = self.neocortex.generalize(from_concept, to_concept)
        if result is not None:
            self.stats['generalizations_made'] += 1
        return result

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'hippocampus_traces': len(self.hippocampus.traces),
            'hippocampus_active': self.hippocampus.get_active_count(),
            'neocortex_patterns': len(self.neocortex.patterns),
        }
