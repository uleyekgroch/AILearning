"""分布式概念空间 — 知识在向量空间中的活的表示

核心理念（vs 三元组知识图谱）
==============================

三元组知识图谱：
  "数学" —[是]→ "学科"   静态记录，检索走固定路径
  "物理" —[是]→ "学科"   另一条独立记录
  关系类型预定义：是、导致、包括...

分布式概念空间：
  "数学" = [0.2, -0.5, 0.8, ...]  一个128维向量
  "物理" = [0.3, -0.4, 0.7, ...]  另一个向量
  它们在向量空间中距离近 → 自然关联
  推理 = 激活扩散（从问题概念向相关概念扩散）

人脑没有三元组。当你想到"数学"，激活的是整个概念网络：
  数字 → 公式 → 课堂 → 逻辑 → 推理 → 物理 → ...

核心操作：
1. register() — 注册概念（来自统计学习或感知体验）
2. activate() — 激活扩散推理（从问题出发，联想相关概念）
3. learn_relation() — Hebbian学习（相关的概念向量互相吸引）
4. analogy() — 向量类比（king - man + woman ≈ queen）
"""

import math
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


@dataclass
class ConceptNode:
    """概念节点 — 概念空间中的一个点"""
    id: str                          # 可读标签（如"数学"）
    vector: torch.Tensor             # 概念向量 (dim,)
    frequency: int = 1               # 被观察到的次数
    strength: float = 1.0            # 概念强度（被巩固的程度）
    last_activated: float = 0.0      # 最后被激活的时间戳
    sensory_anchors: List[str] = field(default_factory=list)  # 感知锚点
    source: str = 'text'             # 来源：text/perception/reasoning
    created_at: float = 0.0

    # Phase 1新增：功能性表征（概念不只是一个标签+向量）
    perceptual_features: Dict = field(default_factory=dict)    # 感知特征绑定
    affordances: List[str] = field(default_factory=list)      # 可供性（Gibson 1977）
    usage_contexts: List[Dict] = field(default_factory=list)  # 使用场景

    def boost(self, amount: float = 0.1):
        """增强概念（每次被使用时调用）"""
        self.frequency += 1
        self.strength = min(self.strength + amount, 2.0)
        self.last_activated = time.time()


@dataclass
class ActivatedConcept:
    """被激活的概念（推理结果）"""
    concept_id: str
    activation: float       # 激活强度 (0~1)
    depth: int              # 扩散深度（0=直接, 1=一次扩散...）
    path: List[str]         # 激活路径


class ConceptSpace:
    """分布式概念空间

    使用方式：
        space = ConceptSpace(dim=128, encoder=my_encoder)

        # 注册概念
        space.register("数学", source='text')
        space.register("物理学", source='text')

        # 学习关系（Hebbian）
        space.learn_relation("数学", "物理学", strength=0.5)

        # 激活扩散推理
        results = space.activate("什么是数学？")
        for ac in results:
            print(f"  {ac.concept_id}: activation={ac.activation:.3f}")

        # 向量类比
        result = space.analogy("数学", "数学家", "物理学")
    """

    def __init__(self, dim: int = 128, encoder=None):
        """
        Args:
            dim: 概念向量维度
            encoder: 可选的文本编码器（LearnableTextEncoder实例）
                     如果提供，用它将文本编码为向量
                     如果不提供，用随机初始化
        """
        self.dim = dim
        self.encoder = encoder

        # 概念存储
        self.concepts: Dict[str, ConceptNode] = {}

        # 关系存储（邻接表 + 权重）
        # relations[a][b] = weight 表示 a 和 b 之间的关联强度
        self.relations: Dict[str, Dict[str, float]] = defaultdict(dict)

        # 向量索引（用于快速最近邻搜索）
        self._vector_matrix: Optional[torch.Tensor] = None  # (N, dim)
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._index_dirty = True  # 需要重建索引

    def register(self, text: str, vector: torch.Tensor = None,
                 source: str = 'text', sensory_anchors: List[str] = None) -> ConceptNode:
        """注册新概念

        如果概念已存在，增强它（boost）而不是重复创建。

        Args:
            text: 概念文本（如"数学"）
            vector: 可选的预计算向量。如果不提供，用encoder编码
            source: 来源（text/perception/reasoning/statistical）
            sensory_anchors: 感知锚点（来自感知体验）

        Returns:
            概念节点
        """
        if text in self.concepts:
            # 已存在 → 增强
            node = self.concepts[text]
            node.boost()
            if sensory_anchors:
                node.sensory_anchors.extend(sensory_anchors)
            return node

        # 新概念 → 编码为向量
        if vector is not None:
            vec = vector.detach().clone()
        elif self.encoder is not None:
            with torch.no_grad():
                vec = self.encoder(text).detach().clone()
        else:
            vec = torch.randn(self.dim)
            vec = F.normalize(vec, p=2, dim=0)

        # 归一化
        vec = F.normalize(vec, p=2, dim=0)

        # 感知概念初始强度补偿（Phase 7）
        # 感知概念频率低但不应被遗忘，给予更高初始强度
        initial_strength = 1.0
        if source == 'perception' or (sensory_anchors and len(sensory_anchors) > 0):
            initial_strength = 1.3

        node = ConceptNode(
            id=text,
            vector=vec,
            source=source,
            sensory_anchors=sensory_anchors or [],
            last_activated=time.time(),
            created_at=time.time(),
            strength=initial_strength,
        )
        self.concepts[text] = node
        self._index_dirty = True
        return node

    def learn_relation(self, concept_a: str, concept_b: str,
                       strength: float = 0.1, bidirectional: bool = True):
        """学习概念间的关系（Hebbian更新）

        核心机制：相关的概念向量互相吸引。
        不是存储三元组(subject, relation, object)，
        而是调整向量使相关概念在空间中更接近。

        Args:
            concept_a: 概念A
            concept_b: 概念B
            strength: 关联强度 (0~1)
            bidirectional: 是否双向更新
        """
        if concept_a not in self.concepts or concept_b not in self.concepts:
            return

        # 更新关系图
        existing = self.relations[concept_a].get(concept_b, 0.0)
        self.relations[concept_a][concept_b] = min(existing + strength, 1.0)
        if bidirectional:
            existing_b = self.relations[concept_b].get(concept_a, 0.0)
            self.relations[concept_b][concept_a] = min(existing_b + strength, 1.0)

        # Hebbian更新：向量互相吸引
        with torch.no_grad():
            vec_a = self.concepts[concept_a].vector
            vec_b = self.concepts[concept_b].vector

            # 向量插值：各自向对方靠近一小步
            lr = strength * 0.02
            delta_a = lr * (vec_b - vec_a)
            delta_b = lr * (vec_a - vec_b)

            self.concepts[concept_a].vector = F.normalize(vec_a + delta_a, p=2, dim=0)
            self.concepts[concept_b].vector = F.normalize(vec_b + delta_b, p=2, dim=0)

        self._index_dirty = True

    def activate(self, query: str, top_k: int = 10,
                 spread_depth: int = 2, decay: float = 0.6) -> List[ActivatedConcept]:
        """激活扩散推理 — 核心推理方法

        模拟人脑的联想过程：
        1. 问题编码为向量
        2. 在概念空间中找到最近的概念（直接激活）
        3. 从被激活的概念出发，沿着关系边扩散激活
        4. 收集所有被激活的概念，按激活强度排序

        Args:
            query: 查询文本
            top_k: 直接激活的top-K概念数
            spread_depth: 扩散深度
            decay: 每次扩散的衰减系数

        Returns:
            被激活的概念列表，按激活强度降序
        """
        if not self.concepts:
            return []

        # 1. 编码查询
        if self.encoder is not None:
            with torch.no_grad():
                query_vec = self.encoder(query).detach()
        else:
            query_vec = torch.randn(self.dim)
        query_vec = F.normalize(query_vec, p=2, dim=0)

        # 2. 直接激活：找与查询最相似的概念
        activated: Dict[str, ActivatedConcept] = {}

        all_vecs = torch.stack([c.vector for c in self.concepts.values()])
        all_ids = list(self.concepts.keys())
        sims = F.cosine_similarity(query_vec.unsqueeze(0), all_vecs, dim=1)

        # 取top_k
        top_values, top_indices = sims.topk(min(top_k, len(all_ids)))
        for val, idx in zip(top_values.tolist(), top_indices.tolist()):
            cid = all_ids[idx]
            # 概念质量因子：短概念(<3字)降权，减少碎片概念主导推理
            quality = 1.0
            if len(cid) < 3:
                quality = 0.5  # 短概念可能是碎片
            if len(cid) >= 4:
                quality = 1.2  # 较长概念通常更有意义

            # 感知提权（Phase 7）：有 sensory_anchors 的概念获得竞争力加成
            # 补偿感知概念频率低（1-3次 vs 学术概念10-20次）的劣势
            node = self.concepts[cid]
            sensory_boost = 1.0
            if node.sensory_anchors:
                # 锚点越多 → 加成越大，上限 1.8
                sensory_boost = min(1.0 + len(node.sensory_anchors) * 0.2, 1.8)
            elif getattr(node, 'source', 'text') == 'perception':
                # 来源是感知但暂无锚点 → 基础加成
                sensory_boost = 1.3

            activation = max(0.0, val) * node.strength * quality * sensory_boost
            activated[cid] = ActivatedConcept(
                concept_id=cid,
                activation=activation,
                depth=0,
                path=[cid],
            )
            self.concepts[cid].boost(0.05)

        # 3. 扩散激活
        for depth in range(1, spread_depth + 1):
            new_activated = {}
            for cid, ac in list(activated.items()):
                if ac.depth != depth - 1:
                    continue  # 只从上一层的节点扩散
                # 沿关系边扩散
                neighbors = self.relations.get(cid, {})
                for neighbor_id, weight in neighbors.items():
                    if neighbor_id in activated:
                        continue  # 已被激活

                    # 计算扩散激活强度
                    spread_activation = ac.activation * weight * decay
                    if spread_activation < 0.05:
                        continue  # 太弱，不扩散

                    # 加上向量相似度加成
                    if neighbor_id in self.concepts:
                        sim = F.cosine_similarity(
                            query_vec.unsqueeze(0),
                            self.concepts[neighbor_id].vector.unsqueeze(0)
                        ).item()
                        sim_bonus = max(0, sim) * 0.3
                        spread_activation += sim_bonus

                    if neighbor_id in new_activated:
                        # 多条路径到达同一节点 → 取最强
                        if spread_activation > new_activated[neighbor_id].activation:
                            new_activated[neighbor_id] = ActivatedConcept(
                                concept_id=neighbor_id,
                                activation=spread_activation,
                                depth=depth,
                                path=ac.path + [neighbor_id],
                            )
                    else:
                        new_activated[neighbor_id] = ActivatedConcept(
                            concept_id=neighbor_id,
                            activation=spread_activation,
                            depth=depth,
                            path=ac.path + [neighbor_id],
                        )

            activated.update(new_activated)

        # 4. 排序返回
        results = sorted(activated.values(), key=lambda x: x.activation, reverse=True)
        return results

    def analogy(self, a: str, b: str, c: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """向量类比推理

        a:b :: c:?
        例如：数学:数学家 :: 物理:?  → 物理学家

        原理：在向量空间中计算 b - a + c，找最近的向量
        """
        if a not in self.concepts or b not in self.concepts or c not in self.concepts:
            return []

        with torch.no_grad():
            vec_a = self.concepts[a].vector
            vec_b = self.concepts[b].vector
            vec_c = self.concepts[c].vector

            # 类比向量
            target = vec_b - vec_a + vec_c
            target = F.normalize(target, p=2, dim=0)

        # 找最近的向量（排除a, b, c自身）
        exclude = {a, b, c}
        results = []
        for cid, node in self.concepts.items():
            if cid in exclude:
                continue
            sim = F.cosine_similarity(target.unsqueeze(0), node.vector.unsqueeze(0)).item()
            results.append((cid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_related(self, concept_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """获取与指定概念最相关的概念（综合向量距离+关系权重）"""
        if concept_id not in self.concepts:
            return []

        node = self.concepts[concept_id]
        results = []

        for cid, other in self.concepts.items():
            if cid == concept_id:
                continue
            # 向量相似度
            sim = F.cosine_similarity(
                node.vector.unsqueeze(0), other.vector.unsqueeze(0)
            ).item()
            # 关系权重加成
            rel_weight = self.relations.get(concept_id, {}).get(cid, 0.0)
            # 综合分数
            score = max(0, sim) * 0.7 + rel_weight * 0.3
            results.append((cid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def consolidate(self, decay: float = 0.95):
        """记忆巩固 — 模拟睡眠中的记忆整理

        弱概念衰减，强概念被保留。
        类似人类睡眠中的记忆巩固过程。
        """
        to_remove = []
        for cid, node in self.concepts.items():
            node.strength *= decay
            if node.strength < 0.1 and node.frequency < 3:
                to_remove.append(cid)

        for cid in to_remove:
            del self.concepts[cid]
            self.relations.pop(cid, None)
            for neighbors in self.relations.values():
                neighbors.pop(cid, None)

        if to_remove:
            self._index_dirty = True

    def get_stats(self) -> Dict:
        """获取概念空间统计"""
        concepts_with_grounding = sum(
            1 for c in self.concepts.values() if c.sensory_anchors
        )
        total_relations = sum(
            len(neighbors) for neighbors in self.relations.values()
        )
        return {
            'total_concepts': len(self.concepts),
            'total_relations': total_relations,
            'concepts_with_grounding': concepts_with_grounding,
            'sources': defaultdict(int, {
                c.source: c.frequency for c in self.concepts.values()
            }),
            'top_by_strength': sorted(
                [(c.id, round(c.strength, 3)) for c in self.concepts.values()],
                key=lambda x: x[1], reverse=True
            )[:10],
        }
