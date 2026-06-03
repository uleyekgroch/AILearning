#!/usr/bin/env python3
"""图式学习 (Schema-Based Learning)

基于论文:
- Nature Communications 2022: "Schemas provide a scaffold for neocortical
  integration of new congruent information"
- Neuron 2018: "Consolidation Promotes the Emergence of Representational
  Overlap in the Hippocampus and Medial Prefrontal Cortex"
- Tse et al. 2007: "Schema-mediated learning in the hippocampus"
- Tse et al. 2011: "Schema-dependent learning in the hippocampus"

核心思想:
  图式(Schema)是已有的知识结构（框架/骨架），新知识通过与图式
  匹配来加速学习：
  1. 与已有图式一致的知识 → 快速整合（不需要海马体中转）
  2. 与已有图式矛盾的知识 → 慢速学习（需要冲突解决）
  3. 完全陌生的知识 → 创建新图式

  类比：学物理的人学数学比学文学的人快，因为物理知识提供了一个
  数学图式（框架），新的数学概念可以"挂"到已有框架上。

  对学习系统的意义:
  - 相似领域的知识学得更快（迁移加速）
  - 系统自动发现和维护知识图式
  - 用图式指导主动推理的学习目标选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import hashlib
import math


@dataclass
class Schema:
    """知识图式

    一个图式是一组相关的概念和关系的抽象框架。
    例如："物理定律"图式包含 "力→加速度"、"能量守恒"等。
    """
    schema_id: str                    # 唯一标识
    name: str                         # 图式名称
    centroid: torch.Tensor            # 中心嵌入（图式的抽象表征）
    members: Set[str] = field(default_factory=set)  # 属于此图式的实体
    relations: List[Tuple[str, str]] = field(default_factory=list)  # 关系模板
    strength: float = 1.0             # 图式强度（被验证次数越多越强）
    created_at: int = 0               # 创建时间
    last_updated: int = 0             # 最后更新时间


class SchemaLearningSystem:
    """图式学习系统

    核心功能:
    1. 自动发现知识图式（通过实体聚类）
    2. 用图式加速新知识整合
    3. 图式冲突检测与解决
    4. 图式驱动的学习策略选择
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 图式库
        self.schemas: Dict[str, Schema] = {}

        # 实体到图式的映射
        self.entity_to_schema: Dict[str, str] = {}

        # 图式发现阈值
        self.similarity_threshold = 0.6  # 高于此值认为是同一图式
        self.merge_threshold = 0.8       # 高于此值合并图式

        # 统计
        self.stats = {
            'schemas_created': 0,
            'schemas_merged': 0,
            'knowledge_accelerated': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
        }

        self._time = 0

    def find_or_create_schema(self, entity_id: str, entity_embedding: torch.Tensor) -> Tuple[str, bool]:
        """为实体找到或创建图式

        如果实体与已有图式相似，归入该图式（加速学习）。
        如果不相似任何图式，创建新图式。

        Returns:
            (schema_id, is_new_schema)
        """
        self._time += 1
        emb = entity_embedding.to(self.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        best_schema = None
        best_sim = 0.0

        # 查找最匹配的图式
        for sid, schema in self.schemas.items():
            schema_emb = schema.centroid.unsqueeze(0) if schema.centroid.dim() == 1 else schema.centroid
            sim = F.cosine_similarity(emb, schema_emb).item()
            if sim > best_sim:
                best_sim = sim
                best_schema = sid

        if best_sim >= self.similarity_threshold and best_schema is not None:
            # 归入已有图式
            schema = self.schemas[best_schema]
            schema.members.add(entity_id)
            self.entity_to_schema[entity_id] = best_schema

            # 更新图式中心（EMA）
            alpha = 1.0 / (1 + len(schema.members))
            with torch.no_grad():
                schema.centroid = (1 - alpha) * schema.centroid + alpha * entity_embedding.detach()
            schema.last_updated = self._time
            schema.strength += 0.1

            return best_schema, False
        else:
            # 创建新图式
            schema_id = f"schema_{self.stats['schemas_created']}"
            schema = Schema(
                schema_id=schema_id,
                name=f"图式-{entity_id[:10]}",
                centroid=entity_embedding.detach().clone(),
                members={entity_id},
                strength=1.0,
                created_at=self._time,
                last_updated=self._time,
            )
            self.schemas[schema_id] = schema
            self.entity_to_schema[entity_id] = schema_id
            self.stats['schemas_created'] += 1

            return schema_id, True

    def compute_learning_acceleration(self, entity_id: str,
                                       entity_embedding: torch.Tensor) -> float:
        """计算图式加速因子

        如果新知识匹配已有图式 → 学习加速（更高优先级，更快整合）
        如果不匹配任何图式 → 正常速度（需要从头学习）

        Returns:
            acceleration: 1.0-3.0 (1.0=无加速, 3.0=最大加速)
        """
        if not self.schemas:
            return 1.0

        emb = entity_embedding.to(self.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        max_sim = 0.0
        for schema in self.schemas.values():
            schema_emb = schema.centroid.unsqueeze(0) if schema.centroid.dim() == 1 else schema.centroid
            sim = F.cosine_similarity(emb, schema_emb).item()
            if sim > max_sim:
                max_sim = sim

        if max_sim >= self.similarity_threshold:
            # 图式匹配 → 加速（图式越强，加速越大）
            best_schema = max(self.schemas.values(), key=lambda s: s.strength)
            acceleration = 1.0 + max_sim * best_schema.strength * 0.5
            self.stats['knowledge_accelerated'] += 1
            return min(3.0, acceleration)
        else:
            return 1.0

    def detect_conflict(self, subject: str, relation: str, obj: str,
                        confidence: float) -> Optional[Dict]:
        """检测新知识是否与已有图式冲突

        如果新关系与图式中已有的关系矛盾（同一subject+relation，不同obj），
        触发冲突解决流程。
        """
        if subject not in self.entity_to_schema:
            return None

        schema_id = self.entity_to_schema[subject]
        schema = self.schemas.get(schema_id)
        if schema is None:
            return None

        # 检查图式中是否有矛盾的关系
        for existing_rel, existing_obj in schema.relations:
            if existing_rel == relation and existing_obj != obj:
                self.stats['conflicts_detected'] += 1
                return {
                    'schema_id': schema_id,
                    'existing_rel': existing_rel,
                    'existing_obj': existing_obj,
                    'new_obj': obj,
                    'conflict_type': 'contradiction',
                }

        return None

    def resolve_conflict(self, conflict: Dict, new_confidence: float) -> str:
        """解决图式冲突

        策略:
        1. 如果新知识置信度更高 → 更新图式
        2. 如果图式更强 → 保持不变
        3. 如果两者相当 → 条件化（保留两者）
        """
        schema = self.schemas.get(conflict['schema_id'])
        if schema is None:
            return 'no_schema'

        # 图式强度 vs 新知识置信度
        if new_confidence > schema.strength * 0.8:
            # 新知识更可靠，更新图式
            for i, (rel, obj) in enumerate(schema.relations):
                if rel == conflict['existing_rel'] and obj == conflict['existing_obj']:
                    schema.relations[i] = (rel, conflict['new_obj'])
                    break
            self.stats['conflicts_resolved'] += 1
            return 'updated'
        else:
            # 保持图式
            return 'preserved'

    def get_related_schemas(self, entity_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """获取与实体相关的图式"""
        if entity_id not in self.entity_to_schema:
            return []

        current_schema_id = self.entity_to_schema[entity_id]
        current = self.schemas[current_schema_id]

        results = []
        for sid, schema in self.schemas.items():
            if sid == current_schema_id:
                continue
            sim = F.cosine_similarity(
                current.centroid.unsqueeze(0),
                schema.centroid.unsqueeze(0)
            ).item()
            if sim > 0.3:
                results.append((sid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def suggest_learning_targets(self, encoder_fn, top_k: int = 5) -> List[Dict]:
        """建议下一步学习目标（基于图式空白）

        找到图式中最弱的领域，建议学习能增强图式的内容。
        """
        if not self.schemas:
            return []

        suggestions = []
        for sid, schema in self.schemas.items():
            # 图式强度低 = 需要更多知识来巩固
            if schema.strength < 3.0:
                suggestions.append({
                    'schema_id': sid,
                    'schema_name': schema.name,
                    'current_strength': schema.strength,
                    'member_count': len(schema.members),
                    'priority': 3.0 - schema.strength,  # 越弱优先级越高
                    'suggestion': f"学习更多关于{schema.name}的知识",
                })

        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        return suggestions[:top_k]

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'total_schemas': len(self.schemas),
            'total_entities_mapped': len(self.entity_to_schema),
            'avg_schema_strength': (
                sum(s.strength for s in self.schemas.values()) / max(1, len(self.schemas))
            ),
        }
