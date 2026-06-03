"""文化创新追踪 — 代际知识传递与棘轮效应

模拟文化累积进化：
- 教师将知识传递给学生（跨代传递）
- 学生在已有基础上产生新创新
- 跟踪代际间的知识增长（棘轮效应）
"""

import random
from typing import Dict, List, Optional

import torch

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation


class CulturalInnovationTracker:
    """文化创新追踪器 — 代际知识积累

    核心思想：
    文化累积类似于棘轮（ratchet），每一代在前一代基础上积累，
    不会完全回退。通过模拟知识传递中的部分丢失与创新增益，
    观察知识复杂度的代际增长曲线。
    """

    def __init__(self, knowledge: KnowledgeGraph):
        """
        参数:
            knowledge: 初始知识图谱（第 0 代的基础知识）
        """
        self.knowledge = knowledge
        self.generations: List[Dict] = []
        self.innovation_log: List[Dict] = []

    # ── 代际运行 ─────────────────────────────────────────────

    def run_generation(self,
                       teacher_knowledge: KnowledgeGraph,
                       student_knowledge: KnowledgeGraph,
                       num_rounds: int = 10) -> Dict:
        """运行一代：教师教导学生，记录学生的知识变化

        流程：
        1. 教师将部分知识传递给学生（传递率 80%）
        2. 学生在接收的知识上自主探索，产生新实体
        3. 记录创新指标

        参数:
            teacher_knowledge: 教师的知识图谱
            student_knowledge: 学生的知识图谱（会在过程中被修改）
            num_rounds: 学生自主探索轮数
        返回:
            本代统计字典
        """
        entities_before = len(student_knowledge.entities)

        # 知识传递
        transferred = self.transfer_knowledge(
            teacher_knowledge, student_knowledge, ratio=0.8
        )

        # 学生自主探索：每轮以一定概率发现新实体
        for _ in range(num_rounds):
            self._student_explore(student_knowledge)

        new_entities = len(student_knowledge.entities) - entities_before - transferred
        new_entities = max(0, new_entities)

        gen_stats = {
            'generation': len(self.generations),
            'transferred_entities': transferred,
            'new_entities': new_entities,
            'total_entities': len(student_knowledge.entities),
            'total_relations': len(student_knowledge.relations),
        }
        self.generations.append(gen_stats)

        # 记录本轮创新详情
        self.innovation_log.append({
            'generation': gen_stats['generation'],
            'new_symbols': new_entities,
            'teacher_entities': len(teacher_knowledge.entities),
            'student_entities': len(student_knowledge.entities),
        })

        return gen_stats

    # ── 知识传递 ─────────────────────────────────────────────

    def transfer_knowledge(self,
                           source: KnowledgeGraph,
                           target: KnowledgeGraph,
                           ratio: float = 0.8) -> int:
        """将源图谱中按置信度排序的前 ratio 比例实体传递到目标图谱

        传递时置信度衰减为原来的 0.8 倍，模拟传递损失。
        同时复制与被传递实体相关的关系。

        参数:
            source: 源知识图谱
            target: 目标知识图谱
            ratio: 传递比例 (0, 1]
        返回:
            实际传递的实体数量
        """
        if not source.entities:
            return 0

        # 按置信度降序排列
        sorted_entities = sorted(
            source.entities.values(),
            key=lambda e: e.confidence,
            reverse=True,
        )

        # 选取前 ratio 比例
        n_transfer = max(1, int(len(sorted_entities) * ratio))
        to_transfer = sorted_entities[:n_transfer]

        transferred_ids: List[str] = []
        for entity in to_transfer:
            if entity.id in target.entities:
                # 已存在则合并，取较高置信度
                existing = target.entities[entity.id]
                existing.confidence = max(existing.confidence, entity.confidence * 0.8)
                existing.properties.update(entity.properties)
            else:
                # 创建衰减副本
                copy = Entity(
                    id=entity.id,
                    type=entity.type,
                    properties=dict(entity.properties),
                    embedding=entity.embedding.clone() if entity.embedding is not None else None,
                    confidence=entity.confidence * 0.8,
                    source='transfer',
                    tags=list(entity.tags),
                )
                target.add_entity(copy)
                transferred_ids.append(entity.id)

        # 复制相关关系（仅两端都在目标图谱中的关系）
        transferred_set = set(transferred_ids) | set(target.entities.keys())
        for rel in source.relations:
            if (rel.source_id in transferred_set
                    and rel.target_id in transferred_set):
                target.add_relation(Relation(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    type=rel.type,
                    confidence=rel.confidence * 0.8,
                    evidence_count=rel.evidence_count,
                    metadata=dict(rel.metadata),
                ))

        return len(transferred_ids)

    # ── 学生探索 ─────────────────────────────────────────────

    def _student_explore(self, knowledge: KnowledgeGraph) -> None:
        """学生自主探索，以概率产生新实体或新关系

        模拟创新机制：
        - 30% 概率基于已有实体派生新属性
        - 20% 概率发现全新的概念
        - 50% 概率无新发现
        """
        roll = random.random()
        if roll < 0.30 and knowledge.entities:
            # 派生：选取已有实体，添加新属性
            entity = random.choice(list(knowledge.entities.values()))
            attr_key = f'derived_{random.randint(100, 999)}'
            entity.properties[attr_key] = random.random()
        elif roll < 0.50 and knowledge.entities:
            # 全新概念
            new_id = f'concept_gen{len(self.generations)}_{random.randint(1000, 9999)}'
            if new_id not in knowledge.entities:
                knowledge.add_entity(Entity(
                    id=new_id,
                    type='concept',
                    properties={'novelty': random.random()},
                    confidence=0.3 + random.random() * 0.4,
                    source='innovation',
                    tags=['novel'],
                ))
                # 尝试连接到已有实体
                existing_ids = [
                    eid for eid in knowledge.entities if eid != new_id
                ]
                if existing_ids:
                    link_id = random.choice(existing_ids)
                    knowledge.add_relation(Relation(
                        source_id=new_id,
                        target_id=link_id,
                        type='related_to',
                        confidence=0.3,
                    ))

    # ── 指标计算 ─────────────────────────────────────────────

    def get_innovation_metrics(self) -> Dict:
        """获取跨代累积创新指标

        返回:
            cumulative_entities: 累积实体总数
            cumulative_relations: 累积关系总数
            innovation_rate: 每代平均新增实体数
            complexity_growth: 复杂度增长列表
            ratchet_score: 棘轮效应得分（知识是否持续增长）
        """
        if not self.generations:
            return {
                'cumulative_entities': len(self.knowledge.entities),
                'cumulative_relations': len(self.knowledge.relations),
                'innovation_rate': 0.0,
                'complexity_growth': [],
                'ratchet_score': 0.0,
            }

        total_new = sum(g['new_entities'] for g in self.generations)
        innovation_rate = total_new / len(self.generations)

        # 棘轮得分：实体数量单调递增的比例
        entity_counts = [g['total_entities'] for g in self.generations]
        monotonic_steps = sum(
            1 for i in range(1, len(entity_counts))
            if entity_counts[i] >= entity_counts[i - 1]
        )
        ratchet_score = (
            monotonic_steps / (len(entity_counts) - 1)
            if len(entity_counts) > 1 else 1.0
        )

        return {
            'cumulative_entities': entity_counts[-1],
            'cumulative_relations': self.generations[-1]['total_relations'],
            'innovation_rate': innovation_rate,
            'complexity_growth': self.get_complexity_growth(),
            'ratchet_score': ratchet_score,
        }

    def get_complexity_growth(self) -> List[float]:
        """追踪各代的知识复杂度（关系/实体比）

        返回:
            各代复杂度列表，complexity = relations / max(entities, 1)
        """
        result = []
        for gen in self.generations:
            entities = max(gen['total_entities'], 1)
            result.append(gen['total_relations'] / entities)
        return result

    # ── 持久化 ───────────────────────────────────────────────

    def save_state(self) -> dict:
        """保存追踪器状态为字典"""
        return {
            'knowledge': self.knowledge.save_state(),
            'generations': [dict(g) for g in self.generations],
            'innovation_log': [dict(log) for log in self.innovation_log],
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复追踪器状态"""
        self.knowledge.load_state(state.get('knowledge', {}))
        self.generations = [dict(g) for g in state.get('generations', [])]
        self.innovation_log = [dict(log) for log in state.get('innovation_log', [])]
