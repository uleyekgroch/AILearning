"""教导学习 — 通过教导他人来巩固自身知识"""

from typing import Any, Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.entity import Entity


class TeachingModule:
    """教导学习模块

    核心原理（Protégé Effect）：
    为了教导别人，需要把知识组织得更系统、更清晰。
    教的过程中会发现自己知识的盲点。
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.taught_topics: List[str] = []
        self.knowledge_refinements: int = 0

    def prepare_lesson(self, topic: str) -> Dict[str, Any]:
        """组织教学序列

        从知识图谱中提取关于某主题的知识，组织成教学序列。
        如果发现知识缺口，标记需要先学习的内容。
        """
        entities = self.graph.query(properties={'topic': topic})
        if not entities:
            entities = self.graph.query(tag=topic)

        related = []
        for entity in entities:
            neighbors = self.graph.get_related(entity.id)
            rels = self.graph.get_relations_of(entity.id)
            related.append({
                'entity_id': entity.id,
                'type': entity.type,
                'properties': dict(entity.properties),
                'connections': len(rels),
                'confidence': entity.confidence,
            })

        gaps = self._identify_teaching_gaps(topic, entities)

        lesson = {
            'topic': topic,
            'core_concepts': related,
            'prerequisites': gaps,
            'readiness': self._assess_readiness(topic, entities),
            'depth': self._estimate_knowledge_depth(topic, entities),
        }
        return lesson

    def teach_learner(self, topic: str, learner_knowledge: KnowledgeGraph) -> Dict:
        """教导另一个 learner，同时巩固自身知识

        教导过程中：
        1. 比较双方知识差异
        2. 找到自身不完整的地方
        3. 通过组织教学强化理解
        """
        my_entities = self.graph.query(tag=topic)
        their_entities = learner_knowledge.query(tag=topic)

        my_ids = {e.id for e in my_entities}
        their_ids = {e.id for e in their_entities}

        i_know_exclusive = my_ids - their_ids
        they_know_exclusive = their_ids - my_ids
        shared = my_ids & their_ids

        taught_content = []
        for eid in i_know_exclusive:
            entity = self.graph.get_entity(eid)
            if entity:
                learner_knowledge.add_entity(Entity(
                    id=entity.id, type=entity.type,
                    properties=dict(entity.properties),
                    confidence=entity.confidence * 0.8,
                    source='taught',
                    tags=list(entity.tags),
                ))
                taught_content.append(eid)

        # 巩固自身知识（教别人强化自己的记忆）
        for eid in shared:
            entity = self.graph.get_entity(eid)
            if entity:
                entity.confidence = min(1.0, entity.confidence + 0.05)
                self.knowledge_refinements += 1

        self.taught_topics.append(topic)

        return {
            'topic': topic,
            'taught_count': len(taught_content),
            'learned_from_them': len(they_know_exclusive),
            'knowledge_reinforced': len(shared),
            'gaps_discovered': self._find_gaps_from_teaching(topic, my_entities),
        }

    def refine_knowledge_through_teaching(self, topic: str) -> float:
        """通过教导过程巩固知识，返回巩固提升量"""
        entities = self.graph.query(tag=topic)
        total_improvement = 0.0

        for entity in entities:
            rels = self.graph.get_relations_of(entity.id)
            if len(rels) > 2 and entity.confidence < 0.9:
                improvement = 0.1 * (1.0 - entity.confidence)
                entity.confidence = min(1.0, entity.confidence + improvement)
                total_improvement += improvement

        return total_improvement

    # ── 内部方法 ──────────────────────────────────────────────

    def _identify_teaching_gaps(self, topic: str,
                                 entities: List[Entity]) -> List[str]:
        gaps = []
        for entity in entities:
            if entity.confidence < 0.5:
                gaps.append(f'{entity.id}: 置信度不足 ({entity.confidence:.2f})')
            rels = self.graph.get_relations_of(entity.id)
            if len(rels) == 0:
                gaps.append(f'{entity.id}: 孤立实体，无关联知识')
        return gaps

    def _assess_readiness(self, topic: str,
                          entities: List[Entity]) -> float:
        if not entities:
            return 0.0
        avg_conf = sum(e.confidence for e in entities) / len(entities)
        total_rels = sum(
            len(self.graph.get_relations_of(e.id)) for e in entities
        )
        connectivity = min(1.0, total_rels / max(len(entities) * 2, 1))
        return 0.6 * avg_conf + 0.4 * connectivity

    def _estimate_knowledge_depth(self, topic: str,
                                   entities: List[Entity]) -> int:
        max_depth = 0
        for entity in entities:
            neighbors = self.graph.get_neighbors(entity.id, depth=3)
            depth = len(neighbors)
            max_depth = max(max_depth, depth)
        return max_depth

    def _find_gaps_from_teaching(self, topic: str,
                                  entities: List[Entity]) -> List[str]:
        gaps = []
        for entity in entities:
            rels = self.graph.get_relations_of(entity.id)
            if entity.confidence < 0.3:
                gaps.append(f'低置信度: {entity.id}')
            if not entity.properties:
                gaps.append(f'无属性: {entity.id}')
        return gaps
