"""迁移学习引擎 — 跨领域知识复用"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation, REL_IS_A, REL_SIMILAR_TO


@dataclass
class StructuralMapping:
    source_domain: str
    target_domain: str
    entity_map: Dict[str, str]        # source_entity_id → target_entity_id
    relation_map: Dict[str, str]      # source_rel_type → target_rel_type
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferredKnowledge:
    source_domain: str
    target_domain: str
    mapping: StructuralMapping
    new_entities: List[str]            # 新创建的实体 ID
    new_relations: List[str]           # 迁移的关系描述
    confidence: float


class TransferEngine:
    """迁移学习引擎

    核心思路：
    1. 找到两个领域之间的结构相似性
    2. 将源领域的知识映射到目标领域
    3. 验证迁移后的知识是否合理
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.transfer_history: List[TransferredKnowledge] = []

    def find_structural_similarity(
        self, domain_a: str, domain_b: str,
    ) -> Optional[StructuralMapping]:
        """发现两个领域之间的结构映射"""
        entities_a = self._get_domain_entities(domain_a)
        entities_b = self._get_domain_entities(domain_b)

        if not entities_a or not entities_b:
            return None

        entity_map = {}
        for ea in entities_a:
            best_match = self._find_best_match(ea, entities_b)
            if best_match:
                entity_map[ea] = best_match

        rel_map = self._map_relations(entity_map)

        confidence = self._compute_mapping_confidence(entity_map, rel_map)

        return StructuralMapping(
            source_domain=domain_a,
            target_domain=domain_b,
            entity_map=entity_map,
            relation_map=rel_map,
            confidence=confidence,
        )

    def transfer_knowledge(
        self, source_domain: str, target_domain: str,
        mapping: Optional[StructuralMapping] = None,
    ) -> Optional[TransferredKnowledge]:
        """将源领域知识迁移到目标领域"""
        if mapping is None:
            mapping = self.find_structural_similarity(source_domain, target_domain)
        if mapping is None:
            return None

        new_entity_ids: List[str] = []
        new_relation_descs: List[str] = []

        for src_id, tgt_id in mapping.entity_map.items():
            src_entity = self.graph.get_entity(src_id)
            if src_entity is None:
                continue

            tgt_entity = self.graph.get_entity(tgt_id)

            # 迁移源实体的属性到目标实体
            for prop_key, prop_val in src_entity.properties.items():
                if tgt_entity and prop_key not in tgt_entity.properties:
                    tgt_entity.properties[prop_key] = prop_val
                    new_relation_descs.append(
                        f'{tgt_id}.{prop_key} = {prop_val}'
                    )

            # 迁移源实体的关系到目标领域
            src_relations = self.graph.get_relations_of(src_id, 'out')
            for rel in src_relations:
                mapped_target = mapping.entity_map.get(rel.target_id)
                if mapped_target and tgt_entity:
                    existing = [r for r in self.graph.get_relations_of(tgt_id, 'out')
                                if r.target_id == mapped_target
                                and r.type == rel.type]
                    if not existing:
                        self.graph.add_relation(Relation(
                            source_id=tgt_id,
                            target_id=mapped_target,
                            type=rel.type,
                            confidence=rel.confidence * mapping.confidence,
                            evidence_count=0,
                            metadata={'transferred_from': src_id, 'source_domain': source_domain},
                        ))
                        new_relation_descs.append(
                            f'{tgt_id} --{rel.type}--> {mapped_target}'
                        )

        # 添加领域间相似关系
        for src_id, tgt_id in mapping.entity_map.items():
            self.graph.add_relation(Relation(
                source_id=src_id, target_id=tgt_id,
                type=REL_SIMILAR_TO,
                confidence=mapping.confidence,
                metadata={'transfer': source_domain},
            ))

        overall_conf = mapping.confidence * (
            len(new_relation_descs) / max(len(mapping.entity_map), 1)
        )

        tk = TransferredKnowledge(
            source_domain=source_domain,
            target_domain=target_domain,
            mapping=mapping,
            new_entities=new_entity_ids,
            new_relations=new_relation_descs,
            confidence=overall_conf,
        )
        self.transfer_history.append(tk)
        return tk

    def validate_transfer(self, transferred: TransferredKnowledge,
                          observations: List[Dict]) -> float:
        """验证迁移的知识是否与实际观察一致"""
        if not observations:
            return transferred.confidence

        matches = 0
        for obs in observations:
            entity_id = obs.get('entity_id', '')
            prop = obs.get('property')
            expected_val = obs.get('value')

            entity = self.graph.get_entity(entity_id)
            if entity and prop and prop in entity.properties:
                if entity.properties[prop] == expected_val:
                    matches += 1

        accuracy = matches / len(observations)
        transferred.confidence = accuracy
        return accuracy

    def suggest_transfer_candidates(self, target_domain: str) -> List[str]:
        """建议可能的源领域"""
        target_entities = self._get_domain_entities(target_domain)
        if not target_entities:
            return []

        scores: Dict[str, float] = {}
        all_types = set()
        for eid in target_entities:
            entity = self.graph.get_entity(eid)
            if entity:
                all_types.add(entity.type)

        for eid in self.graph.entities:
            entity = self.graph.entities[eid]
            domain = self._infer_domain(entity)
            if domain and domain != target_domain:
                if entity.type in all_types:
                    scores[domain] = scores.get(domain, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in ranked[:5]]

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'transfer_history': [
                {
                    'source_domain': tk.source_domain,
                    'target_domain': tk.target_domain,
                    'entity_map': tk.mapping.entity_map,
                    'relation_map': tk.mapping.relation_map,
                    'confidence': tk.confidence,
                    'new_relations': tk.new_relations,
                }
                for tk in self.transfer_history
            ],
        }

    def load_state(self, state: dict) -> None:
        self.transfer_history.clear()
        for td in state.get('transfer_history', []):
            mapping = StructuralMapping(
                source_domain=td['source_domain'],
                target_domain=td['target_domain'],
                entity_map=td.get('entity_map', {}),
                relation_map=td.get('relation_map', {}),
                confidence=td.get('confidence', 0.5),
            )
            self.transfer_history.append(TransferredKnowledge(
                source_domain=td['source_domain'],
                target_domain=td['target_domain'],
                mapping=mapping,
                new_entities=[],
                new_relations=td.get('new_relations', []),
                confidence=td.get('confidence', 0.5),
            ))

    # ── 内部方法 ──────────────────────────────────────────────

    def _get_domain_entities(self, domain: str) -> List[str]:
        entities = self.graph.query(
            properties={'domain': domain},
        )
        if entities:
            return [e.id for e in entities]
        entities = self.graph.query(tag=domain)
        if entities:
            return [e.id for e in entities]
        prefix_entities = [
            eid for eid in self.graph.entities
            if eid.startswith(f'{domain}:')
        ]
        return prefix_entities

    def _find_best_match(self, source_id: str,
                         candidates: List[str]) -> Optional[str]:
        src = self.graph.get_entity(source_id)
        if src is None:
            return None

        best_id: Optional[str] = None
        best_score = -1.0

        for cand_id in candidates:
            cand = self.graph.get_entity(cand_id)
            if cand is None:
                continue
            score = self._entity_similarity(src, cand)
            if score > best_score:
                best_score = score
                best_id = cand_id

        return best_id if best_score > 0.3 else None

    def _entity_similarity(self, a: Entity, b: Entity) -> float:
        if a.type != b.type:
            return 0.0

        shared = set(a.properties.keys()) & set(b.properties.keys())
        total = set(a.properties.keys()) | set(b.properties.keys())
        key_overlap = len(shared) / max(len(total), 1)

        val_match = 0
        for k in shared:
            if a.properties[k] == b.properties[k]:
                val_match += 1
        val_overlap = val_match / max(len(shared), 1)

        return 0.5 * key_overlap + 0.5 * val_overlap

    def _map_relations(self, entity_map: Dict[str, str]) -> Dict[str, str]:
        rel_types: Set[str] = set()
        for src_id in entity_map:
            for rel in self.graph.get_relations_of(src_id, 'out'):
                if rel.target_id in entity_map:
                    rel_types.add(rel.type)
        return {rt: rt for rt in rel_types}

    def _compute_mapping_confidence(self, entity_map: Dict[str, str],
                                     rel_map: Dict[str, str]) -> float:
        if not entity_map:
            return 0.0
        coverage = len(entity_map) / max(len(self._get_domain_entities(
            list(self.graph.get_entity(list(entity_map.keys())[0]).properties.get('domain', ''))
        ) if entity_map else []), 1) if entity_map else 0
        rel_bonus = min(len(rel_map) / 3.0, 1.0) * 0.3
        return min(0.5 + coverage * 0.3 + rel_bonus, 1.0)

    def _infer_domain(self, entity: Entity) -> Optional[str]:
        return entity.properties.get('domain') or entity.properties.get('scope')
