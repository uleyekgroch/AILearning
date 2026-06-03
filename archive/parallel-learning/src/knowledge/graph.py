"""知识图谱 — 实体-关系-属性三元组存储与查询"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from src.knowledge.entity import Entity
from src.knowledge.relation import Relation


class KnowledgeGraph:
    """结构化知识图谱

    存储 Entity（节点）和 Relation（边），
    支持查询、路径查找、聚合等操作。
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        # 索引：加速查询
        self._outgoing: Dict[str, List[Relation]] = defaultdict(list)
        self._incoming: Dict[str, List[Relation]] = defaultdict(list)
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)

    # ── 增删 ──────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> str:
        """添加实体（已存在则合并属性）"""
        if entity.id in self.entities:
            existing = self.entities[entity.id]
            existing.properties.update(entity.properties)
            if entity.embedding is not None:
                existing.embedding = entity.embedding
            for tag in entity.tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)
                    self._tag_index[tag].add(entity.id)
        else:
            self.entities[entity.id] = entity
            self._type_index[entity.type].add(entity.id)
            for tag in entity.tags:
                self._tag_index[tag].add(entity.id)
        return entity.id

    def add_relation(self, relation: Relation) -> None:
        """添加关系（已存在则加强）"""
        # 检查是否已存在相同 source-target-type 的关系
        for r in self._outgoing[relation.source_id]:
            if r.target_id == relation.target_id and r.type == relation.type:
                r.strengthen()
                r.metadata.update(relation.metadata)
                return

        self.relations.append(relation)
        self._outgoing[relation.source_id].append(relation)
        self._incoming[relation.target_id].append(relation)

    def remove_entity(self, entity_id: str) -> None:
        """删除实体及其所有关系"""
        if entity_id not in self.entities:
            return
        entity = self.entities.pop(entity_id)
        self._type_index[entity.type].discard(entity_id)
        for tag in entity.tags:
            self._tag_index[tag].discard(entity_id)
        # 删除相关关系
        self.relations = [r for r in self.relations
                          if r.source_id != entity_id and r.target_id != entity_id]
        self._rebuild_indices()

    # ── 查询 ──────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_related(self, entity_id: str,
                    relation_type: Optional[str] = None) -> List[Entity]:
        """获取与某实体有关系的所有实体"""
        related = []
        for rel in self._outgoing.get(entity_id, []):
            if relation_type is None or rel.type == relation_type:
                target = self.entities.get(rel.target_id)
                if target:
                    related.append(target)
        return related

    def get_relations_of(self, entity_id: str,
                         direction: str = 'both') -> List[Relation]:
        """获取与实体相关的所有关系"""
        result = []
        if direction in ('out', 'both'):
            result.extend(self._outgoing.get(entity_id, []))
        if direction in ('in', 'both'):
            result.extend(self._incoming.get(entity_id, []))
        return result

    def query(self, entity_type: Optional[str] = None,
              properties: Optional[Dict[str, Any]] = None,
              tag: Optional[str] = None) -> List[Entity]:
        """按条件查询实体"""
        candidates = None
        if entity_type:
            candidates = self._type_index.get(entity_type, set())
        if tag:
            tag_set = self._tag_index.get(tag, set())
            candidates = tag_set if candidates is None else candidates & tag_set
        if candidates is None:
            candidates = set(self.entities.keys())

        results = []
        for eid in candidates:
            entity = self.entities.get(eid)
            if entity and (properties is None or self._match_props(entity, properties)):
                results.append(entity)
        return results

    def find_path(self, source_id: str, target_id: str,
                  max_depth: int = 3) -> List[str]:
        """BFS 查找两个实体之间的最短路径"""
        if source_id not in self.entities or target_id not in self.entities:
            return []
        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = [(source_id, [source_id])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth + 1:
                break
            for rel in self._outgoing.get(current, []):
                next_id = rel.target_id
                if next_id == target_id:
                    return path + [next_id]
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
        return []

    def get_neighbors(self, entity_id: str, depth: int = 1) -> Dict[str, Entity]:
        """获取 N 跳邻居"""
        result = {}
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier = set()
            for eid in frontier:
                for rel in self._outgoing.get(eid, []):
                    if rel.target_id not in result and rel.target_id != entity_id:
                        ent = self.entities.get(rel.target_id)
                        if ent:
                            result[rel.target_id] = ent
                            next_frontier.add(rel.target_id)
                for rel in self._incoming.get(eid, []):
                    if rel.source_id not in result and rel.source_id != entity_id:
                        ent = self.entities.get(rel.source_id)
                        if ent:
                            result[rel.source_id] = ent
                            next_frontier.add(rel.source_id)
            frontier = next_frontier
        return result

    def aggregate(self, entity_type: str,
                  property_name: str) -> Dict[str, int]:
        """聚合统计：某类型实体的某属性值分布"""
        counts: Dict[str, int] = defaultdict(int)
        for eid in self._type_index.get(entity_type, set()):
            entity = self.entities.get(eid)
            if entity and property_name in entity.properties:
                val = str(entity.properties[property_name])
                counts[val] += 1
        return dict(counts)

    def get_or_create(self, entity_id: str, entity_type: str,
                      **kwargs) -> Entity:
        """获取或创建实体"""
        if entity_id in self.entities:
            return self.entities[entity_id]
        entity = Entity(id=entity_id, type=entity_type, **kwargs)
        self.add_entity(entity)
        return entity

    # ── 统计 ──────────────────────────────────────────────────

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    def type_distribution(self) -> Dict[str, int]:
        return {t: len(ids) for t, ids in self._type_index.items()}

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'entities': {eid: e.to_dict() for eid, e in self.entities.items()},
            'relations': [r.to_dict() for r in self.relations],
        }

    def load_state(self, state: dict) -> None:
        self.entities.clear()
        self.relations.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._type_index.clear()
        self._tag_index.clear()

        for eid, ed in state.get('entities', {}).items():
            entity = Entity.from_dict(ed)
            self.entities[eid] = entity
            self._type_index[entity.type].add(eid)
            for tag in entity.tags:
                self._tag_index[tag].add(eid)

        for rd in state.get('relations', []):
            rel = Relation.from_dict(rd)
            self.relations.append(rel)
            self._outgoing[rel.source_id].append(rel)
            self._incoming[rel.target_id].append(rel)

    # ── 内部 ──────────────────────────────────────────────────

    def _match_props(self, entity: Entity, properties: Dict) -> bool:
        for key, value in properties.items():
            if entity.properties.get(key) != value:
                return False
        return True

    def _rebuild_indices(self) -> None:
        self._outgoing.clear()
        self._incoming.clear()
        for r in self.relations:
            self._outgoing[r.source_id].append(r)
            self._incoming[r.target_id].append(r)
