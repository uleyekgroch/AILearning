"""合作学习 — 多 Agent 协作、知识交换、共识达成"""

from typing import Any, Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation, REL_SIMILAR_TO


class CollaborationModule:
    """合作学习模块

    三个核心能力：
    1. 知识交换：两个 Agent 互相分享某主题的知识
    2. 共识达成：多个 Agent 讨论后对某主题达成一致
    3. 协作构建：多个 Agent 合作构建更完整的知识结构
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.exchange_history: List[Dict] = []

    def share_knowledge(self, agent_a_graph: KnowledgeGraph,
                        agent_b_graph: KnowledgeGraph,
                        topic: str) -> Dict[str, Any]:
        """两个 Agent 交换某主题的知识"""
        a_entities = agent_a_graph.query(tag=topic)
        b_entities = agent_b_graph.query(tag=topic)

        a_ids = {e.id for e in a_entities}
        b_ids = {e.id for e in b_entities}

        a_only = a_ids - b_ids
        b_only = b_ids - a_ids
        shared = a_ids & b_ids

        conflicts = []
        for eid in shared:
            ea = agent_a_graph.get_entity(eid)
            eb = agent_b_graph.get_entity(eid)
            if ea and eb:
                for key in set(ea.properties) & set(eb.properties):
                    if ea.properties[key] != eb.properties[key]:
                        conflicts.append({
                            'entity': eid,
                            'property': key,
                            'value_a': ea.properties[key],
                            'value_b': eb.properties[key],
                        })

        # 将对方独有知识写入自己的图谱
        transferred_to_a = 0
        for eid in b_only:
            entity = agent_b_graph.get_entity(eid)
            if entity:
                agent_a_graph.add_entity(Entity(
                    id=entity.id, type=entity.type,
                    properties=dict(entity.properties),
                    confidence=entity.confidence * 0.7,
                    source='shared',
                    tags=list(entity.tags),
                ))
                transferred_to_a += 1

        transferred_to_b = 0
        for eid in a_only:
            entity = agent_a_graph.get_entity(eid)
            if entity:
                agent_b_graph.add_entity(Entity(
                    id=entity.id, type=entity.type,
                    properties=dict(entity.properties),
                    confidence=entity.confidence * 0.7,
                    source='shared',
                    tags=list(entity.tags),
                ))
                transferred_to_b += 1

        result = {
            'topic': topic,
            'a_unique': len(a_only),
            'b_unique': len(b_only),
            'shared': len(shared),
            'conflicts': conflicts,
            'transferred_to_a': transferred_to_a,
            'transferred_to_b': transferred_to_b,
        }
        self.exchange_history.append(result)
        return result

    def negotiate_understanding(self, agents_graphs: List[KnowledgeGraph],
                                 topic: str) -> Dict[str, Any]:
        """多个 Agent 讨论后达成共识

        对于同一知识，采用多数投票或置信度加权平均。
        """
        all_entities: Dict[str, List[Entity]] = {}
        for graph in agents_graphs:
            for entity in graph.query(tag=topic):
                all_entities.setdefault(entity.id, []).append(entity)

        consensus: Dict[str, Entity] = {}
        contested = []

        for eid, entities in all_entities.items():
            if len(entities) == 1:
                consensus[eid] = entities[0]
                continue

            merged = Entity(
                id=eid,
                type=entities[0].type,
                properties={},
                confidence=0.0,
                source='consensus',
            )

            # 属性：取置信度加权值
            all_props: Dict[str, List] = {}
            for e in entities:
                for k, v in e.properties.items():
                    all_props.setdefault(k, []).append((v, e.confidence))

            for prop_key, vals in all_props.items():
                val_counts: Dict[str, float] = {}
                for val, conf in vals:
                    str_val = str(val)
                    val_counts[str_val] = val_counts.get(str_val, 0) + conf
                winner = max(val_counts, key=val_counts.get)
                # 尝试还原原始类型
                original_val = vals[0][0]
                if isinstance(original_val, (int, float)):
                    try:
                        winner = type(original_val)(winner)
                    except (ValueError, TypeError):
                        pass
                merged.properties[prop_key] = winner

            avg_conf = sum(e.confidence for e in entities) / len(entities)
            merged.confidence = min(1.0, avg_conf + 0.1 * len(entities))
            consensus[eid] = merged

            if len(entities) > 1:
                contested.append(eid)

        # 将共识写入所有 Agent 的图谱
        for graph in agents_graphs:
            for eid, entity in consensus.items():
                graph.add_entity(entity)

        return {
            'topic': topic,
            'consensus_count': len(consensus),
            'contested': contested,
            'agents_involved': len(agents_graphs),
        }

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'exchange_history': self.exchange_history,
        }

    def load_state(self, state: dict) -> None:
        self.exchange_history = state.get('exchange_history', [])
