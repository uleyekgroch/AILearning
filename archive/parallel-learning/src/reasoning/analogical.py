"""类比推理 — 基于结构相似性的关系映射"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.knowledge.graph import KnowledgeGraph


@dataclass
class AnalogyMapping:
    """类比映射：源域到目标域的实体对应关系"""
    source_domain: str
    target_domain: str
    entity_map: Dict[str, str]  # 源实体ID -> 目标实体ID
    confidence: float


class AnalogicalReasoner:
    """类比推理器：通过关系结构匹配解决 A:B :: C:? 形式的类比问题。"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def solve_analogy(self, a: str, b: str, c: str) -> Optional[str]:
        """解决类比问题 A:B :: C:?

        算法：
        1. 找出 A→B 之间的所有关系类型及其属性
        2. 找出 C 的所有出边（C→X 的关系）
        3. 对每个候选 X，计算其关系与 A→B 关系的匹配度
        4. 返回匹配度最高的 X
        """
        # 获取 A→B 的关系
        a_to_b_relations = self._get_relations_between(a, b)
        if not a_to_b_relations:
            return None

        # 提取 A→B 的关系类型集合
        ab_rel_types = {r.type for r in a_to_b_relations}
        # 同时收集关系元数据用于更精细匹配
        ab_rel_meta = {}
        for r in a_to_b_relations:
            ab_rel_meta[r.type] = r.metadata

        # 获取 C 的所有出边关系
        c_outgoing = self.graph.get_relations_of(c, direction='out')
        if not c_outgoing:
            return None

        # 对每个候选目标评分
        best_target = None
        best_score = 0.0

        for rel in c_outgoing:
            candidate = rel.target_id
            # 不能选 C 本身或 A/B
            if candidate in (a, b, c):
                continue

            score = self._compute_relation_match(
                ab_rel_types, ab_rel_meta,
                {rel.type}, {rel.type: rel.metadata}
            )

            if score > best_score:
                best_score = score
                best_target = candidate

        # 至少要有一定的匹配度才返回
        if best_target is not None and best_score > 0:
            return best_target
        return None

    def find_structural_similarity(self, domain_a_ids: List[str],
                                   domain_b_ids: List[str]) -> AnalogyMapping:
        """基于共享关系类型数量计算两个域之间的结构相似度。

        算法：
        1. 为每个域构建关系类型签名（出边类型 → 目标实体集合的映射）
        2. 计算两个域共享的关系类型数量
        3. 尝试建立实体映射（基于关系结构的最优匹配）
        4. 置信度 = 共享关系类型数 / max(域A关系类型数, 域B关系类型数)
        """
        if not domain_a_ids or not domain_b_ids:
            return AnalogyMapping(
                source_domain='domain_a',
                target_domain='domain_b',
                entity_map={},
                confidence=0.0,
            )

        # 构建每个实体的关系指纹：{关系类型: [目标实体类型]}
        def build_fingerprint(entity_ids: List[str]) -> Dict[str, Dict[str, int]]:
            """返回 entity_id -> {relation_type: count}"""
            fp = {}
            for eid in entity_ids:
                rel_counts: Dict[str, int] = defaultdict(int)
                for rel in self.graph.get_relations_of(eid, direction='both'):
                    rel_counts[rel.type] += 1
                fp[eid] = dict(rel_counts)
            return fp

        fp_a = build_fingerprint(domain_a_ids)
        fp_b = build_fingerprint(domain_b_ids)

        # 收集所有关系类型
        all_rel_types_a = set()
        for fp in fp_a.values():
            all_rel_types_a.update(fp.keys())

        all_rel_types_b = set()
        for fp in fp_b.values():
            all_rel_types_b.update(fp.keys())

        # 共享关系类型
        shared_types = all_rel_types_a & all_rel_types_b
        total_types = all_rel_types_a | all_rel_types_b

        # Jaccard 相似度作为基础置信度
        jaccard = len(shared_types) / len(total_types) if total_types else 0.0

        # 贪心建立实体映射：对 domain_a 中每个实体找 domain_b 中指纹最相似的
        entity_map: Dict[str, str] = {}
        used_b = set()

        for eid_a in domain_a_ids:
            fp_a_entity = fp_a.get(eid_a, {})
            best_match = None
            best_sim = 0.0

            for eid_b in domain_b_ids:
                if eid_b in used_b:
                    continue
                fp_b_entity = fp_b.get(eid_b, {})
                # 计算指纹余弦相似度
                sim = self._fingerprint_similarity(fp_a_entity, fp_b_entity)
                if sim > best_sim:
                    best_sim = sim
                    best_match = eid_b

            if best_match is not None and best_sim > 0:
                entity_map[eid_a] = best_match
                used_b.add(best_match)

        # 最终权重映射数量越多越可信
        mapping_ratio = len(entity_map) / min(len(domain_a_ids), len(domain_b_ids))
        confidence = jaccard * mapping_ratio

        return AnalogyMapping(
            source_domain='domain_a',
            target_domain='domain_b',
            entity_map=entity_map,
            confidence=min(confidence, 1.0),
        )

    # ── 内部方法 ──────────────────────────────────────────────

    def _get_relations_between(self, source_id: str,
                               target_id: str) -> list:
        """获取两个实体之间的所有关系"""
        relations = []
        for rel in self.graph.get_relations_of(source_id, direction='out'):
            if rel.target_id == target_id:
                relations.append(rel)
        return relations

    def _compute_relation_match(self, types_a: set, meta_a: dict,
                                types_b: set, meta_b: dict) -> float:
        """计算两组关系类型集合的匹配度。

        匹配度 = 交集大小 / 并集大小（Jaccard 系数）
        """
        if not types_a or not types_b:
            return 0.0

        intersection = types_a & types_b
        union = types_a | types_b
        jaccard = len(intersection) / len(union) if union else 0.0

        # 对元数据匹配额外加分
        meta_bonus = 0.0
        for rel_type in intersection:
            ma = meta_a.get(rel_type, {})
            mb = meta_b.get(rel_type, {})
            if ma and mb:
                shared_keys = set(ma.keys()) & set(mb.keys())
                if shared_keys:
                    matching = sum(1 for k in shared_keys if ma[k] == mb[k])
                    meta_bonus += matching / len(shared_keys)

        # 最终分数 = Jaccard * 0.7 + 元数据匹配 * 0.3
        return min(1.0, jaccard * 0.7 + meta_bonus * 0.3)

    def _fingerprint_similarity(self, fp_a: Dict[str, int],
                                fp_b: Dict[str, int]) -> float:
        """计算两个关系指纹的余弦相似度"""
        all_keys = set(fp_a.keys()) | set(fp_b.keys())
        if not all_keys:
            return 0.0

        # 构建向量
        vec_a = [fp_a.get(k, 0) for k in all_keys]
        vec_b = [fp_b.get(k, 0) for k in all_keys]

        # 余弦相似度
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)
