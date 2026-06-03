"""元认知监控器 — 评估知识置信度、识别缺口、估计难度"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph


@dataclass
class KnowledgeAssessment:
    """对某一主题的知识评估"""
    topic: str
    confidence: float
    evidence_count: int
    gaps: List[str] = field(default_factory=list)


class MetacognitiveMonitor:
    """元认知监控

    跟踪对各个知识主题的置信度，识别知识缺口，
    并估计学习任务的难度。
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge = knowledge_graph
        self.assessments: Dict[str, KnowledgeAssessment] = {}

    # ── 核心评估 ────────────────────────────────────────────────

    def knowledge_confidence(self, topic: str) -> float:
        """评估对某主题的置信度 [0, 1]

        基于：相关实体数 × 平均 confidence + 关系数 × 平均 confidence，
        归一化到 [0, 1]。
        """
        entity = self.knowledge.get_entity(topic)
        if entity is None:
            return 0.0

        # 实体自身的置信度
        entity_conf = entity.confidence

        # 邻居实体的加权贡献
        neighbors = self.knowledge.get_neighbors(topic, depth=1)
        if neighbors:
            neighbor_avg_conf = sum(n.confidence for n in neighbors.values()) / len(neighbors)
            neighbor_factor = min(1.0, len(neighbors) / 10.0)  # 10 个邻居即满
        else:
            neighbor_avg_conf = 0.0
            neighbor_factor = 0.0

        # 关系的加权贡献
        relations = self.knowledge.get_relations_of(topic, direction='both')
        if relations:
            relation_avg_conf = sum(r.confidence for r in relations) / len(relations)
            relation_factor = min(1.0, len(relations) / 10.0)
        else:
            relation_avg_conf = 0.0
            relation_factor = 0.0

        # 综合置信度：实体自身 + 邻居贡献 + 关系贡献
        score = (
            0.4 * entity_conf
            + 0.3 * neighbor_avg_conf * neighbor_factor
            + 0.3 * relation_avg_conf * relation_factor
        )
        return round(max(0.0, min(1.0, score)), 4)

    def identify_knowledge_gaps(self) -> List[str]:
        """识别知识缺口

        三类缺口：
        1. 低置信度实体（confidence < 0.3）
        2. 孤立实体（无任何关系）
        3. 缺失关键属性的实体（无 properties 或 properties 为空）
        """
        gaps: List[str] = []

        for eid, entity in self.knowledge.entities.items():
            # 低置信度
            if entity.confidence < 0.3:
                gaps.append(eid)
                continue

            # 孤立实体
            relations = self.knowledge.get_relations_of(eid, direction='both')
            if not relations:
                gaps.append(eid)
                continue

            # 缺失关键属性
            if not entity.properties:
                gaps.append(eid)

        return gaps

    def estimate_difficulty(self, task_description: str) -> float:
        """估计任务难度 [0, 1]

        难度 = 涉及的未知实体比例 + 涉及实体数越多越难。
        task_description 中出现的实体 ID 视为"涉及"。
        """
        # 从描述中提取可能的实体 ID（按长度降序匹配，优先长 ID）
        all_ids = sorted(self.knowledge.entities.keys(), key=len, reverse=True)
        involved: List[str] = []
        remaining = task_description
        for eid in all_ids:
            if eid in remaining:
                involved.append(eid)
                remaining = remaining.replace(eid, '')

        if not involved:
            # 没有匹配到已知实体，按未知对待
            return min(1.0, len(task_description) / 50.0 * 0.5 + 0.5)

        # 未知比例
        unknown_count = sum(
            1 for eid in involved
            if self.knowledge.get_entity(eid) is None
            or self.knowledge.get_entity(eid).confidence < 0.3
        )
        unknown_ratio = unknown_count / len(involved)

        # 规模因子：涉及的实体越多，难度越高
        scale_factor = min(1.0, len(involved) / 20.0)

        difficulty = 0.6 * unknown_ratio + 0.4 * scale_factor
        return round(max(0.0, min(1.0, difficulty)), 4)

    def update_assessment(self, topic: str, success: bool) -> None:
        """根据学习结果更新评估

        成功 → +0.1 confidence，失败 → -0.05。
        """
        entity = self.knowledge.get_entity(topic)
        if entity is None:
            return

        if success:
            entity.confidence = min(1.0, entity.confidence + 0.1)
        else:
            entity.confidence = max(0.0, entity.confidence - 0.05)

        # 更新 assessment 缓存
        conf = self.knowledge_confidence(topic)
        relations = self.knowledge.get_relations_of(topic, direction='both')
        self.assessments[topic] = KnowledgeAssessment(
            topic=topic,
            confidence=conf,
            evidence_count=sum(r.evidence_count for r in relations),
            gaps=[topic] if conf < 0.3 else [],
        )

    def assess_readiness(self, required_topics: List[str]) -> float:
        """对一组主题的平均准备程度 [0, 1]"""
        if not required_topics:
            return 1.0

        scores = [self.knowledge_confidence(t) for t in required_topics]
        return round(sum(scores) / len(scores), 4)
