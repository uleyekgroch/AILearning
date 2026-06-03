"""目标分解器 — 将大目标分解为子目标"""

from typing import List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.goals.goal import Goal, GoalStatus


class GoalDecomposer:
    """目标分解器

    分析目标描述中的关键词，关联知识图谱中的实体和概念，
    将大目标分解为可执行的子目标。
    """

    def __init__(self, knowledge_graph: KnowledgeGraph, metacognition=None):
        self.knowledge = knowledge_graph
        self.metacognition = metacognition  # Optional[MetaAssessor]

    def decompose_goal(self, goal: Goal) -> List[Goal]:
        """将目标分解为子目标

        策略：
        1. 从 description 提取关键词（实体 ID）
        2. 找到关键词所属的知识域（按 type 或 tag 分组）
        3. 每个知识域生成一个子目标
        4. 若没有可匹配的实体，按描述中的词组切分生成子目标
        """
        # 提取描述中出现的实体
        involved_entities = self._extract_entities(goal.description)

        if not involved_entities:
            # 没有匹配到实体，按简单规则切分
            return self._fallback_decompose(goal)

        # 按类型分组
        domain_groups: dict = {}  # type -> [entity_id, ...]
        for eid in involved_entities:
            entity = self.knowledge.get_entity(eid)
            if entity:
                domain = entity.type
            else:
                domain = 'unknown'
            domain_groups.setdefault(domain, []).append(eid)

        # 每个域一个子目标
        sub_goals: List[Goal] = []
        counter = 0
        for domain, eids in domain_groups.items():
            counter += 1
            sub_id = f"{goal.id}_sub{counter}"
            sub_desc = f"掌握 {domain} 领域: {', '.join(eids)}"
            sub = Goal(
                id=sub_id,
                description=sub_desc,
                status=GoalStatus.PENDING,
                parent_goal=goal.id,
                required_knowledge=eids,
                priority=goal.priority,
                created_step=goal.created_step,
            )
            sub_goals.append(sub)

        return sub_goals

    def identify_required_knowledge(self, goal: Goal) -> List[str]:
        """分析目标描述涉及的实体，找出知识缺口

        缺口定义：实体不存在于知识图谱中，或置信度 < 0.3。
        """
        involved = self._extract_entities(goal.description)
        gaps: List[str] = []

        for eid in involved:
            entity = self.knowledge.get_entity(eid)
            if entity is None:
                gaps.append(eid)
            elif entity.confidence < 0.3:
                gaps.append(eid)

        # 如果有元认知模块，用它来补充缺口
        if self.metacognition:
            meta_gaps = self.metacognition.monitor.identify_knowledge_gaps()
            # 合并去重
            for g in meta_gaps:
                if g not in gaps and g in involved:
                    gaps.append(g)

        return gaps

    # ── 内部方法 ──────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取已知实体 ID（按长度降序匹配）"""
        all_ids = sorted(self.knowledge.entities.keys(), key=len, reverse=True)
        found: List[str] = []
        remaining = text
        for eid in all_ids:
            if eid in remaining:
                found.append(eid)
                remaining = remaining.replace(eid, '')
        return found

    def _fallback_decompose(self, goal: Goal) -> List[Goal]:
        """当无法匹配到实体时的备用分解策略

        将描述按逗号、句号、顿号等分割，每段作为一个子目标。
        """
        import re
        parts = re.split(r'[,，。、；;]', goal.description)
        parts = [p.strip() for p in parts if len(p.strip()) > 1]

        if not parts:
            # 无法分解，返回自身
            return []

        sub_goals: List[Goal] = []
        for i, part in enumerate(parts, 1):
            sub = Goal(
                id=f"{goal.id}_sub{i}",
                description=part,
                status=GoalStatus.PENDING,
                parent_goal=goal.id,
                priority=goal.priority,
                created_step=goal.created_step,
            )
            sub_goals.append(sub)

        return sub_goals
