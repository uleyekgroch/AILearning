"""学习计划器 — 生成学习步骤和估算学习代价"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.goals.goal import Goal


@dataclass
class LearningPlan:
    """学习计划"""
    goal_id: str
    steps: List[Dict] = field(default_factory=list)
    estimated_effort: float = 0.0


class LearningPlanner:
    """学习计划器

    根据知识缺口和依赖关系，生成有序的学习步骤。
    """

    def __init__(self, knowledge_graph: KnowledgeGraph, metacognition=None):
        self.knowledge = knowledge_graph
        self.metacognition = metacognition  # Optional[MetaAssessor]

    def plan_learning(self, goal: Goal) -> LearningPlan:
        """为指定目标生成学习计划

        流程：
        1. 收集目标所需知识（required_knowledge）
        2. 找出缺口（不存在或低置信度）
        3. 按依赖关系排序（被依赖的先学）
        4. 为每个缺口选择策略
        5. 生成步骤列表
        """
        required = goal.required_knowledge
        if not required:
            # 没有明确的知识需求，从描述推断
            from src.goals.decomposer import GoalDecomposer
            decomposer = GoalDecomposer(self.knowledge, self.metacognition)
            required = decomposer._extract_entities(goal.description)

        # 识别缺口
        gaps = self._identify_gaps(required)

        if not gaps:
            # 没有缺口，直接标记为可执行
            return LearningPlan(
                goal_id=goal.id,
                steps=[{
                    'action': 'execute',
                    'target': goal.description,
                    'strategy': 'practice',
                    'reason': '所有前置知识已具备',
                }],
                estimated_effort=0.2,
            )

        # 按依赖排序
        ordered_gaps = self._topological_sort(gaps)

        # 为每个缺口生成步骤
        steps: List[Dict] = []
        total_effort = 0.0

        for gap_id in ordered_gaps:
            entity = self.knowledge.get_entity(gap_id)
            difficulty = self._estimate_gap_difficulty(gap_id)
            strategy = self._choose_strategy(difficulty, len(gaps))

            step = {
                'action': 'learn',
                'target': gap_id,
                'strategy': strategy,
                'difficulty': round(difficulty, 4),
            }

            # 如果实体有邻居已掌握，用类比策略
            if entity:
                neighbors = self.knowledge.get_neighbors(gap_id, depth=1)
                mastered_neighbors = [
                    n.id for n in neighbors.values()
                    if n.confidence >= 0.5
                ]
                if mastered_neighbors:
                    step['strategy'] = 'analogize'
                    step['analogy_from'] = mastered_neighbors[0]

            steps.append(step)
            total_effort += difficulty

        # 添加最终执行步骤
        steps.append({
            'action': 'execute',
            'target': goal.description,
            'strategy': 'practice',
            'reason': '完成所有前置知识学习',
        })

        avg_effort = total_effort / max(len(ordered_gaps), 1)
        return LearningPlan(
            goal_id=goal.id,
            steps=steps,
            estimated_effort=round(avg_effort, 4),
        )

    # ── 内部方法 ──────────────────────────────────────────────────

    def _identify_gaps(self, required: List[str]) -> List[str]:
        """识别知识缺口"""
        gaps: List[str] = []
        for eid in required:
            entity = self.knowledge.get_entity(eid)
            if entity is None or entity.confidence < 0.3:
                gaps.append(eid)
        return gaps

    def _topological_sort(self, gaps: List[str]) -> List[str]:
        """按依赖关系排序

        规则：如果一个实体的邻居也在缺口列表中，则邻居先学。
        使用 Kahn 算法（BFS 拓扑排序）。
        """
        gap_set = set(gaps)

        # 构建依赖图：gap → 它依赖的其他 gaps
        deps: Dict[str, set] = {g: set() for g in gaps}
        for gap in gaps:
            relations = self.knowledge.get_relations_of(gap, direction='in')
            for rel in relations:
                if rel.source_id in gap_set:
                    deps[gap].add(rel.source_id)

        # Kahn 算法
        in_degree = {g: len(deps[g]) for g in gaps}
        queue = [g for g in gaps if in_degree[g] == 0]
        result: List[str] = []

        while queue:
            # 优先选择置信度更高的（已有部分基础）
            queue.sort(key=lambda g: (
                self.knowledge.get_entity(g).confidence
                if self.knowledge.get_entity(g) else 0.0
            ))
            node = queue.pop(0)
            result.append(node)

            for g in gaps:
                if node in deps[g]:
                    deps[g].discard(node)
                    in_degree[g] -= 1
                    if in_degree[g] == 0:
                        queue.append(g)

        # 处理循环依赖：未排到的直接追加
        for g in gaps:
            if g not in result:
                result.append(g)

        return result

    def _estimate_gap_difficulty(self, gap_id: str) -> float:
        """估计单个缺口的难度"""
        entity = self.knowledge.get_entity(gap_id)
        if entity is None:
            return 0.9  # 完全未知 → 很难

        # 已有部分置信度 → 较容易
        base_diff = 1.0 - entity.confidence

        # 邻居多 → 有参照 → 较容易
        neighbors = self.knowledge.get_neighbors(gap_id, depth=1)
        neighbor_bonus = min(0.3, len(neighbors) * 0.05)

        return max(0.1, min(1.0, base_diff - neighbor_bonus))

    def _choose_strategy(self, difficulty: float, gap_count: int) -> str:
        """为单个缺口选择策略"""
        if difficulty >= 0.7:
            return 'decompose'
        if difficulty >= 0.5:
            return 'analogize'
        if gap_count >= 5:
            return 'practice'
        return 'explore'
