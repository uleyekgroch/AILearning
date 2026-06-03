"""通用主动选择器 — 基于多信号评分选择最有学习价值的知识单元

核心思想：不是随机选择学习内容，而是综合考虑：
1. 不确定性 — 掌握度低的优先
2. 好奇心 — 预测误差高的优先
3. 知识缺口 — 孤立节点、缺失关系优先
4. 前置知识 — 前置已满足的优先
5. 难度 — 最近发展区（ZPD）内的优先
6. 频率/重要性 — 高频/核心概念优先

使用 epsilon-greedy 策略平衡探索与利用。
"""

import random
from typing import Dict, List, Optional

from src.knowledge.unit import KnowledgeUnit, MasteryLevel


class ActiveSelector:
    """通用主动选择器

    基于多信号评分，从候选知识单元中选择最有学习价值的一个。
    """

    def __init__(self,
                 epsilon: float = 0.1,
                 weight_uncertainty: float = 0.30,
                 weight_gap: float = 0.20,
                 weight_prerequisite: float = 0.20,
                 weight_difficulty: float = 0.15,
                 weight_frequency: float = 0.15):
        """
        Args:
            epsilon: 探索率 [0, 1]，epsilon% 的概率随机选择
            weight_uncertainty: 不确定性权重
            weight_gap: 知识缺口权重
            weight_prerequisite: 前置知识权重
            weight_difficulty: 难度匹配权重
            weight_frequency: 频率/重要性权重
        """
        self.epsilon = epsilon
        self.w_uncertainty = weight_uncertainty
        self.w_gap = weight_gap
        self.w_prerequisite = weight_prerequisite
        self.w_difficulty = weight_difficulty
        self.w_frequency = weight_frequency

    def select_next(self, candidates: List[KnowledgeUnit],
                    all_units: Optional[Dict[str, KnowledgeUnit]] = None,
                    current_level: float = 0.5) -> Optional[KnowledgeUnit]:
        """选择下一个最有学习价值的知识单元

        Args:
            candidates: 候选知识单元列表
            all_units: 所有知识单元字典（id -> unit），用于检查前置知识
            current_level: 当前学习者水平 [0, 1]，用于 ZPD 计算

        Returns:
            选中的知识单元，或 None（无可用候选）
        """
        if not candidates:
            return None

        if all_units is None:
            all_units = {}

        # 过滤掉已精通的
        learnable = [u for u in candidates if u.mastery_level != MasteryLevel.MASTERED]
        if not learnable:
            # 所有都已精通，随机选一个复习
            return random.choice(candidates)

        # epsilon-greedy 探索
        if random.random() < self.epsilon:
            return random.choice(learnable)

        # 计算每个候选的得分
        scored = []
        for unit in learnable:
            score = self._compute_score(unit, all_units, current_level)
            scored.append((unit, score))

        # 选择得分最高的
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def select_batch(self, candidates: List[KnowledgeUnit],
                     batch_size: int = 10,
                     all_units: Optional[Dict[str, KnowledgeUnit]] = None,
                     current_level: float = 0.5) -> List[KnowledgeUnit]:
        """选择一批有学习价值的知识单元

        Args:
            candidates: 候选知识单元列表
            batch_size: 批次大小
            all_units: 所有知识单元字典
            current_level: 当前学习者水平

        Returns:
            选中的知识单元列表
        """
        if not candidates:
            return []

        if all_units is None:
            all_units = {}

        # 过滤掉已精通的
        learnable = [u for u in candidates if u.mastery_level != MasteryLevel.MASTERED]
        if not learnable:
            return candidates[:batch_size]

        # 计算得分并排序
        scored = []
        for unit in learnable:
            score = self._compute_score(unit, all_units, current_level)
            scored.append((unit, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 选择 top-N
        selected = [u for u, s in scored[:batch_size]]

        # epsilon-greedy: 替换部分为随机探索
        n_explore = max(1, int(batch_size * self.epsilon))
        for i in range(n_explore):
            if len(learnable) > batch_size:
                random_unit = random.choice(learnable)
                if random_unit not in selected:
                    selected[-(i + 1)] = random_unit

        return selected

    def _compute_score(self, unit: KnowledgeUnit,
                       all_units: Dict[str, KnowledgeUnit],
                       current_level: float) -> float:
        """计算知识单元的综合学习价值得分

        Args:
            unit: 知识单元
            all_units: 所有知识单元字典
            current_level: 当前学习者水平

        Returns:
            综合得分 [0, 1]
        """
        # 1. 不确定性：掌握度越低，不确定性越高
        uncertainty = 1.0 - unit.mastery

        # 2. 知识缺口：孤立节点（无相关概念）得分更高
        gap = self._gap_score(unit)

        # 3. 前置知识：前置已满足的优先
        prerequisite = self._prerequisite_score(unit, all_units)

        # 4. 难度匹配：最近发展区（ZPD）内的优先
        difficulty = self._zpd_score(unit, current_level)

        # 5. 频率/重要性：低难度（更基础）的概念更重要
        frequency = 1.0 - unit.difficulty

        # 加权组合
        score = (
            self.w_uncertainty * uncertainty +
            self.w_gap * gap +
            self.w_prerequisite * prerequisite +
            self.w_difficulty * difficulty +
            self.w_frequency * frequency
        )

        return score

    def _gap_score(self, unit: KnowledgeUnit) -> float:
        """计算知识缺口得分

        孤立节点（无相关概念、无前置知识）得分高。
        有关系但关系少的得分中等。
        """
        n_relations = len(unit.related) + len(unit.prerequisites) + len(unit.is_a) + len(unit.part_of)
        if n_relations == 0:
            return 1.0  # 完全孤立
        elif n_relations <= 2:
            return 0.7  # 关系较少
        elif n_relations <= 5:
            return 0.4  # 关系适中
        else:
            return 0.2  # 关系丰富

    def _prerequisite_score(self, unit: KnowledgeUnit,
                            all_units: Dict[str, KnowledgeUnit]) -> float:
        """计算前置知识得分

        前置知识全部满足 → 1.0
        部分满足 → 按比例
        无前置知识 → 0.8（可以直接学）
        """
        if not unit.prerequisites:
            return 0.8  # 无前置知识，可以直接学

        met_count = 0
        for prereq_id in unit.prerequisites:
            prereq = all_units.get(prereq_id)
            if prereq is None:
                met_count += 1  # 前置不存在，视为满足
            elif prereq.mastery >= 0.5:
                met_count += 1

        return met_count / len(unit.prerequisites)

    def _zpd_score(self, unit: KnowledgeUnit, current_level: float) -> float:
        """计算最近发展区（ZPD）得分

        ZPD = 当前水平附近 ± 0.2 的范围。
        在 ZPD 内的知识单元得分最高。
        太简单或太难的得分低。
        """
        difficulty = unit.difficulty
        diff = abs(difficulty - current_level)

        if diff <= 0.15:
            return 1.0  # 完美匹配 ZPD
        elif diff <= 0.3:
            return 0.7  # 接近 ZPD
        elif diff <= 0.5:
            return 0.4  # 偏离 ZPD
        else:
            return 0.1  # 远离 ZPD

    def get_priority_report(self, candidates: List[KnowledgeUnit],
                            all_units: Optional[Dict[str, KnowledgeUnit]] = None,
                            current_level: float = 0.5,
                            top_n: int = 10) -> List[Dict]:
        """获取优先级报告

        Args:
            candidates: 候选知识单元列表
            all_units: 所有知识单元字典
            current_level: 当前学习者水平
            top_n: 返回前 N 个

        Returns:
            按得分排序的报告列表
        """
        if all_units is None:
            all_units = {}

        scored = []
        for unit in candidates:
            if unit.mastery_level == MasteryLevel.MASTERED:
                continue
            score = self._compute_score(unit, all_units, current_level)
            scored.append({
                'id': unit.id,
                'name': unit.name,
                'domain': unit.domain,
                'score': round(score, 4),
                'mastery': round(unit.mastery, 4),
                'mastery_level': unit.mastery_level.name,
                'difficulty': unit.difficulty,
                'prerequisites_met': unit.is_prerequisites_met(all_units),
            })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_n]
