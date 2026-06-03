"""专业水平评估 — CEFR A1-C2 + 专业水平

CEFR (Common European Framework of Reference for Languages) 等级：
- A1 (Beginner): 约 600 词，能理解基础日常表达
- A2 (Elementary): 约 1200 词，能理解常见句子和表达
- B1 (Intermediate): 约 2000 词，能理解标准语言的主要内容
- B2 (Upper Intermediate): 约 3000 词，能理解复杂文本的主要内容
- C1 (Advanced): 约 5000 词，能理解多种高难度文本
- C2 (Proficiency): 约 8000+ 词，能轻松理解几乎所有内容

评估维度：
1. 词汇量 — 知道多少词
2. 词汇深度 — 对每个词的理解程度
3. 语义网络 — 词汇之间的关联密度
4. 搭配知识 — 词组和搭配的掌握
5. 词族知识 — 同一词根的不同形式
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.knowledge.unit import KnowledgeUnit, MasteryLevel


@dataclass
class ProficiencyReport:
    """专业水平评估报告"""
    level: str                          # CEFR 等级
    score: float                        # 总分 [0, 1]
    receptive_vocabulary: int           # 接收词汇量
    productive_vocabulary: int          # 产出词汇量
    semantic_depth: float               # 语义深度 [0, 1]
    collocation_knowledge: float        # 搭配知识 [0, 1]
    word_family_coverage: float         # 词族覆盖率 [0, 1]
    dimensions: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    level_distribution: Dict[str, int] = field(default_factory=dict)


class ProficiencyTester:
    """专业水平评估器"""

    def __init__(self):
        # CEFR 等级阈值
        self.level_thresholds = {
            'A1': 0.10,
            'A2': 0.20,
            'B1': 0.35,
            'B2': 0.50,
            'C1': 0.70,
            'C2': 0.85,
            'Professional': 0.95,
        }

        # 各维度权重
        self.dimension_weights = {
            'vocabulary_breadth': 0.30,   # 词汇广度
            'vocabulary_depth': 0.25,     # 词汇深度
            'semantic_network': 0.20,     # 语义网络
            'collocation': 0.15,          # 搭配知识
            'word_family': 0.10,          # 词族知识
        }

    def assess(self, units: List[KnowledgeUnit]) -> ProficiencyReport:
        """评估英语水平

        Args:
            units: 英语知识单元列表

        Returns:
            专业水平评估报告
        """
        if not units:
            return self._empty_report()

        # 1. 词汇广度
        vocab_breadth = self._assess_vocabulary_breadth(units)

        # 2. 词汇深度
        vocab_depth = self._assess_vocabulary_depth(units)

        # 3. 语义网络
        semantic_network = self._assess_semantic_network(units)

        # 4. 搭配知识
        collocation = self._assess_collocation(units)

        # 5. 词族知识
        word_family = self._assess_word_family(units)

        # 计算总分
        dimensions = {
            'vocabulary_breadth': vocab_breadth,
            'vocabulary_depth': vocab_depth,
            'semantic_network': semantic_network,
            'collocation': collocation,
            'word_family': word_family,
        }

        total_score = sum(
            dimensions[dim] * self.dimension_weights[dim]
            for dim in dimensions
        )

        # 确定 CEFR 等级
        level = self._determine_level(total_score)

        # 统计
        receptive = self._count_receptive(units)
        productive = self._count_productive(units)

        # 等级分布
        level_dist = {}
        for u in units:
            cefr = u.metadata.get('cefr', 'B1')
            level_dist[cefr] = level_dist.get(cefr, 0) + 1

        # 生成建议
        recommendations = self._generate_recommendations(dimensions, level)

        return ProficiencyReport(
            level=level,
            score=round(total_score, 4),
            receptive_vocabulary=receptive,
            productive_vocabulary=productive,
            semantic_depth=round(vocab_depth, 4),
            collocation_knowledge=round(collocation, 4),
            word_family_coverage=round(word_family, 4),
            dimensions={k: round(v, 4) for k, v in dimensions.items()},
            recommendations=recommendations,
            level_distribution=level_dist,
        )

    def _assess_vocabulary_breadth(self, units: List[KnowledgeUnit]) -> float:
        """评估词汇广度

        基于已掌握的词汇数量和 CEFR 等级覆盖。
        """
        if not units:
            return 0.0

        # 按 CEFR 等级分组
        by_cefr = {}
        for u in units:
            cefr = u.metadata.get('cefr', 'B1')
            if cefr not in by_cefr:
                by_cefr[cefr] = []
            by_cefr[cefr].append(u)

        # 计算每个等级的掌握率
        level_scores = {}
        for cefr, cefr_units in by_cefr.items():
            mastered = sum(1 for u in cefr_units
                          if u.mastery_level.value >= MasteryLevel.RECOGNIZED.value)
            level_scores[cefr] = mastered / len(cefr_units) if cefr_units else 0

        # 加权计算（A1 权重最低，C2 权重最高）
        level_weights = {
            'A1': 0.05, 'A2': 0.10, 'B1': 0.15,
            'B2': 0.20, 'C1': 0.25, 'C2': 0.25,
        }

        total = 0.0
        weight_sum = 0.0
        for cefr, score in level_scores.items():
            weight = level_weights.get(cefr, 0.15)
            total += score * weight
            weight_sum += weight

        return total / weight_sum if weight_sum > 0 else 0.0

    def _assess_vocabulary_depth(self, units: List[KnowledgeUnit]) -> float:
        """评估词汇深度

        基于每个词汇的掌握程度（不仅仅是"知道"，而是"精通"）。
        """
        if not units:
            return 0.0

        # 计算平均掌握度
        avg_mastery = sum(u.mastery for u in units) / len(units)

        # 计算精通率
        mastered = sum(1 for u in units
                       if u.mastery_level.value >= MasteryLevel.APPLIED.value)
        mastery_rate = mastered / len(units)

        # 综合深度
        return 0.6 * avg_mastery + 0.4 * mastery_rate

    def _assess_semantic_network(self, units: List[KnowledgeUnit]) -> float:
        """评估语义网络

        基于词汇之间的关联密度和质量。
        """
        if not units:
            return 0.0

        # 计算平均关系数
        total_relations = sum(
            len(u.related) + len(u.prerequisites) + len(u.is_a)
            for u in units
        )
        avg_relations = total_relations / len(units)

        # 归一化（假设 10 个关系为满分）
        density = min(1.0, avg_relations / 10.0)

        # 计算孤立节点率
        isolated = sum(1 for u in units if not u.related and not u.prerequisites)
        isolation_rate = isolated / len(units)

        return density * (1.0 - 0.5 * isolation_rate)

    def _assess_collocation(self, units: List[KnowledgeUnit]) -> float:
        """评估搭配知识

        基于示例和解释的丰富度。
        """
        if not units:
            return 0.0

        # 计算有示例的词汇比例
        with_examples = sum(1 for u in units if u.examples)
        example_rate = with_examples / len(units)

        # 计算平均示例数
        avg_examples = sum(len(u.examples) for u in units) / len(units)

        # 归一化（假设 3 个示例为满分）
        example_depth = min(1.0, avg_examples / 3.0)

        return 0.5 * example_rate + 0.5 * example_depth

    def _assess_word_family(self, units: List[KnowledgeUnit]) -> float:
        """评估词族知识

        基于词族关联的覆盖度。
        """
        if not units:
            return 0.0

        # 计算有词族关联的词汇比例
        with_family = sum(1 for u in units
                          if u.metadata.get('word_family'))
        family_rate = with_family / len(units)

        return family_rate

    def _count_receptive(self, units: List[KnowledgeUnit]) -> int:
        """计算接收词汇量（能识别的）"""
        return sum(1 for u in units
                   if u.mastery_level.value >= MasteryLevel.RECOGNIZED.value)

    def _count_productive(self, units: List[KnowledgeUnit]) -> int:
        """计算产出词汇量（能使用的）"""
        return sum(1 for u in units
                   if u.mastery_level.value >= MasteryLevel.APPLIED.value)

    def _determine_level(self, score: float) -> str:
        """确定 CEFR 等级"""
        level = 'A1'
        for l, threshold in sorted(self.level_thresholds.items(),
                                    key=lambda x: x[1]):
            if score >= threshold:
                level = l
        return level

    def _generate_recommendations(self, dimensions: Dict[str, float],
                                   level: str) -> List[str]:
        """生成学习建议"""
        recommendations = []

        if dimensions['vocabulary_breadth'] < 0.3:
            recommendations.append("扩大词汇量：学习更多基础词汇")

        if dimensions['vocabulary_depth'] < 0.3:
            recommendations.append("加深理解：对已学词汇进行更多练习")

        if dimensions['semantic_network'] < 0.3:
            recommendations.append("建立联系：学习词汇之间的关联")

        if dimensions['collocation'] < 0.3:
            recommendations.append("学习搭配：掌握常用词组和搭配")

        if dimensions['word_family'] < 0.3:
            recommendations.append("学习词族：掌握同一词根的不同形式")

        if level in ['A1', 'A2']:
            recommendations.append("重点学习高频基础词汇")
        elif level in ['B1', 'B2']:
            recommendations.append("扩展中级词汇，加强语义网络")
        elif level in ['C1', 'C2']:
            recommendations.append("深化专业词汇，提高产出能力")

        return recommendations

    def _empty_report(self) -> ProficiencyReport:
        """空报告"""
        return ProficiencyReport(
            level='A1',
            score=0.0,
            receptive_vocabulary=0,
            productive_vocabulary=0,
            semantic_depth=0.0,
            collocation_knowledge=0.0,
            word_family_coverage=0.0,
        )
