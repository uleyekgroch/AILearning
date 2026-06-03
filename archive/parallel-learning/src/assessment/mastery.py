"""通用掌握度评估 — 评估知识单元的掌握程度

评估维度：
1. 识别 — 看到能认出
2. 理解 — 知道含义
3. 应用 — 能使用
4. 分析 — 能分析关系
5. 综合 — 能综合运用
6. 教授 — 能教授他人

基于 Bloom's Taxonomy 的认知层次。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.knowledge.unit import KnowledgeUnit, MasteryLevel


@dataclass
class AssessmentResult:
    """评估结果"""
    unit_id: str
    unit_name: str
    domain: str
    overall_mastery: float
    mastery_level: MasteryLevel
    dimensions: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MasteryAssessor:
    """通用掌握度评估器

    基于多个维度评估知识单元的掌握程度。
    """

    def __init__(self):
        # 各维度的权重
        self.dimension_weights = {
            'recognition': 0.15,   # 识别
            'comprehension': 0.25, # 理解
            'application': 0.25,   # 应用
            'analysis': 0.15,      # 分析
            'synthesis': 0.10,     # 综合
            'teaching': 0.10,      # 教授
        }

    def assess(self, unit: KnowledgeUnit) -> AssessmentResult:
        """评估单个知识单元的掌握程度

        Args:
            unit: 要评估的知识单元

        Returns:
            评估结果
        """
        dimensions = {}

        # 1. 识别：基于接触次数和成功率
        dimensions['recognition'] = self._assess_recognition(unit)

        # 2. 理解：基于练习次数和成功率
        dimensions['comprehension'] = self._assess_comprehension(unit)

        # 3. 应用：基于掌握度
        dimensions['application'] = self._assess_application(unit)

        # 4. 分析：基于相关概念的掌握度
        dimensions['analysis'] = self._assess_analysis(unit)

        # 5. 综合：基于知识图谱中的关系丰富度
        dimensions['synthesis'] = self._assess_synthesis(unit)

        # 6. 教授：基于掌握度达到 MASTERED 的时间
        dimensions['teaching'] = self._assess_teaching(unit)

        # 计算加权总分
        overall = sum(
            dimensions[dim] * self.dimension_weights[dim]
            for dim in dimensions
        )

        # 确定掌握等级
        level = self._determine_level(overall)

        # 识别优势和弱点
        strengths = [dim for dim, score in dimensions.items() if score >= 0.7]
        weaknesses = [dim for dim, score in dimensions.items() if score < 0.3]

        # 生成建议
        recommendations = self._generate_recommendations(unit, dimensions)

        return AssessmentResult(
            unit_id=unit.id,
            unit_name=unit.name,
            domain=unit.domain,
            overall_mastery=round(overall, 4),
            mastery_level=level,
            dimensions={k: round(v, 4) for k, v in dimensions.items()},
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def assess_domain(self, units: List[KnowledgeUnit]) -> Dict:
        """评估整个领域的掌握程度

        Args:
            units: 该领域的所有知识单元

        Returns:
            领域评估报告
        """
        if not units:
            return {
                'domain': 'unknown',
                'total_units': 0,
                'avg_mastery': 0.0,
                'level_distribution': {},
                'coverage': 0.0,
            }

        domain = units[0].domain
        results = [self.assess(u) for u in units]

        # 统计
        avg_mastery = sum(r.overall_mastery for r in results) / len(results)

        level_dist = {}
        for level in MasteryLevel:
            count = sum(1 for r in results if r.mastery_level == level)
            level_dist[level.name] = count

        # 覆盖率：至少 EXPOSED 的比例
        exposed_count = sum(1 for u in units if u.mastery_level.value >= MasteryLevel.EXPOSED.value)
        coverage = exposed_count / len(units) if units else 0.0

        # 知识图谱密度：平均每个单元的关系数
        avg_relations = sum(
            len(u.related) + len(u.prerequisites) + len(u.is_a)
            for u in units
        ) / len(units)

        return {
            'domain': domain,
            'total_units': len(units),
            'avg_mastery': round(avg_mastery, 4),
            'level_distribution': level_dist,
            'coverage': round(coverage, 4),
            'avg_relations': round(avg_relations, 2),
            'strongest_units': [
                {'id': r.unit_id, 'name': r.unit_name, 'mastery': r.overall_mastery}
                for r in sorted(results, key=lambda x: x.overall_mastery, reverse=True)[:5]
            ],
            'weakest_units': [
                {'id': r.unit_id, 'name': r.unit_name, 'mastery': r.overall_mastery}
                for r in sorted(results, key=lambda x: x.overall_mastery)[:5]
            ],
        }

    def assess_all_domains(self, units_by_domain: Dict[str, List[KnowledgeUnit]]) -> Dict:
        """评估所有领域的掌握程度

        Args:
            units_by_domain: 按领域分组的知识单元字典

        Returns:
            全领域评估报告
        """
        domain_reports = {}
        for domain, units in units_by_domain.items():
            domain_reports[domain] = self.assess_domain(units)

        # 总体统计
        total_units = sum(r['total_units'] for r in domain_reports.values())
        total_mastered = sum(
            r['level_distribution'].get('MASTERED', 0)
            for r in domain_reports.values()
        )

        return {
            'total_units': total_units,
            'total_mastered': total_mastered,
            'mastery_rate': round(total_mastered / total_units, 4) if total_units > 0 else 0.0,
            'domains': domain_reports,
        }

    # ── 内部评估方法 ──────────────────────────────────────────

    def _assess_recognition(self, unit: KnowledgeUnit) -> float:
        """评估识别能力

        基于接触次数：接触越多，识别能力越强。
        """
        if unit.exposure_count == 0:
            return 0.0

        # 对数衰减：前几次接触效果最大
        score = math.log(1 + unit.exposure_count) / math.log(1 + 20)
        return min(1.0, score)

    def _assess_comprehension(self, unit: KnowledgeUnit) -> float:
        """评估理解能力

        基于练习成功率：成功率越高，理解越深。
        """
        if unit.practice_count == 0:
            return 0.0

        success_rate = unit.success_count / unit.practice_count
        # 练习次数加权
        practice_factor = min(1.0, unit.practice_count / 10.0)

        return success_rate * practice_factor

    def _assess_application(self, unit: KnowledgeUnit) -> float:
        """评估应用能力

        基于掌握度本身。
        """
        return unit.mastery

    def _assess_analysis(self, unit: KnowledgeUnit) -> float:
        """评估分析能力

        基于相关概念的数量和质量。
        """
        n_relations = len(unit.related) + len(unit.is_a) + len(unit.part_of)
        if n_relations == 0:
            return 0.0

        # 关系越多，分析能力越强
        score = min(1.0, n_relations / 10.0)
        return score

    def _assess_synthesis(self, unit: KnowledgeUnit) -> float:
        """评估综合能力

        基于知识图谱中的关系丰富度和跨领域连接。
        """
        total_relations = (
            len(unit.related) + len(unit.prerequisites) +
            len(unit.is_a) + len(unit.part_of) + len(unit.has_part)
        )

        if total_relations == 0:
            return 0.0

        # 跨领域关系加分
        cross_domain = 0
        for rel_id in unit.related:
            if ':' in rel_id:
                rel_domain = rel_id.split(':')[0]
                if rel_domain != unit.domain:
                    cross_domain += 1

        base_score = min(1.0, total_relations / 15.0)
        cross_bonus = min(0.2, cross_domain * 0.05)

        return min(1.0, base_score + cross_bonus)

    def _assess_teaching(self, unit: KnowledgeUnit) -> float:
        """评估教学能力

        基于是否达到 MASTERED 级别。
        """
        if unit.mastery_level == MasteryLevel.MASTERED:
            return 1.0
        elif unit.mastery_level == MasteryLevel.APPLIED:
            return 0.6
        elif unit.mastery_level == MasteryLevel.UNDERSTOOD:
            return 0.3
        else:
            return 0.0

    def _determine_level(self, overall: float) -> MasteryLevel:
        """根据总分确定掌握等级"""
        if overall >= 0.9:
            return MasteryLevel.MASTERED
        elif overall >= 0.7:
            return MasteryLevel.APPLIED
        elif overall >= 0.5:
            return MasteryLevel.UNDERSTOOD
        elif overall >= 0.3:
            return MasteryLevel.RECOGNIZED
        elif overall >= 0.1:
            return MasteryLevel.EXPOSED
        else:
            return MasteryLevel.UNKNOWN

    def _generate_recommendations(self, unit: KnowledgeUnit,
                                   dimensions: Dict[str, float]) -> List[str]:
        """生成学习建议"""
        recommendations = []

        if dimensions['recognition'] < 0.3:
            recommendations.append(f"增加接触次数（当前 {unit.exposure_count} 次）")

        if dimensions['comprehension'] < 0.3:
            recommendations.append(f"增加练习次数（当前 {unit.practice_count} 次）")

        if dimensions['application'] < 0.3:
            recommendations.append("通过更多练习提高应用能力")

        if dimensions['analysis'] < 0.3:
            recommendations.append("学习相关概念，建立知识网络")

        if dimensions['synthesis'] < 0.3:
            recommendations.append("探索跨领域联系")

        if not recommendations:
            recommendations.append("继续巩固，尝试教授他人")

        return recommendations
