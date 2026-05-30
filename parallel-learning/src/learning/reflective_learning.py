"""经验反思学习 — 从交互中提取改进策略

基于 Experiential Reflective Learning (arXiv:2603.24639) 的核心思想：
- 每次学习后进行"经验反思"
- 分析自身行为并提取改进策略
- 从多个经验中合成新知识

与自改进系统的区别：
- 自改进系统：调整参数（脚手架更新）
- 反思学习：生成新知识（归纳推理）

设计原则：
- 反思不是简单的统计，而是模式发现
- 从多个成功/失败案例中归纳规律
- 生成可迁移的策略知识
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class LearningExperience:
    """学习经验"""
    text: str
    entities: List[str]
    triples: List[Tuple]
    verification_passed: bool
    score: float
    timestamp: float


@dataclass
class ReflectionInsight:
    """反思洞见"""
    insight: str
    confidence: float
    evidence: List[str]
    category: str  # 'pattern', 'gap', 'strategy'


class ReflectiveLearningSystem:
    """经验反思学习系统

    从多个学习经验中提取规律和策略。
    """

    def __init__(self):
        # 经验库
        self.experiences: List[LearningExperience] = []

        # 反思洞见
        self.insights: List[ReflectionInsight] = []

        # 模式统计
        self.pattern_stats = {
            'relation_frequency': defaultdict(int),
            'entity_frequency': defaultdict(int),
            'success_patterns': defaultdict(int),
            'failure_patterns': defaultdict(int),
        }

        # 反思间隔
        self.reflect_interval = 5
        self.experience_count = 0

    def record_experience(self, text: str, entities: List[str],
                         triples: List[Tuple], verification_passed: bool,
                         score: float):
        """记录学习经验"""
        experience = LearningExperience(
            text=text,
            entities=entities,
            triples=triples,
            verification_passed=verification_passed,
            score=score,
            timestamp=time.time(),
        )
        self.experiences.append(experience)
        self.experience_count += 1

        # 更新统计
        for triple in triples:
            if len(triple) >= 3:
                self.pattern_stats['relation_frequency'][triple[1]] += 1
                self.pattern_stats['entity_frequency'][triple[0]] += 1
                self.pattern_stats['entity_frequency'][triple[2]] += 1

        if verification_passed:
            for triple in triples:
                if len(triple) >= 3:
                    self.pattern_stats['success_patterns'][triple[1]] += 1
        else:
            for triple in triples:
                if len(triple) >= 3:
                    self.pattern_stats['failure_patterns'][triple[1]] += 1

    def should_reflect(self) -> bool:
        """判断是否应该进行反思"""
        return self.experience_count % self.reflect_interval == 0 and self.experience_count > 0

    def reflect(self) -> List[ReflectionInsight]:
        """进行反思，提取洞见

        Returns:
            新发现的洞见列表
        """
        new_insights = []

        # 1. 分析成功模式
        success_insights = self._analyze_success_patterns()
        new_insights.extend(success_insights)

        # 2. 分析失败模式
        failure_insights = self._analyze_failure_patterns()
        new_insights.extend(failure_insights)

        # 3. 分析知识空白
        gap_insights = self._analyze_knowledge_gaps()
        new_insights.extend(gap_insights)

        # 4. 合成新策略
        strategy_insights = self._synthesize_strategies()
        new_insights.extend(strategy_insights)

        # 存储洞见
        self.insights.extend(new_insights)

        return new_insights

    def _analyze_success_patterns(self) -> List[ReflectionInsight]:
        """分析成功模式"""
        insights = []

        # 找出高频成功关系
        success = self.pattern_stats['success_patterns']
        total = self.pattern_stats['relation_frequency']

        for rel, count in success.items():
            if count >= 3:
                total_count = total.get(rel, 1)
                success_rate = count / total_count
                if success_rate > 0.7:
                    insights.append(ReflectionInsight(
                        insight=f"关系'{rel}'的学习成功率很高({success_rate:.1%})，可以优先使用",
                        confidence=success_rate,
                        evidence=[f"成功{count}次/总计{total_count}次"],
                        category='pattern',
                    ))

        return insights

    def _analyze_failure_patterns(self) -> List[ReflectionInsight]:
        """分析失败模式"""
        insights = []

        # 找出高频失败关系
        failure = self.pattern_stats['failure_patterns']
        total = self.pattern_stats['relation_frequency']

        for rel, count in failure.items():
            if count >= 3:
                total_count = total.get(rel, 1)
                failure_rate = count / total_count
                if failure_rate > 0.5:
                    insights.append(ReflectionInsight(
                        insight=f"关系'{rel}'的学习失败率很高({failure_rate:.1%})，需要改进提取策略",
                        confidence=failure_rate,
                        evidence=[f"失败{count}次/总计{total_count}次"],
                        category='gap',
                    ))

        return insights

    def _analyze_knowledge_gaps(self) -> List[ReflectionInsight]:
        """分析知识空白"""
        insights = []

        # 找出低频实体（可能是知识空白）
        entity_freq = self.pattern_stats['entity_frequency']
        if len(entity_freq) > 5:
            avg_freq = sum(entity_freq.values()) / len(entity_freq)
            low_freq_entities = [e for e, f in entity_freq.items() if f < avg_freq * 0.3]

            if low_freq_entities:
                insights.append(ReflectionInsight(
                    insight=f"发现{len(low_freq_entities)}个低频实体，可能是知识空白: {', '.join(low_freq_entities[:5])}",
                    confidence=0.6,
                    evidence=[f"低频实体: {low_freq_entities[:5]}"],
                    category='gap',
                ))

        return insights

    def _synthesize_strategies(self) -> List[ReflectionInsight]:
        """合成新策略"""
        insights = []

        # 分析最近的经验趋势
        recent = self.experiences[-10:]
        if len(recent) >= 5:
            success_count = sum(1 for e in recent if e.verification_passed)
            success_rate = success_count / len(recent)

            if success_rate > 0.8:
                insights.append(ReflectionInsight(
                    insight=f"最近学习成功率很高({success_rate:.1%})，可以增加学习难度",
                    confidence=success_rate,
                    evidence=[f"最近{len(recent)}次学习，成功{success_count}次"],
                    category='strategy',
                ))
            elif success_rate < 0.3:
                insights.append(ReflectionInsight(
                    insight=f"最近学习成功率很低({success_rate:.1%})，需要降低难度或改进策略",
                    confidence=1.0 - success_rate,
                    evidence=[f"最近{len(recent)}次学习，成功{success_count}次"],
                    category='strategy',
                ))

        return insights

    def get_insights(self, category: Optional[str] = None) -> List[ReflectionInsight]:
        """获取洞见"""
        if category:
            return [i for i in self.insights if i.category == category]
        return self.insights

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_experiences': len(self.experiences),
            'total_insights': len(self.insights),
            'success_rate': self._success_rate(),
            'top_relations': self._top_relations(),
            'top_entities': self._top_entities(),
        }

    def _success_rate(self) -> float:
        """计算成功率"""
        if not self.experiences:
            return 0.0
        return sum(1 for e in self.experiences if e.verification_passed) / len(self.experiences)

    def _top_relations(self) -> List[Tuple[str, int]]:
        """获取高频关系"""
        return sorted(self.pattern_stats['relation_frequency'].items(),
                     key=lambda x: x[1], reverse=True)[:5]

    def _top_entities(self) -> List[Tuple[str, int]]:
        """获取高频实体"""
        return sorted(self.pattern_stats['entity_frequency'].items(),
                     key=lambda x: x[1], reverse=True)[:5]

    def get_report(self) -> str:
        """获取反思报告"""
        stats = self.get_stats()
        lines = [
            "=== 经验反思学习报告 ===",
            f"总经验数: {stats['total_experiences']}",
            f"总洞见数: {stats['total_insights']}",
            f"学习成功率: {stats['success_rate']:.1%}",
            "",
            "高频关系:",
        ]
        for rel, count in stats['top_relations']:
            lines.append(f"  {rel}: {count}次")

        lines.append("")
        lines.append("高频实体:")
        for entity, count in stats['top_entities']:
            lines.append(f"  {entity}: {count}次")

        if self.insights:
            lines.append("")
            lines.append("最近洞见:")
            for insight in self.insights[-5:]:
                lines.append(f"  [{insight.category}] {insight.insight}")

        return '\n'.join(lines)
