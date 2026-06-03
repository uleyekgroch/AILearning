"""自我迭代器 — 评估和改进学习过程

核心思想：系统评估自己的学习效果，自动改进策略。
像人类一样：反思→发现不足→调整方法→再次尝试。

与现有系统的区别：
- 现有：MetaAssessor 评估知识置信度
- 新系统：评估学习过程本身的效果，改进学习策略
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LearningReport:
    """学习报告"""
    prediction_accuracy: float    # 预测准确率
    knowledge_coverage: float     # 知识覆盖度
    concept_quality: float        # 概念质量
    learning_efficiency: float    # 学习效率
    surprise_trend: float         # 惊讶度趋势（负=在学习）
    weaknesses: List[str]         # 薄弱环节
    recommendations: List[str]    # 改进建议


@dataclass
class Weakness:
    """薄弱环节"""
    area: str
    severity: float               # 严重程度 [0, 1]
    description: str
    suggested_action: str


class SelfIterator:
    """自我迭代器 — 改进学习过程"""

    def __init__(self):
        # 评估历史
        self.evaluation_history: List[LearningReport] = []
        self.strategy_effectiveness: Dict[str, List[float]] = {}

        # 改进记录
        self.improvements_made: List[Dict] = []

    def evaluate_learning(self, world_model, concept_former,
                          explorer, language_system) -> LearningReport:
        """评估当前学习效果"""

        # 1. 预测准确率
        prediction_accuracy = self._evaluate_prediction_accuracy(world_model)

        # 2. 知识覆盖度
        knowledge_coverage = self._evaluate_knowledge_coverage(world_model)

        # 3. 概念质量
        concept_quality = self._evaluate_concept_quality(concept_former)

        # 4. 学习效率
        learning_efficiency = self._evaluate_learning_efficiency(explorer)

        # 5. 惊讶度趋势
        surprise_trend = self._evaluate_surprise_trend(explorer)

        # 6. 识别薄弱环节
        weaknesses = self._identify_weaknesses(
            world_model, concept_former, explorer, language_system
        )

        # 7. 生成建议
        recommendations = self._generate_recommendations(weaknesses)

        report = LearningReport(
            prediction_accuracy=prediction_accuracy,
            knowledge_coverage=knowledge_coverage,
            concept_quality=concept_quality,
            learning_efficiency=learning_efficiency,
            surprise_trend=surprise_trend,
            weaknesses=[w.area for w in weaknesses],
            recommendations=recommendations,
        )
        self.evaluation_history.append(report)

        return report

    def improve_strategy(self, report: LearningReport,
                         world_model, explorer) -> List[Dict]:
        """根据评估报告改进学习策略"""
        improvements = []

        for weakness_area in report.weaknesses:
            if weakness_area == 'low_prediction_accuracy':
                # 增加探索多样性
                improvement = {
                    'area': weakness_area,
                    'action': '增加探索率',
                    'before': explorer.exploration_rate,
                    'after': min(0.5, explorer.exploration_rate + 0.1),
                }
                explorer.exploration_rate = improvement['after']
                improvements.append(improvement)

            elif weakness_area == 'poor_concept_formation':
                # 降低概念形成阈值，更容易创建新概念
                improvement = {
                    'area': weakness_area,
                    'action': '降低概念相似度阈值',
                }
                improvements.append(improvement)

            elif weakness_area == 'high_surprise':
                # 增加世界模型更新频率
                improvement = {
                    'area': weakness_area,
                    'action': '增加模型更新频率',
                }
                improvements.append(improvement)

        self.improvements_made.extend(improvements)
        return improvements

    def get_iteration_stats(self) -> Dict:
        """获取迭代统计"""
        if not self.evaluation_history:
            return {'evaluations': 0}

        latest = self.evaluation_history[-1]
        return {
            'evaluations': len(self.evaluation_history),
            'improvements': len(self.improvements_made),
            'latest_accuracy': latest.prediction_accuracy,
            'latest_coverage': latest.knowledge_coverage,
            'latest_efficiency': latest.learning_efficiency,
            'accuracy_trend': self._compute_trend('prediction_accuracy'),
            'coverage_trend': self._compute_trend('knowledge_coverage'),
        }

    # ── 内部方法 ──────────────────────────────────────────────

    def _evaluate_prediction_accuracy(self, world_model) -> float:
        """评估预测准确率"""
        if world_model.total_predictions == 0:
            return 0.0
        # 惊讶度越低，准确率越高
        return max(0, 1.0 - world_model.avg_surprise)

    def _evaluate_knowledge_coverage(self, world_model) -> float:
        """评估知识覆盖度"""
        if world_model.total_observations == 0:
            return 0.0
        # 规则数 / 观察数
        return min(1.0, len(world_model.rules) / max(world_model.total_observations * 0.1, 1))

    def _evaluate_concept_quality(self, concept_former) -> float:
        """评估概念质量"""
        if not concept_former.concepts:
            return 0.0
        # 平均置信度
        avg_conf = sum(c.confidence for c in concept_former.concepts.values()) / len(concept_former.concepts)
        # 平均例子数
        avg_examples = sum(c.example_count for c in concept_former.concepts.values()) / len(concept_former.concepts)
        return (avg_conf + min(1.0, avg_examples / 5)) / 2

    def _evaluate_learning_efficiency(self, explorer) -> float:
        """评估学习效率"""
        stats = explorer.get_exploration_stats()
        if stats.get('total_explorations', 0) == 0:
            return 0.0
        return stats.get('knowledge_discovered', 0) / max(stats['total_explorations'], 1)

    def _evaluate_surprise_trend(self, explorer) -> float:
        """评估惊讶度趋势（负=在学习）"""
        if len(explorer.surprise_history) < 20:
            return 0.0
        recent = explorer.surprise_history[-20:]
        first_half = sum(recent[:10]) / 10
        second_half = sum(recent[10:]) / 10
        return second_half - first_half

    def _identify_weaknesses(self, world_model, concept_former,
                             explorer, language_system) -> List[Weakness]:
        """识别薄弱环节"""
        weaknesses = []

        # 预测准确率低
        accuracy = self._evaluate_prediction_accuracy(world_model)
        if accuracy < 0.5:
            weaknesses.append(Weakness(
                area='low_prediction_accuracy',
                severity=1.0 - accuracy,
                description=f'预测准确率 {accuracy:.3f}，需要更多探索',
                suggested_action='增加探索多样性',
            ))

        # 概念质量差
        concept_quality = self._evaluate_concept_quality(concept_former)
        if concept_quality < 0.4:
            weaknesses.append(Weakness(
                area='poor_concept_formation',
                severity=1.0 - concept_quality,
                description=f'概念质量 {concept_quality:.3f}，需要更多例子',
                suggested_action='收集更多具体例子',
            ))

        # 惊讶度持续高
        if world_model.avg_surprise > 0.5:
            weaknesses.append(Weakness(
                area='high_surprise',
                severity=world_model.avg_surprise,
                description=f'平均惊讶度 {world_model.avg_surprise:.3f}，世界模型不准确',
                suggested_action='更新世界模型规则',
            ))

        # 语言接地不足
        if language_system:
            lang_stats = language_system.get_stats()
            if lang_stats.get('grounded_words', 0) < 10:
                weaknesses.append(Weakness(
                    area='insufficient_grounding',
                    severity=0.5,
                    description='词汇接地不足',
                    suggested_action='增加语言与世界模型的关联',
                ))

        return weaknesses

    def _generate_recommendations(self, weaknesses: List[Weakness]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        for w in weaknesses:
            recommendations.append(f"{w.area}: {w.suggested_action}")
        return recommendations

    def _compute_trend(self, metric: str) -> float:
        """计算指标趋势"""
        if len(self.evaluation_history) < 2:
            return 0.0
        values = [getattr(r, metric, 0) for r in self.evaluation_history[-10:]]
        if len(values) < 2:
            return 0.0
        return values[-1] - values[0]
