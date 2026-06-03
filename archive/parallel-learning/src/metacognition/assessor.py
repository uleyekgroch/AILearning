"""元认知评估器 — 整合监控和策略选择的高层接口"""

from typing import Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.metacognition.monitor import MetacognitiveMonitor, KnowledgeAssessment
from src.metacognition.strategy import StrategySelector, LearningStrategy


class MetaAssessor:
    """元认知评估器

    将 MetacognitiveMonitor 和 StrategySelector 组合在一起，
    提供自评估和学习规划的高层接口。
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.monitor = MetacognitiveMonitor(knowledge_graph)
        self.strategy = StrategySelector()

    def self_evaluate(self, topics: List[str] = None) -> Dict[str, float]:
        """返回每个 topic 的 confidence

        若 topics 为 None，则评估知识图谱中所有实体。
        """
        if topics is None:
            topics = list(self.monitor.knowledge.entities.keys())

        result: Dict[str, float] = {}
        for topic in topics:
            result[topic] = self.monitor.knowledge_confidence(topic)

        return result

    def plan_next_learning(self, task: str = None) -> Dict:
        """规划下一步学习

        流程：识别缺口 → 估计难度 → 选策略
        返回 {gaps, difficulty, strategy, readiness}
        """
        # 1. 识别缺口
        gaps = self.monitor.identify_knowledge_gaps()

        # 2. 估计难度（基于任务描述或缺口情况）
        if task:
            difficulty = self.monitor.estimate_difficulty(task)
        else:
            # 无明确任务时，根据缺口比例估计难度
            total = self.monitor.knowledge.entity_count
            if total > 0:
                difficulty = len(gaps) / total
            else:
                difficulty = 1.0  # 空图 → 最大难度

        # 3. 选择策略
        chosen = self.strategy.choose_strategy(difficulty, gaps)

        # 4. 评估整体准备程度
        readiness = self.monitor.assess_readiness(gaps) if gaps else 1.0

        return {
            'gaps': gaps,
            'difficulty': round(difficulty, 4),
            'strategy': {
                'name': chosen.name,
                'description': chosen.description,
                'effectiveness': chosen.effectiveness,
            },
            'readiness': round(readiness, 4),
        }

    def save_state(self) -> dict:
        """保存状态"""
        return {
            'assessments': {
                topic: {
                    'topic': a.topic,
                    'confidence': a.confidence,
                    'evidence_count': a.evidence_count,
                    'gaps': a.gaps,
                }
                for topic, a in self.monitor.assessments.items()
            },
            'strategy_stats': self.strategy.get_strategy_stats(),
        }

    def load_state(self, state: dict) -> None:
        """加载状态"""
        # 恢复 assessments
        self.monitor.assessments.clear()
        for topic, data in state.get('assessments', {}).items():
            self.monitor.assessments[topic] = KnowledgeAssessment(
                topic=data['topic'],
                confidence=data['confidence'],
                evidence_count=data['evidence_count'],
                gaps=data.get('gaps', []),
            )

        # 恢复策略 effectiveness
        stats = state.get('strategy_stats', {})
        for name, info in stats.items():
            if name in self.strategy.STRATEGIES:
                self.strategy.STRATEGIES[name].effectiveness = info.get('effectiveness', 0.5)
