"""
元认知系统 - 基于2024-2026年最新研究

Metacognition for Bounded and Effectively Self-Governing AI
MetaCognition Patterns for AI Agent Self-Monitoring
Language Models Are Capable of Metacognitive Monitoring and Control
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CognitiveState(Enum):
    """认知状态"""
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    CONFUSED = "confused"
    EXPLORING = "exploring"
    REFLECTING = "reflecting"


class LearningStrategy(Enum):
    """学习策略"""
    CONSERVATIVE = "conservative"  # 保守：依赖已有知识
    EXPLORATORY = "exploratory"  # 探索：尝试新方法
    ANALYTICAL = "analytical"  # 分析：仔细推理
    INTUITIVE = "intuitive"  # 直观：快速响应


@dataclass
class MetacognitiveBelief:
    """元认知信念"""
    topic: str
    confidence: float  # [0, 1]
    uncertainty: float  # [0, 1]
    last_updated: int
    source: str  # 知识来源
    reliability: float  # 来源可靠性


class MetacognitiveMonitor:
    """
    元认知监控器

    监控自身的认知状态，评估置信度和不确定性
    """

    def __init__(self):
        # 自我知识模型
        self.self_knowledge = {}

        # 不确定性追踪
        self.uncertainty_history = defaultdict(deque)

        # 置信度校准
        self.calibration_history = []

        # 认知状态
        self.current_state = CognitiveState.EXPLORING

    def monitor(self, task: dict) -> dict:
        """
        监控当前认知状态

        Args:
            task: {query, context, expected_difficulty}

        Returns:
            监控结果 {confidence, uncertainty, should_delegate}
        """
        query = task.get('query', '')
        context = task.get('context', '')

        # 评估置信度
        confidence = self._assess_confidence(query, context)

        # 评估不确定性
        uncertainty = self._assess_uncertainty(query, context)

        # 决定是否需要委托
        should_delegate = uncertainty > 0.5 or confidence < 0.3

        # 更新认知状态
        self._update_state(confidence, uncertainty)

        return {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'should_delegate': should_delegate,
            'cognitive_state': self.current_state,
        }

    def _assess_confidence(self, query: str, context: str) -> float:
        """评估置信度"""
        # 基于熟悉度
        familiarity = self._assess_familiarity(query)

        # 基于上下文丰富度
        context_richness = len(context) / 100.0 if context else 0.0
        context_richness = min(context_richness, 1.0)

        # 基于历史表现
        historical_accuracy = self._get_historical_accuracy(query)

        # 组合评估
        confidence = (
            0.4 * familiarity +
            0.2 * context_richness +
            0.4 * historical_accuracy
        )

        return np.clip(confidence, 0.0, 1.0)

    def _assess_uncertainty(self, query: str, context: str) -> float:
        """评估不确定性"""
        # 证据不确定性（缺乏信息）
        evidence_uncertainty = 1.0 - self._assess_familiarity(query)

        # 上下文不确定性（上下文冲突）
        context_uncertainty = self._detect_context_conflicts(context)

        # 模型不确定性（知识不足）
        model_uncertainty = self._assess_model_gaps(query)

        # 组合不确定性
        uncertainty = (
            0.5 * evidence_uncertainty +
            0.3 * context_uncertainty +
            0.2 * model_uncertainty
        )

        return np.clip(uncertainty, 0.0, 1.0)

    def _assess_familiarity(self, query: str) -> float:
        """评估查询的熟悉度"""
        # 简化：基于查询中的关键词
        keywords = query.split()
        if not keywords:
            return 0.0

        # 统计熟悉的关键词比例
        familiar_count = 0
        for kw in keywords:
            if kw in self.self_knowledge:
                familiar_count += 1

        return familiar_count / len(keywords)

    def _detect_context_conflicts(self, context: str) -> float:
        """检测上下文冲突"""
        # 简化：检查是否有矛盾的信息
        return 0.0  # 暂时返回0

    def _assess_model_gaps(self, query: str) -> float:
        """评估模型知识缺口"""
        # 检查查询领域是否有足够知识
        domain = query.split()[0] if query else ''
        if domain and domain not in self.self_knowledge:
            return 0.5
        return 0.0

    def _get_historical_accuracy(self, query: str) -> float:
        """获取历史准确率"""
        # 简化：返回默认值
        return 0.7

    def _update_state(self, confidence: float, uncertainty: float):
        """更新认知状态"""
        if uncertainty > 0.6:
            self.current_state = CognitiveState.CONFUSED
        elif uncertainty > 0.3:
            self.current_state = CognitiveState.UNCERTAIN
        elif confidence > 0.7:
            self.current_state = CognitiveState.CONFIDENT
        else:
            self.current_state = CognitiveState.EXPLORING


class MetacognitiveRegulator:
    """
    元认知调节器

    基于监控结果调节认知策略
    """

    def __init__(self):
        self.strategy = LearningStrategy.EXPLORATORY
        self.strategy_history = []
        self.performance_history = deque(maxlen=100)

    def regulate(self, monitoring_result: dict) -> LearningStrategy:
        """
        调节学习策略

        Args:
            monitoring_result: 来自监控器的结果

        Returns:
            选择的学习策略
        """
        uncertainty = monitoring_result.get('uncertainty', 0.5)
        confidence = monitoring_result.get('confidence', 0.5)
        state = monitoring_result.get('cognitive_state', CognitiveState.UNCERTAIN)

        # 策略选择逻辑
        if state == CognitiveState.CONFUSED:
            # 困惑状态：保守策略
            self.strategy = LearningStrategy.CONSERVATIVE
        elif state == CognitiveState.UNCERTAIN:
            # 不确定状态：分析策略
            self.strategy = LearningStrategy.ANALYTICAL
        elif confidence > 0.7:
            # 高置信度：直观策略
            self.strategy = LearningStrategy.INTUITIVE
        else:
            # 默认：探索策略
            self.strategy = LearningStrategy.EXPLORATORY

        # 记录策略历史
        self.strategy_history.append(self.strategy)

        return self.strategy

    def report_performance(self, task: dict, success: bool):
        """报告任务表现，用于未来调节"""
        self.performance_history.append({
            'task': task,
            'strategy': self.strategy,
            'success': success,
        })


class MetacognitiveSystem:
    """
    完整的元认知系统

    整合监控和调节，实现自我监控和自适应
    """

    def __init__(self):
        self.monitor = MetacognitiveMonitor()
        self.regulator = MetacognitiveRegulator()

        # 元认知知识
        self.beliefs = {}

        # 反思历史
        self.reflection_history = []

    def process(self, task: dict) -> dict:
        """
        处理任务，应用元认知

        Args:
            task: {query, context, options}

        Returns:
            处理结果
        """
        # 1. 监控
        monitoring = self.monitor.monitor(task)

        # 2. 调节
        strategy = self.regulator.regulate(monitoring)

        # 3. 更新信念
        self._update_beliefs(task, monitoring)

        return {
            'monitoring': monitoring,
            'strategy': strategy,
            'should_delegate': monitoring['should_delegate'],
        }

    def _update_beliefs(self, task: dict, monitoring: dict):
        """更新元认知信念"""
        query = task.get('query', '')
        confidence = monitoring.get('confidence', 0.5)
        uncertainty = monitoring.get('uncertainty', 0.5)

        # 创建或更新信念
        if query not in self.beliefs:
            self.beliefs[query] = MetacognitiveBelief(
                topic=query,
                confidence=confidence,
                uncertainty=uncertainty,
                last_updated=0,
                source='direct',
                reliability=0.5
            )
        else:
            # 更新现有信念
            belief = self.beliefs[query]
            belief.confidence = 0.7 * belief.confidence + 0.3 * confidence
            belief.uncertainty = 0.7 * belief.uncertainty + 0.3 * uncertainty

    def reflect(self) -> list:
        """反思：总结最近的认知表现"""
        if not self.regulator.performance_history:
            return []

        # 分析最近表现
        recent = list(self.regulator.performance_history)[-10:]

        success_rate = sum(1 for p in recent if p['success']) / len(recent)

        # 策略效果分析
        strategy_performance = defaultdict(list)
        for p in recent:
            strategy_performance[p['strategy']].append(p['success'])

        insights = []
        for strategy, successes in strategy_performance.items():
            rate = sum(successes) / len(successes)
            insights.append({
                'strategy': strategy,
                'success_rate': rate,
                'recommendation': 'increase' if rate > 0.7 else 'decrease'
            })

        self.reflection_history.append(insights)

        return insights

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'num_beliefs': len(self.beliefs),
            'current_state': self.monitor.current_state,
            'current_strategy': self.regulator.strategy,
            'avg_confidence': np.mean([b.confidence for b in self.beliefs.values()])
                                if self.beliefs else 0.0,
            'reflections': len(self.reflection_history),
        }


if __name__ == '__main__':
    print("=== 元认知系统 ===")
    print()
    print("基于2024-2026年最新研究：")
    print("1. 元认知使AI有界自治")
    print("2. AI智能体元认知模式")
    print("3. LLM具备元认知监控与控制")
    print()
    print("核心特性：")
    print("- 自我监控：评估置信度与不确定性")
    print("- 策略调节：选择最优学习策略")
    print("- 信念更新：维护元认知知识")
    print("- 反思能力：总结与改进")
