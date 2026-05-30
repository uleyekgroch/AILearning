#!/usr/bin/env python3
"""元认知自我调控 (Metacognitive Self-Regulation)

基于论文:
- NeurIPS 2025: "Language Models Are Capable of Metacognitive Monitoring and Control"
- Educational Psychology Review 2024: "Executive Functions, Metacognition,
  Self-Regulation, and Self-Regulated Learning"
- Flavell 1979: "Metacognition and cognitive monitoring"

核心思想:
  元认知是"对认知的认知"——系统不仅学习知识，还监控自己学习的效果：
  1. 监控(Monitoring): "我学得好吗？"——评估当前学习效果
  2. 控制(Control): "我该改变策略吗？"——根据监控调整学习参数
  3. 评估(Evaluation): "这次学习成功吗？"——判断是否需要重新学习

  自我调控循环:
  Plan → Act → Monitor → Evaluate → Adjust → Re-Plan

  这与现有self_improvement的区别:
  - self_improvement: 静态参数调整（tunable_params）
  - metacognitive_regulation: 动态策略切换（实时决策）

  对学习系统的意义:
  - 动态调整学习率（学得好→降低，学得差→提高）
  - 决定何时切换学习领域（避免过度专注）
  - 触发反思学习（连续失败→停下来反思）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import math


class LearningPhase(Enum):
    """学习阶段"""
    ACQUISITION = 'acquisition'    # 新知识获取
    CONSOLIDATION = 'consolidation'  # 知识巩固
    REFLECTION = 'reflection'      # 反思调整
    EXPLORATION = 'exploration'    # 新领域探索


class LearningStrategy(Enum):
    """学习策略"""
    FAST = 'fast'          # 快速模式（低精度，高覆盖）
    THOROUGH = 'thorough'  # 精确模式（高精度，深学习）
    REVIEW = 'review'      # 复习模式（巩固已有知识）
    EXPLORE = 'explore'    # 探索模式（新领域）


@dataclass
class MetacognitiveState:
    """元认知状态快照"""
    phase: LearningPhase
    strategy: LearningStrategy
    recent_success_rate: float     # 最近成功率
    knowledge_confidence: float    # 知识置信度
    learning_velocity: float       # 学习速度（成功/时间）
    domain_coverage: float         # 领域覆盖度
    uncertainty_level: float       # 不确定性水平
    adjustment_count: int = 0      # 策略调整次数


class MetacognitiveRegulator:
    """元认知自我调控器

    核心循环:
    1. 监控学习效果（成功率、置信度、速度）
    2. 评估当前策略是否有效
    3. 必要时调整策略
    4. 追踪调整效果
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 状态
        self.current_phase = LearningPhase.ACQUISITION
        self.current_strategy = LearningStrategy.FAST

        # 学习历史（用于监控）
        self.success_history: List[float] = []
        self.confidence_history: List[float] = []
        self.velocity_history: List[float] = []

        # 领域统计
        self.domain_stats: Dict[str, Dict] = {}

        # 可调参数（由元认知动态调整）
        self.dynamic_params = {
            'learning_rate': 1e-4,          # 当前学习率
            'exploration_ratio': 0.3,       # 探索vs利用比例
            'consolidation_interval': 500,   # 巩固间隔
            'confidence_threshold': 0.5,    # 置信度阈值
            'max_retry': 3,                 # 最大重试次数
        }

        # 调整历史
        self.adjustment_log: List[Dict] = []

        # 统计
        self.stats = {
            'total_evaluations': 0,
            'strategy_switches': 0,
            'phase_transitions': 0,
            'reflections_triggered': 0,
            'params_adjusted': 0,
        }

    def monitor(self, success: bool, confidence: float,
                domain: str = '', elapsed: float = 0.0) -> MetacognitiveState:
        """监控学习效果

        每次学习后调用，记录效果并评估当前状态。

        Args:
            success: 本次学习是否成功
            confidence: 置信度(0-1)
            domain: 学习领域
            elapsed: 本次学习耗时

        Returns:
            当前元认知状态
        """
        # 记录历史
        self.success_history.append(1.0 if success else 0.0)
        self.confidence_history.append(confidence)
        if elapsed > 0:
            velocity = (1.0 if success else 0.0) / elapsed
            self.velocity_history.append(velocity)

        # 限制历史长度
        max_history = 100
        if len(self.success_history) > max_history:
            self.success_history = self.success_history[-max_history:]
        if len(self.confidence_history) > max_history:
            self.confidence_history = self.confidence_history[-max_history:]
        if len(self.velocity_history) > max_history:
            self.velocity_history = self.velocity_history[-max_history:]

        # 更新领域统计
        if domain:
            if domain not in self.domain_stats:
                self.domain_stats[domain] = {'count': 0, 'success': 0}
            self.domain_stats[domain]['count'] += 1
            if success:
                self.domain_stats[domain]['success'] += 1

        # 计算当前指标
        recent_success_rate = self._recent_average(self.success_history, 20)
        recent_confidence = self._recent_average(self.confidence_history, 20)
        recent_velocity = self._recent_average(self.velocity_history, 20) if self.velocity_history else 0.0
        coverage = len(self.domain_stats) / max(1, len(self.domain_stats))
        uncertainty = 1.0 - recent_confidence

        state = MetacognitiveState(
            phase=self.current_phase,
            strategy=self.current_strategy,
            recent_success_rate=recent_success_rate,
            knowledge_confidence=recent_confidence,
            learning_velocity=recent_velocity,
            domain_coverage=coverage,
            uncertainty_level=uncertainty,
        )

        self.stats['total_evaluations'] += 1

        # 评估并调整
        self._evaluate_and_adjust(state)

        return state

    def _evaluate_and_adjust(self, state: MetacognitiveState):
        """评估当前状态并调整策略

        核心决策逻辑:
        - 成功率高 → 可以进入更深的学习模式
        - 成功率低 → 降低难度或切换领域
        - 连续失败 → 触发反思
        - 速度下降 → 可能需要巩固
        """
        adjustments = []

        # 策略1: 成功率太低 → 切换到复习模式
        if state.recent_success_rate < 0.3 and len(self.success_history) >= 10:
            if self.current_strategy != LearningStrategy.REVIEW:
                self.current_strategy = LearningStrategy.REVIEW
                self.dynamic_params['learning_rate'] *= 0.5  # 降低学习率
                self.dynamic_params['confidence_threshold'] *= 0.8  # 降低阈值
                adjustments.append('strategy→review (low success rate)')

        # 策略2: 成功率高 → 切换到精确模式
        elif state.recent_success_rate > 0.7 and len(self.success_history) >= 10:
            if self.current_strategy == LearningStrategy.FAST:
                self.current_strategy = LearningStrategy.THOROUGH
                self.dynamic_params['confidence_threshold'] = min(0.8, self.dynamic_params['confidence_threshold'] + 0.1)
                adjustments.append('strategy→thorough (high success rate)')

        # 策略3: 不确定性高 → 增加探索
        if state.uncertainty_level > 0.6:
            self.dynamic_params['exploration_ratio'] = min(0.7, self.dynamic_params['exploration_ratio'] + 0.1)
            adjustments.append('exploration↑ (high uncertainty)')

        # 策略4: 速度下降 → 需要巩固
        if len(self.velocity_history) >= 10:
            recent_velocity = self._recent_average(self.velocity_history, 5)
            older_velocity = self._recent_average(self.velocity_history[-20:-5], 10) if len(self.velocity_history) >= 20 else recent_velocity
            if recent_velocity < older_velocity * 0.5:
                self.current_phase = LearningPhase.CONSOLIDATION
                self.dynamic_params['consolidation_interval'] = max(100, self.dynamic_params['consolidation_interval'] // 2)
                adjustments.append('phase→consolidation (velocity drop)')

        # 策略5: 连续失败 → 触发反思
        if len(self.success_history) >= 5:
            last_5 = self.success_history[-5:]
            if all(s == 0.0 for s in last_5):
                self.current_phase = LearningPhase.REFLECTION
                self.stats['reflections_triggered'] += 1
                adjustments.append('phase→reflection (5 consecutive failures)')

        # 记录调整
        if adjustments:
            self.adjustment_log.append({
                'adjustments': adjustments,
                'success_rate': state.recent_success_rate,
                'strategy': self.current_strategy.value,
                'phase': self.current_phase.value,
            })
            self.stats['strategy_switches'] += len(adjustments)
            self.stats['params_adjusted'] += len(adjustments)

    def get_learning_params(self) -> Dict:
        """获取当前元认知调整后的学习参数"""
        return {
            **self.dynamic_params,
            'phase': self.current_phase.value,
            'strategy': self.current_strategy.value,
            'should_consolidate': self.current_phase == LearningPhase.CONSOLIDATION,
            'should_reflect': self.current_phase == LearningPhase.REFLECTION,
            'should_explore': self.current_phase == LearningPhase.EXPLORATION,
        }

    def _recent_average(self, history: list, window: int) -> float:
        """计算最近N个值的平均值"""
        if not history:
            return 0.0
        recent = history[-window:]
        return sum(recent) / len(recent)

    def get_domain_priorities(self) -> List[Tuple[str, float]]:
        """获取领域学习优先级

        返回应该优先学习的领域（成功率低的领域优先级更高）。
        """
        priorities = []
        for domain, stats in self.domain_stats.items():
            success_rate = stats['success'] / max(1, stats['count'])
            # 成功率低的领域需要更多学习
            priority = 1.0 - success_rate
            priorities.append((domain, priority))

        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities[:10]

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'current_phase': self.current_phase.value,
            'current_strategy': self.current_strategy.value,
            'domains_tracked': len(self.domain_stats),
            'recent_success_rate': self._recent_average(self.success_history, 20),
            'recent_confidence': self._recent_average(self.confidence_history, 20),
        }
