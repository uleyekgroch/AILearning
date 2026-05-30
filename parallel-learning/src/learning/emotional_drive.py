"""情感驱动引擎 — 从好奇心到完整情感系统

实现情感驱动学习的核心能力：
1. 好奇心：对新奇事物的探索欲望
2. 成就感：完成任务的满足感
3. 挫败感：失败时的调整动力
4. 社交情感：与他人互动的情感

设计原则：
- 情感是学习的驱动力，不是副产品
- 不同情感影响不同的学习策略
- 情感状态应该可调节
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class EmotionalState:
    """情感状态"""
    curiosity: float = 0.5  # 好奇心
    satisfaction: float = 0.5  # 满足感
    frustration: float = 0.0  # 挫败感
    social_connection: float = 0.5  # 社交连接
    confidence: float = 0.5  # 自信
    timestamp: float = 0.0


@dataclass
class EmotionalEvent:
    """情感事件"""
    event_type: str
    description: str
    emotional_impact: Dict[str, float]
    timestamp: float


class EmotionalDriveEngine:
    """情感驱动引擎

    管理和利用情感状态来驱动学习。
    """

    def __init__(self):
        # 当前情感状态
        self.state = EmotionalState(timestamp=time.time())

        # 情感历史
        self.history: List[EmotionalState] = []

        # 情感事件
        self.events: List[EmotionalEvent] = []

        # 情感衰减率
        self.decay_rates = {
            'curiosity': 0.01,
            'satisfaction': 0.02,
            'frustration': 0.05,
            'social_connection': 0.01,
            'confidence': 0.01,
        }

        # 情感阈值
        self.thresholds = {
            'high_curiosity': 0.7,
            'low_satisfaction': 0.3,
            'high_frustration': 0.6,
            'low_confidence': 0.3,
        }

    def update_state(self, event: EmotionalEvent):
        """更新情感状态

        Args:
            event: 情感事件
        """
        # 记录事件
        self.events.append(event)

        # 更新情感状态
        for emotion, impact in event.emotional_impact.items():
            if hasattr(self.state, emotion):
                current = getattr(self.state, emotion)
                new_value = max(0.0, min(1.0, current + impact))
                setattr(self.state, emotion, new_value)

        # 更新时间戳
        self.state.timestamp = time.time()

        # 记录历史
        self.history.append(EmotionalState(
            curiosity=self.state.curiosity,
            satisfaction=self.state.satisfaction,
            frustration=self.state.frustration,
            social_connection=self.state.social_connection,
            confidence=self.state.confidence,
            timestamp=self.state.timestamp,
        ))

    def decay_emotions(self):
        """衰减情感状态"""
        for emotion, rate in self.decay_rates.items():
            if hasattr(self.state, emotion):
                current = getattr(self.state, emotion)
                # 向0.5衰减（中性状态）
                target = 0.5
                new_value = current + (target - current) * rate
                setattr(self.state, emotion, new_value)

    def on_learning_success(self, description: str):
        """学习成功时的情感反应"""
        event = EmotionalEvent(
            event_type='success',
            description=description,
            emotional_impact={
                'satisfaction': 0.2,
                'confidence': 0.1,
                'curiosity': 0.05,
                'frustration': -0.1,
            },
            timestamp=time.time(),
        )
        self.update_state(event)

    def on_learning_failure(self, description: str):
        """学习失败时的情感反应"""
        event = EmotionalEvent(
            event_type='failure',
            description=description,
            emotional_impact={
                'frustration': 0.15,
                'confidence': -0.1,
                'satisfaction': -0.05,
            },
            timestamp=time.time(),
        )
        self.update_state(event)

    def on_new_discovery(self, description: str):
        """新发现时的情感反应"""
        event = EmotionalEvent(
            event_type='discovery',
            description=description,
            emotional_impact={
                'curiosity': 0.3,
                'satisfaction': 0.15,
                'confidence': 0.05,
            },
            timestamp=time.time(),
        )
        self.update_state(event)

    def on_social_interaction(self, description: str, positive: bool = True):
        """社交互动时的情感反应"""
        impact = {
            'social_connection': 0.2 if positive else -0.15,
            'satisfaction': 0.1 if positive else -0.05,
        }

        event = EmotionalEvent(
            event_type='social',
            description=description,
            emotional_impact=impact,
            timestamp=time.time(),
        )
        self.update_state(event)

    def get_exploration_drive(self) -> float:
        """获取探索驱动力"""
        # 好奇心高 + 挫败感低 = 高探索驱动力
        return self.state.curiosity * (1.0 - self.state.frustration)

    def get_persistence_drive(self) -> float:
        """获取坚持驱动力"""
        # 自信高 + 满足感中等 = 高坚持驱动力
        return self.state.confidence * 0.7 + self.state.satisfaction * 0.3

    def get_social_drive(self) -> float:
        """获取社交驱动力"""
        return self.state.social_connection

    def should_explore(self) -> bool:
        """判断是否应该探索"""
        return self.get_exploration_drive() > 0.5

    def should_persist(self) -> bool:
        """判断是否应该坚持"""
        return self.get_persistence_drive() > 0.4

    def get_recommended_action(self) -> str:
        """获取推荐行动"""
        if self.state.curiosity > self.thresholds['high_curiosity']:
            return 'explore_new'
        elif self.state.frustration > self.thresholds['high_frustration']:
            return 'take_break'
        elif self.state.confidence < self.thresholds['low_confidence']:
            return 'practice_easy'
        elif self.state.satisfaction > 0.7:
            return 'increase_difficulty'
        else:
            return 'continue_learning'

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'curiosity': self.state.curiosity,
            'satisfaction': self.state.satisfaction,
            'frustration': self.state.frustration,
            'social_connection': self.state.social_connection,
            'confidence': self.state.confidence,
            'exploration_drive': self.get_exploration_drive(),
            'persistence_drive': self.get_persistence_drive(),
            'total_events': len(self.events),
        }

    def get_report(self) -> str:
        """获取报告"""
        stats = self.get_stats()
        lines = [
            "=== 情感驱动引擎报告 ===",
            f"好奇心: {stats['curiosity']:.2f}",
            f"满足感: {stats['satisfaction']:.2f}",
            f"挫败感: {stats['frustration']:.2f}",
            f"社交连接: {stats['social_connection']:.2f}",
            f"自信: {stats['confidence']:.2f}",
            "",
            f"探索驱动力: {stats['exploration_drive']:.2f}",
            f"坚持驱动力: {stats['persistence_drive']:.2f}",
            f"推荐行动: {self.get_recommended_action()}",
        ]

        if self.events:
            lines.append("")
            lines.append("最近事件:")
            for event in self.events[-3:]:
                lines.append(f"  [{event.event_type}] {event.description}")

        return '\n'.join(lines)
