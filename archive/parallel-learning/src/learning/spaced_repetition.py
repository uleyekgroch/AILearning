"""间隔重复调度器 — 基于 SM-2 算法

SM-2 算法是 SuperMemo 系列的核心，用于优化间隔重复调度。
它根据每次复习的质量评分动态调整下次复习间隔。

核心公式：
- 质量 q ∈ [0, 5]：0=完全忘记，5=完美回忆
- 间隔 I(n) = I(n-1) × EF（n=1 时 I(1)=1, n=2 时 I(2)=6）
- 难度因子 EF = EF + (0.1 - (5-q) × (0.08 + (5-q) × 0.02))
- EF 最小值 = 1.3

参考：https://supermemo.guru/wiki/SuperMemo_Algorithm
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.knowledge.unit import KnowledgeUnit, MasteryLevel


@dataclass
class ScheduleEntry:
    """调度条目"""
    unit_id: str
    interval: float = 1.0        # 当前间隔（天）
    ease_factor: float = 2.5     # 难度因子
    repetitions: int = 0         # 成功重复次数
    next_review: float = 0.0     # 下次复习时间（时间戳）
    last_review: float = 0.0     # 上次复习时间
    last_quality: int = 0        # 上次质量评分

    def to_dict(self) -> dict:
        return {
            'unit_id': self.unit_id,
            'interval': self.interval,
            'ease_factor': self.ease_factor,
            'repetitions': self.repetitions,
            'next_review': self.next_review,
            'last_review': self.last_review,
            'last_quality': self.last_quality,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ScheduleEntry':
        return cls(
            unit_id=d.get('unit_id', ''),
            interval=d.get('interval', 1.0),
            ease_factor=d.get('ease_factor', 2.5),
            repetitions=d.get('repetitions', 0),
            next_review=d.get('next_review', 0.0),
            last_review=d.get('last_review', 0.0),
            last_quality=d.get('last_quality', 0),
        )


class SpacedRepetitionScheduler:
    """间隔重复调度器

    基于 SM-2 算法，根据复习质量动态调整间隔。
    """

    def __init__(self, initial_interval: float = 1.0,
                 easy_bonus: float = 1.3,
                 min_ease: float = 1.3,
                 max_interval: float = 365.0):
        """
        Args:
            initial_interval: 初始间隔（天）
            easy_bonus: 简单奖励因子
            min_ease: 最小难度因子
            max_interval: 最大间隔（天）
        """
        self.initial_interval = initial_interval
        self.easy_bonus = easy_bonus
        self.min_ease = min_ease
        self.max_interval = max_interval

        # 调度表：unit_id -> ScheduleEntry
        self.schedule: Dict[str, ScheduleEntry] = {}

    def schedule_review(self, unit_id: str, quality: int) -> ScheduleEntry:
        """安排一次复习

        Args:
            unit_id: 知识单元 ID
            quality: 复习质量 [0, 5]
                0 = 完全忘记
                1 = 错误，但看到答案后想起
                2 = 错误，但答案很熟悉
                3 = 正确，但很困难
                4 = 正确，有些犹豫
                5 = 完美回忆

        Returns:
            更新后的调度条目
        """
        quality = max(0, min(5, quality))

        if unit_id not in self.schedule:
            self.schedule[unit_id] = ScheduleEntry(unit_id=unit_id)

        entry = self.schedule[unit_id]
        now = time.time()

        if quality < 3:
            # 质量太低，重置
            entry.repetitions = 0
            entry.interval = self.initial_interval
        else:
            # 成功回忆
            if entry.repetitions == 0:
                entry.interval = 1.0
            elif entry.repetitions == 1:
                entry.interval = 6.0
            else:
                entry.interval = entry.interval * entry.ease_factor

            entry.repetitions += 1

        # 更新难度因子
        ef = entry.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        entry.ease_factor = max(self.min_ease, ef)

        # 简单奖励
        if quality == 5:
            entry.interval *= self.easy_bonus

        # 限制最大间隔
        entry.interval = min(self.max_interval, entry.interval)

        # 更新时间
        entry.last_review = now
        entry.next_review = now + entry.interval * 86400  # 转换为秒
        entry.last_quality = quality

        return entry

    def get_due_reviews(self, limit: int = 10) -> List[ScheduleEntry]:
        """获取到期的复习

        Args:
            limit: 最大返回数量

        Returns:
            到期的调度条目列表
        """
        now = time.time()
        due = []

        for entry in self.schedule.values():
            if entry.next_review <= now:
                due.append(entry)

        # 按到期时间排序（最早到期的优先）
        due.sort(key=lambda e: e.next_review)

        return due[:limit]

    def get_overdue_reviews(self) -> List[ScheduleEntry]:
        """获取过期的复习（到期但未复习的）"""
        now = time.time()
        overdue = []

        for entry in self.schedule.values():
            if entry.next_review < now:
                overdue.append(entry)

        # 按过期程度排序（最过期的优先）
        overdue.sort(key=lambda e: now - e.next_review, reverse=True)

        return overdue

    def get_upcoming_reviews(self, days: float = 7.0) -> List[ScheduleEntry]:
        """获取即将到来的复习

        Args:
            days: 未来天数

        Returns:
            即将到来的调度条目列表
        """
        now = time.time()
        future = now + days * 86400

        upcoming = []
        for entry in self.schedule.values():
            if now < entry.next_review <= future:
                upcoming.append(entry)

        upcoming.sort(key=lambda e: e.next_review)
        return upcoming

    def get_retention_rate(self) -> float:
        """估算记忆保持率

        基于间隔和难度因子估算。
        """
        if not self.schedule:
            return 0.0

        now = time.time()
        total_retention = 0.0

        for entry in self.schedule.values():
            if entry.next_review <= now:
                # 已过期，保持率下降
                overdue_days = (now - entry.next_review) / 86400
                retention = math.exp(-overdue_days / (entry.interval * 2))
            else:
                # 未到期，保持率高
                retention = 0.9 + 0.1 * (entry.ease_factor / 2.5)

            total_retention += min(1.0, retention)

        return total_retention / len(self.schedule)

    def update_from_unit(self, unit: KnowledgeUnit) -> None:
        """从知识单元更新调度

        根据知识单元的掌握度和练习历史更新调度。
        """
        unit_id = unit.id

        if unit_id not in self.schedule:
            self.schedule[unit_id] = ScheduleEntry(unit_id=unit_id)

        entry = self.schedule[unit_id]

        # 根据掌握度估算质量
        if unit.mastery >= 0.9:
            quality = 5
        elif unit.mastery >= 0.7:
            quality = 4
        elif unit.mastery >= 0.5:
            quality = 3
        elif unit.mastery >= 0.3:
            quality = 2
        elif unit.mastery >= 0.1:
            quality = 1
        else:
            quality = 0

        # 更新调度
        self.schedule_review(unit_id, quality)

    def get_schedule_stats(self) -> Dict:
        """获取调度统计"""
        now = time.time()
        total = len(self.schedule)

        if total == 0:
            return {
                'total': 0,
                'due': 0,
                'overdue': 0,
                'upcoming_7d': 0,
                'avg_interval': 0.0,
                'avg_ease': 0.0,
                'retention_rate': 0.0,
            }

        due = sum(1 for e in self.schedule.values() if e.next_review <= now)
        overdue = sum(1 for e in self.schedule.values() if e.next_review < now)
        upcoming = sum(1 for e in self.schedule.values()
                       if now < e.next_review <= now + 7 * 86400)

        avg_interval = sum(e.interval for e in self.schedule.values()) / total
        avg_ease = sum(e.ease_factor for e in self.schedule.values()) / total

        return {
            'total': total,
            'due': due,
            'overdue': overdue,
            'upcoming_7d': upcoming,
            'avg_interval': round(avg_interval, 2),
            'avg_ease': round(avg_ease, 4),
            'retention_rate': round(self.get_retention_rate(), 4),
        }

    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            'schedule': {uid: e.to_dict() for uid, e in self.schedule.items()},
            'stats': self.get_schedule_stats(),
        }

    def from_dict(self, d: dict) -> None:
        """从字典导入"""
        for uid, entry_dict in d.get('schedule', {}).items():
            self.schedule[uid] = ScheduleEntry.from_dict(entry_dict)
