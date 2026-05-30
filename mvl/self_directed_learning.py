"""
自主目标设定 — Agent 自己决定学什么

核心思想：
当前所有学习都是外部驱动的（实验者选择场景和目标）。
真实的婴儿会主动选择探索什么——先爬再走、先抓再扔。
Agent 应该能评估自己的知识状态，识别薄弱环节，针对性练习。

机制：
1. KnowledgeAssessor — 评估每个属性维度的掌握程度
2. GoalSelector — 选择最需要练习的维度
3. SelfDirectedLearner — 编排自主学习循环
"""

import math
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from language_emergence import (
    LanguageAgent, EmergingLanguage, cross_language_round
)
from language_rich_scene import (
    generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES, RICH_ATTRIBUTES
)


class KnowledgeAssessor:
    """
    知识评估器 — 评估 Agent 在每个属性维度上的掌握程度

    利用 EmergingLanguage.dimension_stats 中已有的
    per-dimension frequency 和 success_rate 作为知识差距信号。
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def assess(self) -> Dict[str, float]:
        """
        返回每个属性维度的优先级分数（越高越需要练习）

        优先级公式：
        - 未见过的属性 → priority = 2.0（最高优先，始终优先于已见属性）
        - 已知属性 → priority = (1 - success_rate) * log(1 + frequency)
          频率越高且成功率越低 → 优先级越高

        注意：unseen 的 priority 2.0 > 已知属性的最大可能值
        （(1-0)*log(1+freq) 需要 freq > 6.4 才能超过 2.0），
        但 epsilon-greedy 保证 10% 随机探索不会完全卡住。
        """
        priorities = {}
        for attr in ALL_ATTRIBUTE_NAMES:
            stats = self.language.dimension_stats.get(attr)
            if stats is None or stats['frequency'] == 0:
                priorities[attr] = 2.0
            else:
                sr = stats['success_rate']
                freq = stats['frequency']
                priorities[attr] = (1.0 - sr) * math.log(1 + freq)
        return priorities

    def get_knowledge_summary(self) -> Dict[str, dict]:
        """返回每个属性维度的详细知识状态"""
        summary = {}
        for attr in ALL_ATTRIBUTE_NAMES:
            stats = self.language.dimension_stats.get(attr)
            if stats is None:
                summary[attr] = {'status': 'unseen', 'frequency': 0,
                                 'success_rate': 0.0, 'priority': 1.0}
            else:
                sr = stats['success_rate']
                freq = stats['frequency']
                priority = (1.0 - sr) * math.log(1 + freq)
                if sr >= 0.8:
                    status = 'mastered'
                elif sr >= 0.5:
                    status = 'learning'
                else:
                    status = 'weak'
                summary[attr] = {
                    'status': status,
                    'frequency': freq,
                    'success_rate': round(sr, 3),
                    'priority': round(priority, 3),
                }
        return summary


class GoalSelector:
    """
    目标选择器 — 选择最需要练习的属性维度

    策略：
    - 90% 按优先级选 top-N（exploitation）
    - 10% 随机选（exploration，避免陷入局部）
    """

    def __init__(self, assessor: KnowledgeAssessor, epsilon: float = 0.1):
        self.assessor = assessor
        self.epsilon = epsilon
        self.goal_history = []  # 记录每次选择的目标

    def select_goals(self, num_goals: int = 2) -> List[str]:
        """选择最需要练习的属性维度"""
        priorities = self.assessor.assess()

        if random.random() < self.epsilon:
            # 探索：随机选择
            goals = random.sample(ALL_ATTRIBUTE_NAMES,
                                  min(num_goals, len(ALL_ATTRIBUTE_NAMES)))
        else:
            # 利用：按优先级排序选 top-N（同优先级随机打乱，避免卡住）
            items = list(priorities.items())
            random.shuffle(items)  # 打破同优先级的确定性顺序
            sorted_attrs = sorted(items, key=lambda x: x[1], reverse=True)
            goals = [attr for attr, _ in sorted_attrs[:num_goals]]

        self.goal_history.append(goals)
        return goals

    def generate_practice_scene(self, num_objects: int = 8) -> List[Dict]:
        """
        生成针对薄弱维度的练习场景

        关键：场景中只保留目标属性，迫使 Agent 用这些属性描述物体。
        这模拟了真实学习中的"聚焦练习"——只给你看颜色，你必须学会用颜色区分。
        """
        goals = self.select_goals()
        scene = generate_rich_scene_v2(
            num_objects=num_objects,
            attribute_names=goals
        )
        # 过滤：只保留目标属性
        filtered = []
        for obj in scene:
            filtered.append({k: v for k, v in obj.items() if k in goals})
        return filtered


class SelfDirectedLearner:
    """
    自主学习编排器 — Agent 自己决定学什么并执行学习

    每轮：
    1. 评估知识状态
    2. 选择薄弱维度
    3. 生成针对性场景
    4. 执行通信游戏
    5. 从结果中学习
    """

    def __init__(self, speaker: LanguageAgent, listener: LanguageAgent,
                 epsilon: float = 0.1):
        self.speaker = speaker
        self.listener = listener
        self.goal_selector = GoalSelector(
            KnowledgeAssessor(speaker.language), epsilon=epsilon
        )

    def run_round(self) -> bool:
        """一轮自主学习"""
        scene = self.goal_selector.generate_practice_scene()
        target_idx = random.randint(0, len(scene) - 1)
        return cross_language_round(
            self.speaker, self.listener, scene, target_idx
        )

    def run_session(self, num_rounds: int,
                    log_interval: int = 50) -> dict:
        """
        运行多轮自主学习

        返回：
        - success_rate: 累积成功率
        - window_rates: 每 log_interval 轮的窗口成功率
        - goal_history: 目标选择历史
        - knowledge_snapshots: 定期知识状态快照
        """
        successes = 0
        window_successes = 0
        window_total = 0
        window_rates = []
        knowledge_snapshots = {}

        for r in range(num_rounds):
            success = self.run_round()
            if success:
                successes += 1
                window_successes += 1
            window_total += 1

            if (r + 1) % log_interval == 0:
                rate = window_successes / window_total if window_total > 0 else 0
                window_rates.append(rate)
                window_successes = 0
                window_total = 0

                # 记录知识快照
                assessor = KnowledgeAssessor(self.speaker.language)
                knowledge_snapshots[r + 1] = assessor.get_knowledge_summary()

        return {
            'success_rate': successes / num_rounds if num_rounds > 0 else 0,
            'window_rates': window_rates,
            'total_rounds': num_rounds,
            'vocabulary_size': len(self.speaker.language.vocabulary),
            'goal_history': self.goal_selector.goal_history,
            'knowledge_snapshots': knowledge_snapshots,
        }


def run_random_learning(speaker: LanguageAgent, listener: LanguageAgent,
                        num_rounds: int, num_attrs: int = 3,
                        log_interval: int = 50) -> dict:
    """随机学习基线：每轮随机选属性维度"""
    successes = 0
    window_successes = 0
    window_total = 0
    window_rates = []

    for r in range(num_rounds):
        attrs = random.sample(ALL_ATTRIBUTE_NAMES,
                              min(num_attrs, len(ALL_ATTRIBUTE_NAMES)))
        scene = generate_rich_scene_v2(num_objects=8, attribute_names=attrs)
        target_idx = random.randint(0, len(scene) - 1)
        success = cross_language_round(speaker, listener, scene, target_idx)
        if success:
            successes += 1
            window_successes += 1
        window_total += 1

        if (r + 1) % log_interval == 0:
            rate = window_successes / window_total if window_total > 0 else 0
            window_rates.append(rate)
            window_successes = 0
            window_total = 0

    return {
        'success_rate': successes / num_rounds if num_rounds > 0 else 0,
        'window_rates': window_rates,
        'total_rounds': num_rounds,
        'vocabulary_size': len(speaker.language.vocabulary),
    }


def run_uniform_learning(speaker: LanguageAgent, listener: LanguageAgent,
                         num_rounds: int, num_attrs: int = 3,
                         log_interval: int = 50) -> dict:
    """均匀学习基线：轮转所有属性维度"""
    successes = 0
    window_successes = 0
    window_total = 0
    window_rates = []
    attr_idx = 0

    for r in range(num_rounds):
        # 轮转选择属性
        attrs = []
        for i in range(num_attrs):
            attrs.append(ALL_ATTRIBUTE_NAMES[(attr_idx + i) % len(ALL_ATTRIBUTE_NAMES)])
        attr_idx = (attr_idx + 1) % len(ALL_ATTRIBUTE_NAMES)

        scene = generate_rich_scene_v2(num_objects=8, attribute_names=attrs)
        target_idx = random.randint(0, len(scene) - 1)
        success = cross_language_round(speaker, listener, scene, target_idx)
        if success:
            successes += 1
            window_successes += 1
        window_total += 1

        if (r + 1) % log_interval == 0:
            rate = window_successes / window_total if window_total > 0 else 0
            window_rates.append(rate)
            window_successes = 0
            window_total = 0

    return {
        'success_rate': successes / num_rounds if num_rounds > 0 else 0,
        'window_rates': window_rates,
        'total_rounds': num_rounds,
        'vocabulary_size': len(speaker.language.vocabulary),
    }
