"""
课程涌现 — 从低歧义到高歧义的自发课程

核心发现（v1 → v2 的转折）：
v1 用"复杂度"（物体数+属性数）衡量难度，结果"困难"场景反而
成功率最高（100%）——因为更多属性 = 更多区分信息 = 更容易。

真正的难度来源是歧义度（ambiguity）：场景中多个物体共享属性值，
导致 Speaker 无法用单个符号唯一标识目标。

v2 设计：
- 固定属性数量，控制共享属性数量
- 共享越多 = 歧义越高 = 需要更多符号组合 = 越难
- 测试渐进歧义是否加速学习

理论：
- Vygotsky 最近发展区：在能力边缘学习
- Bengio 课程学习：从易到难
- Piaget 发展阶段：不能跳级
"""

import random
import numpy as np
from typing import List, Dict
from collections import deque

from language_emergence import (
    LanguageAgent, cross_language_round
)
from language_rich_scene import (
    generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES, RICH_ATTRIBUTES
)


# 歧义等级：共享属性越多，有效区分维度越少，难度越高
AMBIGUITY_LEVELS = {
    1: {'num_objects': 3, 'total_attrs': 4, 'num_shared': 0},
    2: {'num_objects': 4, 'total_attrs': 4, 'num_shared': 1},
    3: {'num_objects': 5, 'total_attrs': 4, 'num_shared': 1},
    4: {'num_objects': 5, 'total_attrs': 3, 'num_shared': 1},
    5: {'num_objects': 6, 'total_attrs': 3, 'num_shared': 2},
    6: {'num_objects': 8, 'total_attrs': 3, 'num_shared': 2},
    7: {'num_objects': 10, 'total_attrs': 3, 'num_shared': 2},
}


def generate_ambiguous_scene(num_objects: int, total_attrs: int,
                              num_shared: int) -> List[Dict]:
    """
    生成控制歧义度的场景

    num_shared 个属性所有物体共享相同值 → 这些属性无区分力
    剩余 (total_attrs - num_shared) 个属性各物体独立取值 → 唯一区分维度

    例：num_objects=6, total_attrs=3, num_shared=2
    → 所有物体共享 color=red, shape=circle
    → 只有 size 不同 → 如果 size 只有 3 种值但有 6 个物体 → 必有重复
    """
    attrs = ALL_ATTRIBUTE_NAMES[:total_attrs]
    shared_attrs = attrs[:num_shared]
    unique_attrs = attrs[num_shared:]

    # 共享属性：所有物体取同一个值
    shared_values = {}
    for attr in shared_attrs:
        values = RICH_ATTRIBUTES[attr]
        shared_values[attr] = random.choice(values)

    # 独立属性：每个物体随机取值
    scene = []
    for i in range(num_objects):
        obj = dict(shared_values)
        for attr in unique_attrs:
            values = RICH_ATTRIBUTES[attr]
            obj[attr] = random.choice(values)
        scene.append(obj)

    return scene


def compute_ambiguity(scene: List[Dict]) -> float:
    """
    计算场景的歧义度（0-1）

    0 = 每个物体都唯一（无歧义）
    1 = 所有物体完全相同（无法区分）
    """
    if len(scene) <= 1:
        return 0.0

    num_attrs = len(scene[0]) if scene else 0
    total_shared = 0
    for i in range(len(scene)):
        for j in range(i + 1, len(scene)):
            shared = sum(1 for k in scene[i]
                         if k in scene[j] and scene[i][k] == scene[j][k])
            total_shared += shared

    max_shared = len(scene) * (len(scene) - 1) / 2 * num_attrs
    return total_shared / max(1, max_shared)


def count_unique_objects(scene: List[Dict]) -> int:
    """统计场景中唯一物体的数量"""
    seen = set()
    for obj in scene:
        key = tuple(sorted(obj.items()))
        seen.add(key)
    return len(seen)


class ProgressiveCurriculum:
    """渐进课程：线性递增歧义度"""

    def __init__(self, total_rounds: int = 500,
                 start_level: int = 1, end_level: int = 7):
        self.total_rounds = total_rounds
        self.start_level = start_level
        self.end_level = end_level

    def generate_scene(self, round_num: int) -> List[Dict]:
        progress = round_num / max(1, self.total_rounds - 1)
        level = self.start_level + progress * (self.end_level - self.start_level)
        level = max(1, min(7, round(level)))
        params = AMBIGUITY_LEVELS.get(level, AMBIGUITY_LEVELS[7])
        return generate_ambiguous_scene(**params)


class SelfPacedCurriculum:
    """自主节奏课程：Agent 根据表现调整歧义度"""

    def __init__(self, window: int = 10,
                 increase_threshold: float = 0.85,
                 decrease_threshold: float = 0.50):
        self.current_level = 1.0
        self.window = window
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.recent_results = deque(maxlen=window)
        self.level_history = []

    def update(self, success: bool):
        self.recent_results.append(success)
        if len(self.recent_results) >= self.window:
            rate = sum(self.recent_results) / len(self.recent_results)
            if rate >= self.increase_threshold:
                self.current_level = min(7.0, self.current_level + 0.3)
            elif rate < self.decrease_threshold:
                self.current_level = max(1.0, self.current_level - 0.5)
        self.level_history.append(self.current_level)

    def generate_scene(self) -> List[Dict]:
        lower = max(1, int(self.current_level))
        upper = min(7, lower + 1)
        frac = self.current_level - lower

        lp = AMBIGUITY_LEVELS[lower]
        up = AMBIGUITY_LEVELS[upper]

        num_objects = max(2, round(lp['num_objects'] * (1 - frac) +
                                    up['num_objects'] * frac))
        total_attrs = max(2, min(10, round(lp['total_attrs'] * (1 - frac) +
                                            up['total_attrs'] * frac)))
        num_shared = max(0, min(total_attrs - 1, round(
            lp['num_shared'] * (1 - frac) + up['num_shared'] * frac)))

        return generate_ambiguous_scene(num_objects, total_attrs, num_shared)

    def get_avg_level(self) -> float:
        if not self.level_history:
            return 1.0
        return sum(self.level_history) / len(self.level_history)


def _generate_by_level(level: int) -> List[Dict]:
    """按歧义等级生成场景"""
    params = AMBIGUITY_LEVELS.get(level, AMBIGUITY_LEVELS[4])
    return generate_ambiguous_scene(**params)


def run_curriculum(speaker: LanguageAgent, listener: LanguageAgent,
                   num_rounds: int, strategy: str = 'random',
                   log_interval: int = 50) -> dict:
    """
    运行课程学习实验

    策略：
    - 'random': 随机歧义度（基线）
    - 'easy': 固定低歧义（等级 1）
    - 'hard': 固定高歧义（等级 7）
    - 'progressive': 线性递增歧义
    - 'self_paced': 自主节奏
    """
    successes = 0
    window_successes = 0
    window_total = 0
    window_rates = []
    ambiguity_history = []

    curriculum = None
    if strategy == 'progressive':
        curriculum = ProgressiveCurriculum(num_rounds)
    elif strategy == 'self_paced':
        curriculum = SelfPacedCurriculum()

    for r in range(num_rounds):
        if strategy == 'random':
            level = random.randint(1, 7)
            scene = _generate_by_level(level)
        elif strategy == 'easy':
            scene = _generate_by_level(1)
        elif strategy == 'hard':
            scene = _generate_by_level(7)
        elif strategy == 'progressive':
            scene = curriculum.generate_scene(r)
        elif strategy == 'self_paced':
            scene = curriculum.generate_scene()
        else:
            scene = _generate_by_level(4)

        ambiguity_history.append(compute_ambiguity(scene))

        target_idx = random.randint(0, len(scene) - 1)
        success = cross_language_round(speaker, listener, scene, target_idx)

        if success:
            successes += 1
            window_successes += 1
        window_total += 1

        if strategy == 'self_paced':
            curriculum.update(success)

        if (r + 1) % log_interval == 0:
            rate = window_successes / window_total if window_total > 0 else 0
            window_rates.append(rate)
            window_successes = 0
            window_total = 0

    result = {
        'strategy': strategy,
        'success_rate': successes / num_rounds if num_rounds > 0 else 0,
        'window_rates': window_rates,
        'total_rounds': num_rounds,
        'vocabulary_size': len(speaker.language.vocabulary),
        'avg_ambiguity': sum(ambiguity_history) / len(ambiguity_history),
    }

    if strategy == 'self_paced' and curriculum:
        result['avg_level'] = curriculum.get_avg_level()

    return result
