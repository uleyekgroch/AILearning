"""
递归从句接地模块：相对从句从区分需求中涌现

核心思想：
相对从句（"the ball that was pushed"）的意义来自嵌套描述需求。
当多个物体共享静态属性但动作不同时，
主句"ball"无法区分 → 需要从句"that was pushed"来过滤。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener,
    RELATIVE_MARKERS, ACTIONS, ACTION_EFFECTS,
    _symbol_category, generate_rich_scene,
)


def generate_recursive_scene(num_objects: int = 8,
                              complexity: str = 'medium') -> Tuple[List[Dict[str, str]], int]:
    """
    生成需要相对从句才能区分的场景

    设计：多个物体共享相同的静态属性（如都是红色圆形），
    但有不同的动作（一个被推，一个被拉）。
    主句"red circle"无法区分 → 需要从句"that push"来过滤。
    """
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle']
    sizes = ['big', 'small']
    action_types = list(ACTIONS)

    # 选择共享的静态属性
    shared_color = np.random.choice(colors)
    shared_shape = np.random.choice(shapes)
    shared_size = np.random.choice(sizes)

    # 所有相似物体共享相同的静态属性，但有不同的动作
    num_similar = min(5, num_objects - 1)
    scene = []

    for i in range(num_similar):
        action = action_types[i % len(action_types)]
        effect = ACTION_EFFECTS.get(action, 'unknown')
        scene.append({
            'color': shared_color,
            'shape': shared_shape,
            'size': shared_size,
            'action': action,
            'action_effect': effect,
        })

    # 添加一些不同的物体（填充场景）
    for i in range(num_similar, num_objects):
        scene.append({
            'color': np.random.choice(colors),
            'shape': np.random.choice(shapes),
            'size': np.random.choice(sizes),
        })

    # 随机选择一个相似物体作为目标
    target_idx = np.random.randint(num_similar)

    return scene, target_idx


class RecursiveCommunicationGame:
    """递归从句交流游戏：验证相对从句的涌现"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.game_log = []
        self.relative_used = 0
        self.relative_success = 0

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int) -> bool:
        if target_idx >= len(scene_features):
            return False

        target = scene_features[target_idx]

        # speaker 描述（传入 target_idx 以启用从句）
        utterance = self.speaker.describe(target, scene_features, target_idx)
        if not utterance:
            return False

        # 检测是否使用了相对从句
        used_relative = any(s in RELATIVE_MARKERS for s in utterance)
        if used_relative:
            self.relative_used += 1

        # listener 解释
        chosen_idx = self.listener.interpret(utterance, scene_features)
        success = (chosen_idx == target_idx)

        if used_relative and success:
            self.relative_success += 1

        # 更新语言统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        if len(utterance) > 1:
            self.language.multi_symbol_games += 1

        self.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                self.language.record_collocation(utterance[i], utterance[i + 1], success)
            self.language.record_ngram(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
            'used_relative': used_relative,
        })
        return success

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['relative_used'] = self.relative_used
        stats['relative_success'] = self.relative_success
        stats['relative_rate'] = self.relative_used / max(1, len(self.game_log))
        return stats


class RecursiveLanguageAgent:
    """拥有独立从句语言的 agent"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)

    def speak(self, scene: List[Dict[str, str]], target_idx: int) -> List[str]:
        if target_idx < len(scene):
            return self.speaker.describe(scene[target_idx], scene, target_idx)
        return []

    def listen(self, utterance: List[str], scene: List[Dict[str, str]]) -> Optional[int]:
        return self.listener.interpret(utterance, scene)

    def update(self, utterance: List[str], success: bool):
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                self.language.record_collocation(utterance[i], utterance[i + 1], success)
            self.language.record_ngram(utterance, success)


def cross_recursive_round(speaker_agent: RecursiveLanguageAgent,
                          listener_agent: RecursiveLanguageAgent,
                          scene: List[Dict[str, str]],
                          target_idx: int) -> bool:
    """跨语言从句交流一轮"""
    utterance = speaker_agent.speak(scene, target_idx)
    if not utterance:
        return False

    chosen_idx = listener_agent.listen(utterance, scene)
    success = (chosen_idx == target_idx)

    speaker_agent.update(utterance, success)
    listener_agent.update(utterance, success)
    return success
