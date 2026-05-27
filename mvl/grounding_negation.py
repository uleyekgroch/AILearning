"""
否定接地模块：否定符号从区分需求中涌现

核心思想：
否定（"not", "no"）的意义来自排除需求。
当正向描述无法唯一标识目标时，否定成为必要工具。
"not red" = 排除所有红色物体后剩下的那个。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener,
    NEGATION_MARKERS, _symbol_category, generate_rich_scene,
)


def generate_negation_scene(num_objects: int = 8,
                             complexity: str = 'medium') -> Tuple[List[Dict[str, str]], int]:
    """
    生成否定更高效的场景

    设计：所有物体共享 2 个特征维度，第 3 个维度有差异。
    目标是唯一的 minority，正向描述需要 2 符号，否定只需 2 符号（"not majority"）。
    但否定在语义上更简洁——排除而非包含。
    """
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle']
    sizes = ['big', 'small']

    majority_color = np.random.choice(colors)
    minority_color = np.random.choice([c for c in colors if c != majority_color])
    shared_shape = np.random.choice(shapes)
    shared_size = np.random.choice(sizes)

    scene = []
    target_idx = num_objects - 1

    for i in range(num_objects):
        if i == target_idx:
            scene.append({
                'color': minority_color,
                'shape': shared_shape,
                'size': shared_size,
            })
        else:
            scene.append({
                'color': majority_color,
                'shape': shared_shape,
                'size': shared_size,
            })

    return scene, target_idx


class NegationCommunicationGame:
    """否定交流游戏：验证否定符号的涌现"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.game_log = []
        self.negation_used = 0
        self.negation_success = 0

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int) -> bool:
        if target_idx >= len(scene_features):
            return False

        target = scene_features[target_idx]

        # speaker 描述（传入 target_idx 以启用否定/从句）
        utterance = self.speaker.describe(target, scene_features, target_idx)
        if not utterance:
            return False

        # 检测是否使用了否定
        used_negation = any(s in NEGATION_MARKERS for s in utterance)
        if used_negation:
            self.negation_used += 1

        # listener 解释
        chosen_idx = self.listener.interpret(utterance, scene_features)
        success = (chosen_idx == target_idx)

        if used_negation and success:
            self.negation_success += 1

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
            'used_negation': used_negation,
        })
        return success

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['negation_used'] = self.negation_used
        stats['negation_success'] = self.negation_success
        stats['negation_rate'] = self.negation_used / max(1, len(self.game_log))
        return stats


class NegationLanguageAgent:
    """拥有独立否定语言的 agent"""

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


def cross_negation_round(speaker_agent: NegationLanguageAgent,
                         listener_agent: NegationLanguageAgent,
                         scene: List[Dict[str, str]],
                         target_idx: int) -> bool:
    """跨语言否定交流一轮"""
    utterance = speaker_agent.speak(scene, target_idx)
    if not utterance:
        return False

    chosen_idx = listener_agent.listen(utterance, scene)
    success = (chosen_idx == target_idx)

    speaker_agent.update(utterance, success)
    listener_agent.update(utterance, success)
    return success
