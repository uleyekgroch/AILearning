"""
动作接地模块：动词符号从行为-效果循环中涌现

核心思想：
动作符号（push, pull, grab 等）的意义来自 agent 执行动作后的感知变化。
"push" = 物体位移的可靠预测，"move" = agent 自身位置变化。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener,
    ACTIONS, ACTION_EFFECTS, _symbol_category, generate_rich_scene
)


@dataclass
class ActionEvent:
    """一个动作事件：谁对什么做了什么"""
    action_type: str           # 'push', 'move', etc.
    target_obj_idx: int        # 被作用的物体索引（-1 表示无目标）
    direction: str             # 'up', 'down', 'left', 'right'
    effect_description: str    # 'displacement', 'translation', etc.

    def to_feature_dict(self) -> Dict[str, str]:
        return {
            'action': self.action_type,
            'action_effect': self.effect_description,
            'direction': self.direction,
        }


def generate_action_scene(num_objects: int = 6,
                          complexity: str = 'medium') -> Tuple[List[Dict[str, str]], List[ActionEvent]]:
    """生成带动作标注的场景"""
    base_scene = generate_rich_scene(complexity)[:num_objects]

    action_types = list(ACTIONS)
    directions = ['up', 'down', 'left', 'right']

    action_events = []
    for i, obj_features in enumerate(base_scene):
        if np.random.random() < 0.5:
            action_type = np.random.choice(action_types)
            direction = np.random.choice(directions)
            effect = ACTION_EFFECTS.get(action_type, 'unknown')

            obj_features['action'] = action_type
            obj_features['action_effect'] = effect
            obj_features['direction'] = direction

            action_events.append(ActionEvent(
                action_type=action_type,
                target_obj_idx=i,
                direction=direction,
                effect_description=effect,
            ))

    return base_scene, action_events


class ActionCommunicationGame:
    """动作交流游戏：描述"发生了什么"，listener 识别目标物体"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   action_events: List[ActionEvent],
                   target_event_idx: int) -> bool:
        if target_event_idx >= len(action_events):
            return False

        target_event = action_events[target_event_idx]
        target_obj_idx = target_event.target_obj_idx

        # speaker 描述：合并物体特征和动作特征
        if target_obj_idx < len(scene_features):
            combined = {**scene_features[target_obj_idx], **target_event.to_feature_dict()}
        else:
            combined = target_event.to_feature_dict()

        utterance = self.speaker.describe(combined, scene_features)
        if not utterance:
            return False

        # listener 解释
        chosen_idx = self.listener.interpret(utterance, scene_features)
        success = (chosen_idx == target_obj_idx)

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
            'target_obj_idx': target_obj_idx,
            'action': target_event.action_type,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success


class ActionLanguageAgent:
    """拥有独立动作语言的 agent"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)

    def speak(self, scene: List[Dict[str, str]], event: ActionEvent) -> List[str]:
        if event.target_obj_idx < len(scene):
            combined = {**scene[event.target_obj_idx], **event.to_feature_dict()}
        else:
            combined = event.to_feature_dict()
        return self.speaker.describe(combined, scene)

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


def cross_action_round(speaker_agent: ActionLanguageAgent,
                       listener_agent: ActionLanguageAgent,
                       scene: List[Dict[str, str]],
                       event: ActionEvent) -> bool:
    """跨语言动作交流一轮"""
    utterance = speaker_agent.speak(scene, event)
    if not utterance:
        return False

    chosen_idx = listener_agent.listen(utterance, scene)
    success = (chosen_idx == event.target_obj_idx)

    speaker_agent.update(utterance, success)
    listener_agent.update(utterance, success)
    return success
