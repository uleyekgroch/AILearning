"""
时态接地模块：时态符号从事件时间维度中涌现

核心思想：
时态标记（past, present, future）的意义来自事件的时间位置。
"past" = 已完成的动作，"future" = 将要发生的动作。
时态通过独立 dict 键注入，匹配逻辑天然支持。
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener,
    TENSE_MARKERS, ACTIONS, ACTION_EFFECTS,
    _symbol_category, generate_rich_scene,
)


@dataclass
class TemporalEvent:
    """带时态的动作事件"""
    action_type: str
    tense: str              # 'past', 'present', 'future'
    target_obj_idx: int
    direction: str
    effect_description: str

    def to_feature_dict(self) -> Dict[str, str]:
        return {
            'action': self.action_type,
            'action_effect': self.effect_description,
            'direction': self.direction,
            'tense': self.tense,
        }


TENSE_ACTIONS = {
    'past': {'push': 'pushed', 'pull': 'pulled', 'grab': 'grabbed',
             'drop': 'dropped', 'move': 'moved', 'go': 'went',
             'stop': 'stopped', 'turn': 'turned'},
    'present': {'push': 'pushing', 'pull': 'pulling', 'grab': 'grabbing',
                'drop': 'dropping', 'move': 'moving', 'go': 'going',
                'stop': 'stopping', 'turn': 'turning'},
    'future': {'push': 'will_push', 'pull': 'will_pull', 'grab': 'will_grab',
               'drop': 'will_drop', 'move': 'will_move', 'go': 'will_go',
               'stop': 'will_stop', 'turn': 'will_turn'},
}


def generate_temporal_scene(num_objects: int = 6,
                             complexity: str = 'medium') -> Tuple[List[Dict[str, str]], List[TemporalEvent]]:
    """生成带时态标注的场景"""
    base_scene = generate_rich_scene(complexity)[:num_objects]

    action_types = list(ACTIONS)
    directions = ['up', 'down', 'left', 'right']
    tenses = list(TENSE_MARKERS)

    temporal_events = []
    for i, obj_features in enumerate(base_scene):
        if np.random.random() < 0.6:
            action_type = np.random.choice(action_types)
            tense = np.random.choice(tenses)
            direction = np.random.choice(directions)
            effect = ACTION_EFFECTS.get(action_type, 'unknown')

            obj_features['action'] = action_type
            obj_features['action_effect'] = effect
            obj_features['direction'] = direction
            obj_features['tense'] = tense

            temporal_events.append(TemporalEvent(
                action_type=action_type,
                tense=tense,
                target_obj_idx=i,
                direction=direction,
                effect_description=effect,
            ))

    return base_scene, temporal_events


class TemporalCommunicationGame:
    """时态交流游戏：描述"做了什么/在做什么/将做什么"，listener 识别目标"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   temporal_events: List[TemporalEvent],
                   target_event_idx: int) -> bool:
        if target_event_idx >= len(temporal_events):
            return False

        target_event = temporal_events[target_event_idx]
        target_obj_idx = target_event.target_obj_idx

        if target_obj_idx >= len(scene_features):
            return False

        # speaker 描述（包含时态）
        combined = {**scene_features[target_obj_idx], **target_event.to_feature_dict()}
        utterance = self.speaker.describe(combined, scene_features, target_obj_idx)
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
            'tense': target_event.tense,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success


class TemporalLanguageAgent:
    """拥有独立时态语言的 agent"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)

    def speak(self, scene: List[Dict[str, str]], event: TemporalEvent) -> List[str]:
        if event.target_obj_idx < len(scene):
            combined = {**scene[event.target_obj_idx], **event.to_feature_dict()}
        else:
            combined = event.to_feature_dict()
        return self.speaker.describe(combined, scene, event.target_obj_idx)

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


def cross_temporal_round(speaker_agent: TemporalLanguageAgent,
                         listener_agent: TemporalLanguageAgent,
                         scene: List[Dict[str, str]],
                         event: TemporalEvent) -> bool:
    """跨语言时态交流一轮"""
    utterance = speaker_agent.speak(scene, event)
    if not utterance:
        return False

    chosen_idx = listener_agent.listen(utterance, scene)
    success = (chosen_idx == event.target_obj_idx)

    speaker_agent.update(utterance, success)
    listener_agent.update(utterance, success)
    return success
