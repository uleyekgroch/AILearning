"""
叙事与篇章模块：超越句子的语言结构

核心思想：
叙事 = 事件序列 + 因果/时序连接
当单句无法描述完整场景时，叙事结构从交流压力中涌现。

叙事结构：
- 事件 1: "red ball pushed"
- 事件 2: "blue ball moved"
- 连接: "then" (时序) 或 "because" (因果)

涌现条件：
1. 单句无法描述完整场景（信息量超过单句表达力）
2. 事件间有因果/时序依赖
3. Listener 需要理解叙事结构才能正确回应
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
    RELATIVE_MARKERS, NEGATION_MARKERS,
)

# 叙事连接词
NARRATIVE_MARKERS = {'then', 'because', 'so', 'but', 'and'}
TEMPORAL_MARKERS = {'then', 'before', 'after'}
CAUSAL_MARKERS = {'because', 'so', 'therefore'}


class Event:
    """叙事中的单个事件"""

    def __init__(self, subject: Dict[str, str], action: str,
                 object_features: Optional[Dict[str, str]] = None):
        """
        参数：
            subject: 执行者特征 {color: 'red', shape: 'ball'}
            action: 动作符号
            object_features: 被作用对象特征（可选）
        """
        self.subject = subject
        self.action = action
        self.object_features = object_features

    def to_symbols(self) -> List[str]:
        """转换为符号序列"""
        symbols = list(self.subject.values())
        symbols.append(self.action)
        if self.object_features:
            symbols.extend(self.object_features.values())
        return symbols

    def __repr__(self):
        subj = ', '.join(f'{k}:{v}' for k, v in self.subject.items())
        obj = ''
        if self.object_features:
            obj = ' → ' + ', '.join(f'{k}:{v}' for k, v in self.object_features.items())
        return f"Event({subj} {self.action}{obj})"


class Narrative:
    """叙事 = 事件序列 + 连接"""

    def __init__(self, events: List[Event],
                 connections: List[str]):
        """
        参数：
            events: 事件列表
            connections: 连接词列表（长度 = len(events) - 1）
        """
        self.events = events
        self.connections = connections

    def to_symbols(self) -> List[str]:
        """转换为完整的符号序列"""
        symbols = []
        for i, event in enumerate(self.events):
            symbols.extend(event.to_symbols())
            if i < len(self.connections):
                symbols.append(self.connections[i])
        return symbols

    def __repr__(self):
        parts = []
        for i, event in enumerate(self.events):
            parts.append(str(event))
            if i < len(self.connections):
                parts.append(self.connections[i])
        return ' '.join(parts)


class NarrativeSpeaker:
    """叙事 Speaker：将场景转化为叙事"""

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def narrate(self, events: List[Event],
                target_sequence: List[int]) -> List[str]:
        """
        将事件序列转化为符号叙事

        参数：
            events: 所有事件
            target_sequence: 目标事件顺序（索引列表）

        返回：
            符号序列，包含事件描述和连接词
        """
        if not events or not target_sequence:
            return []

        # 构建叙事
        symbols = []
        for i, idx in enumerate(target_sequence):
            if idx >= len(events):
                continue
            event = events[idx]
            # 添加事件符号
            event_symbols = event.to_symbols()
            symbols.extend(event_symbols)

            # 添加连接词（如果不是最后一个）
            if i < len(target_sequence) - 1:
                # 选择连接词
                connector = self._choose_connector(events, target_sequence, i)
                symbols.append(connector)

        return symbols

    def _choose_connector(self, events: List[Event],
                          sequence: List[int], position: int) -> str:
        """选择连接词"""
        # 优先使用已知的连接词
        known_connectors = [s for s in self.language.vocabulary
                           if s in NARRATIVE_MARKERS]
        if known_connectors:
            return np.random.choice(known_connectors)

        # 默认使用 "then"
        return 'then'


class NarrativeListener:
    """叙事 Listener：理解叙事结构"""

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def interpret(self, utterance: List[str],
                  events: List[Event]) -> Optional[List[int]]:
        """
        理解叙事，返回事件顺序

        参数：
            utterance: 符号序列
            events: 所有候选事件

        返回：
            事件索引列表（按叙事顺序）
        """
        if not utterance or not events:
            return None

        # 分割叙事为事件和连接
        event_segments = self._split_narrative(utterance)

        # 匹配每个事件段到候选事件
        matched_sequence = []
        for segment in event_segments:
            best_match = self._match_event(segment, events)
            if best_match is not None:
                matched_sequence.append(best_match)

        return matched_sequence if matched_sequence else None

    def _split_narrative(self, utterance: List[str]) -> List[List[str]]:
        """将叙事分割为事件段"""
        segments = []
        current_segment = []

        for sym in utterance:
            if sym in NARRATIVE_MARKERS:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(sym)

        if current_segment:
            segments.append(current_segment)

        return segments

    def _match_event(self, segment: List[str],
                     events: List[Event]) -> Optional[int]:
        """将符号段匹配到最相似的事件"""
        best_score = -1
        best_idx = None

        for i, event in enumerate(events):
            event_symbols = set(event.to_symbols())
            segment_set = set(segment)
            # 计算交集大小
            overlap = len(event_symbols & segment_set)
            if overlap > best_score:
                best_score = overlap
                best_idx = i

        return best_idx if best_score > 0 else None


class NarrativeGame:
    """叙事交流游戏"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = NarrativeSpeaker(self.language)
        self.listener = NarrativeListener(self.language)
        self.game_log = []
        self.narrative_used = 0
        self.narrative_success = 0

    def play_round(self, events: List[Event],
                   target_sequence: List[int]) -> bool:
        """
        进行一轮叙事游戏

        参数：
            events: 所有事件
            target_sequence: 目标事件顺序

        返回：
            是否成功
        """
        # Speaker 生成叙事
        utterance = self.speaker.narrate(events, target_sequence)
        if not utterance:
            return False

        # 检查是否使用了叙事连接词
        used_narrative = any(s in NARRATIVE_MARKERS for s in utterance)
        if used_narrative:
            self.narrative_used += 1

        # Listener 理解叙事
        interpreted = self.listener.interpret(utterance, events)
        success = (interpreted == target_sequence)

        if used_narrative and success:
            self.narrative_success += 1

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
            'target_sequence': target_sequence,
            'utterance': utterance,
            'interpreted': interpreted,
            'success': success,
            'used_narrative': used_narrative,
            'utterance_length': len(utterance),
        })
        return success

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['narrative_used'] = self.narrative_used
        stats['narrative_success'] = self.narrative_success
        stats['narrative_rate'] = self.narrative_used / max(1, len(self.game_log))
        return stats


def generate_simple_narrative() -> Tuple[List[Event], List[int]]:
    """
    生成简单叙事场景：2 个事件，有时序关系

    例如：
    - 事件 0: red ball pushed
    - 事件 1: blue ball moved
    - 目标顺序: [0, 1]（先推后移）
    """
    colors = list(COLORS)
    shapes = list(SHAPES)
    actions = ['push', 'pull', 'grab', 'drop', 'lift', 'throw']

    np.random.shuffle(colors)
    np.random.shuffle(shapes)
    np.random.shuffle(actions)

    event1 = Event(
        subject={'color': colors[0], 'shape': shapes[0]},
        action=actions[0],
    )
    event2 = Event(
        subject={'color': colors[1], 'shape': shapes[1]},
        action=actions[1],
    )

    events = [event1, event2]
    target_sequence = [0, 1]  # 固定顺序

    return events, target_sequence


def generate_causal_narrative() -> Tuple[List[Event], List[int]]:
    """
    生成因果叙事场景：2 个事件，有因果关系

    例如：
    - 事件 0: red ball pushed
    - 事件 1: blue ball moved (因为被推)
    - 因果关系: push → move
    """
    colors = list(COLORS)
    shapes = list(SHAPES)

    np.random.shuffle(colors)
    np.random.shuffle(shapes)

    # 因果动作对
    causal_pairs = [
        ('push', 'move'),
        ('pull', 'slide'),
        ('hit', 'break'),
        ('drop', 'fall'),
        ('lift', 'rise'),
    ]
    cause_action, effect_action = causal_pairs[np.random.randint(len(causal_pairs))]

    event1 = Event(
        subject={'color': colors[0], 'shape': shapes[0]},
        action=cause_action,
    )
    event2 = Event(
        subject={'color': colors[1], 'shape': shapes[1]},
        action=effect_action,
    )

    events = [event1, event2]
    target_sequence = [0, 1]  # 因果顺序

    return events, target_sequence


def generate_multi_event_narrative(num_events: int = 3) -> Tuple[List[Event], List[int]]:
    """
    生成多事件叙事场景

    例如（3 事件）：
    - 事件 0: red ball pushed
    - 事件 1: blue ball moved
    - 事件 2: green ball fallen
    - 目标顺序: [0, 1, 2]
    """
    colors = list(COLORS)
    shapes = list(SHAPES)
    actions = ['push', 'pull', 'grab', 'drop', 'lift', 'throw',
               'move', 'slide', 'fall', 'rise', 'break', 'stop']

    np.random.shuffle(colors)
    np.random.shuffle(shapes)
    np.random.shuffle(actions)

    events = []
    for i in range(num_events):
        event = Event(
            subject={'color': colors[i % len(colors)], 'shape': shapes[i % len(shapes)]},
            action=actions[i % len(actions)],
        )
        events.append(event)

    target_sequence = list(range(num_events))

    return events, target_sequence


def test_narrative():
    """测试叙事机制"""
    print("=== 叙事机制测试 ===")

    # 简单叙事
    events, sequence = generate_simple_narrative()
    print(f"\n事件:")
    for i, e in enumerate(events):
        print(f"  {i}: {e}")
    print(f"目标顺序: {sequence}")

    game = NarrativeGame()
    utterance = game.speaker.narrate(events, sequence)
    print(f"叙事: {utterance}")

    interpreted = game.listener.interpret(utterance, events)
    print(f"理解: {interpreted}")
    print(f"正确: {interpreted == sequence}")

    # 因果叙事
    events, sequence = generate_causal_narrative()
    print(f"\n因果叙事:")
    for i, e in enumerate(events):
        print(f"  {i}: {e}")

    utterance = game.speaker.narrate(events, sequence)
    print(f"叙事: {utterance}")

    interpreted = game.listener.interpret(utterance, events)
    print(f"理解: {interpreted}")
    print(f"正确: {interpreted == sequence}")


if __name__ == '__main__':
    test_narrative()
