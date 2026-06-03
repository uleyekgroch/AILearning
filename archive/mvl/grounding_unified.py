"""
统一接地场景生成：组合静态属性、动作、情感、因果

核心思想：
真实场景中，符号接地是多维度同时发生的。
一个物体既有颜色（静态），又被推了（动作），
agent 此时感到好奇（情感），因为推导致了移动（因果）。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener, _symbol_category,
    generate_rich_scene, ACTIONS, ACTION_EFFECTS, EMOTIONS, CAUSAL_MARKERS,
)
from grounding_actions import ActionEvent, generate_action_scene
from grounding_emotions import EmotionMapper, EmotionState
from grounding_causal import CausalRule, CausalGroundingModule, CausalExpression


def generate_grounded_scene(
    num_objects: int = 6,
    complexity: str = 'medium',
    include_static: bool = True,
    include_actions: bool = True,
    include_emotions: bool = True,
    include_causal: bool = True,
) -> Tuple[List[Dict[str, str]], Dict]:
    """
    生成统一的接地场景

    返回：
        enriched_scene: 包含所有接地维度的场景
        metadata: 动作事件、情感状态、因果规则等元数据
    """
    # 基础场景（静态属性）
    if include_static:
        base_scene = generate_rich_scene(complexity)[:num_objects]
    else:
        base_scene = [{'obj_id': str(i)} for i in range(num_objects)]

    metadata = {}
    enriched = [dict(obj) for obj in base_scene]

    # 动作接地
    if include_actions:
        action_types = list(ACTIONS)
        directions = ['up', 'down', 'left', 'right']
        action_events = []
        for i, obj in enumerate(enriched):
            if np.random.random() < 0.5:
                action = np.random.choice(action_types)
                direction = np.random.choice(directions)
                effect = ACTION_EFFECTS.get(action, 'unknown')
                obj['action'] = action
                obj['action_effect'] = effect
                obj['direction'] = direction
                action_events.append(ActionEvent(
                    action_type=action,
                    target_obj_idx=i,
                    direction=direction,
                    effect_description=effect,
                ))
        metadata['action_events'] = action_events

    # 情感接地
    if include_emotions:
        emotion = np.random.choice(list(EMOTIONS))
        for obj in enriched:
            obj['emotion'] = emotion
        metadata['emotion'] = emotion

    # 因果接地
    if include_causal:
        action_types = list(ACTIONS)
        effects = list(ACTION_EFFECTS.values())
        causal_module = CausalGroundingModule()
        causal_rules = []
        for i, obj in enumerate(enriched):
            if np.random.random() < 0.4:
                cause = np.random.choice(action_types)
                effect = np.random.choice(effects)
                if np.random.random() < 0.7:
                    causal_module.observe(cause, effect)
                else:
                    causal_module.observe_non_occurrence(cause, effect)

                key = (cause, effect)
                if key in causal_module.rules:
                    rule = causal_module.rules[key]
                    expr = CausalExpression.from_rule(rule)
                    obj['causal_cause'] = cause
                    obj['causal_effect'] = effect
                    obj['causal_marker'] = expr.marker
                    causal_rules.append(rule)
        metadata['causal_rules'] = causal_rules
        metadata['causal_module'] = causal_module

    return enriched, metadata


class UnifiedCommunicationGame:
    """统一交流游戏：同时包含所有接地维度"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.game_log = []

    def play_round(self, scene: List[Dict[str, str]],
                   target_idx: int) -> bool:
        if target_idx >= len(scene) or not scene:
            return False

        target = scene[target_idx]
        utterance = self.speaker.describe(target, scene)
        if not utterance:
            return False

        chosen_idx = self.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        if len(utterance) > 1:
            self.language.multi_symbol_games += 1
        if len(utterance) >= 3:
            self.language.tri_symbol_games += 1

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
        })
        return success

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['games_played'] = len(self.game_log)
        return stats


def run_unified_experiment(num_rounds: int = 500,
                           num_objects: int = 6,
                           complexity: str = 'medium') -> Dict:
    """运行统一接地实验"""
    game = UnifiedCommunicationGame()

    successes = []
    for round_idx in range(num_rounds):
        scene, metadata = generate_grounded_scene(
            num_objects=num_objects,
            complexity=complexity,
            include_static=True,
            include_actions=True,
            include_emotions=True,
            include_causal=True,
        )
        target_idx = np.random.randint(len(scene))
        success = game.play_round(scene, target_idx)
        successes.append(success)

    # 统计各符号类别的使用
    vocab = game.language.vocabulary
    category_counts = {}
    for sym in vocab:
        cat = _symbol_category(sym)
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + vocab[sym]['frequency']

    return {
        'success_rate': sum(successes) / len(successes),
        'vocab_size': game.language.get_vocabulary_size(),
        'combination_rate': game.language.get_combination_rate(),
        'tri_symbol_rate': game.language.get_tri_symbol_rate(),
        'ngram_patterns': len(game.language.ngram_patterns),
        'category_usage': category_counts,
        'stats': game.get_stats(),
    }
