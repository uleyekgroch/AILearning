"""
Phase 24: 统一语言系统 (Unified Language System)

将 Phase 9-23 的所有符号系统整合到一个通信游戏中，
验证多种符号能否在单次通信中组合使用。
"""

import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from language_emergence import (
    EmergingLanguage, COLORS, SHAPES, SIZES, MATERIALS,
    AUDITORY_SYMBOLS, TACTILE_SYMBOLS,
    NEGATION_MARKERS, TENSE_MARKERS, CAUSAL_REASONING_MARKERS,
    PERSPECTIVE_MARKERS, TOOL_MARKERS, ABSTRACT_MARKERS,
    _symbol_category
)

# ============================================================
# 统一物体：融合视觉 + 听觉 + 触觉 + 功能
# ============================================================

VISUAL_FEATURES = {
    'color': list(COLORS),
    'shape': list(SHAPES),
    'size': list(SIZES),
    'material': list(MATERIALS),
}

AUDITORY_FEATURES = {
    'sound': ['loud', 'quiet', 'sharp', 'soft', 'buzz', 'click', 'hum'],
}

TACTILE_FEATURES = {
    'texture': ['rough', 'smooth', 'hard', 'soft_tactile', 'hot', 'cold'],
}

AFFORDANCES = ['reach', 'contain', 'cut', 'hit', 'support', 'measure']


@dataclass
class UnifiedObject:
    """融合多模态特征的物体"""
    visual: Dict[str, str]           # color, shape, size, material
    auditory: Dict[str, str]         # sound
    tactile: Dict[str, str]          # texture
    affordances: List[str]           # 功能
    causal_effect: Optional[str] = None  # 因果效果标记

    def to_visual_symbols(self) -> List[str]:
        return list(self.visual.values())

    def to_auditory_symbols(self) -> List[str]:
        return list(self.auditory.values())

    def to_tactile_symbols(self) -> List[str]:
        return list(self.tactile.values())

    def to_affordance_symbols(self) -> List[str]:
        return list(self.affordances)

    def to_all_symbols(self) -> List[str]:
        return (self.to_visual_symbols() + self.to_auditory_symbols() +
                self.to_tactile_symbols() + self.to_affordance_symbols())

    def visual_signature(self) -> frozenset:
        return frozenset(self.visual.items())


@dataclass
class UnifiedScene:
    """统一场景：包含多种歧义类型"""
    objects: List[UnifiedObject]
    target_idx: int
    ambiguity_types: Set[str]       # 'visual_ambiguous', 'subset', 'causal', 'confidence', 'crossmodal', 'tool'
    speaker_confidence: float = 0.8
    causal_chain: Optional[Tuple[str, str]] = None  # (cause_symbol, effect_description)

    @property
    def target(self) -> UnifiedObject:
        return self.objects[self.target_idx]


# ============================================================
# 场景生成器
# ============================================================

def _random_visual() -> Dict[str, str]:
    return {k: random.choice(v) for k, v in VISUAL_FEATURES.items()}

def _random_auditory() -> Dict[str, str]:
    return {k: random.choice(v) for k, v in AUDITORY_FEATURES.items()}

def _random_tactile() -> Dict[str, str]:
    return {k: random.choice(v) for k, v in TACTILE_FEATURES.items()}

def _random_affordances(n: int = 1) -> List[str]:
    return random.sample(AFFORDANCES, min(n, len(AFFORDANCES)))


def generate_unified_scenario(
    ambiguity_types: Optional[Set[str]] = None,
    num_objects: int = 3,
    speaker_confidence: float = 0.8,
) -> UnifiedScene:
    """
    生成统一场景。

    ambiguity_types 可包含：
      - 'visual_ambiguous': 目标与干扰物视觉相同
      - 'subset': 目标视觉特征是干扰物子集（需要否定）
      - 'crossmodal': 视觉模糊，需要听觉/触觉
      - 'causal': 存在因果关系
      - 'confidence': Speaker 置信度不同
      - 'tool': 目标需要功能描述（外观歧义+功能不同）
    """
    if ambiguity_types is None:
        ambiguity_types = {'visual_ambiguous', 'crossmodal'}

    objects = []
    target_visual = _random_visual()
    target_auditory = _random_auditory()
    target_tactile = _random_tactile()
    target_affordances = _random_affordances(2)

    target = UnifiedObject(
        visual=target_visual.copy(),
        auditory=target_auditory.copy(),
        tactile=target_tactile.copy(),
        affordances=target_affordances.copy(),
    )
    objects.append(target)

    for i in range(num_objects - 1):
        obj_visual = target_visual.copy()
        obj_auditory = _random_auditory()
        obj_tactile = _random_tactile()
        obj_affordances = _random_affordances(2)

        # 根据歧义类型调整干扰物
        if 'visual_ambiguous' in ambiguity_types and i == 0:
            # 第一个干扰物：视觉完全相同
            pass  # obj_visual 已经是 target 的副本

        elif 'subset' in ambiguity_types and i == 0:
            # 子集关系：干扰物有目标的所有特征 + 额外特征
            extra_color = random.choice([c for c in COLORS if c != target_visual['color']])
            obj_visual['extra_color'] = extra_color

        elif 'tool' in ambiguity_types and i == 0:
            # 工具场景：视觉相同但功能不同
            obj_affordances = [a for a in AFFORDANCES if a not in target_affordances][:2]

        else:
            # 默认：部分视觉差异
            change_key = random.choice(list(VISUAL_FEATURES.keys()))
            options = [v for v in VISUAL_FEATURES[change_key] if v != target_visual[change_key]]
            if options:
                obj_visual[change_key] = random.choice(options)

        obj = UnifiedObject(
            visual=obj_visual,
            auditory=obj_auditory,
            tactile=obj_tactile,
            affordances=obj_affordances,
        )
        objects.append(obj)

    causal_chain = None
    if 'causal' in ambiguity_types:
        causal_chain = ('hit', 'buzz')

    # 打乱物体顺序，避免 target 总在 index 0
    target_obj = objects[0]
    random.shuffle(objects)
    new_target_idx = objects.index(target_obj)

    return UnifiedScene(
        objects=objects,
        target_idx=new_target_idx,
        ambiguity_types=ambiguity_types,
        speaker_confidence=speaker_confidence,
        causal_chain=causal_chain,
    )


# ============================================================
# 统一 Speaker：选择最优描述策略
# ============================================================

class UnifiedSpeaker:
    """
    统一 Speaker：尝试所有策略，选择最短成功描述。

    策略优先级（从短到长）：
    1. 视觉唯一 → 纯视觉
    2. 视觉模糊 → 跨模态（听觉/触觉）
    3. 子集关系 → 否定
    4. 工具歧义 → 功能描述
    5. 因果关系 → "because" 连接
    6. 置信度 → 视角标记前缀
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.strategy_counts = {
            'visual': 0,
            'crossmodal': 0,
            'negation': 0,
            'tool': 0,
            'causal': 0,
            'confidence': 0,
        }

    def describe(self, scene: UnifiedScene) -> List[str]:
        target = scene.target
        candidates = scene.objects

        # 策略 1: 视觉唯一
        if self._is_visual_unique(target, candidates):
            utterance = target.to_visual_symbols()
            self.strategy_counts['visual'] += 1
            return utterance

        utterance = []

        # 策略 2: 跨模态（视觉模糊时）
        if 'crossmodal' in scene.ambiguity_types or 'visual_ambiguous' in scene.ambiguity_types:
            auditory = target.to_auditory_symbols()
            tactile = target.to_tactile_symbols()
            # 加入能区分的非视觉特征
            for sym in auditory + tactile:
                if self._symbol_discriminates(sym, target, candidates):
                    utterance.append(sym)
            if utterance:
                self.strategy_counts['crossmodal'] += 1

        # 策略 3: 否定（子集关系时）— 组合视觉+否定
        if 'subset' in scene.ambiguity_types:
            neg = self._try_negation(target, candidates)
            if neg:
                # 先描述目标视觉特征，再加否定排除子集干扰物
                if not utterance:
                    utterance = target.to_visual_symbols()
                utterance.extend(neg)
                self.strategy_counts['negation'] += 1

        # 策略 4: 功能描述（工具歧义时）
        if 'tool' in scene.ambiguity_types and not utterance:
            func = self._try_functional(target)
            if func:
                utterance.extend(func)
                self.strategy_counts['tool'] += 1

        # 如果还没找到描述，用全部视觉特征
        if not utterance:
            utterance = target.to_visual_symbols()

        # 策略 5: 因果关系
        if 'causal' in scene.ambiguity_types and scene.causal_chain:
            cause, effect = scene.causal_chain
            utterance.extend(['because', cause, effect])
            self.strategy_counts['causal'] += 1

        # 策略 6: 置信度标记
        if 'confidence' in scene.ambiguity_types:
            marker = self._choose_confidence_marker(scene.speaker_confidence)
            utterance.insert(0, marker)
            self.strategy_counts['confidence'] += 1

        return utterance

    def _is_visual_unique(self, target: UnifiedObject, candidates: List[UnifiedObject]) -> bool:
        sig = target.visual_signature()
        return sum(1 for c in candidates if c.visual_signature() == sig) == 1

    def _symbol_discriminates(self, sym: str, target: UnifiedObject, candidates: List[UnifiedObject]) -> bool:
        """检查某个符号是否能区分目标和其他物体"""
        target_syms = set(target.to_auditory_symbols() + target.to_tactile_symbols())
        if sym not in target_syms:
            return False
        for c in candidates:
            if c is target:
                continue
            other_syms = set(c.to_auditory_symbols() + c.to_tactile_symbols())
            if sym in other_syms:
                return False
        return True

    def _try_negation(self, target: UnifiedObject, candidates: List[UnifiedObject]) -> Optional[List[str]]:
        """尝试否定策略：找到目标没有但其他物体有的特征"""
        target_syms = set(target.to_all_symbols())
        for c in candidates:
            if c is target:
                continue
            other_syms = set(c.to_all_symbols())
            unique_to_other = other_syms - target_syms
            if unique_to_other:
                neg_feature = random.choice(list(unique_to_other))
                return ['not', neg_feature]
        return None

    def _try_functional(self, target: UnifiedObject) -> Optional[List[str]]:
        """尝试功能描述策略"""
        if target.affordances:
            return ['use', target.affordances[0], 'for'] + target.to_visual_symbols()[:2]
        return None

    def _choose_confidence_marker(self, confidence: float) -> str:
        if confidence > 0.8:
            return 'know'
        elif confidence > 0.4:
            return 'think'
        else:
            return 'believe'


# ============================================================
# 统一 Listener：综合所有解释策略
# ============================================================

class UnifiedListener:
    """
    统一 Listener：按优先级尝试所有解释方法。

    1. 视觉匹配
    2. 跨模态匹配
    3. 否定过滤
    4. 功能匹配
    5. 视角标记加分
    """

    CONFIDENCE_BOOST = 1.5   # "know" 加分
    CONFIDENCE_MILD = 0.5    # "think" 加分

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def interpret(self, utterance: List[str], objects: List[UnifiedObject]) -> int:
        if not objects:
            return 0

        scores = [0.0] * len(objects)
        confidence_marker = None

        # 检测视角标记
        perspective_markers = {'know', 'think', 'believe'}
        filtered_utterance = []
        for sym in utterance:
            if sym in perspective_markers:
                confidence_marker = sym
            else:
                filtered_utterance.append(sym)

        # 检测因果标记
        causal_markers = {'because', 'so', 'therefore'}
        has_causal = any(s in causal_markers for s in filtered_utterance)
        causal_parts = []
        if has_causal:
            because_idx = next(i for i, s in enumerate(filtered_utterance) if s in causal_markers)
            causal_parts = filtered_utterance[because_idx + 1:]
            filtered_utterance = filtered_utterance[:because_idx]

        # 检测否定
        negation_markers = {'not', 'no', "n't"}
        has_negation = any(s in negation_markers for s in filtered_utterance)
        neg_feature = None
        if has_negation:
            neg_idx = next(i for i, s in enumerate(filtered_utterance) if s in negation_markers)
            if neg_idx + 1 < len(filtered_utterance):
                neg_feature = filtered_utterance[neg_idx + 1]
            filtered_utterance = filtered_utterance[:neg_idx] + filtered_utterance[neg_idx + 2:]

        # 检测功能标记
        tool_markers = {'use', 'for'}
        has_tool = any(s in tool_markers for s in filtered_utterance)
        if has_tool:
            tool_parts = [s for s in filtered_utterance if s in tool_markers or s in AFFORDANCES]
            filtered_utterance = [s for s in filtered_utterance if s not in tool_markers and s not in AFFORDANCES]

        # 主匹配：视觉 + 跨模态
        utterance_set = set(filtered_utterance)
        for i, obj in enumerate(objects):
            # 视觉匹配
            visual_set = set(obj.to_visual_symbols())
            scores[i] += len(utterance_set & visual_set) * 2.0

            # 跨模态匹配
            auditory_set = set(obj.to_auditory_symbols())
            tactile_set = set(obj.to_tactile_symbols())
            scores[i] += len(utterance_set & auditory_set) * 2.0
            scores[i] += len(utterance_set & tactile_set) * 2.0

            # 功能匹配
            if has_tool:
                affordance_set = set(obj.affordances)
                scores[i] += len(set(tool_parts) & affordance_set) * 2.0

            # 否定过滤：减去有 neg_feature 的物体
            if has_negation and neg_feature:
                all_obj_syms = set(obj.to_all_symbols())
                if neg_feature in all_obj_syms:
                    scores[i] -= 10.0

            # 视角标记加分
            if confidence_marker == 'know':
                scores[i] += self.CONFIDENCE_BOOST
            elif confidence_marker == 'think':
                scores[i] += self.CONFIDENCE_MILD

        return scores.index(max(scores))


# ============================================================
# 统一通信游戏
# ============================================================

class UnifiedCommunicationGame:
    """统一通信游戏：整合所有符号系统"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = UnifiedSpeaker(self.language)
        self.listener = UnifiedListener(self.language)
        self.total_games = 0
        self.total_successes = 0
        self.strategy_usage = {
            'visual': 0,
            'crossmodal': 0,
            'negation': 0,
            'tool': 0,
            'causal': 0,
            'confidence': 0,
        }
        self.symbol_categories_used = set()

    def play_round(self, scene: UnifiedScene) -> bool:
        self.total_games += 1

        # Speaker 描述
        utterance = self.speaker.describe(scene)

        # 记录使用的符号类别
        for sym in utterance:
            cat = _symbol_category(sym)
            if cat:
                self.symbol_categories_used.add(cat)

        # Listener 解释
        chosen_idx = self.listener.interpret(utterance, scene.objects)
        success = chosen_idx == scene.target_idx

        # 记录
        self.language.record_usage(utterance, success)

        if success:
            self.total_successes += 1

        # 累计策略使用
        for k, v in self.speaker.strategy_counts.items():
            self.strategy_usage[k] = v

        return success

    def get_stats(self) -> Dict:
        lang_stats = self.language.get_stats()
        return {
            'total_games': self.total_games,
            'total_successes': self.total_successes,
            'success_rate': self.total_successes / max(1, self.total_games),
            'vocabulary_size': lang_stats['vocabulary_size'],
            'strategy_usage': self.strategy_usage.copy(),
            'symbol_categories_used': sorted(self.symbol_categories_used),
            'num_categories': len(self.symbol_categories_used),
        }


class BaselineVisualGame:
    """基线：只使用视觉描述"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.total_games = 0
        self.total_successes = 0

    def play_round(self, scene: UnifiedScene) -> bool:
        self.total_games += 1
        target = scene.target

        # 只用视觉
        utterance = target.to_visual_symbols()

        # Listener 简单匹配（平局随机选择）
        best_score = -1
        best_candidates = []
        utterance_set = set(utterance)
        for i, obj in enumerate(scene.objects):
            score = len(utterance_set & set(obj.to_visual_symbols()))
            if score > best_score:
                best_score = score
                best_candidates = [i]
            elif score == best_score:
                best_candidates.append(i)
        best_idx = random.choice(best_candidates)

        success = best_idx == scene.target_idx
        self.language.record_usage(utterance, success)
        if success:
            self.total_successes += 1
        return success

    def get_stats(self) -> Dict:
        return {
            'total_games': self.total_games,
            'total_successes': self.total_successes,
            'success_rate': self.total_successes / max(1, self.total_games),
            'vocabulary_size': self.language.get_stats()['vocabulary_size'],
        }


# ============================================================
# 测试
# ============================================================

def test_unified():
    """快速测试"""
    random.seed(42)
    game = UnifiedCommunicationGame()

    # 测试单模块场景
    print("=== 单模块：视觉模糊 + 跨模态 ===")
    for i in range(100):
        scene = generate_unified_scenario(
            ambiguity_types={'visual_ambiguous', 'crossmodal'},
            num_objects=3,
        )
        game.play_round(scene)

    stats = game.get_stats()
    print(f"成功率: {stats['success_rate']:.1%}")
    print(f"词汇量: {stats['vocabulary_size']}")
    print(f"策略使用: {stats['strategy_usage']}")
    print(f"符号类别: {stats['symbol_categories_used']}")
    print(f"类别数: {stats['num_categories']}")

    # 测试全模块场景
    print("\n=== 全模块 ===")
    game2 = UnifiedCommunicationGame()
    for i in range(200):
        scene = generate_unified_scenario(
            ambiguity_types={'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'},
            num_objects=4,
            speaker_confidence=random.uniform(0.2, 0.95),
        )
        game2.play_round(scene)

    stats2 = game2.get_stats()
    print(f"成功率: {stats2['success_rate']:.1%}")
    print(f"词汇量: {stats2['vocabulary_size']}")
    print(f"策略使用: {stats2['strategy_usage']}")
    print(f"符号类别: {stats2['symbol_categories_used']}")
    print(f"类别数: {stats2['num_categories']}")


if __name__ == '__main__':
    test_unified()
