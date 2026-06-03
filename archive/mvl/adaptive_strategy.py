"""
Phase 25: 自适应策略选择 (Adaptive Strategy Selection)

核心改进：让语言系统本身学习。
- Speaker 的策略选择基于历史成功率自适应
- Listener 的评分权重基于通信反馈调整
- 歧义类型-策略偏好矩阵从经验中学习

解决 Phase 24 的根本问题：统计被跟踪但从未驱动行为。
"""

import random
import math
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from language_emergence import EmergingLanguage, _symbol_category
from grounding_unified_language import (
    UnifiedObject, UnifiedScene, UnifiedSpeaker, UnifiedListener,
    UnifiedCommunicationGame, generate_unified_scenario, AFFORDANCES,
)


def _softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """数值稳定的 softmax"""
    if not values:
        return []
    scaled = [v / max(temperature, 0.01) for v in values]
    max_v = max(scaled)
    exps = [math.exp(v - max_v) for v in scaled]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


# ============================================================
# 经验记录
# ============================================================

@dataclass
class CommunicationExperience:
    """一次通信的经验记录"""
    ambiguity_types: Set[str]
    strategy_used: str
    utterance: List[str]
    success: bool
    round_num: int
    chosen_idx: int
    correct_idx: int


# ============================================================
# 自适应 Speaker
# ============================================================

class AdaptiveUnifiedSpeaker(UnifiedSpeaker):
    """
    自适应 Speaker：策略选择基于历史成功率。

    与固定优先级的 UnifiedSpeaker 不同：
    - 维护每个策略的经验权重
    - 用 softmax 采样决定策略顺序（早期探索，后期利用）
    - 按歧义类型跟踪策略成功率
    """

    def __init__(self, language: EmergingLanguage):
        super().__init__(language)
        self.total_rounds = 0

        # 策略权重：初始均匀
        self.strategy_weights = {
            'visual': 1.0,
            'crossmodal': 1.0,
            'negation': 1.0,
            'tool': 1.0,
            'causal': 1.0,
            'confidence': 1.0,
        }

        # 歧义类型 → 策略 → 成功/总计
        self.ambiguity_strategy_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

        # 最近一轮使用的策略（用于反馈）
        self._last_strategy = None
        self._contributing_strategies: Set[str] = set()
        self._failed_strategies: Set[str] = set()

    def describe(self, scene: UnifiedScene) -> List[str]:
        """
        重写 describe：组合所有可用策略，基于权重决定组合顺序。

        与固定系统的关键区别：
        - 固定系统：硬编码优先级（视觉→跨模态→否定→工具→因果→置信度）
        - 自适应系统：基于权重排序，但尝试所有可用策略并组合结果

        功劳分配：记录哪些策略真正贡献了描述。
        """
        self.total_rounds += 1
        target = scene.target
        candidates = scene.objects
        self._contributing_strategies = set()
        self._failed_strategies = set()

        # 视觉唯一 → 直接用视觉
        if self._is_visual_unique(target, candidates):
            utterance = target.to_visual_symbols()
            self._last_strategy = 'visual'
            self._contributing_strategies = {'visual'}
            self.strategy_counts['visual'] += 1
        else:
            # 获取所有可用策略并按权重排序
            all_strategies = self._get_all_strategies(scene)
            ordered = self._select_strategy_order(all_strategies)

            # 按排序顺序尝试所有策略，组合结果
            utterance = []

            for strategy in ordered:
                result = self._try_strategy(strategy, target, candidates, scene)
                if result is not None:
                    utterance.extend(result)
                    self._contributing_strategies.add(strategy)
                else:
                    self._failed_strategies.add(strategy)

            if not utterance:
                utterance = target.to_visual_symbols()

            self._last_strategy = '+'.join(sorted(self._contributing_strategies))
            for s in self._contributing_strategies:
                self.strategy_counts[s] += 1

        # 因果
        if 'causal' in scene.ambiguity_types and scene.causal_chain:
            cause, effect = scene.causal_chain
            utterance.extend(['because', cause, effect])
            self.strategy_counts['causal'] += 1

        # 置信度
        if 'confidence' in scene.ambiguity_types:
            marker = self._choose_confidence_marker(scene.speaker_confidence)
            utterance.insert(0, marker)
            self.strategy_counts['confidence'] += 1

        return utterance

    def _get_all_strategies(self, scene: UnifiedScene) -> List[str]:
        """获取所有可用策略（按歧义类型）"""
        available = []

        if ('crossmodal' in scene.ambiguity_types or
                'visual_ambiguous' in scene.ambiguity_types):
            available.append('crossmodal')

        if 'subset' in scene.ambiguity_types:
            available.append('negation')

        if 'tool' in scene.ambiguity_types:
            available.append('tool')

        # 视觉总是可用
        available.append('visual')

        return available

    def _select_strategy_order(self, available: List[str]) -> List[str]:
        """
        基于权重选择策略顺序。

        温度参数随经验递减：
        - 早期（T=2.0）：高探索，均匀采样
        - 后期（T=0.5）：高利用，偏向高权重策略
        """
        temp = max(0.5, 2.0 - self.total_rounds / 500)
        weights = [self.strategy_weights.get(s, 1.0) for s in available]
        probs = _softmax(weights, temp)

        # 采样排列（不放回）
        result = []
        remaining = list(range(len(available)))
        remaining_probs = list(probs)

        while remaining:
            total = sum(remaining_probs)
            if total == 0:
                normed = [1.0 / len(remaining)] * len(remaining)
            else:
                normed = [p / total for p in remaining_probs]

            idx = random.choices(remaining, weights=normed, k=1)[0]
            result.append(available[idx])
            pos = remaining.index(idx)
            remaining.pop(pos)
            remaining_probs.pop(pos)

        return result

    def _try_strategy(self, strategy: str, target: UnifiedObject,
                      candidates: List[UnifiedObject],
                      scene: UnifiedScene) -> Optional[List[str]]:
        """
        尝试策略，返回描述片段或 None。

        视觉策略只在能唯一标识时返回结果。
        其他策略返回能区分目标的符号。
        """
        if strategy == 'visual':
            if self._is_visual_unique(target, candidates):
                return target.to_visual_symbols()
            return None

        elif strategy == 'crossmodal':
            utterance = []
            auditory = target.to_auditory_symbols()
            tactile = target.to_tactile_symbols()
            for sym in auditory + tactile:
                if self._symbol_discriminates(sym, target, candidates):
                    utterance.append(sym)
            return utterance if utterance else None

        elif strategy == 'negation':
            neg = self._try_negation(target, candidates)
            return neg  # 返回 ['not', feature] 或 None

        elif strategy == 'tool':
            return self._try_functional(target)

        return None

    def update_weights(self, contributing_strategies: Set[str], success: bool,
                       ambiguity_types: Optional[Set[str]] = None,
                       failed_strategies: Optional[Set[str]] = None):
        """
        基于通信结果更新策略权重。

        精确功劳分配：
        - 成功 → 强化所有贡献策略
        - 失败 → 弱化所有贡献策略
        - 失败的策略（返回 None）→ 总是弱化（与成功/失败无关）
        """
        lr = 0.1
        reward = 1.0 if success else -0.5

        # 更新贡献策略
        for strategy in contributing_strategies:
            self.strategy_weights[strategy] += lr * reward
            self.strategy_weights[strategy] = max(0.1, self.strategy_weights[strategy])

        # 惩罚失败策略（尝试了但无法区分目标）
        if failed_strategies:
            penalty = -0.3  # 比失败的贡献策略惩罚轻，但明确降低
            for strategy in failed_strategies:
                self.strategy_weights[strategy] += lr * penalty
                self.strategy_weights[strategy] = max(0.1, self.strategy_weights[strategy])

        # 更新歧义类型-策略统计
        if ambiguity_types:
            for amb_type in ambiguity_types:
                if amb_type not in self.ambiguity_strategy_stats:
                    self.ambiguity_strategy_stats[amb_type] = {}
                for strategy in contributing_strategies:
                    if strategy not in self.ambiguity_strategy_stats[amb_type]:
                        self.ambiguity_strategy_stats[amb_type][strategy] = {
                            'successes': 0, 'total': 0
                        }
                    self.ambiguity_strategy_stats[amb_type][strategy]['total'] += 1
                    if success:
                        self.ambiguity_strategy_stats[amb_type][strategy]['successes'] += 1

    def get_strategy_preferences(self) -> Dict[str, Dict[str, float]]:
        """获取每个歧义类型的策略偏好（成功率）"""
        result = {}
        for amb_type, strategies in self.ambiguity_strategy_stats.items():
            result[amb_type] = {}
            for strategy, stats in strategies.items():
                if stats['total'] > 0:
                    result[amb_type][strategy] = stats['successes'] / stats['total']
        return result


# ============================================================
# 自适应 Listener
# ============================================================

class AdaptiveUnifiedListener(UnifiedListener):
    """
    自适应 Listener：评分权重基于通信反馈调整。

    与固定权重的 UnifiedListener 不同：
    - 评分系数是可学习参数
    - 失败时调整权重让正确答案得分更高
    """

    def __init__(self, language: EmergingLanguage):
        super().__init__(language)

        # 可学习的评分权重（初始值与父类相同）
        self.scoring_weights = {
            'visual_match': 2.0,
            'auditory_match': 2.0,
            'tactile_match': 2.0,
            'affordance_match': 2.0,
            'negation_penalty': -10.0,
            'confidence_know': 1.5,
            'confidence_think': 0.5,
        }

        # 学习率
        self.lr = 0.05
        self.total_updates = 0

    def interpret(self, utterance: List[str], objects: List[UnifiedObject]) -> int:
        """重写 interpret：使用可学习权重"""
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
        if has_causal:
            because_idx = next(i for i, s in enumerate(filtered_utterance) if s in causal_markers)
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
        tool_parts = []
        if has_tool:
            tool_parts = [s for s in filtered_utterance if s in tool_markers or s in AFFORDANCES]
            filtered_utterance = [s for s in filtered_utterance if s not in tool_markers and s not in AFFORDANCES]

        # 主匹配：使用可学习权重
        utterance_set = set(filtered_utterance)
        for i, obj in enumerate(objects):
            # 视觉匹配
            visual_set = set(obj.to_visual_symbols())
            scores[i] += len(utterance_set & visual_set) * self.scoring_weights['visual_match']

            # 跨模态匹配
            auditory_set = set(obj.to_auditory_symbols())
            tactile_set = set(obj.to_tactile_symbols())
            scores[i] += len(utterance_set & auditory_set) * self.scoring_weights['auditory_match']
            scores[i] += len(utterance_set & tactile_set) * self.scoring_weights['tactile_match']

            # 功能匹配
            if has_tool:
                affordance_set = set(obj.affordances)
                scores[i] += len(set(tool_parts) & affordance_set) * self.scoring_weights['affordance_match']

            # 否定过滤
            if has_negation and neg_feature:
                all_obj_syms = set(obj.to_all_symbols())
                if neg_feature in all_obj_syms:
                    scores[i] -= abs(self.scoring_weights['negation_penalty'])

            # 视角标记加分
            if confidence_marker == 'know':
                scores[i] += self.scoring_weights['confidence_know']
            elif confidence_marker == 'think':
                scores[i] += self.scoring_weights['confidence_think']

        return scores.index(max(scores))

    def update_weights(self, utterance: List[str], objects: List[UnifiedObject],
                       chosen_idx: int, correct_idx: int, success: bool):
        """
        基于通信结果调整评分权重。

        失败时：分析正确答案和错误答案的得分差异，调整权重缩小差距。
        """
        self.total_updates += 1

        if success or not objects:
            return

        # 计算当前各物体得分
        scores = self._compute_scores(utterance, objects)

        # 正确答案的得分
        correct_score = scores[correct_idx]
        chosen_score = scores[chosen_idx]

        # 如果正确答案得分已经更高（随机平局选错了），不调整
        if correct_score >= chosen_score:
            return

        # 分析哪些特征能提高正确答案的得分
        correct_obj = objects[correct_idx]
        chosen_obj = objects[chosen_idx]

        utterance_set = set(utterance) - {'not', 'no', 'because', 'so', 'know', 'think', 'believe', 'use', 'for'}

        # 正确答案独有匹配
        correct_visual = set(correct_obj.to_visual_symbols()) & utterance_set
        chosen_visual = set(chosen_obj.to_visual_symbols()) & utterance_set

        correct_auditory = set(correct_obj.to_auditory_symbols()) & utterance_set
        chosen_auditory = set(chosen_obj.to_auditory_symbols()) & utterance_set

        # 如果正确答案在某模态有更多匹配，提高该模态权重
        if len(correct_visual) > len(chosen_visual):
            self.scoring_weights['visual_match'] += self.lr
        if len(correct_auditory) > len(chosen_auditory):
            self.scoring_weights['auditory_match'] += self.lr

        # 检查触觉
        correct_tactile = set(correct_obj.to_tactile_symbols()) & utterance_set
        chosen_tactile = set(chosen_obj.to_tactile_symbols()) & utterance_set
        if len(correct_tactile) > len(chosen_tactile):
            self.scoring_weights['tactile_match'] += self.lr

        # 权重下限
        for key in self.scoring_weights:
            if 'penalty' not in key:
                self.scoring_weights[key] = max(0.1, self.scoring_weights[key])

    def _compute_scores(self, utterance: List[str],
                        objects: List[UnifiedObject]) -> List[float]:
        """计算各物体得分（与 interpret 相同逻辑）"""
        scores = [0.0] * len(objects)
        confidence_marker = None

        perspective_markers = {'know', 'think', 'believe'}
        filtered = []
        for sym in utterance:
            if sym in perspective_markers:
                confidence_marker = sym
            else:
                filtered.append(sym)

        causal_markers = {'because', 'so', 'therefore'}
        if any(s in causal_markers for s in filtered):
            idx = next(i for i, s in enumerate(filtered) if s in causal_markers)
            filtered = filtered[:idx]

        negation_markers = {'not', 'no', "n't"}
        has_negation = any(s in negation_markers for s in filtered)
        neg_feature = None
        if has_negation:
            neg_idx = next(i for i, s in enumerate(filtered) if s in negation_markers)
            if neg_idx + 1 < len(filtered):
                neg_feature = filtered[neg_idx + 1]
            filtered = filtered[:neg_idx] + filtered[neg_idx + 2:]

        tool_markers = {'use', 'for'}
        has_tool = any(s in tool_markers for s in filtered)
        tool_parts = []
        if has_tool:
            tool_parts = [s for s in filtered if s in tool_markers or s in AFFORDANCES]
            filtered = [s for s in filtered if s not in tool_markers and s not in AFFORDANCES]

        utterance_set = set(filtered)
        for i, obj in enumerate(objects):
            scores[i] += len(utterance_set & set(obj.to_visual_symbols())) * self.scoring_weights['visual_match']
            scores[i] += len(utterance_set & set(obj.to_auditory_symbols())) * self.scoring_weights['auditory_match']
            scores[i] += len(utterance_set & set(obj.to_tactile_symbols())) * self.scoring_weights['tactile_match']

            if has_tool:
                scores[i] += len(set(tool_parts) & set(obj.affordances)) * self.scoring_weights['affordance_match']

            if has_negation and neg_feature:
                if neg_feature in set(obj.to_all_symbols()):
                    scores[i] -= abs(self.scoring_weights['negation_penalty'])

            if confidence_marker == 'know':
                scores[i] += self.scoring_weights['confidence_know']
            elif confidence_marker == 'think':
                scores[i] += self.scoring_weights['confidence_think']

        return scores


# ============================================================
# 自适应通信游戏
# ============================================================

class AdaptiveCommunicationGame:
    """
    自适应通信游戏：集成反馈闭环。

    与 UnifiedCommunicationGame 的关键区别：
    - play_round() 包含反馈到 speaker 和 listener
    - 记录经验用于分析
    - 统计自适应指标
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = AdaptiveUnifiedSpeaker(self.language)
        self.listener = AdaptiveUnifiedListener(self.language)
        self.total_games = 0
        self.total_successes = 0
        self.strategy_usage = {
            'visual': 0, 'crossmodal': 0, 'negation': 0,
            'tool': 0, 'causal': 0, 'confidence': 0,
        }
        self.symbol_categories_used = set()

        # 经验记录
        self.experiences: List[CommunicationExperience] = []

        # 滑动窗口成功率（用于收敛分析）
        self.window_size = 50
        self.recent_outcomes: List[bool] = []

    def play_round(self, scene: UnifiedScene) -> bool:
        """执行一轮通信，包含反馈闭环"""
        self.total_games += 1

        # Speaker 描述
        utterance = self.speaker.describe(scene)
        contributing = self.speaker._contributing_strategies.copy()
        failed = self.speaker._failed_strategies.copy()

        # 记录符号类别
        for sym in utterance:
            cat = _symbol_category(sym)
            if cat:
                self.symbol_categories_used.add(cat)

        # Listener 解释
        chosen_idx = self.listener.interpret(utterance, scene.objects)
        success = chosen_idx == scene.target_idx

        # 记录语言统计
        self.language.record_usage(utterance, success)

        if success:
            self.total_successes += 1

        # === 反馈闭环 ===
        # 1. 更新 Speaker 策略权重（精确功劳分配 + 失败惩罚）
        self.speaker.update_weights(
            contributing, success, scene.ambiguity_types, failed
        )

        # 2. 更新 Listener 评分权重
        self.listener.update_weights(
            utterance, scene.objects, chosen_idx, scene.target_idx, success
        )

        # 记录经验
        self.experiences.append(CommunicationExperience(
            ambiguity_types=scene.ambiguity_types.copy(),
            strategy_used='+'.join(sorted(contributing)),
            utterance=utterance,
            success=success,
            round_num=self.total_games,
            chosen_idx=chosen_idx,
            correct_idx=scene.target_idx,
        ))

        # 更新滑动窗口
        self.recent_outcomes.append(success)
        if len(self.recent_outcomes) > self.window_size:
            self.recent_outcomes.pop(0)

        # 累计策略使用
        for k, v in self.speaker.strategy_counts.items():
            self.strategy_usage[k] = v

        return success

    def get_recent_success_rate(self) -> float:
        """最近 N 轮的成功率"""
        if not self.recent_outcomes:
            return 0.0
        return sum(self.recent_outcomes) / len(self.recent_outcomes)

    def get_stats(self) -> Dict:
        lang_stats = self.language.get_stats()
        return {
            'total_games': self.total_games,
            'total_successes': self.total_successes,
            'success_rate': self.total_successes / max(1, self.total_games),
            'recent_success_rate': self.get_recent_success_rate(),
            'vocabulary_size': lang_stats['vocabulary_size'],
            'strategy_usage': self.strategy_usage.copy(),
            'strategy_weights': self.speaker.strategy_weights.copy(),
            'listener_weights': self.listener.scoring_weights.copy(),
            'symbol_categories_used': sorted(self.symbol_categories_used),
            'num_categories': len(self.symbol_categories_used),
            'strategy_preferences': self.speaker.get_strategy_preferences(),
        }


# ============================================================
# 测试
# ============================================================

def test_adaptive():
    """快速测试自适应系统"""
    random.seed(42)
    game = AdaptiveCommunicationGame()

    print("=== 自适应策略系统测试 ===")
    print(f"初始策略权重: {game.speaker.strategy_weights}")
    print(f"初始 Listener 权重: {game.listener.scoring_weights}")
    print()

    # 混合场景测试
    for i in range(2000):
        amb_types = random.choice([
            {'visual_ambiguous', 'crossmodal'},
            {'subset'},
            {'tool'},
            {'causal'},
            {'confidence'},
            {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'},
        ])
        scene = generate_unified_scenario(
            ambiguity_types=amb_types,
            num_objects=3,
            speaker_confidence=random.uniform(0.2, 0.95),
        )
        game.play_round(scene)

        if (i + 1) % 200 == 0:
            stats = game.get_stats()
            print(f"轮次 {i+1}: 成功率={stats['success_rate']:.1%}, "
                  f"近期成功率={stats['recent_success_rate']:.1%}")

    # 最终统计
    stats = game.get_stats()
    print(f"\n=== 最终结果 ===")
    print(f"总成功率: {stats['success_rate']:.1%}")
    print(f"近期成功率: {stats['recent_success_rate']:.1%}")
    print(f"策略权重: {stats['strategy_weights']}")
    print(f"Listener 权重: {stats['listener_weights']}")
    print(f"策略使用: {stats['strategy_usage']}")
    print(f"符号类别: {stats['num_categories']}")

    # 策略偏好
    prefs = stats['strategy_preferences']
    if prefs:
        print(f"\n歧义类型-策略偏好:")
        for amb_type, strategies in prefs.items():
            best = max(strategies, key=strategies.get) if strategies else 'N/A'
            print(f"  {amb_type}: 最优={best}, 详情={strategies}")


if __name__ == '__main__':
    test_adaptive()
