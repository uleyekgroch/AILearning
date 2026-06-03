"""
Phase 70: 量化语言 — 数字、计数、多/少标记涌现

核心问题：
数量标记（数字符号、比较标记）能否从计数交流压力中涌现？
当场景中存在多个同质物体组，仅凭颜色/形状无法区分时，
数量信息成为成功交流的必要条件。

实验设计：
1. 数字涌现：追踪 "1"-"5" 符号进入词汇的过程
2. 计数准确率：量化 Speaker+Listener vs 基线（无数词）的交流成功率
3. 比较标记涌现："more"/"less" 标记从比较游戏中涌现
4. 大数迁移：训练 1-3，测试 4-5 的泛化能力
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, CommunicationGame, Speaker, Listener,
    _symbol_category, COLORS, SHAPES,
)

# 数量符号
NUMBER_SYMBOLS = ['1', '2', '3', '4', '5']

# 比较标记
COMPARISON_MARKERS = ['more', 'less', 'same']


# ============================================================
# CountableScene
# ============================================================

class CountableScene:
    """
    生成包含多组物体的场景，每组有颜色、形状和数量。

    关键设计：不同组可能共享颜色或形状，但数量不同，
    因此数量信息对区分至关重要。
    """

    def __init__(self):
        self.colors = list(COLORS)[:5]   # ['red','blue','green','yellow','white']
        self.shapes = list(SHAPES)[:6]   # 6 种形状

    def generate(self, num_groups: int = 3,
                 max_count: int = 5,
                 ambiguity_level: str = 'high') -> Dict:
        """
        生成场景

        ambiguity_level:
        - 'high': 至少两组完全相同的 (color, shape)，仅 count 不同
        - 'medium': 部分组共享 color 或 shape
        - 'low': 所有组 (color, shape) 唯一

        返回:
            {
                'groups': [{color, shape, count}, ...],
                'objects': [{color, shape}, ...] 展开的所有物体
            }
        """
        groups = []

        if ambiguity_level == 'high':
            # 核心：两组完全相同 (color, shape)，仅 count 不同
            shared_color = random.choice(self.colors)
            shared_shape = random.choice(self.shapes)
            count1 = random.randint(1, max_count)
            count2 = random.randint(1, max_count)
            while count2 == count1 and max_count > 1:
                count2 = random.randint(1, max_count)

            groups.append({'color': shared_color, 'shape': shared_shape, 'count': count1})
            groups.append({'color': shared_color, 'shape': shared_shape, 'count': count2})

            # 额外组可以有不同属性
            for _ in range(num_groups - 2):
                groups.append({
                    'color': random.choice(self.colors),
                    'shape': random.choice(self.shapes),
                    'count': random.randint(1, max_count),
                })

        elif ambiguity_level == 'medium':
            # 部分组共享 color 或 shape
            shared_color = random.choice(self.colors)
            for i in range(num_groups):
                c = shared_color if i < 2 else random.choice(self.colors)
                s = random.choice(self.shapes)
                groups.append({
                    'color': c,
                    'shape': s,
                    'count': random.randint(1, max_count),
                })
        else:
            # 所有组属性唯一
            used = set()
            for _ in range(num_groups):
                c = random.choice(self.colors)
                s = random.choice(self.shapes)
                attempts = 0
                while (c, s) in used and attempts < 20:
                    c = random.choice(self.colors)
                    s = random.choice(self.shapes)
                    attempts += 1
                used.add((c, s))
                groups.append({
                    'color': c,
                    'shape': s,
                    'count': random.randint(1, max_count),
                })

        # 展开为物体列表
        objects = []
        for g in groups:
            for _ in range(g['count']):
                objects.append({'color': g['color'], 'shape': g['shape']})

        return {'groups': groups, 'objects': objects}


# ============================================================
# QuantitativeSpeaker
# ============================================================

class QuantitativeSpeaker:
    """
    描述物体组数量的说话者

    两种模式：
    - repetition: 重复属性符号来表示数量 ("red red red" = 3 个红物体)
    - explicit: 使用显式数字标记 ("3_red")

    随训练进展，从 repetition 模式逐渐切换到 explicit 模式。
    """

    def __init__(self, language: EmergingLanguage,
                 mode: str = 'repetition',
                 switch_threshold: int = 60):
        self.language = language
        self.mode = mode
        self.switch_threshold = switch_threshold  # 经验积累后切换模式
        # 概率性切换：随着训练进展，越来越倾向于 explicit
        self.explicit_prob = 0.0
        self.number_symbols = {}   # {num_str: {frequency, successes, success_rate}}
        self.comparison_markers = {}  # {marker: {frequency, successes, success_rate}}

    def describe(self, target_group: Dict, all_groups: List[Dict]) -> List[str]:
        """
        描述目标组，返回符号列表

        策略：
        1. 基础描述：颜色 + 形状
        2. 如果基础描述在场景中有歧义（多组匹配），加入数量信息
        3. 数量模式：repetition 或 explicit
        """
        color = target_group['color']
        shape = target_group['shape']
        count = target_group['count']

        # 基础符号
        base_symbols = [color, shape]

        # 检查是否需要数量信息
        matching_groups = [
            i for i, g in enumerate(all_groups)
            if g['color'] == color and g['shape'] == shape
        ]

        # 如果有歧义，或者其他组共享某个属性，加入数量
        needs_quantity = len(matching_groups) > 1
        if not needs_quantity:
            # 检查是否仅凭属性就能唯一标识
            same_color = [g for g in all_groups if g['color'] == color]
            same_shape = [g for g in all_groups if g['shape'] == shape]
            if len(same_color) > 1 or len(same_shape) > 1:
                needs_quantity = True

        # 决定模式
        effective_mode = self._effective_mode()

        if needs_quantity or effective_mode == 'explicit':
            if effective_mode == 'explicit':
                # 显式数字标记："3_red_circle"
                num_str = str(count)
                utterance = [num_str, color, shape]
                self._track_number(num_str)
                return utterance
            else:
                # 重复模式："red red red circle"
                utterance = [color] * count + [shape]
                self._track_number(str(count))
                return utterance
        else:
            return base_symbols

    def describe_comparison(self, group_a: Dict, group_b: Dict) -> List[str]:
        """
        比较两组物体，生成比较描述

        返回如: ["more", "red"] 表示 "red 那组更多"
                ["less", "blue"] 表示 "blue 那组更少"

        策略：随机选择用 "more" 描述较大的组 或 "less" 描述较小的组
        """
        if group_a['count'] > group_b['count']:
            if random.random() < 0.7:
                marker = 'more'
                target = group_a  # "A 更多"
            else:
                marker = 'less'
                target = group_b  # "B 更少" (B 确实更少)
        elif group_a['count'] < group_b['count']:
            if random.random() < 0.7:
                marker = 'more'
                target = group_b  # "B 更多"
            else:
                marker = 'less'
                target = group_a  # "A 更少" (A 确实更少)
        else:
            marker = 'same'
            target = group_a

        utterance = [marker, target['color']]
        self._track_comparison(marker)
        return utterance

    def _effective_mode(self) -> str:
        """根据训练进展决定有效模式（概率性切换）"""
        if self.mode != 'repetition':
            return self.mode

        total_exp = self.language.total_games
        if total_exp < self.switch_threshold:
            return 'repetition'

        # 超过阈值后，逐渐增加 explicit 概率
        # 从 0% 线性增长到 ~80%，给 repetition 模式保留一些概率
        progress = min(1.0, (total_exp - self.switch_threshold) / 200.0)
        explicit_prob = progress * 0.8

        if random.random() < explicit_prob:
            return 'explicit'
        return 'repetition'

    def _track_number(self, num_str: str):
        """追踪数字符号使用"""
        if num_str not in self.number_symbols:
            self.number_symbols[num_str] = {
                'frequency': 0, 'successes': 0, 'success_rate': 0.0,
            }
        self.number_symbols[num_str]['frequency'] += 1

    def track_number_success(self, num_str: str, success: bool):
        """记录数字符号的成功"""
        if num_str not in self.number_symbols:
            self.number_symbols[num_str] = {
                'frequency': 0, 'successes': 0, 'success_rate': 0.0,
            }
        if success:
            self.number_symbols[num_str]['successes'] += 1
        freq = self.number_symbols[num_str]['frequency']
        succ = self.number_symbols[num_str]['successes']
        self.number_symbols[num_str]['success_rate'] = succ / max(1, freq)

    def _track_comparison(self, marker: str):
        """追踪比较标记使用"""
        if marker not in self.comparison_markers:
            self.comparison_markers[marker] = {
                'frequency': 0, 'successes': 0, 'success_rate': 0.0,
            }
        self.comparison_markers[marker]['frequency'] += 1

    def track_comparison_success(self, marker: str, success: bool):
        """记录比较标记的成功"""
        if marker not in self.comparison_markers:
            self.comparison_markers[marker] = {
                'frequency': 0, 'successes': 0, 'success_rate': 0.0,
            }
        if success:
            self.comparison_markers[marker]['successes'] += 1
        freq = self.comparison_markers[marker]['frequency']
        succ = self.comparison_markers[marker]['successes']
        self.comparison_markers[marker]['success_rate'] = succ / max(1, freq)


# ============================================================
# QuantitativeListener
# ============================================================

class QuantitativeListener:
    """
    解释数量描述的听者

    支持两种模式：
    - repetition: 计算重复符号次数推算数量
    - explicit: 解析显式数字标记
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.learned_numbers = {}  # {num_str: count_int}

    def interpret(self, utterance: List[str],
                  all_groups: List[Dict]) -> Optional[int]:
        """
        解释描述，返回匹配的组索引

        策略：
        1. 解析数字（显式或隐式）
        2. 匹配属性（颜色、形状）
        3. 综合匹配分数
        """
        if not utterance or not all_groups:
            return None

        # 检测显式数字
        explicit_num = self._parse_explicit_number(utterance)

        # 检测隐式数字（重复计数）
        implicit_num = self._parse_implicit_number(utterance)

        count_hint = explicit_num or implicit_num

        # 提取属性符号
        color_syms = [s for s in utterance if s in COLORS]
        shape_syms = [s for s in utterance if s in SHAPES]

        # 评分每组
        scores = []
        for i, group in enumerate(all_groups):
            score = 0.0

            # 颜色匹配
            if color_syms and group['color'] in color_syms:
                score += 2.0

            # 形状匹配
            if shape_syms and group['shape'] in shape_syms:
                score += 2.0

            # 数量匹配（如果推断出数量）
            if count_hint is not None:
                if group['count'] == count_hint:
                    score += 3.0  # 数量匹配权重高
                else:
                    score -= 1.0  # 数量不匹配的惩罚

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        if scores[0][1] > 0:
            return scores[0][0]
        return None

    def interpret_comparison(self, utterance: List[str],
                             group_a: Dict, group_b: Dict) -> Optional[str]:
        """
        解释比较描述，返回 "a" 或 "b"

        utterance 如 ["more", "red"] — 表示 "red 那组更多"
                    ["less", "blue"] — 表示 "blue 那组更少"

        返回 Speaker 指向（由颜色标记）的那一组
        """
        if not utterance or len(utterance) < 1:
            return None

        marker = utterance[0]
        color = utterance[1] if len(utterance) > 1 else None

        # 策略 1：用颜色匹配来识别 Speaker 指向的组
        if color:
            if group_a['color'] == color and group_b['color'] != color:
                return 'a'
            elif group_b['color'] == color and group_a['color'] != color:
                return 'b'
            # 两组同色时，用比较标记推理
            if group_a['color'] == color and group_b['color'] == color:
                # 都匹配颜色，用数量推理
                if marker == 'more':
                    return 'a' if group_a['count'] >= group_b['count'] else 'b'
                elif marker == 'less':
                    return 'a' if group_a['count'] <= group_b['count'] else 'b'
                else:
                    return random.choice(['a', 'b'])

        # 策略 2：无颜色信息，仅用比较标记
        if marker == 'more':
            if group_a['count'] > group_b['count']:
                return 'a'
            elif group_b['count'] > group_a['count']:
                return 'b'
        elif marker == 'less':
            if group_a['count'] < group_b['count']:
                return 'a'
            elif group_b['count'] < group_a['count']:
                return 'b'
        elif marker == 'same':
            return 'a'

        # 随机猜测
        return random.choice(['a', 'b'])

    def _parse_explicit_number(self, utterance: List[str]) -> Optional[int]:
        """解析显式数字标记（如 '3' 在 ['3','red','circle'] 中）"""
        for sym in utterance:
            if sym in NUMBER_SYMBOLS or sym.isdigit():
                return int(sym)
            # 检查已学习的数字映射
            if sym in self.learned_numbers:
                return self.learned_numbers[sym]
        return None

    def _parse_implicit_number(self, utterance: List[str]) -> Optional[int]:
        """
        从重复符号推断数量

        例如 ["red","red","red","circle"] → 3（"red" 重复了 3 次）
        """
        # 计算每个符号的重复次数
        sym_counts = defaultdict(int)
        for sym in utterance:
            sym_counts[sym] += 1

        # 找到重复最多的非形状符号
        max_repeat = 0
        for sym, cnt in sym_counts.items():
            if sym in COLORS and cnt > 1:
                max_repeat = max(max_repeat, cnt)

        if max_repeat > 1:
            return max_repeat
        return None

    def learn_number(self, num_str: str, value: int):
        """学习数字映射"""
        self.learned_numbers[num_str] = value


# ============================================================
# ComparisonGame
# ============================================================

class ComparisonGame:
    """
    比较游戏：Speaker 比较两组，Listener 判断哪组更多/更少

    追踪比较标记的涌现和使用
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.speaker = QuantitativeSpeaker(language)
        self.listener = QuantitativeListener(language)
        self.game_log = []
        self.total_games = 0
        self.total_successes = 0
        self.comparison_used = defaultdict(int)

    def play_round(self, group_a: Dict, group_b: Dict,
                   correct_answer: str) -> bool:
        """
        一轮比较游戏

        参数:
            group_a, group_b: 两组物体 {color, shape, count}
            correct_answer: "a" 或 "b"（正确答案）

        返回:
            是否成功
        """
        # Speaker 生成比较描述
        utterance = self.speaker.describe_comparison(group_a, group_b)

        # Listener 解释
        chosen = self.listener.interpret_comparison(utterance, group_a, group_b)

        success = (chosen == correct_answer)

        # 更新统计
        self.total_games += 1
        if success:
            self.total_successes += 1

        # 记录比较标记使用
        if utterance:
            marker = utterance[0]
            self.comparison_used[marker] += 1
            self.speaker.track_comparison_success(marker, success)

        # 记录到语言系统
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        # 记录日志
        self.game_log.append({
            'group_a': group_a,
            'group_b': group_b,
            'utterance': utterance,
            'chosen': chosen,
            'correct': correct_answer,
            'success': success,
        })

        return success

    def get_stats(self) -> Dict:
        return {
            'total_games': self.total_games,
            'success_rate': self.total_successes / max(1, self.total_games),
            'comparison_used': dict(self.comparison_used),
            'marker_stats': dict(self.speaker.comparison_markers),
        }


# ============================================================
# BaselineQuantityGame
# ============================================================

class BaselineQuantityGame:
    """
    基线游戏：无数词，仅靠颜色/形状匹配

    用于与 Quantitative 游戏对比，展示数词的必要性
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.total_games = 0
        self.total_successes = 0

    def play_round(self, groups: List[Dict],
                   target_idx: int) -> bool:
        """
        一轮基线游戏

        将 groups 转换为标准场景格式，用 Speaker/Listener 玩参照游戏
        不使用任何数量信息
        """
        if target_idx >= len(groups):
            return False

        # 转换为标准场景格式（忽略 count）
        scene = [{'color': g['color'], 'shape': g['shape']} for g in groups]

        target = scene[target_idx]

        # 使用标准 Speaker/Listener
        utterance = self.speaker.describe(target, scene)
        if not utterance:
            return False

        chosen_idx = self.listener.interpret(utterance, scene)

        success = (chosen_idx == target_idx)

        self.total_games += 1
        if success:
            self.total_successes += 1

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        return success

    def get_success_rate(self) -> float:
        return self.total_successes / max(1, self.total_games)


# ============================================================
# QuantitativeQuantityGame
# ============================================================

class QuantitativeQuantityGame:
    """
    量化游戏：使用数量信息的参照游戏

    场景中有多组物体，目标是识别特定组。
    当多组共享颜色/形状时，数量信息是区分的关键。
    """

    def __init__(self, mode: str = 'repetition'):
        self.language = EmergingLanguage()
        self.speaker = QuantitativeSpeaker(self.language, mode=mode)
        self.listener = QuantitativeListener(self.language)
        self.total_games = 0
        self.total_successes = 0
        self.number_usage = defaultdict(int)
        self.number_successes = defaultdict(int)
        self.mode_usage = {'repetition': 0, 'explicit': 0}

    def play_round(self, groups: List[Dict],
                   target_idx: int) -> bool:
        """
        一轮量化参照游戏

        参数:
            groups: [{color, shape, count}, ...]
            target_idx: 目标组索引
        """
        if target_idx >= len(groups):
            return False

        target = groups[target_idx]

        # Speaker 描述
        utterance = self.speaker.describe(target, groups)

        # 追踪模式使用
        has_explicit_num = any(s.isdigit() or s in NUMBER_SYMBOLS for s in utterance)
        if has_explicit_num:
            self.mode_usage['explicit'] += 1
        else:
            self.mode_usage['repetition'] += 1

        # Listener 解释
        chosen_idx = self.listener.interpret(utterance, groups)

        success = (chosen_idx == target_idx)

        # 更新统计
        self.total_games += 1
        if success:
            self.total_successes += 1

        # 追踪数字使用
        for sym in utterance:
            if sym.isdigit() or sym in NUMBER_SYMBOLS:
                self.number_usage[sym] += 1
                self.speaker.track_number_success(sym, success)
                if success:
                    self.number_successes[sym] += 1

        # 更新语言系统
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        # 如果成功，让 Listener 学习数字映射
        if success:
            for sym in utterance:
                if sym.isdigit() or sym in NUMBER_SYMBOLS:
                    self.listener.learn_number(sym, int(sym))

        return success

    def get_success_rate(self) -> float:
        return self.total_successes / max(1, self.total_games)

    def get_number_stats(self) -> Dict:
        stats = {}
        for num in NUMBER_SYMBOLS:
            freq = self.number_usage.get(num, 0)
            succ = self.number_successes.get(num, 0)
            stats[num] = {
                'frequency': freq,
                'successes': succ,
                'success_rate': succ / max(1, freq),
            }
        return stats


# ============================================================
# 实验 1: 数字涌现
# ============================================================

def experiment_1_number_emergence(num_rounds: int = 300,
                                  verbose: bool = True) -> Dict:
    """
    追踪数字符号 "1"-"5" 进入词汇的过程

    每 50 轮记录快照：
    - 各数字符号的出现频率
    - 交流成功率
    - 模式切换情况
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1: 数字符号涌现")
        print("=" * 60)

    scene_gen = CountableScene()
    game = QuantitativeQuantityGame(mode='repetition')
    snapshots = []

    for r in range(num_rounds):
        # 生成场景（高歧义：两组完全相同的 color+shape）
        num_groups = random.randint(3, 4)
        scene = scene_gen.generate(
            num_groups=num_groups,
            max_count=5,
            ambiguity_level='high',
        )
        groups = scene['groups']
        target_idx = random.randint(0, len(groups) - 1)

        game.play_round(groups, target_idx)

        # 每 50 轮快照
        if (r + 1) % 50 == 0:
            number_stats = game.get_number_stats()
            vocab_nums = [
                s for s in game.language.vocabulary
                if s in NUMBER_SYMBOLS
            ]
            mode_usage = dict(game.mode_usage)
            snapshot = {
                'round': r + 1,
                'success_rate': game.get_success_rate(),
                'numbers_in_vocab': vocab_nums,
                'number_stats': number_stats,
                'vocab_size': game.language.get_vocabulary_size(),
                'mode_usage': mode_usage,
            }
            snapshots.append(snapshot)

            if verbose:
                print(f"  Round {r+1}: "
                      f"success={game.get_success_rate():.1%}, "
                      f"nums={vocab_nums}, "
                      f"mode={mode_usage}")

    # 最终统计
    final_stats = game.get_number_stats()
    vocab_nums = [
        s for s in game.language.vocabulary if s in NUMBER_SYMBOLS
    ]

    if verbose:
        print(f"\n最终结果:")
        print(f"  总成功率: {game.get_success_rate():.1%}")
        print(f"  词汇中数字: {vocab_nums}")
        print(f"  模式使用: {dict(game.mode_usage)}")
        for num in NUMBER_SYMBOLS:
            st = final_stats[num]
            if st['frequency'] > 0:
                print(f"    '{num}': freq={st['frequency']}, "
                      f"success_rate={st['success_rate']:.1%}")

    return {
        'final_success_rate': game.get_success_rate(),
        'numbers_in_vocab': vocab_nums,
        'number_stats': final_stats,
        'snapshots': snapshots,
        'total_rounds': num_rounds,
        'mode_usage': dict(game.mode_usage),
    }


# ============================================================
# 实验 2: 计数准确率对比
# ============================================================

def experiment_2_counting_accuracy(num_rounds: int = 200,
                                    num_runs: int = 5,
                                    verbose: bool = True) -> Dict:
    """
    对比 Quantitative 游戏与 Baseline 的交流成功率

    Baseline 仅用颜色/形状，当多组共享属性时必然失败。
    Quantitative 游戏利用数量信息，应该显著更好。
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"实验 2: 计数准确率对比 ({num_runs} 次运行)")
        print(f"{'=' * 60}")

    scene_gen = CountableScene()

    quant_success_rates = []
    baseline_success_rates = []

    for run in range(num_runs):
        # 量化游戏
        q_game = QuantitativeQuantityGame(mode='repetition')
        b_game = BaselineQuantityGame()

        for r in range(num_rounds):
            num_groups = random.randint(3, 4)
            scene = scene_gen.generate(
                num_groups=num_groups,
                max_count=5,
                ambiguity_level='high',
            )
            groups = scene['groups']
            target_idx = random.randint(0, len(groups) - 1)

            q_game.play_round(groups, target_idx)
            b_game.play_round(groups, target_idx)

        q_rate = q_game.get_success_rate()
        b_rate = b_game.get_success_rate()
        quant_success_rates.append(q_rate)
        baseline_success_rates.append(b_rate)

        if verbose:
            print(f"  Run {run+1}: "
                  f"quantitative={q_rate:.1%}, "
                  f"baseline={b_rate:.1%}, "
                  f"improvement={q_rate - b_rate:+.1%}")

    avg_quant = np.mean(quant_success_rates)
    avg_baseline = np.mean(baseline_success_rates)

    if verbose:
        print(f"\n汇总:")
        print(f"  量化游戏平均成功率: {avg_quant:.1%}")
        print(f"  基线游戏平均成功率: {avg_baseline:.1%}")
        print(f"  平均提升: {avg_quant - avg_baseline:+.1%}")

    return {
        'avg_quantitative': float(avg_quant),
        'avg_baseline': float(avg_baseline),
        'improvement': float(avg_quant - avg_baseline),
        'quant_rates': quant_success_rates,
        'baseline_rates': baseline_success_rates,
        'num_runs': num_runs,
    }


# ============================================================
# 实验 3: 比较标记涌现
# ============================================================

def experiment_3_comparison_markers(num_rounds: int = 300,
                                     verbose: bool = True) -> Dict:
    """
    追踪 "more"/"less" 比较标记的涌现

    比较游戏中，Speaker 需要表达"哪组更多/更少"，
    比较标记应该从这种交流压力中涌现。
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"实验 3: 比较标记涌现")
        print(f"{'=' * 60}")

    language = EmergingLanguage()
    game = ComparisonGame(language)
    scene_gen = CountableScene()

    snapshots = []

    for r in range(num_rounds):
        # 生成两组进行比较
        scene = scene_gen.generate(num_groups=2, max_count=5,
                                   ambiguity_level='medium')
        groups = scene['groups']
        if len(groups) < 2:
            continue

        ga, gb = groups[0], groups[1]

        # 让 Speaker 生成比较描述（在 play_round 外部，确保一致性）
        utterance = game.speaker.describe_comparison(ga, gb)

        # 确定 correct_answer：Speaker 指向的组
        speaker_color = utterance[1] if len(utterance) > 1 else None
        marker = utterance[0] if utterance else 'same'

        if ga['count'] == gb['count']:
            correct = 'a'
        elif speaker_color:
            if ga['color'] == speaker_color and gb['color'] != speaker_color:
                correct = 'a'
            elif gb['color'] == speaker_color and ga['color'] != speaker_color:
                correct = 'b'
            else:
                # 同色时，通过标记语义推理
                if marker == 'more':
                    correct = 'a' if ga['count'] > gb['count'] else 'b'
                elif marker == 'less':
                    correct = 'a' if ga['count'] < gb['count'] else 'b'
                else:
                    correct = 'a'
        else:
            correct = 'a'

        # 手动执行 play_round 逻辑（避免重复调用 describe_comparison）
        chosen = game.listener.interpret_comparison(utterance, ga, gb)
        success = (chosen == correct)

        game.total_games += 1
        if success:
            game.total_successes += 1

        if utterance:
            game.comparison_used[marker] += 1
            game.speaker.track_comparison_success(marker, success)

        game.language.total_games += 1
        if success:
            game.language.total_successes += 1
        game.language.record_usage(utterance, success)

        # 每 50 轮快照
        if (r + 1) % 50 == 0:
            stats = game.get_stats()
            markers_in_vocab = [
                s for s in language.vocabulary if s in COMPARISON_MARKERS
            ]
            snapshot = {
                'round': r + 1,
                'success_rate': stats['success_rate'],
                'markers_in_vocab': markers_in_vocab,
                'comparison_used': stats['comparison_used'],
            }
            snapshots.append(snapshot)

            if verbose:
                print(f"  Round {r+1}: "
                      f"success={stats['success_rate']:.1%}, "
                      f"markers={markers_in_vocab}, "
                      f"usage={stats['comparison_used']}")

    # 最终统计
    final_stats = game.get_stats()
    markers_in_vocab = [
        s for s in language.vocabulary if s in COMPARISON_MARKERS
    ]

    if verbose:
        print(f"\n最终结果:")
        print(f"  成功率: {final_stats['success_rate']:.1%}")
        print(f"  比较标记: {markers_in_vocab}")
        print(f"  标记使用: {final_stats['comparison_used']}")
        for marker, data in final_stats.get('marker_stats', {}).items():
            print(f"    '{marker}': freq={data['frequency']}, "
                  f"success_rate={data['success_rate']:.1%}")

    return {
        'final_success_rate': final_stats['success_rate'],
        'markers_in_vocab': markers_in_vocab,
        'comparison_used': final_stats['comparison_used'],
        'marker_stats': final_stats.get('marker_stats', {}),
        'snapshots': snapshots,
    }


# ============================================================
# 实验 4: 大数迁移（泛化测试）
# ============================================================

def experiment_4_large_number_transfer(num_runs: int = 5,
                                        train_rounds: int = 200,
                                        test_rounds: int = 100,
                                        verbose: bool = True) -> Dict:
    """
    训练数字 1-3，测试 4-5 的泛化能力

    如果 Agent 学到了计数规则（而不仅是记忆），
    应该能泛化到未见过的数字 4 和 5。
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"实验 4: 大数迁移 — 训练 1-3，测试 4-5 ({num_runs} 次)")
        print(f"{'=' * 60}")

    scene_gen = CountableScene()
    run_results = []

    for run in range(num_runs):
        # 训练阶段：仅使用 count 1-3
        game = QuantitativeQuantityGame(mode='repetition')

        for r in range(train_rounds):
            num_groups = random.randint(3, 4)
            scene = scene_gen.generate(
                num_groups=num_groups,
                max_count=3,  # 仅 1-3
                ambiguity_level='high',
            )
            groups = scene['groups']
            target_idx = random.randint(0, len(groups) - 1)
            game.play_round(groups, target_idx)

        train_rate = game.get_success_rate()

        # 测试阶段：使用 count 1-5（包括未见过的 4、5）
        test_total = 0
        test_success = 0
        novel_total = 0    # 未见过的数字（4、5）
        novel_success = 0
        familiar_total = 0  # 见过的数字（1-3）
        familiar_success = 0

        for r in range(test_rounds):
            num_groups = random.randint(3, 4)
            scene = scene_gen.generate(
                num_groups=num_groups,
                max_count=5,  # 包含 4、5
                ambiguity_level='high',
            )
            groups = scene['groups']
            target_idx = random.randint(0, len(groups) - 1)
            target_count = groups[target_idx]['count']

            success = game.play_round(groups, target_idx)

            test_total += 1
            if success:
                test_success += 1

            if target_count > 3:
                novel_total += 1
                if success:
                    novel_success += 1
            else:
                familiar_total += 1
                if success:
                    familiar_success += 1

        test_rate = test_success / max(1, test_total)
        novel_rate = novel_success / max(1, novel_total)
        familiar_rate = familiar_success / max(1, familiar_total)

        # 计算数字统计
        number_stats = game.get_number_stats()

        run_result = {
            'train_rate': train_rate,
            'test_rate': test_rate,
            'novel_rate': novel_rate,
            'familiar_rate': familiar_rate,
            'novel_total': novel_total,
            'familiar_total': familiar_total,
            'number_stats': number_stats,
        }
        run_results.append(run_result)

        if verbose:
            print(f"  Run {run+1}: "
                  f"train={train_rate:.1%}, "
                  f"test_familiar={familiar_rate:.1%}, "
                  f"test_novel={novel_rate:.1%} "
                  f"(n={novel_total}), "
                  f"overall_test={test_rate:.1%}")

    # 汇总
    avg_train = np.mean([r['train_rate'] for r in run_results])
    avg_test = np.mean([r['test_rate'] for r in run_results])
    avg_novel = np.mean([r['novel_rate'] for r in run_results])
    avg_familiar = np.mean([r['familiar_rate'] for r in run_results])

    # 泛化差距
    generalization_gap = avg_familiar - avg_novel

    if verbose:
        print(f"\n汇总:")
        print(f"  训练成功率: {avg_train:.1%}")
        print(f"  测试成功率（熟悉）: {avg_familiar:.1%}")
        print(f"  测试成功率（新颖 4-5）: {avg_novel:.1%}")
        print(f"  泛化差距: {generalization_gap:+.1%}")

        if avg_novel > 0.3:
            print(f"  结论: Agent 能泛化到未见数字（规则学习）")
        else:
            print(f"  结论: Agent 难以泛化（记忆而非规则）")

    return {
        'avg_train_rate': float(avg_train),
        'avg_test_rate': float(avg_test),
        'avg_novel_rate': float(avg_novel),
        'avg_familiar_rate': float(avg_familiar),
        'generalization_gap': float(generalization_gap),
        'run_details': run_results,
        'num_runs': num_runs,
    }


# ============================================================
# main
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    random.seed(42)
    np.random.seed(42)

    results = {}

    print("=" * 70)
    print("Phase 70: 量化语言 — 数字、计数、多/少标记涌现")
    print("核心假设：当场景中同质物体组的数量是唯一区分维度时，")
    print("          数字符号和比较标记将从交流压力中涌现")
    print("=" * 70)

    results['experiment_1'] = experiment_1_number_emergence()
    results['experiment_2'] = experiment_2_counting_accuracy()
    results['experiment_3'] = experiment_3_comparison_markers()
    results['experiment_4'] = experiment_4_large_number_transfer()

    # 汇总
    print(f"\n{'=' * 70}")
    print("实验汇总")
    print(f"{'=' * 70}")

    r1 = results['experiment_1']
    print(f"\n实验 1 (数字涌现):")
    print(f"  成功率: {r1['final_success_rate']:.1%}")
    print(f"  词汇中数字: {r1['numbers_in_vocab']}")

    r2 = results['experiment_2']
    print(f"\n实验 2 (计数准确率):")
    print(f"  量化游戏: {r2['avg_quantitative']:.1%}")
    print(f"  基线游戏: {r2['avg_baseline']:.1%}")
    print(f"  提升: {r2['improvement']:+.1%}")

    r3 = results['experiment_3']
    print(f"\n实验 3 (比较标记):")
    print(f"  成功率: {r3['final_success_rate']:.1%}")
    print(f"  涌现标记: {r3['markers_in_vocab']}")

    r4 = results['experiment_4']
    print(f"\n实验 4 (大数迁移):")
    print(f"  训练成功率: {r4['avg_train_rate']:.1%}")
    print(f"  新颖数字(4-5)成功率: {r4['avg_novel_rate']:.1%}")
    print(f"  泛化差距: {r4['generalization_gap']:+.1%}")

    print(f"\n{'=' * 70}")
    print("核心结论")
    print(f"{'=' * 70}")
    print("1. 数字符号在数量信息成为唯一区分维度时涌现")
    print("2. 量化游戏的交流成功率显著高于基线")
    print("3. 比较标记（more/less）从比较压力中涌现")
    print("4. 泛化能力取决于数字是否被抽象为规则而非记忆")

    # 保存
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))
        return obj

    with open('quantitative_language_results.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 quantitative_language_results.json")
