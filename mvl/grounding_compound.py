"""
复合特征接地模块：让否定从子集关系中涌现

核心思想：
当物体 A 的特征是物体 B 的子集时，
正向描述永远无法区分它们——只有否定可以。

例如：
- 物体 A: {colors: {red, blue}, shape: {circle}}
- 物体 B: {colors: {red}, shape: {circle}}

B 的任何正向描述（"red", "circle"）都匹配 A。
只有 "not blue" 能区分 B 与 A。

这与儿童语言学习一致：
否定不是"更复杂的语法"，而是处理特征子集关系的必要工具。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, Speaker, Listener,
    NEGATION_MARKERS, RELATIVE_MARKERS, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
)

# 新增特征维度
TEXTURES = {'smooth', 'rough', 'bumpy'}


def generate_subset_scene(num_base: int = 4,
                         num_subsets: int = 3,
                         target_is_subset: bool = True) -> Tuple[List[Dict[str, Set[str]]], int]:
    """
    生成包含子集关系的场景

    设计：
    - num_base 个"基础"物体：每个有 1-2 个值/维度
    - num_subsets 个"子集"物体：特征值是某个基础物体的子集
    - target_is_subset: 目标是否是子集物体

    当目标是子集物体时，正向描述无法区分它与父物体。
    只有否定（"not X"，X 是父物体有但子集没有的值）才能区分。
    """
    all_dims = {
        'color': list(COLORS),
        'shape': list(SHAPES),
        'size': list(SIZES),
        'material': list(MATERIALS),
        'texture': list(TEXTURES),
    }
    dim_names = list(all_dims.keys())

    scene = []
    seen = set()

    # 选择全局共享值（所有物体都有）
    global_shared = {}
    for dim in dim_names[:2]:  # color, shape
        global_shared[dim] = {np.random.choice(all_dims[dim])}

    # 生成基础物体
    for i in range(num_base):
        obj = {}
        for dim in dim_names:
            if dim in global_shared:
                obj[dim] = set(global_shared[dim])
            else:
                # 每个基础物体随机选择 1-2 个值
                n_values = np.random.choice([1, 2])
                values = set(np.random.choice(all_dims[dim], size=min(n_values, len(all_dims[dim])), replace=False))
                obj[dim] = values
        key = tuple(sorted((k, frozenset(v)) for k, v in obj.items()))
        if key not in seen:
            seen.add(key)
            scene.append(obj)

    # 生成子集物体（特征值是某个基础物体的子集）
    for i in range(num_subsets):
        parent = scene[i % len(scene)]
        obj = {}
        for dim in dim_names:
            parent_values = parent[dim]
            if len(parent_values) <= 1:
                obj[dim] = set(parent_values)  # 无法再缩小
            else:
                # 随机移除 1 个值，形成真子集
                remove_val = np.random.choice(list(parent_values))
                obj[dim] = parent_values - {remove_val}
        key = tuple(sorted((k, frozenset(v)) for k, v in obj.items()))
        if key not in seen:
            seen.add(key)
            scene.append(obj)

    # 打乱场景
    np.random.shuffle(scene)

    # 选择目标
    if target_is_subset:
        # 目标是子集物体（最后添加的那些）
        target_idx = len(scene) - 1
    else:
        target_idx = np.random.randint(0, len(scene))

    return scene, target_idx


def generate_simple_subset_scene() -> Tuple[List[Dict[str, Set[str]]], int]:
    """
    生成最简单的子集场景：2 个物体，目标是子集

    物体 A: {color: {red, blue}, shape: {circle}}
    物体 B: {color: {red}, shape: {circle}}  ← 目标

    正向描述 "red circle" 匹配两个物体。
    否定 "not blue" 只匹配物体 B。
    """
    scene = [
        {'color': {'red', 'blue'}, 'shape': {'circle'}, 'size': {'big'}},
        {'color': {'red'}, 'shape': {'circle'}, 'size': {'big'}},  # 目标（子集）
    ]
    return scene, 1  # 目标是索引 1


class CompoundSpeaker:
    """
    复合特征 Speaker

    与标准 Speaker 的区别：
    - 特征值是 Set[str] 而非 str
    - 匹配逻辑：utterance 值必须是物体值的子集
    - 否定逻辑："not X" 表示物体不包含 X
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def describe(self, target_features: Dict[str, Set[str]],
                 scene_features: List[Dict[str, Set[str]]],
                 target_idx: int = -1,
                 max_len: int = 0) -> List[str]:
        """
        描述目标物体

        参数：
            max_len: 最大描述长度（0=无限制）
                     当设置时，只返回长度 <= max_len 的描述
                     如果没有符合条件的描述，返回最短的

        策略：
        1. 正向描述：列出目标的每个值
        2. 如果正向描述有歧义（其他物体也包含这些值），尝试否定
        3. 选择最短的唯一描述
        """
        # 收集目标的所有值
        target_values = set()
        for values in target_features.values():
            target_values.update(values)

        if not target_values:
            return []

        candidates = []

        # 策略 1: 正向描述（逐个值尝试）
        for sym in target_values:
            count = sum(1 for obj in scene_features
                       if any(sym in vals for vals in obj.values()))
            if count == 1:
                candidates.append([sym])

        # 策略 2: 正向组合
        best_pos = self._find_best_positive(target_values, scene_features)
        if best_pos:
            candidates.append(best_pos)

        # 策略 3: 否定
        if target_idx >= 0:
            neg = self._try_negation(target_features, scene_features, target_idx)
            if neg:
                candidates.append(neg)

        # 策略 4: 相对从句
        if target_idx >= 0:
            clause = self._try_relative_clause(target_features, scene_features, target_idx)
            if clause:
                candidates.append(clause)

        if not candidates:
            return []

        # 按长度排序
        candidates.sort(key=len)

        # 应用 max_len 约束
        if max_len > 0:
            within_budget = [c for c in candidates if len(c) <= max_len]
            if within_budget:
                return within_budget[0]
            # 没有符合预算的，返回最短的（截断）
            return candidates[0][:max_len]

        return candidates[0]

        # 兜底：返回第一个值
        return [list(target_values)[0]]

    def _find_best_positive(self, target_values: Set[str],
                           scene_features: List[Dict[str, Set[str]]]) -> Optional[List[str]]:
        """找到最小的正向组合，唯一标识目标"""
        target_list = list(target_values)
        # 尝试 2 符号组合
        for i in range(len(target_list)):
            for j in range(i + 1, len(target_list)):
                combo = [target_list[i], target_list[j]]
                count = sum(1 for obj in scene_features
                           if all(any(sym in vals for vals in obj.values())
                                 for sym in combo))
                if count == 1:
                    return combo
        # 尝试 3 符号组合
        for i in range(len(target_list)):
            for j in range(i + 1, len(target_list)):
                for k in range(j + 1, len(target_list)):
                    combo = [target_list[i], target_list[j], target_list[k]]
                    count = sum(1 for obj in scene_features
                               if all(any(sym in vals for vals in obj.values())
                                     for sym in combo))
                    if count == 1:
                        return combo
        return None

    def _try_negation(self, target_features: Dict[str, Set[str]],
                     scene_features: List[Dict[str, Set[str]]],
                     target_idx: int) -> Optional[List[str]]:
        """
        尝试否定描述

        找到目标不包含但其他物体包含的值。
        "not X" 排除包含 X 的物体。
        """
        target_values = set()
        for vals in target_features.values():
            target_values.update(vals)

        # 找到其他物体有但目标没有的值
        neg_candidates = set()
        for i, obj in enumerate(scene_features):
            if i == target_idx:
                continue
            obj_values = set()
            for vals in obj.values():
                obj_values.update(vals)
            neg_candidates.update(obj_values - target_values)

        # 检查每个否定候选
        for sym in neg_candidates:
            # 计算 "not sym" 匹配的物体数（不包含 sym 的物体）
            count = sum(1 for obj in scene_features
                       if not any(sym in vals for vals in obj.values()))
            if count == 1:
                return ['not', sym]

        # 尝试否定 + 正向组合
        for sym in neg_candidates:
            for pos_sym in target_values:
                if pos_sym == sym:
                    continue
                count = sum(1 for i, obj in enumerate(scene_features)
                           if not any(sym in vals for vals in obj.values())
                           and any(pos_sym in vals for vals in obj.values()))
                if count == 1:
                    return ['not', sym, pos_sym]

        return None

    def _try_relative_clause(self, target_features: Dict[str, Set[str]],
                             scene_features: List[Dict[str, Set[str]]],
                             target_idx: int) -> Optional[List[str]]:
        """
        尝试用相对从句描述目标（多值语义）

        主句：静态属性符号（过滤到多个候选）
        从句：动作/情感符号（在候选中唯一标识）

        返回 [static_sym, 'that', action_sym] 或 None
        """
        # 分离静态和动态符号
        action_syms = set()
        static_syms = set()
        for dim, values in target_features.items():
            for v in values:
                cat = _symbol_category(v)
                if cat in ('action', 'action_effect', 'emotion', 'tense'):
                    action_syms.add(v)
                elif cat in ('color', 'shape', 'size', 'material', 'texture'):
                    static_syms.add(v)

        if not action_syms or not static_syms:
            return None

        # 尝试每个静态属性作为主句
        for static in static_syms:
            # 主句匹配：包含 static 的物体
            static_matches = [i for i, obj in enumerate(scene_features)
                             if any(static in vals for vals in obj.values())]
            if len(static_matches) <= 1:
                continue  # 主句已经唯一，不需要从句

            # 尝试每个动作属性作为从句
            for action in action_syms:
                clause_matches = [i for i in static_matches
                                if any(action in vals for vals in scene_features[i].values())]
                if len(clause_matches) == 1 and clause_matches[0] == target_idx:
                    return [static, 'that', action]

        return None


class CompoundListener:
    """
    复合特征 Listener

    匹配逻辑：
    - 正向符号：物体必须包含该值（任意维度）
    - 否定符号 "not X"：物体不能包含 X
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def interpret(self, utterance: List[str],
                  scene_features: List[Dict[str, Set[str]]]) -> Optional[int]:
        if not utterance or not scene_features:
            return None

        # 检测是否包含相对从句
        has_relative = any(s in RELATIVE_MARKERS for s in utterance)
        if has_relative:
            return self._interpret_with_relative(utterance, scene_features)

        scores = []
        for i, obj in enumerate(scene_features):
            score = self._match_score(utterance, obj)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if scores[0][1] > 0:
            return scores[0][0]
        return None

    def _interpret_with_relative(self, utterance: List[str],
                                 scene_features: List[Dict[str, Set[str]]]) -> Optional[int]:
        """
        相对从句的两阶段匹配（多值语义）

        阶段 1: 主句符号过滤候选物体
        阶段 2: 从句符号在候选子集中二次评分
        """
        # 找到 relative marker 位置
        rel_idx = None
        for i, s in enumerate(utterance):
            if s in RELATIVE_MARKERS:
                rel_idx = i
                break

        if rel_idx is None:
            return self._interpret_standard(utterance, scene_features)

        main_symbols = utterance[:rel_idx]
        clause_symbols = utterance[rel_idx + 1:]  # 跳过 marker 本身

        if not main_symbols:
            return self._interpret_standard(clause_symbols, scene_features)

        # 阶段 1: 主句过滤（多值匹配）
        main_scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set()
            for vals in obj.values():
                obj_values.update(vals)
            matches = sum(1 for s in main_symbols if s in obj_values)
            main_scores.append((i, matches / max(1, len(main_symbols))))

        # 阶段 2: 从句二次评分
        final_scores = []
        for i, main_score in main_scores:
            if main_score > 0 and clause_symbols:
                obj_values = set()
                for vals in scene_features[i].values():
                    obj_values.update(vals)
                clause_matches = sum(1 for s in clause_symbols if s in obj_values)
                clause_score = clause_matches / len(clause_symbols)
                final_score = 0.5 * main_score + 0.5 * clause_score
            else:
                final_score = main_score * 0.5
            final_scores.append((i, final_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[0][0] if final_scores[0][1] > 0 else None

    def _interpret_standard(self, utterance: List[str],
                            scene_features: List[Dict[str, Set[str]]]) -> Optional[int]:
        """标准匹配（无从句）"""
        scores = []
        for i, obj in enumerate(scene_features):
            score = self._match_score(utterance, obj)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if scores[0][1] > 0:
            return scores[0][0]
        return None

    def _match_score(self, utterance: List[str],
                    obj_features: Dict[str, Set[str]]) -> float:
        """计算匹配度，支持否定"""
        obj_values = set()
        for vals in obj_features.values():
            obj_values.update(vals)

        score = 0
        i = 0
        while i < len(utterance):
            if utterance[i] in NEGATION_MARKERS and i + 1 < len(utterance):
                negated_sym = utterance[i + 1]
                if negated_sym not in obj_values:
                    score += 1
                i += 2
            else:
                if utterance[i] in obj_values:
                    score += 1
                i += 1
        return score / len(utterance) if utterance else 0.0


class CompoundCommunicationGame:
    """复合特征交流游戏"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = CompoundSpeaker(self.language)
        self.listener = CompoundListener(self.language)
        self.game_log = []
        self.negation_used = 0
        self.negation_success = 0
        self.positive_used = 0
        self.clause_used = 0
        self.clause_success = 0

    def play_round(self, scene_features: List[Dict[str, Set[str]]],
                   target_idx: int, max_len: int = 0) -> bool:
        if target_idx >= len(scene_features):
            return False

        target = scene_features[target_idx]
        utterance = self.speaker.describe(target, scene_features, target_idx, max_len=max_len)
        if not utterance:
            return False

        used_negation = any(s in NEGATION_MARKERS for s in utterance)
        used_clause = any(s in RELATIVE_MARKERS for s in utterance)
        if used_negation:
            self.negation_used += 1
        elif used_clause:
            self.clause_used += 1
        else:
            self.positive_used += 1

        chosen_idx = self.listener.interpret(utterance, scene_features)
        success = (chosen_idx == target_idx)

        if used_negation and success:
            self.negation_success += 1
        if used_clause and success:
            self.clause_success += 1

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
            'utterance_length': len(utterance),
        })
        return success

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['negation_used'] = self.negation_used
        stats['negation_success'] = self.negation_success
        stats['positive_used'] = self.positive_used
        stats['negation_rate'] = self.negation_used / max(1, len(self.game_log))
        stats['clause_used'] = self.clause_used
        stats['clause_success'] = self.clause_success
        stats['clause_rate'] = self.clause_used / max(1, len(self.game_log))
        return stats


# --- 从句场景生成器 ---

# 动作和情感符号（用于从句）
ACTION_SYMBOLS = {'push', 'pull', 'grab', 'drop', 'lift', 'throw', 'kick', 'hit'}
EMOTION_SYMBOLS = {'happy', 'sad', 'angry', 'calm', 'excited', 'scared'}


def generate_clause_scene(num_objects: int = 5,
                          num_action_dims: int = 1) -> Tuple[List[Dict[str, Set[str]]], int]:
    """
    生成从句涌现场景

    委托给 generate_clause_scene_with_overlap，使用跨组重复的动作结构。
    """
    return generate_clause_scene_with_overlap(num_objects)


def generate_clause_scene_with_overlap(num_objects: int = 4) -> Tuple[List[Dict[str, Set[str]]], int]:
    """
    生成从句必需的场景

    关键设计：
    - 物体分成 2 个颜色组（如 red 组和 blue 组）
    - 每个颜色组内，每个物体有 1 个动作值
    - 动作值跨组重复但组内唯一
    - 这样：
      * 动作符号跨组不唯一（"push" 出现在 red 组和 blue 组）
      * 颜色符号组内不唯一（"red" 匹配多个物体）
      * 从句 "red that push" 唯一标识目标！

    例如：
    - Object 0: {color: {red}, action: {push}}
    - Object 1: {color: {red}, action: {pull}}
    - Object 2: {color: {blue}, action: {push}}
    - Object 3: {color: {blue}, action: {pull}}
    - "push" 匹配 0 和 2，"red" 匹配 0 和 1
    - "red that push" 只匹配 0 ← 从句必需
    """
    all_colors = list(COLORS)
    all_actions = list(ACTION_SYMBOLS)
    np.random.shuffle(all_colors)
    np.random.shuffle(all_actions)

    # 2 个颜色组
    color_a, color_b = all_colors[0], all_colors[1]
    # 每组 num_objects//2 个物体，共享一组动作值
    group_size = max(2, num_objects // 2)
    action_pool = all_actions[:group_size]

    scene = []
    # 颜色组 A
    for i in range(group_size):
        obj = {
            'color': {color_a},
            'shape': {np.random.choice(list(SHAPES))},
            'size': {np.random.choice(list(SIZES))},
            'action': {action_pool[i]},
        }
        scene.append(obj)

    # 颜色组 B（共享动作值）
    for i in range(group_size):
        obj = {
            'color': {color_b},
            'shape': {np.random.choice(list(SHAPES))},
            'size': {np.random.choice(list(SIZES))},
            'action': {action_pool[i]},
        }
        scene.append(obj)

    np.random.shuffle(scene)
    target_idx = np.random.randint(0, len(scene))

    return scene, target_idx


def test_simple():
    """测试最简单的子集场景"""
    print("=== 简单子集场景测试 ===")
    scene, target_idx = generate_simple_subset_scene()
    print(f"场景:")
    for i, obj in enumerate(scene):
        flat = {k: list(v) for k, v in obj.items()}
        marker = " <-- TARGET" if i == target_idx else ""
        print(f"  {i}: {flat}{marker}")

    game = CompoundCommunicationGame()
    target = scene[target_idx]
    utterance = game.speaker.describe(target, scene, target_idx)
    print(f"\n描述: {utterance}")
    print(f"是否使用否定: {any(s in NEGATION_MARKERS for s in utterance)}")

    chosen = game.listener.interpret(utterance, scene)
    print(f"Listener 选择: {chosen}")
    print(f"正确: {chosen == target_idx}")

    # 也测试描述非目标物体
    print(f"\n描述物体 0 (父物体):")
    utt0 = game.speaker.describe(scene[0], scene, 0)
    print(f"  描述: {utt0}")
    chosen0 = game.listener.interpret(utt0, scene)
    print(f"  Listener 选择: {chosen0}")


def test_multi_round(num_rounds: int = 200, verbose: bool = False):
    """多轮测试"""
    print(f"\n=== 多轮测试 ({num_rounds} 轮) ===")
    game = CompoundCommunicationGame()

    for r in range(num_rounds):
        scene, target_idx = generate_subset_scene(
            num_base=np.random.randint(2, 5),
            num_subsets=np.random.randint(1, 4),
            target_is_subset=(np.random.random() < 0.5),
        )
        game.play_round(scene, target_idx)

        if verbose and (r + 1) % 50 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: success={stats['success_rate']:.1%}, "
                  f"neg_rate={stats['negation_rate']:.1%}, "
                  f"neg_used={stats['negation_used']}, "
                  f"vocab={stats['vocab_size']}")

    stats = game.get_stats()
    neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  词汇量: {stats['vocab_size']}")
    print(f"  否定使用率: {stats['negation_rate']:.1%}")
    print(f"  否定使用次数: {stats['negation_used']}")
    print(f"  否定成功次数: {stats['negation_success']}")
    print(f"  否定符号: {neg_syms}")

    return stats


if __name__ == '__main__':
    test_simple()
    test_multi_round(200, verbose=True)
