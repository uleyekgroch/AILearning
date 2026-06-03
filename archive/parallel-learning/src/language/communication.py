"""
通信协议：实现 ILanguage 接口的通信层

包含：
- CommunicationProtocol: 说话者/听者策略 + 参照游戏
- 场景生成辅助函数
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from itertools import combinations
import random

from src.core.interfaces import ILanguage
from src.language.emergence import (
    EmergingLanguage, _symbol_category, _expand_compound, _is_compound,
    NEGATION_MARKERS, RELATIVE_MARKERS, COLORS, SHAPES, SIZES, MATERIALS,
)
from src.language.grammar import GrammarSystem


class CommunicationProtocol(ILanguage):
    """
    通信协议：实现 ILanguage 接口

    组合 EmergingLanguage + GrammarSystem，
    提供 produce / comprehend / play_round 的完整协议。
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.grammar = GrammarSystem()

    # -------------------------------------------------------------------
    # ILanguage 接口
    # -------------------------------------------------------------------

    def produce(self, intention: Dict) -> List[str]:
        """
        从内部意图产生话语（符号序列）

        策略：
        1. 单符号能唯一标识 -> 用单符号
        2. 组合能唯一标识 -> 用组合
        3. 否定策略
        """
        symbols = [v for v in intention.values() if v]
        if not symbols:
            return []
        return symbols

    def comprehend(self, utterance: List[str], context: Dict) -> Dict:
        """
        理解话语，返回解析后的意图/指称

        支持否定标记和复合符号展开。
        """
        if not utterance:
            return {'matched': False, 'symbols': []}

        # 检测是否有否定标记
        has_negation = any(s in NEGATION_MARKERS for s in utterance)
        has_relative = any(s in RELATIVE_MARKERS for s in utterance)

        result = {
            'matched': True,
            'symbols': utterance,
            'has_negation': has_negation,
            'has_relative': has_relative,
        }

        # 如果有场景上下文，尝试匹配物体
        scene = context.get('scene', [])
        if scene:
            best_idx = self._match_to_scene(utterance, scene)
            result['best_match_idx'] = best_idx

        return result

    def get_vocabulary(self) -> Dict[str, Any]:
        """获取当前词汇表状态"""
        return self.language.vocabulary

    # -------------------------------------------------------------------
    # 参照游戏
    # -------------------------------------------------------------------

    def play_round(self, speaker, listener,
                   scene: List[Dict[str, str]], target_idx: int) -> bool:
        """
        一轮参照游戏

        参数：
            speaker: CommunicationProtocol (说话者)
            listener: CommunicationProtocol (听者)
            scene: 场景中所有物体的特征
            target_idx: 目标物体的索引

        返回：
            是否成功
        """
        if target_idx >= len(scene) or not scene:
            return False

        target = scene[target_idx]

        # 暴露所有目标符号（像儿童听到/看到环境中的词汇）
        for val in target.values():
            if val:
                speaker.language.expose_symbol(val)
                if listener is not speaker:
                    listener.language.expose_symbol(val)

        # 说话者描述目标
        utterance = self._describe(speaker, target, scene, target_idx)
        if not utterance:
            return False

        # 听者解释并选择
        chosen_idx = self._interpret(listener, utterance, scene)

        # 判断成功
        success = (chosen_idx == target_idx)

        # 更新双方语言统计
        self._update_stats(speaker, utterance, success, target, scene)
        if listener is not speaker:
            self._update_stats(listener, utterance, success, target, scene)

        return success

    # -------------------------------------------------------------------
    # 多模态游戏（完形填空 / 句子重组 / 听力匹配）
    # -------------------------------------------------------------------

    def play_cloze_game(self, sentence: str, target_word: str,
                        options: list) -> bool:
        """完形填空游戏 — 真正尝试从选项中选出正确答案

        学习闭环（像老师纠正学生）：
        1. 尝试选择 → 2. 得到反馈（对/错 + 正确答案）→ 3. 强化正确词、弱化错误词
        """
        self.language.expose_symbol(target_word)
        for opt in options:
            self.language.expose_symbol(opt)

        # 第一步：真正尝试匹配
        chosen = self._try_cloze_match(target_word, options)
        success = (chosen == target_word)

        # 第二步：从结果学习 — 关键差异化机制
        # 成功 → 正确词得到大幅加强
        # 失败 → 正确词得到"纠正学习"（暴露但小奖励），错误词被弱化
        self._learn_from_cloze_result(target_word, chosen, success)

        return success

    def _try_cloze_match(self, target: str, options: list) -> str:
        """真正尝试从选项中选词 — 基于掌握度的加权选择"""
        if not options:
            return ''

        scores = []
        for opt in options:
            mastery = self._get_receptive_mastery(opt)
            scores.append(mastery)

        total = sum(scores)
        if total < 1e-6:
            return random.choice(options)

        import math
        temperature = 0.3  # 低温度 = 更确定性
        exp_scores = [math.exp(s / max(temperature, 0.01)) for s in scores]
        exp_total = sum(exp_scores)

        r = random.random() * exp_total
        cumsum = 0
        for opt, es in zip(options, exp_scores):
            cumsum += es
            if r <= cumsum:
                return opt
        return options[-1]

    def _learn_from_cloze_result(self, target: str, chosen: str, success: bool):
        """从完形填空结果学习 — 差异化更新

        像老师纠正：
        - 答对了 → "很好！" → 正确词大幅强化
        - 答错了 → "不对，这是 cat 不是 dog" → 正确词学习（暴露），错误词弱化
        """
        # 正确词：无论成败都学习（答错时 = 老师告诉你正确答案）
        target_data = self.language.vocabulary.get(target)
        if target_data is None:
            target_data = {'frequency': 0, 'successes': 0, 'success_rate': 0.0, 'exposures': 0}
            self.language.vocabulary[target] = target_data

        target_data['frequency'] = target_data.get('frequency', 0) + 1
        if success:
            target_data['successes'] = target_data.get('successes', 0) + 1
        target_data['success_rate'] = target_data['successes'] / target_data['frequency']

        # 错误选择：如果有错误，弱化被错误选择的词
        if not success and chosen != target:
            wrong_data = self.language.vocabulary.get(chosen)
            if wrong_data:
                # 错误选择被"惩罚"：增加失败计数
                wrong_data['frequency'] = wrong_data.get('frequency', 0) + 1
                wrong_data['success_rate'] = wrong_data['successes'] / wrong_data['frequency']

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1

    def _learn_from_game_result(self, word: str, success: bool):
        """从游戏结果学习 — 更新掌握度"""
        data = self.language.vocabulary.get(word)
        if data is None:
            data = {'frequency': 0, 'successes': 0, 'success_rate': 0.0, 'exposures': 0}
            self.language.vocabulary[word] = data

        data['frequency'] = data.get('frequency', 0) + 1
        if success:
            data['successes'] = data.get('successes', 0) + 1
        data['success_rate'] = data['successes'] / data['frequency']

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1

    def play_sentence_game(self, target_words: list,
                           available_words: list) -> bool:
        """句子重组游戏 — 基于掌握度真正尝试排列词序

        学习闭环：尝试排列 → 比较正确顺序 → 学习差异
        """
        for w in target_words:
            self.language.expose_symbol(w)

        # 第一步：基于掌握度尝试排列
        produced = self._try_sentence_production(target_words, available_words)

        # 第二步：与目标比较
        produced_set = set(produced)
        target_set = set(target_words)
        overlap = len(produced_set & target_set) / max(len(target_set), 1)

        # 词序准确性
        order_correct = sum(1 for a, b in zip(produced, target_words) if a == b)
        order_ratio = order_correct / max(len(target_words), 1)

        # 综合：词选对了 AND 顺序也对
        score = 0.5 * overlap + 0.5 * order_ratio
        success = score > 0.7

        # 第三步：学习
        for w in target_words:
            self._learn_from_game_result(w, success)
        if len(target_words) >= 2:
            self.language.multi_symbol_games += 1
            self.language.record_ngram(target_words, success)
            self.language.record_usage(target_words, success)

        return success

    def _try_sentence_production(self, target_words: list,
                                  available_words: list) -> list:
        """尝试排列词序 — 基于掌握度的启发式"""
        # 能掌握的词才放对位置
        produced = []
        for w in target_words:
            mastery = self._get_productive_mastery(w, self.language)
            if random.random() < mastery:
                produced.append(w)
            # 低掌握度的词可能被遗漏或替换

        # 如果能掌握的词太少，从 available 中随机补充
        if len(produced) < len(target_words) // 2:
            extras = [w for w in available_words if w not in produced]
            random.shuffle(extras)
            produced.extend(extras[:len(target_words) - len(produced)])

        return produced

    def play_listening_game(self, sentence: str,
                            text_options: list) -> bool:
        """听力匹配游戏 — 基于词汇掌握度真正尝试匹配

        学习闭环：听 → 尝试识别 → 得到反馈 → 差异化学习
        """
        words = [w for w in sentence.lower().split() if w.isalpha()]
        for w in words:
            self.language.expose_symbol(w)

        # 第一步：计算对每个选项的识别分数
        best_idx = 0
        best_score = -1.0
        for i, opt in enumerate(text_options):
            opt_words = [w for w in opt.lower().split() if w.isalpha()]
            score = sum(self._get_receptive_mastery(ow) for ow in opt_words)
            score += random.gauss(0, 0.05)  # 少量噪声
            if score > best_score:
                best_score = score
                best_idx = i

        # 第二步：判断是否选对
        chosen = text_options[best_idx] if best_idx < len(text_options) else ''
        success = (chosen == sentence)

        # 第三步：差异化学习
        for w in words:
            self._learn_from_cloze_result(w, w if success else '', success)

        return success

    # -------------------------------------------------------------------
    # 内部辅助：说话者策略
    # -------------------------------------------------------------------

    def _describe(self, speaker, target_features: Dict[str, str],
                  scene_features: List[Dict[str, str]],
                  target_idx: int = -1) -> List[str]:
        """说话者描述目标物体（受产出掌握度门控）"""
        lang = speaker.language
        target_symbols = [v for v in target_features.values() if v]
        if not target_symbols:
            return []

        # 词汇门控：只"说出"已掌握的符号
        available = []
        for sym in target_symbols:
            mastery = self._get_productive_mastery(sym, lang)
            if random.random() < mastery:
                available.append(sym)

        if not available:
            # 什么都说不出来 — 猜一个
            available = [random.choice(target_symbols)]

        candidates = []

        # 策略 1: 单符号（仅当掌握度高时）
        for sym in available:
            if self._is_unique(sym, scene_features):
                mastery = self._get_productive_mastery(sym, lang)
                if mastery > 0.5:
                    candidates.append([sym])

        # 策略 2: 组合（低掌握度词汇需要更多上下文）
        best_combo = self._find_best_combination(available, scene_features, lang)
        if best_combo:
            candidates.append(self._order_symbols(best_combo, lang))

        # 策略 3: 否定
        if target_idx >= 0:
            neg_result = self._try_negation(target_features, scene_features, target_idx)
            if neg_result:
                candidates.append(neg_result)

        # 选择最短且可靠的候选
        if candidates:
            candidates.sort(key=lambda c: (len(c), -self._candidate_score(c, lang)))
            return candidates[0]

        # 回退：返回最佳组合
        return self._order_symbols(best_combo or available[:2], lang)

    def _get_productive_mastery(self, symbol: str, lang) -> float:
        """产出掌握度：能"说出"这个符号的概率

        产出比感受难（像儿童能听懂但说不出）。
        需要大量的成功经历才能产出。
        """
        data = lang.vocabulary.get(symbol)
        if data is None:
            return 0.02
        freq = data.get('frequency', 0)
        successes = data.get('successes', 0)
        rate = data.get('success_rate', 0)

        if freq <= 0:
            return 0.02

        import math
        # 产出需要更多成功经历（门槛更高）
        base = min(0.3, 0.08 * math.log2(1 + exposures)) if (exposures := data.get('exposures', freq)) > 0 else 0
        success_bonus = min(0.6, 0.12 * math.log2(1 + successes))

        return min(0.85, max(0.02, base + success_bonus))

    def _get_receptive_mastery(self, symbol: str) -> float:
        """感受掌握度：能"听懂"这个符号的概率

        像儿童学词的间隔重复：
        - 暴露次数给基础识别能力
        - 成功使用经历（正确次数）给信心
        - 最近成功比很久以前成功更重要
        """
        data = self.language.vocabulary.get(symbol)
        if data is None:
            return 0.08  # 完全没听过

        freq = data.get('frequency', 0)
        exposures = data.get('exposures', freq)
        successes = data.get('successes', 0)

        if exposures <= 0:
            return 0.08

        import math

        # 基础：暴露次数的对数增长（能"认出"这个词）
        base = min(0.4, 0.1 * math.log2(1 + exposures))

        # 核心加分：成功次数的对数增长（真正"掌握"了）
        # 这是区分已学/未学的关键
        success_bonus = min(0.55, 0.15 * math.log2(1 + successes))

        return min(0.95, base + success_bonus)

    def _is_unique(self, symbol: str, scene: List[Dict[str, str]]) -> bool:
        """检查符号是否唯一标识场景中的一个物体"""
        components = _expand_compound(symbol)
        return sum(
            1 for obj in scene
            if all(c in obj.values() for c in components)
        ) == 1

    def _find_best_combination(self, symbols: List[str],
                                scene: List[Dict[str, str]],
                                lang: EmergingLanguage) -> Optional[List[str]]:
        """找到最小的能唯一标识目标的符号组合"""
        for n in range(2, len(symbols) + 1):
            for combo in combinations(symbols, n):
                combo_list = list(combo)
                matches = self._count_matches(combo_list, scene)
                if matches <= 1:
                    return combo_list
        return symbols[:2]

    def _count_matches(self, combo: List[str], scene: List[Dict[str, str]]) -> int:
        """计算场景中有多少物体匹配该组合"""
        return sum(
            1 for obj in scene
            if all(any(c in obj.values() for c in _expand_compound(sym))
                   for sym in combo)
        )

    def _order_symbols(self, symbols: List[str], lang: EmergingLanguage) -> List[str]:
        """按属性类别层级排列符号"""
        if len(symbols) <= 1:
            return symbols

        by_cat = defaultdict(list)
        for sym in symbols:
            cat = _symbol_category(sym) or 'other'
            by_cat[cat].append(sym)

        preferred = lang.get_preferred_modifier_order()
        result = []
        for cat in preferred:
            result.extend(by_cat.pop(cat, []))
        for remaining in by_cat.values():
            result.extend(remaining)

        return result

    def _try_negation(self, target_features: Dict[str, str],
                       scene_features: List[Dict[str, str]],
                       target_idx: int) -> Optional[List[str]]:
        """尝试用否定描述目标"""
        target_values = set(target_features.values())
        neg_candidates = set()
        for i, obj in enumerate(scene_features):
            if i == target_idx:
                continue
            obj_values = set(obj.values())
            neg_candidates.update(obj_values - target_values)

        for sym in neg_candidates:
            neg_count = sum(1 for obj in scene_features
                           if sym not in set(obj.values()))
            if neg_count == 1:
                return ['not', sym]

        # 否定 + 正向组合
        for sym in neg_candidates:
            for pos_sym in target_values:
                if pos_sym == sym:
                    continue
                combo_count = sum(
                    1 for obj in scene_features
                    if sym not in set(obj.values()) and pos_sym in obj.values()
                )
                if combo_count == 1:
                    return ['not', sym, pos_sym]

        return None

    def _candidate_score(self, symbols: List[str], lang: EmergingLanguage) -> float:
        """计算候选描述的词汇成功率分数"""
        if not lang.vocabulary:
            return 0.0
        rates = []
        for sym in symbols:
            data = lang.vocabulary.get(sym)
            if data and data['frequency'] > 0:
                rates.append(data['success_rate'])
            else:
                rates.append(0.0)
        return sum(rates) / len(rates) if rates else 0.0

    # -------------------------------------------------------------------
    # 内部辅助：听者策略
    # -------------------------------------------------------------------

    def _interpret(self, listener, utterance: List[str],
                   scene_features: List[Dict[str, str]]) -> Optional[int]:
        """听者解释描述并选择物体"""
        if not utterance or not scene_features:
            return None

        # 检测相对从句
        has_relative = any(s in RELATIVE_MARKERS for s in utterance)
        if has_relative:
            return self._interpret_with_relative(utterance, scene_features)

        # 标准匹配
        scores = []
        for i, obj in enumerate(scene_features):
            score = self._match_score(utterance, obj)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if scores[0][1] > 0:
            return scores[0][0]
        return None

    def _interpret_with_relative(self, utterance: List[str],
                                  scene_features: List[Dict[str, str]]) -> Optional[int]:
        """相对从句的两阶段匹配"""
        rel_idx = None
        for i, s in enumerate(utterance):
            if s in RELATIVE_MARKERS:
                rel_idx = i
                break

        if rel_idx is None:
            return None

        main_symbols = utterance[:rel_idx]
        clause_symbols = utterance[rel_idx + 1:]

        if not main_symbols:
            return None

        main_scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set(obj.values())
            matches = sum(1 for s in main_symbols if s in obj_values)
            main_scores.append((i, matches / max(1, len(main_symbols))))

        final_scores = []
        for i, main_score in main_scores:
            if main_score > 0 and clause_symbols:
                obj_values = set(scene_features[i].values())
                clause_matches = sum(1 for s in clause_symbols if s in obj_values)
                clause_score = clause_matches / len(clause_symbols)
                final_score = 0.5 * main_score + 0.5 * clause_score
            else:
                final_score = main_score * 0.5
            final_scores.append((i, final_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[0][0] if final_scores[0][1] > 0 else None

    def _match_score(self, utterance: List[str], obj_features: Dict[str, str]) -> float:
        """计算描述与物体的匹配度（受感受掌握度门控）"""
        obj_values = set(obj_features.values())
        score = 0
        i = 0
        while i < len(utterance):
            if utterance[i] in NEGATION_MARKERS and i + 1 < len(utterance):
                negated_sym = utterance[i + 1]
                components = _expand_compound(negated_sym)
                mastery = self._get_receptive_mastery(negated_sym)
                if random.random() < mastery and not all(c in obj_values for c in components):
                    score += 1
                i += 2
            else:
                components = _expand_compound(utterance[i])
                mastery = self._get_receptive_mastery(utterance[i])
                if random.random() < mastery and all(c in obj_values for c in components):
                    score += 1
                i += 1
        return score / len(utterance) if utterance else 0.0

    def _match_to_scene(self, utterance: List[str], scene: List[Dict[str, str]]) -> Optional[int]:
        """将话语匹配到场景中的物体"""
        if not scene:
            return None
        scores = []
        for i, obj in enumerate(scene):
            score = self._match_score(utterance, obj)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores[0][1] > 0 else None

    # -------------------------------------------------------------------
    # 统计更新
    # -------------------------------------------------------------------

    def _update_stats(self, protocol, utterance: List[str], success: bool,
                      target: Dict, scene: List[Dict[str, str]]):
        """更新语言统计"""
        lang = protocol.language
        lang.total_games += 1
        if success:
            lang.total_successes += 1
        if len(utterance) > 1:
            lang.multi_symbol_games += 1
        if len(utterance) >= 3:
            lang.tri_symbol_games += 1

        lang.record_usage(utterance, success)

        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                lang.record_collocation(utterance[i], utterance[i + 1], success)
            lang.record_ngram(utterance, success)
            lang.record_compound_cooccurrence(utterance, success)

        lang.check_compound_formation(utterance, success)

        # 语法系统学习
        protocol.grammar.learn_pattern(utterance, success)

        # 词序记录
        order = self._detect_word_order(utterance)
        if order:
            lang.record_word_order(order, success)

        # 形容词层级
        for i in range(len(utterance) - 1):
            cat_a = _symbol_category(utterance[i])
            cat_b = _symbol_category(utterance[i + 1])
            if cat_a and cat_b and cat_a != cat_b and cat_a != 'shape' and cat_b != 'shape':
                lang.record_modifier_order(cat_a, cat_b, success)

    def _detect_word_order(self, utterance: List[str]) -> Optional[str]:
        """检测词序类型"""
        if len(utterance) < 2:
            return None
        cats = [_symbol_category(s) for s in utterance]
        if 'shape' in cats:
            head_idx = cats.index('shape')
            if head_idx > 0 and head_idx == len(utterance) - 1:
                return 'modifier_first'
            if head_idx == 0 and all(c != 'shape' for c in cats[1:]):
                return 'head_first'
        return None


# ---------------------------------------------------------------------------
# 场景生成辅助
# ---------------------------------------------------------------------------

def generate_scene(num_objects: int = 4, complexity: str = 'medium') -> List[Dict[str, str]]:
    """
    生成场景

    复杂度级别：
    - 'simple':  颜色+形状
    - 'medium':  颜色+形状+大小
    - 'complex': 颜色+形状+大小+材质
    """
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle']
    sizes = ['big', 'small']
    materials = ['metal', 'wood', 'plastic']

    if complexity == 'simple':
        pool = [
            {'color': c, 'shape': s}
            for c in colors for s in shapes
        ]
    elif complexity == 'medium':
        pool = [
            {'color': c, 'shape': s, 'size': sz}
            for c in colors for s in shapes for sz in sizes
        ]
    elif complexity == 'complex':
        pool = [
            {'color': c, 'shape': s, 'size': sz, 'material': m}
            for c in colors for s in shapes for sz in sizes for m in materials
        ]
    else:
        pool = [
            {'color': c, 'shape': s}
            for c in colors for s in shapes
        ]

    random.shuffle(pool)
    return pool[:min(num_objects, len(pool))]


def compute_ambiguity(scene: List[Dict[str, str]]) -> float:
    """
    计算场景歧义度

    歧义度 = 平均每个符号匹配的物体数 - 1
    值越大表示越需要组合描述。
    """
    if not scene:
        return 0.0

    # 统计每个符号出现的次数
    symbol_counts = defaultdict(int)
    for obj in scene:
        for val in obj.values():
            symbol_counts[val] += 1

    if not symbol_counts:
        return 0.0

    # 平均每个符号覆盖的物体数
    avg_coverage = sum(symbol_counts.values()) / len(symbol_counts)
    # 归一化到 [0, 1]
    ambiguity = (avg_coverage - 1) / max(1, len(scene) - 1)
    return max(0.0, min(1.0, ambiguity))
