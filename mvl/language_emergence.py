"""
语言涌现模块：从符号到语法的自发涌现

核心思想：
语言不是发明的，是从交流需求中涌现的。
当单个符号不足以区分场景时，agent被迫组合符号。
组合的成功模式固化为语法规则。

理论基础：
- 参照游戏（Reference Game）：交流压力驱动语言进化
- 组合性（Compositionality）：意义 = 部分之和
- 词序涌现：从消歧需求中产生语法规则
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations

# 符号分类常量
COLORS = {'red', 'blue', 'green', 'yellow', 'white', 'black'}
SHAPES = {'circle', 'square', 'triangle', 'star', 'diamond', 'sphere', 'cube', 'cylinder'}
SIZES = {'tiny', 'small', 'medium', 'big', 'huge', 'mid'}
MATERIALS = {'metal', 'wood', 'plastic', 'stone', 'fabric', 'glass', 'rubber'}
TEXTURES = {'smooth', 'rough', 'bumpy', 'fuzzy'}
WEIGHTS = {'light', 'medium', 'heavy'}
TEMPERATURES = {'hot', 'warm', 'cold'}
BRIGHTNESSES = {'bright', 'dim', 'dark'}
PATTERNS = {'solid', 'striped', 'spotted'}
ORIGINS = {'natural', 'artificial', 'magical'}
DEFAULT_MODIFIER_ORDER = ['size', 'color', 'material']

# 动作符号常量
ACTIONS = {'push', 'pull', 'grab', 'drop', 'move', 'go', 'stop', 'turn', 'throw'}
ACTION_EFFECTS = {
    'push': 'displacement',
    'pull': 'attraction',
    'grab': 'attached',
    'drop': 'detached',
    'move': 'translation',
    'go': 'directional',
    'stop': 'halt',
    'turn': 'rotation',
    'throw': 'trajectory',
}
ACTION_EFFECT_VALUES = set(ACTION_EFFECTS.values())

# 情感符号常量
EMOTIONS = {'happy', 'calm', 'curious', 'frustrated', 'scared', 'surprised', 'bored', 'confident'}

# 因果标记常量
CAUSAL_MARKERS = {'then', 'because', 'so', 'when', 'if', 'cause', 'result'}

# 因果推理标记（Phase 19）
CAUSAL_REASONING_MARKERS = {'because', 'so', 'therefore', 'thus', 'hence'}
TEMPORAL_MARKERS = {'then', 'after', 'before', 'when'}

# 视角标记（Phase 20）
PERSPECTIVE_MARKERS = {'know', 'think', 'believe', 'see', 'hear', 'feel'}

# 抽象标记（Phase 21）
ABSTRACT_MARKERS = {'like', 'same', 'similar', 'different', 'analogous'}

# 工具标记（Phase 22）
TOOL_MARKERS = {'use', 'for', 'need', 'can', 'tool', 'help'}

# 听觉符号（Phase 23）
AUDITORY_SYMBOLS = {'loud', 'quiet', 'sharp', 'soft', 'high', 'low', 'buzz', 'click', 'hum', 'crack'}

# 触觉符号（Phase 23）
TACTILE_SYMBOLS = {'rough', 'smooth', 'hard', 'soft_tactile', 'hot', 'cold', 'wet', 'dry', 'sharp_tactile', 'fuzzy'}

# 否定标记
NEGATION_MARKERS = {'not', 'no', 'exclude'}

# 时态标记
TENSE_MARKERS = {'past', 'present', 'future'}

# 相对从句标记
RELATIVE_MARKERS = {'that', 'which', 'who'}


def _symbol_category(sym: str) -> Optional[str]:
    """推断符号所属的属性类别"""
    if sym in COLORS:
        return 'color'
    if sym in SIZES:
        return 'size'
    if sym in MATERIALS:
        return 'material'
    if sym in SHAPES:
        return 'shape'
    if sym in TEXTURES:
        return 'texture'
    if sym in WEIGHTS:
        return 'weight'
    if sym in TEMPERATURES:
        return 'temperature'
    if sym in BRIGHTNESSES:
        return 'brightness'
    if sym in PATTERNS:
        return 'pattern'
    if sym in ORIGINS:
        return 'origin'
    if sym in ACTIONS:
        return 'action'
    if sym in ACTION_EFFECT_VALUES:
        return 'action_effect'
    if sym in EMOTIONS:
        return 'emotion'
    if sym in CAUSAL_MARKERS:
        return 'causal'
    if sym in CAUSAL_REASONING_MARKERS:
        return 'causal_reasoning'
    if sym in TEMPORAL_MARKERS:
        return 'temporal'
    if sym in PERSPECTIVE_MARKERS:
        return 'perspective'
    if sym in ABSTRACT_MARKERS:
        return 'abstract'
    if sym in TOOL_MARKERS:
        return 'tool'
    if sym in NEGATION_MARKERS:
        return 'negation'
    if sym in TENSE_MARKERS:
        return 'tense'
    if sym in RELATIVE_MARKERS:
        return 'relative'
    if sym in AUDITORY_SYMBOLS:
        return 'auditory'
    if sym in TACTILE_SYMBOLS:
        return 'tactile'
    return None


def _is_compound(sym: str) -> bool:
    """判断符号是否为复合符号（如 'red-cube'）"""
    return '-' in sym and _symbol_category(sym) is None


def _expand_compound(sym: str) -> List[str]:
    """展开复合符号为组件列表；非复合符号返回自身"""
    if _is_compound(sym):
        return sym.split('-')
    return [sym]


class EmergingLanguage:
    """
    涌现语言系统

    从交流经验中自发涌现的语言：
    - 词汇表：符号 → 使用统计
    - 组合规则：哪些符号经常一起出现
    - 词序偏好：哪种排列更有效
    - 语法规则：从成功模式中提取的规律
    """

    def __init__(self):
        # 词汇表：符号 → 使用统计
        self.vocabulary = {}  # {symbol: {frequency, success_rate, contexts}}

        # 组合规则：(sym_a, sym_b) → 成功率
        self.collocations = {}  # {(sym_a, sym_b): {count, successes}}

        # 词序偏好：modifier_first vs head_first
        self.word_order_scores = {'modifier_first': 0.0, 'head_first': 0.0}
        self.word_order_counts = {'modifier_first': 0, 'head_first': 0}

        # 语法规则：从成功模式中提取
        self.grammar_rules = []  # [(pattern, confidence, count), ...]

        # n-gram 模式：支持 3+ 符号组合
        self.ngram_patterns = {}  # {('big','red','circle'): {count, successes}}

        # 形容词层级偏好：相邻词对的成功率
        self.modifier_order = {}  # {('size','color'): {count, successes}, ...}

        # 维度级统计：按特征维度（shape/material/color/size）追踪成功率
        # 用于在多个同等效果的组合中，优先选择来自高成功率维度的符号
        self.dimension_stats = {}  # {dim: {frequency, successes, success_rate}}

        # 复合符号：频繁成功的多符号组合被压缩为单一符号
        # 如 "red-cube" 由 "red" + "cube" 频繁共现而生成
        self.compounds = {}  # {compound_sym: {components, frequency, successes, success_rate}}

        # 复合符号共现追踪：用于递归复合（3+ 组件）
        # 记录复合符号与其他符号在同一成功描述中的共现
        self.compound_cooccurrence = {}  # {(compound, other): {count, successes}}

        # 统计
        self.total_games = 0
        self.total_successes = 0
        self.multi_symbol_games = 0  # 使用多符号描述的游戏数
        self.tri_symbol_games = 0    # 使用 3 符号描述的游戏数
        self.inner_speech_games = 0  # 使用内部语言的游戏数

    def save_state(self) -> dict:
        """导出所有学习到的状态（用于迁移学习）"""
        return {
            'vocabulary': self.vocabulary,
            'collocations': {str(k): v for k, v in self.collocations.items()},
            'word_order_scores': self.word_order_scores,
            'word_order_counts': self.word_order_counts,
            'grammar_rules': self.grammar_rules,
            'ngram_patterns': {str(k): v for k, v in self.ngram_patterns.items()},
            'modifier_order': {str(k): v for k, v in self.modifier_order.items()},
            'dimension_stats': self.dimension_stats,
            'compounds': self.compounds,
            'compound_cooccurrence': {str(k): v for k, v in self.compound_cooccurrence.items()},
            'total_games': self.total_games,
            'total_successes': self.total_successes,
            'multi_symbol_games': self.multi_symbol_games,
            'tri_symbol_games': self.tri_symbol_games,
            'inner_speech_games': self.inner_speech_games,
        }

    def load_state(self, state: dict):
        """从导出的状态恢复（用于迁移学习）"""
        self.vocabulary = state.get('vocabulary', {})
        # collocations 的 key 是 tuple，需要从字符串恢复
        raw_col = state.get('collocations', {})
        self.collocations = {}
        for k, v in raw_col.items():
            if isinstance(k, str):
                # 解析 "('a', 'b')" 格式
                import ast
                try:
                    key = ast.literal_eval(k)
                except (ValueError, SyntaxError):
                    key = k
            else:
                key = k
            self.collocations[key] = v
        self.word_order_scores = state.get('word_order_scores', {'modifier_first': 0.0, 'head_first': 0.0})
        self.word_order_counts = state.get('word_order_counts', {'modifier_first': 0, 'head_first': 0})
        self.grammar_rules = state.get('grammar_rules', [])
        # ngram_patterns 和 modifier_order 同样需要 tuple 恢复
        raw_ngram = state.get('ngram_patterns', {})
        self.ngram_patterns = {}
        for k, v in raw_ngram.items():
            if isinstance(k, str):
                import ast
                try:
                    key = ast.literal_eval(k)
                except (ValueError, SyntaxError):
                    key = k
            else:
                key = k
            self.ngram_patterns[key] = v
        raw_mod = state.get('modifier_order', {})
        self.modifier_order = {}
        for k, v in raw_mod.items():
            if isinstance(k, str):
                import ast
                try:
                    key = ast.literal_eval(k)
                except (ValueError, SyntaxError):
                    key = k
            else:
                key = k
            self.modifier_order[key] = v
        self.dimension_stats = state.get('dimension_stats', {})
        self.compounds = state.get('compounds', {})
        raw_cc = state.get('compound_cooccurrence', {})
        self.compound_cooccurrence = {}
        for k, v in raw_cc.items():
            if isinstance(k, str):
                import ast
                try:
                    key = ast.literal_eval(k)
                except (ValueError, SyntaxError):
                    key = k
            else:
                key = k
            self.compound_cooccurrence[key] = v
        self.total_games = state.get('total_games', 0)
        self.total_successes = state.get('total_successes', 0)
        self.multi_symbol_games = state.get('multi_symbol_games', 0)
        self.tri_symbol_games = state.get('tri_symbol_games', 0)
        self.inner_speech_games = state.get('inner_speech_games', 0)

    def record_usage(self, symbols: List[str], success: bool):
        """记录一次符号使用（符号级 + 维度级 + 复合符号更新）"""
        for sym in symbols:
            if sym not in self.vocabulary:
                self.vocabulary[sym] = {
                    'frequency': 0,
                    'successes': 0,
                    'success_rate': 0.0,
                }
            self.vocabulary[sym]['frequency'] += 1
            if success:
                self.vocabulary[sym]['successes'] += 1
            self.vocabulary[sym]['success_rate'] = (
                self.vocabulary[sym]['successes'] / self.vocabulary[sym]['frequency']
            )

            # 更新复合符号的 last_used
            if sym in self.compounds:
                self.compounds[sym]['last_used'] = self.total_games

            # 维度级统计
            dim = _symbol_category(sym)
            if dim:
                if dim not in self.dimension_stats:
                    self.dimension_stats[dim] = {
                        'frequency': 0, 'successes': 0, 'success_rate': 0.0
                    }
                self.dimension_stats[dim]['frequency'] += 1
                if success:
                    self.dimension_stats[dim]['successes'] += 1
                self.dimension_stats[dim]['success_rate'] = (
                    self.dimension_stats[dim]['successes'] /
                    self.dimension_stats[dim]['frequency']
                )

    def record_collocation(self, sym_a: str, sym_b: str, success: bool):
        """记录一次符号组合"""
        key = (sym_a, sym_b)
        if key not in self.collocations:
            self.collocations[key] = {'count': 0, 'successes': 0}
        self.collocations[key]['count'] += 1
        if success:
            self.collocations[key]['successes'] += 1

    def record_word_order(self, order: str, success: bool):
        """记录词序使用"""
        if order in self.word_order_counts:
            self.word_order_counts[order] += 1
            if success:
                self.word_order_scores[order] += 1.0

    def record_ngram(self, symbols: List[str], success: bool):
        """记录 n-gram 模式（支持 3+ 符号）"""
        if len(symbols) >= 2:
            key = tuple(symbols)
            if key not in self.ngram_patterns:
                self.ngram_patterns[key] = {'count': 0, 'successes': 0}
            self.ngram_patterns[key]['count'] += 1
            if success:
                self.ngram_patterns[key]['successes'] += 1

    def check_compound_formation(self, symbols: List[str], success: bool):
        """
        检查符号序列是否应该形成复合符号

        两种路径：
        1. 原始 n-gram：连续符号序列频繁成功
        2. 递归复合：已有复合符号 + 新符号频繁共现
        """
        if not success or len(symbols) < 2:
            return

        # 路径 1：原始 n-gram
        for n in range(2, len(symbols) + 1):
            for i in range(len(symbols) - n + 1):
                subsym = symbols[i:i + n]
                if any(_is_compound(s) for s in subsym):
                    continue
                key = tuple(subsym)
                ngram_data = self.ngram_patterns.get(key)
                if ngram_data is None:
                    continue
                count = ngram_data['count']
                successes = ngram_data['successes']
                rate = successes / count if count > 0 else 0

                min_count = max(2, 7 - n)
                if count >= min_count and rate >= 0.8:
                    compound_sym = '-'.join(subsym)
                    if compound_sym not in self.compounds:
                        self.compounds[compound_sym] = {
                            'components': subsym,
                            'frequency': count,
                            'successes': successes,
                            'success_rate': rate,
                            'last_used': self.total_games,
                        }

        # 路径 2：递归复合——已有复合符号频繁与另一个符号共现
        for sym in symbols:
            if not _is_compound(sym):
                continue
            comp_data = self.compounds.get(sym)
            if comp_data is None:
                continue
            for other in symbols:
                if other == sym or _is_compound(other):
                    continue
                # 检查两种顺序的共现
                for pattern in [(sym, other), (other, sym)]:
                    cc_data = self.compound_cooccurrence.get(pattern)
                    if cc_data is None:
                        continue
                    count = cc_data['count']
                    successes = cc_data['successes']
                    rate = successes / count if count > 0 else 0
                    if count >= 2 and rate >= 0.8:
                        if pattern[0] == other:
                            new_components = [other] + comp_data['components']
                        else:
                            new_components = comp_data['components'] + [other]
                        new_compound = '-'.join(new_components)
                        if new_compound not in self.compounds:
                            self.compounds[new_compound] = {
                                'components': new_components,
                                'frequency': count,
                                'successes': successes,
                                'success_rate': rate,
                                'last_used': self.total_games,
                            }

    def record_compound_cooccurrence(self, symbols: List[str], success: bool):
        """记录复合符号与其他符号的共现（用于递归复合）"""
        compounds_in = [s for s in symbols if _is_compound(s)]
        others = [s for s in symbols if not _is_compound(s)]
        if not compounds_in or not others:
            return
        for comp in compounds_in:
            for other in others:
                key = (comp, other)
                if key not in self.compound_cooccurrence:
                    self.compound_cooccurrence[key] = {'count': 0, 'successes': 0}
                self.compound_cooccurrence[key]['count'] += 1
                if success:
                    self.compound_cooccurrence[key]['successes'] += 1

    def prune_compounds(self, max_age: int = 100):
        """
        淘汰长期不使用的复合符号

        移除最近 max_age 轮内未被使用的复合符号。
        """
        to_remove = []
        for sym, data in self.compounds.items():
            last_used = data.get('last_used', 0)
            if self.total_games - last_used > max_age:
                to_remove.append(sym)
        for sym in to_remove:
            del self.compounds[sym]

    def mutate(self, mutation_rate: float = 0.05,
               borrow_from: Optional['EmergingLanguage'] = None):
        """
        语言变异算子（用于文化演化模拟）

        三种变异：
        1. 简化：高频复合符号被简化为新符号
        2. 借词：从另一个语言中采纳符号
        3. 语义漂移：符号成功率随机偏移

        参数：
            mutation_rate: 变异概率（0-1）
            borrow_from: 借词来源语言（可选）
        """
        import random as _random

        # 1. 简化：高频复合符号 → 缩写
        for sym, data in list(self.compounds.items()):
            if data['frequency'] >= 10 and _random.random() < mutation_rate:
                components = data['components']
                # 取每个组件的前 2 个字符
                short = ''.join(c[:2] for c in components)
                if short not in self.vocabulary and short not in self.compounds:
                    self.compounds[short] = {
                        'components': components,
                        'frequency': data['frequency'],
                        'successes': data['successes'],
                        'success_rate': data['success_rate'],
                        'last_used': self.total_games,
                    }

        # 2. 借词
        if borrow_from and _random.random() < mutation_rate * 2:
            for sym, data in borrow_from.vocabulary.items():
                if sym not in self.vocabulary and _random.random() < 0.1:
                    self.vocabulary[sym] = {
                        'frequency': max(1, data['frequency'] // 4),
                        'successes': max(1, data['successes'] // 4),
                        'success_rate': data['success_rate'],
                    }

        # 3. 语义漂移
        for sym, data in self.vocabulary.items():
            if _random.random() < mutation_rate:
                drift = _random.gauss(0, 0.05)
                data['success_rate'] = max(0.0, min(1.0,
                    data['success_rate'] + drift))

    def record_modifier_order(self, cat_a: str, cat_b: str, success: bool):
        """记录形容词层级偏好（相邻词对的类别关系）"""
        key = (cat_a, cat_b)
        if key not in self.modifier_order:
            self.modifier_order[key] = {'count': 0, 'successes': 0}
        self.modifier_order[key]['count'] += 1
        if success:
            self.modifier_order[key]['successes'] += 1

    def get_preferred_order(self) -> str:
        """获取偏好词序"""
        if sum(self.word_order_counts.values()) < 5:
            return 'modifier_first'  # 默认

        rates = {}
        for order in self.word_order_scores:
            count = self.word_order_counts[order]
            if count > 0:
                rates[order] = self.word_order_scores[order] / count
            else:
                rates[order] = 0.0

        return max(rates, key=rates.get)

    def get_order_consistency(self) -> float:
        """词序一致性：0=随机，1=完全一致"""
        total = sum(self.word_order_counts.values())
        if total < 5:
            return 0.0

        max_count = max(self.word_order_counts.values())
        return max_count / total

    def get_vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def get_combination_rate(self) -> float:
        """组合率：使用多符号描述的比例"""
        if self.total_games == 0:
            return 0.0
        return self.multi_symbol_games / self.total_games

    def extract_grammar_rules(self, min_confidence: float = 0.6):
        """从成功的组合中提取语法规则"""
        rules = []
        for (sym_a, sym_b), stats in self.collocations.items():
            if stats['count'] >= 3:
                confidence = stats['successes'] / stats['count']
                if confidence >= min_confidence:
                    rules.append({
                        'pattern': (sym_a, sym_b),
                        'confidence': confidence,
                        'count': stats['count'],
                    })
        self.grammar_rules = sorted(rules, key=lambda r: r['confidence'], reverse=True)
        return self.grammar_rules

    def get_preferred_modifier_order(self) -> List[str]:
        """获取偏好形容词层级排序"""
        if not self.modifier_order:
            return ['size', 'color', 'material']

        # 按成功率排序各对
        rates = {}
        for (cat_a, cat_b), stats in self.modifier_order.items():
            if stats['count'] >= 3:
                rate = stats['successes'] / stats['count']
                rates[(cat_a, cat_b)] = rate

        if not rates:
            return ['size', 'color', 'material']

        # 构建排序：统计每个类别排在前面的次数
        cat_wins = defaultdict(float)
        for (cat_a, cat_b), rate in rates.items():
            cat_wins[cat_a] += rate

        sorted_cats = sorted(cat_wins.keys(), key=lambda c: cat_wins[c], reverse=True)
        return sorted_cats

    def get_tri_symbol_rate(self) -> float:
        """3 符号组合率"""
        if self.total_games == 0:
            return 0.0
        return self.tri_symbol_games / self.total_games

    def get_stats(self) -> Dict:
        return {
            'vocabulary_size': self.get_vocabulary_size(),
            'total_games': self.total_games,
            'total_successes': self.total_successes,
            'success_rate': self.total_successes / max(1, self.total_games),
            'combination_rate': self.get_combination_rate(),
            'tri_symbol_rate': self.get_tri_symbol_rate(),
            'order_consistency': self.get_order_consistency(),
            'preferred_order': self.get_preferred_order(),
            'preferred_modifier_order': self.get_preferred_modifier_order(),
            'grammar_rules': len(self.grammar_rules),
            'ngram_patterns': len(self.ngram_patterns),
        }


class Speaker:
    """
    说话者：将感知转化为符号序列

    策略：
    1. 如果单个符号能唯一标识目标，用单个符号
    2. 如果单个符号有歧义，尝试 2 符号组合
    3. 如果 2 符号仍有歧义，尝试 3 符号组合
    4. 词序基于属性类别层级排列
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def describe(self, target_features: Dict[str, str],
                 scene_features: List[Dict[str, str]],
                 target_idx: int = -1,
                 scene_model: Optional[Dict] = None,
                 listener_vocab: Optional[Dict] = None) -> List[str]:
        """
        描述目标物体，返回符号列表

        参数：
            scene_model: 可选的场景心理模型（来自 inner_describe），
                        提供符号区分度信息，帮助选择更优描述
            listener_vocab: 可选的听者词汇表（教学模式），
                           优先使用听者已知的符号，每次最多引入 1 个新符号

        策略：
        1. 单符号能唯一标识 → 用单符号
        2. 组合能唯一标识 → 用组合
        3. 组合仍有歧义 → 尝试否定
        4. 否定也不够 → 尝试相对从句
        """
        target_symbols = []
        for key, value in target_features.items():
            if value:
                target_symbols.append(value)

        if not target_symbols:
            return []

        # 尝试所有策略，选择最短的描述
        candidates = []

        # 策略 0: 复合符号（优先使用已生成的复合符号）
        for compound_sym, data in self.language.compounds.items():
            components = data.get('components', [])
            # 检查复合符号的所有组件是否都在目标特征中
            if all(c in target_symbols for c in components):
                # 检查复合符号是否能唯一标识目标
                if self._is_unique(compound_sym, scene_features):
                    candidates.append([compound_sym])
                else:
                    # 复合符号不够唯一，尝试 + 其他符号
                    remaining = [s for s in target_symbols
                                 if s not in components and not _is_compound(s)]
                    for extra in remaining:
                        combo = [compound_sym, extra]
                        if self._count_matches(combo, scene_features) <= 1:
                            candidates.append(combo)

        # 策略 1: 单符号（多个可选时优先选词汇成功率高的）
        unique_syms = [sym for sym in target_symbols if self._is_unique(sym, scene_features)]
        if unique_syms:
            if self.language.vocabulary:
                unique_syms.sort(key=lambda s: -self.language.vocabulary.get(s, {}).get('success_rate', 0.0))
            candidates.append([unique_syms[0]])

        # 策略 2: 正向组合
        best_combo = self._find_best_combination(target_symbols, scene_features)
        best_matches = self._count_matches(best_combo, scene_features)
        if best_matches <= 1:
            candidates.append(self._order_symbols(best_combo))

        # 策略 3: 否定（如果更短）
        if target_idx >= 0:
            neg_result = self._try_negation(target_features, scene_features, target_idx)
            if neg_result:
                candidates.append(neg_result)

        # 策略 4: 相对从句（如果更短）
        if target_idx >= 0:
            rel_result = self._try_relative_clause(target_features, scene_features, target_idx)
            if rel_result:
                candidates.append(rel_result)

        # 选择最短的候选描述（同长度时优先选词汇成功率高的）
        if candidates:
            if listener_vocab:
                # 教学模式：优先选听者已知的符号，最多引入 1 个新符号
                def _teaching_key(c):
                    unknown = sum(1 for s in c if s not in listener_vocab)
                    return (unknown, len(c), -self._candidate_score(c))
                candidates.sort(key=_teaching_key)
                # 如果最佳候选引入 >1 个新符号，尝试精简
                best = candidates[0]
                unknown_count = sum(1 for s in best if s not in listener_vocab)
                if unknown_count > 1:
                    # 尝试只保留听者已知的符号 + 1 个新符号
                    known = [s for s in best if s in listener_vocab]
                    new_syms = [s for s in best if s not in listener_vocab]
                    if known and new_syms:
                        return known + [new_syms[0]]
                return best
            elif scene_model:
                # 有内部语言：优先选区分度高的描述
                candidates.sort(key=lambda c: (
                    len(c),
                    -self._candidate_score_with_model(c, scene_model)
                ))
            else:
                candidates.sort(key=lambda c: (len(c), -self._candidate_score(c)))
            return candidates[0]

        # 所有策略都失败，返回最佳正向组合
        return self._order_symbols(best_combo)

    def _try_negation(self, target_features: Dict[str, str],
                       scene_features: List[Dict[str, str]],
                       target_idx: int) -> Optional[List[str]]:
        """
        尝试用否定描述目标

        找到目标不具有但其他物体具有的特征，
        用 "not" + 该特征来排除。
        """
        target_values = set(target_features.values())

        # 找到所有其他物体有但目标没有的符号
        neg_candidates = set()
        for i, obj in enumerate(scene_features):
            if i == target_idx:
                continue
            obj_values = set(obj.values())
            neg_candidates.update(obj_values - target_values)

        # 检查每个否定候选能否唯一标识
        for sym in neg_candidates:
            # 计算 "not sym" 匹配的物体数
            neg_count = sum(1 for obj in scene_features
                           if sym not in set(obj.values()))
            if neg_count == 1:
                return ['not', sym]

        # 尝试否定 + 正向组合
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

    def _try_relative_clause(self, target_features: Dict[str, str],
                              scene_features: List[Dict[str, str]],
                              target_idx: int) -> Optional[List[str]]:
        """
        尝试用相对从句描述目标

        主句：静态属性（过滤到几个候选）
        从句：动作/情感属性（在候选中唯一标识）
        """
        action_syms = [v for k, v in target_features.items()
                       if _symbol_category(v) in ('action', 'action_effect', 'emotion', 'tense')]
        static_syms = [v for k, v in target_features.items()
                       if _symbol_category(v) in ('color', 'shape', 'size', 'material')]

        if not action_syms or not static_syms:
            return None

        # 尝试每个静态属性作为主句
        for static in static_syms:
            static_matches = [i for i, obj in enumerate(scene_features)
                             if static in obj.values()]
            if len(static_matches) <= 1:
                continue  # 主句已经唯一，不需要从句

            # 尝试每个动作属性作为从句
            for action in action_syms:
                clause_matches = [i for i in static_matches
                                if action in scene_features[i].values()]
                if len(clause_matches) == 1 and clause_matches[0] == target_idx:
                    return [static, 'that', action]

        return None

    def _candidate_score(self, symbols: List[str]) -> float:
        """
        计算候选描述的词汇成功率分数（用于同长度候选的排序）

        优先使用符号级成功率；符号完全无数据时，回退到维度级成功率。
        维度级是回退机制，不是融合机制——避免粗糙的维度信息稀释精确的符号信息。
        维度统计需要最小频率阈值（10 次）才可用，避免早期噪声。
        """
        if not self.language.vocabulary and not self.language.dimension_stats:
            return 0.0
        dim_min_freq = 10  # 维度统计最小可信频率
        rates = []
        for sym in symbols:
            sym_data = self.language.vocabulary.get(sym)
            if sym_data and sym_data['frequency'] > 0:
                rates.append(sym_data['success_rate'])
            else:
                # 回退：用维度级成功率估计未见过的符号
                dim = _symbol_category(sym)
                dim_data = self.language.dimension_stats.get(dim) if dim else None
                if dim_data and dim_data['frequency'] >= dim_min_freq:
                    rates.append(dim_data['success_rate'])
                else:
                    rates.append(0.0)
        return sum(rates) / len(rates) if rates else 0.0

    def _candidate_score_with_model(self, symbols: List[str],
                                     scene_model: Dict) -> float:
        """
        结合场景心理模型的候选评分

        综合词汇成功率和符号区分度：
        - 词汇成功率：历史交流经验
        - 符号区分度：该符号在当前场景中能区分多少物体
        """
        base_score = self._candidate_score(symbols)
        disc = scene_model.get('symbol_discriminability', {})
        if not disc:
            return base_score
        # 区分度越高越好（1.0 = 唯一，0.5 = 2个物体共享）
        disc_scores = [disc.get(s, 0.0) for s in symbols]
        avg_disc = sum(disc_scores) / len(disc_scores) if disc_scores else 0.0
        # 综合：60% 词汇成功率 + 40% 区分度
        return 0.6 * base_score + 0.4 * avg_disc

    def _is_unique(self, symbol: str, scene: List[Dict[str, str]]) -> bool:
        """检查符号是否唯一标识场景中的一个物体（支持复合符号展开）"""
        components = _expand_compound(symbol)
        return sum(
            1 for obj in scene
            if all(c in obj.values() for c in components)
        ) == 1

    def _find_best_combination(self, symbols: List[str],
                               scene: List[Dict[str, str]]) -> List[str]:
        """找到最小的能唯一标识目标的符号组合"""
        best = symbols[:2]
        best_matches = self._count_matches(best, scene)

        # 从 2 符号到全部符号，找到最小的唯一组合
        for n in range(2, len(symbols) + 1):
            combo = self._find_best_ncombo(symbols, n, scene)
            if combo:
                matches = self._count_matches(combo, scene)
                if matches < best_matches:
                    best = combo
                    best_matches = matches
                if matches <= 1:
                    break  # 已唯一，不需要更多符号

        return best

    def _find_best_ncombo(self, symbols: List[str], n: int,
                          scene: List[Dict[str, str]]) -> Optional[List[str]]:
        """找到最佳的 n 符号组合（匹配数最少），同等效果时随机选择"""
        candidates = []
        best_matches = 999

        for combo in combinations(symbols, n):
            combo = list(combo)
            matches = self._count_matches(combo, scene)
            if matches < best_matches:
                best_matches = matches
                candidates = [combo]
            elif matches == best_matches:
                candidates.append(combo)

        if not candidates:
            return None
        # 优先选词汇成功率高的组合，无词汇信息时随机选择（漂变机制）
        if self.language.vocabulary:
            candidates.sort(key=lambda c: -self._candidate_score(c))
            return candidates[0]
        return candidates[np.random.randint(len(candidates))]

    def _count_matches(self, combo: List[str], scene: List[Dict[str, str]]) -> int:
        """计算场景中有多少物体匹配该组合（支持复合符号展开）"""
        return sum(
            1 for obj in scene
            if all(any(c in obj.values() for c in _expand_compound(sym))
                   for sym in combo)
        )

    def _order_symbols(self, symbols: List[str]) -> List[str]:
        """按属性类别层级排列符号，加入随机漂变"""
        if len(symbols) <= 1:
            return symbols

        by_cat = defaultdict(list)
        for sym in symbols:
            cat = _symbol_category(sym) or 'other'
            by_cat[cat].append(sym)

        # 获取偏好层级
        preferred_order = self.language.get_preferred_modifier_order()

        # 随机漂变：当语言经验不足时，随机打乱顺序
        # 经验越多，越倾向于使用偏好顺序
        total_games = self.language.total_games
        if total_games < 50:
            # 早期：完全随机探索不同词序
            categories = [c for c in preferred_order if c in by_cat]
            if 'shape' in by_cat:
                categories.append('shape')
            np.random.shuffle(categories)
        elif total_games < 200:
            # 中期：以 40% 概率使用偏好顺序，60% 概率随机
            if np.random.random() < 0.4:
                categories = [c for c in preferred_order if c in by_cat]
                if 'shape' in by_cat:
                    categories.append('shape')
            else:
                categories = [c for c in preferred_order if c in by_cat]
                if 'shape' in by_cat:
                    categories.append('shape')
                np.random.shuffle(categories)
        else:
            # 后期：主要使用偏好顺序，但保持 20% 随机性
            categories = [c for c in preferred_order if c in by_cat]
            if 'shape' in by_cat:
                categories.append('shape')
            if np.random.random() < 0.2:
                np.random.shuffle(categories)

        result = []
        for cat in categories:
            result.extend(by_cat.pop(cat, []))
        # 剩余类别
        for remaining in by_cat.values():
            result.extend(remaining)

        return result

    def inner_describe(self, scene_features: List[Dict[str, str]]) -> Dict:
        """
        内部语言：对场景中每个物体进行内部描述，建立心理模型

        返回 scene_model:
        - obj_descriptions: {obj_idx: [symbols]} 每个物体的最佳描述
        - symbol_discriminability: {symbol: score} 符号的区分度
        - shared_symbols: set 多个物体共享的符号
        - unique_symbols: {obj_idx: set} 每个物体独有的符号
        """
        obj_descriptions = {}
        symbol_usage = defaultdict(int)  # 每个符号被几个物体使用

        for idx, obj in enumerate(scene_features):
            symbols = [v for v in obj.values() if v]
            obj_descriptions[idx] = symbols
            for s in symbols:
                symbol_usage[s] += 1

        # 区分度：只被 1 个物体使用的符号区分度最高
        shared_symbols = {s for s, count in symbol_usage.items() if count > 1}
        unique_symbols = {}
        for idx, symbols in obj_descriptions.items():
            unique_symbols[idx] = {s for s in symbols if s not in shared_symbols}

        # 符号区分度 = 1 / 使用该符号的物体数
        symbol_discriminability = {
            s: 1.0 / count for s, count in symbol_usage.items()
        }

        return {
            'obj_descriptions': obj_descriptions,
            'symbol_discriminability': symbol_discriminability,
            'shared_symbols': shared_symbols,
            'unique_symbols': unique_symbols,
        }


class Listener:
    """
    听者：将符号序列解释为物体

    策略：
    1. 匹配符号与物体特征（支持否定和从句）
    2. 选择匹配度最高的物体
    3. 如果多个物体匹配，选择最"典型"的
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def interpret(self, utterance: List[str],
                  scene_features: List[Dict[str, str]]) -> Optional[int]:
        """
        解释描述，选择物体

        参数：
            utterance: 符号列表
            scene_features: 场景中所有物体的特征

        返回：
            选中的物体索引，或 None
        """
        if not utterance or not scene_features:
            return None

        # 检测是否包含相对从句
        has_relative = any(s in RELATIVE_MARKERS for s in utterance)
        if has_relative:
            return self._interpret_with_relative(utterance, scene_features)

        # 标准匹配（支持否定）
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
        """
        相对从句的两阶段匹配

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

        # 阶段 1: 主句过滤
        main_scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set(obj.values())
            matches = sum(1 for s in main_symbols if s in obj_values)
            main_scores.append((i, matches / max(1, len(main_symbols))))

        # 阶段 2: 从句二次评分
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

    def _interpret_standard(self, utterance: List[str],
                             scene_features: List[Dict[str, str]]) -> Optional[int]:
        """标准匹配（无从句）"""
        scores = []
        for i, obj in enumerate(scene_features):
            score = self._match_score(utterance, obj)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if scores[0][1] > 0:
            return scores[0][0]
        return None

    def _match_score(self, utterance: List[str], obj_features: Dict[str, str]) -> float:
        """
        计算描述与物体的匹配度

        支持否定：遇到 negation marker 时，下一个符号反转匹配
        支持复合符号：展开后检查每个组件是否匹配
        """
        obj_values = set(obj_features.values())
        score = 0
        i = 0
        while i < len(utterance):
            if utterance[i] in NEGATION_MARKERS and i + 1 < len(utterance):
                # 否定：如果物体不包含下一个符号，得分 +1
                negated_sym = utterance[i + 1]
                components = _expand_compound(negated_sym)
                if not all(c in obj_values for c in components):
                    score += 1
                i += 2
            else:
                # 复合符号：所有组件都匹配才算匹配
                components = _expand_compound(utterance[i])
                if all(c in obj_values for c in components):
                    score += 1
                i += 1
        return score / len(utterance) if utterance else 0.0


class CommunicationGame:
    """
    参照游戏：语言涌现的核心机制

    流程：
    1. 生成场景（多个物体）
    2. 选择目标物体
    3. Speaker 描述目标
    4. Listener 根据描述选择物体
    5. 判断是否成功
    6. 更新语言统计
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int) -> bool:
        """
        一轮参照游戏

        参数：
            scene_features: 场景中所有物体的特征
            target_idx: 目标物体的索引

        返回：
            是否成功
        """
        if target_idx >= len(scene_features) or not scene_features:
            return False

        target = scene_features[target_idx]

        # 内部语言：Speaker 先对场景进行心理建模
        scene_model = self.speaker.inner_describe(scene_features)
        self.language.inner_speech_games += 1

        # Speaker 描述目标（使用场景心理模型辅助）
        utterance = self.speaker.describe(target, scene_features,
                                          scene_model=scene_model)

        if not utterance:
            return False

        # Listener 解释并选择
        chosen_idx = self.listener.interpret(utterance, scene_features)

        # 判断是否成功
        success = (chosen_idx == target_idx)

        # 更新语言统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        if len(utterance) > 1:
            self.language.multi_symbol_games += 1
        if len(utterance) >= 3:
            self.language.tri_symbol_games += 1

        # 记录符号使用
        self.language.record_usage(utterance, success)

        # 记录 2-gram 组合
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                self.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )

        # 记录 n-gram 模式（3+ 符号）
        if len(utterance) >= 2:
            self.language.record_ngram(utterance, success)

        # 记录复合符号共现（用于递归复合）
        self.language.record_compound_cooccurrence(utterance, success)

        # 检查是否形成复合符号
        self.language.check_compound_formation(utterance, success)

        # 定期淘汰不活跃的复合符号（每 50 轮检查一次）
        if self.language.total_games % 50 == 0:
            self.language.prune_compounds(max_age=100)

        # 记录词序
        order = self._detect_order(utterance, target)
        if order:
            self.language.record_word_order(order, success)

        # 记录形容词层级偏好
        self._record_modifier_order(utterance, success)

        # 记录日志
        self.game_log.append({
            'target': target,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })

        return success

    def play_meta_round(self, scene_features: List[Dict[str, str]],
                        target_idx: int,
                        other_game: 'CommunicationGame',
                        failed_utterance: List[str] = None,
                        failed_chosen_idx: int = None) -> bool:
        """
        元语言回合：两个 Agent 讨论语言映射

        核心机制：当通信失败时，双方通过元语言反馈共同学习

        流程：
        1. Speaker 描述目标
        2. Listener 解释
        3. 如果失败，双方都从这次交互中学习：
           - Speaker 记录这次描述（失败）
           - Listener 记录这次描述（成功——因为描述确实指向了目标）
           - 双方记录相同的共现模式
        4. 重试：Speaker 用教学模式（优先 Listener 已知的符号）

        参数：
            other_game: 另一个 Agent 的 CommunicationGame
            failed_utterance: 之前失败的描述
            failed_chosen_idx: 之前 Listener 的错误选择

        返回：
            是否成功（包括元语言纠正后的成功）
        """
        if target_idx >= len(scene_features) or not scene_features:
            return False

        target = scene_features[target_idx]

        # Step 1: 获取描述
        if failed_utterance is not None:
            utterance = failed_utterance
        else:
            utterance = self.speaker.describe(target, scene_features)
            if not utterance:
                return False

        # Step 2: 获取解释结果
        if failed_chosen_idx is not None:
            chosen_idx = failed_chosen_idx
            success = False
        else:
            chosen_idx = other_game.listener.interpret(utterance, scene_features)
            success = (chosen_idx == target_idx)

        # 更新 Speaker 统计（失败）
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        if not success and chosen_idx is not None:
            # Step 3: 元语言反馈——双方共同学习

            # Speaker 记录：这个描述失败了
            # Listener 记录：这个描述其实是正确的（因为 Speaker 说的是对的）
            # 这样双方都从这次交互中学习，词汇逐渐趋同
            other_game.language.record_usage(utterance, True)

            # 双方记录相同的共现模式
            if len(utterance) >= 2:
                for i in range(len(utterance) - 1):
                    # Speaker 记录失败的共现
                    self.language.record_collocation(
                        utterance[i], utterance[i + 1], False
                    )
                    # Listener 记录成功的共现（元语言反馈告诉他这是正确的）
                    other_game.language.record_collocation(
                        utterance[i], utterance[i + 1], True
                    )
                self.language.record_ngram(utterance, False)
                other_game.language.record_ngram(utterance, True)

            # Step 4: 重试——用教学模式（优先 Listener 已知的符号）
            new_utterance = self.speaker.describe(
                target, scene_features,
                listener_vocab=other_game.language.vocabulary
            )
            if new_utterance:
                new_chosen = other_game.listener.interpret(new_utterance, scene_features)
                new_success = (new_chosen == target_idx)
                # 更新双方（重试也算一轮游戏）
                self.language.total_games += 1
                other_game.language.total_games += 1
                if new_success:
                    self.language.total_successes += 1
                    other_game.language.total_successes += 1
                    success = True
                self.language.record_usage(new_utterance, new_success)
                other_game.language.record_usage(new_utterance, new_success)
                if len(new_utterance) >= 2:
                    for i in range(len(new_utterance) - 1):
                        self.language.record_collocation(
                            new_utterance[i], new_utterance[i + 1], new_success
                        )
                        other_game.language.record_collocation(
                            new_utterance[i], new_utterance[i + 1], new_success
                        )

        # 记录日志
        self.game_log.append({
            'target': target,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
            'meta': True,
        })

        return success

    def play_round_frozen(self, scene_features: List[Dict[str, str]],
                           target_idx: int) -> bool:
        """一轮参照游戏（不更新语言统计，用于迁移测试）"""
        if target_idx >= len(scene_features) or not scene_features:
            return False

        target = scene_features[target_idx]
        utterance = self.speaker.describe(target, scene_features)
        if not utterance:
            return False

        chosen_idx = self.listener.interpret(utterance, scene_features)
        return chosen_idx == target_idx

    def _detect_order(self, utterance: List[str],
                      target_features: Dict[str, str]) -> Optional[str]:
        """检测词序类型（支持多层级形容词）"""
        if len(utterance) < 2:
            return None

        cats = [_symbol_category(s) for s in utterance]

        # 检测 modifier + head 模式
        has_head = 'shape' in cats
        if has_head:
            head_idx = cats.index('shape')
            modifiers_before = head_idx  # shape 之前的修饰语数量
            modifiers_after = len(utterance) - head_idx - 1

            if modifiers_before > 0 and modifiers_after == 0:
                return 'modifier_first'
            if modifiers_after > 0 and modifiers_before == 0:
                return 'head_first'

        # 2 修饰语：检测两个修饰语的相对顺序
        if len(utterance) == 2:
            cat_0, cat_1 = cats[0], cats[1]
            if cat_0 and cat_1 and cat_0 != cat_1:
                if cat_0 == 'shape':
                    return 'head_first'
                if cat_1 == 'shape':
                    return 'modifier_first'

        return None

    def _record_modifier_order(self, utterance: List[str], success: bool):
        """记录相邻词对的形容词层级偏好"""
        for i in range(len(utterance) - 1):
            cat_a = _symbol_category(utterance[i])
            cat_b = _symbol_category(utterance[i + 1])
            if cat_a and cat_b and cat_a != cat_b and cat_a != 'shape' and cat_b != 'shape':
                self.language.record_modifier_order(cat_a, cat_b, success)

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['games_played'] = len(self.game_log)
        return stats


def generate_scene(num_objects: int = 4,
                   num_colors: int = 3,
                   num_shapes: int = 3) -> List[Dict[str, str]]:
    """
    生成场景

    创建一组物体，每个物体有颜色和形状。
    确保至少有一些歧义（需要组合描述）。
    """
    colors = ['red', 'blue', 'green', 'yellow'][:num_colors]
    shapes = ['circle', 'square', 'triangle'][:num_shapes]

    scene = []
    for i in range(num_objects):
        color = colors[i % len(colors)]
        shape = shapes[i % len(shapes)]
        scene.append({'color': color, 'shape': shape})

    return scene


def generate_ambiguous_scene(min_objects: int = 4) -> List[Dict[str, str]]:
    """
    生成有歧义的场景

    确保单符号无法唯一标识目标——
    这是语言组合涌现的必要条件。
    """
    colors = ['red', 'blue', 'green']
    shapes = ['circle', 'square', 'triangle']

    scene = []
    for c in colors:
        for s in shapes:
            scene.append({'color': c, 'shape': s})

    np.random.shuffle(scene)
    return scene[:max(min_objects, 6)]


def generate_rich_scene(complexity: str = 'medium') -> List[Dict[str, str]]:
    """
    生成丰富属性的场景

    复杂度级别：
    - 'simple':  4色×3形 = 12 物体（2符号足够）
    - 'medium':  4色×3形×2大小 = 24 物体（可能需要3符号）
    - 'complex': 从 72 种组合中采样 36 物体（需要3符号）
    - 'extreme': 4色×3形×2大小×3材质 = 72 物体（必须3符号）
    """
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle']
    sizes = ['big', 'small']
    materials = ['metal', 'wood', 'plastic']

    if complexity == 'simple':
        scene = [{'color': c, 'shape': s} for c in colors for s in shapes]

    elif complexity == 'medium':
        scene = [
            {'color': c, 'shape': s, 'size': sz}
            for c in colors for s in shapes for sz in sizes
        ]

    elif complexity == 'complex':
        # 从全集中采样 36 个
        all_combos = [
            {'color': c, 'shape': s, 'size': sz}
            for c in colors for s in shapes for sz in sizes
        ]
        np.random.shuffle(all_combos)
        scene = all_combos[:36]

    elif complexity == 'extreme':
        scene = [
            {'color': c, 'shape': s, 'size': sz, 'material': m}
            for c in colors for s in shapes for sz in sizes for m in materials
        ]

    else:
        scene = [{'color': c, 'shape': s} for c in colors for s in shapes]

    np.random.shuffle(scene)
    return scene


class LanguageAgent:
    """
    拥有独立语言的 agent

    每个 agent 有自己的 EmergingLanguage、Speaker、Listener。
    交流时：speaker 用自己的语言描述，listener 用自己的语言解释。
    成功/失败分别更新两个 agent 的语言。
    """

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.communication_log = []

    def speak(self, target_features: Dict[str, str],
              scene_features: List[Dict[str, str]]) -> List[str]:
        """用自己的语言描述目标"""
        return self.speaker.describe(target_features, scene_features)

    def listen(self, utterance: List[str],
               scene_features: List[Dict[str, str]]) -> Optional[int]:
        """用自己的语言解释描述，返回选中的物体索引"""
        return self.listener.interpret(utterance, scene_features)

    def update_from_communication(self, utterance: List[str], success: bool):
        """根据交流结果更新自己的语言统计"""
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
                self.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
            self.language.record_ngram(utterance, success)

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['agent_id'] = self.id
        stats['total_communications'] = len(self.communication_log)
        return stats


def cross_language_round(speaker_agent: LanguageAgent,
                         listener_agent: LanguageAgent,
                         scene_features: List[Dict[str, str]],
                         target_idx: int) -> bool:
    """
    跨语言交流一轮

    speaker 用自己的语言描述目标，listener 用自己的语言解释。
    成功/失败分别更新两个 agent 的语言。

    返回：是否成功
    """
    if target_idx >= len(scene_features) or not scene_features:
        return False

    target = scene_features[target_idx]

    # speaker 描述
    utterance = speaker_agent.speak(target, scene_features)
    if not utterance:
        return False

    # listener 解释
    chosen_idx = listener_agent.listen(utterance, scene_features)
    success = (chosen_idx == target_idx)

    # 分别更新两个 agent 的语言
    speaker_agent.update_from_communication(utterance, success)
    listener_agent.update_from_communication(utterance, success)

    # 记录词序和形容词层级（从 speaker 的视角）
    order = _detect_order_for_agent(utterance, target)
    if order:
        speaker_agent.language.record_word_order(order, success)
        listener_agent.language.record_word_order(order, success)

    _record_modifier_order_for_agent(speaker_agent, utterance, success)
    _record_modifier_order_for_agent(listener_agent, utterance, success)

    # 记录日志
    log_entry = {
        'speaker': speaker_agent.id,
        'listener': listener_agent.id,
        'target': target,
        'utterance': utterance,
        'chosen': chosen_idx,
        'success': success,
    }
    speaker_agent.communication_log.append(log_entry)
    listener_agent.communication_log.append(log_entry)

    return success


def _detect_order_for_agent(utterance: List[str],
                            target_features: Dict[str, str]) -> Optional[str]:
    """检测词序类型"""
    if len(utterance) < 2:
        return None

    cats = [_symbol_category(s) for s in utterance]

    has_head = 'shape' in cats
    if has_head:
        head_idx = cats.index('shape')
        modifiers_before = head_idx
        modifiers_after = len(utterance) - head_idx - 1

        if modifiers_before > 0 and modifiers_after == 0:
            return 'modifier_first'
        if modifiers_after > 0 and modifiers_before == 0:
            return 'head_first'

    if len(utterance) == 2:
        cat_0, cat_1 = cats[0], cats[1]
        if cat_0 and cat_1 and cat_0 != cat_1:
            if cat_0 == 'shape':
                return 'head_first'
            if cat_1 == 'shape':
                return 'modifier_first'

    return None


def _record_modifier_order_for_agent(agent: LanguageAgent,
                                     utterance: List[str], success: bool):
    """记录形容词层级偏好"""
    for i in range(len(utterance) - 1):
        cat_a = _symbol_category(utterance[i])
        cat_b = _symbol_category(utterance[i + 1])
        if cat_a and cat_b and cat_a != cat_b and cat_a != 'shape' and cat_b != 'shape':
            agent.language.record_modifier_order(cat_a, cat_b, success)


def compute_language_similarity(lang_a: EmergingLanguage,
                                lang_b: EmergingLanguage) -> Dict:
    """
    计算两个语言的相似度

    使用频率分布相似度（余弦相似度），而非简单的 Jaccard 相似度。
    因为所有 agent 都有相同的符号集合，差异在于使用频率和组合模式。

    返回：
        vocab_freq_similarity: 词汇使用频率的余弦相似度
        collocation_freq_similarity: 组合频率的余弦相似度
        order_match: 词序是否一致
        modifier_order_match: 形容词层级是否一致
        overall_similarity: 综合相似度
    """
    # 词汇使用频率相似度
    all_symbols = set(lang_a.vocabulary.keys()) | set(lang_b.vocabulary.keys())
    if all_symbols:
        freq_a = np.array([lang_a.vocabulary.get(s, {}).get('frequency', 0) for s in all_symbols])
        freq_b = np.array([lang_b.vocabulary.get(s, {}).get('frequency', 0) for s in all_symbols])
        norm_a = np.linalg.norm(freq_a)
        norm_b = np.linalg.norm(freq_b)
        if norm_a > 0 and norm_b > 0:
            vocab_freq_similarity = float(np.dot(freq_a, freq_b) / (norm_a * norm_b))
        else:
            vocab_freq_similarity = 0.0
    else:
        vocab_freq_similarity = 0.0

    # 组合模式频率相似度
    all_collocations = set(lang_a.collocations.keys()) | set(lang_b.collocations.keys())
    if all_collocations:
        coll_freq_a = np.array([lang_a.collocations.get(c, {}).get('count', 0) for c in all_collocations])
        coll_freq_b = np.array([lang_b.collocations.get(c, {}).get('count', 0) for c in all_collocations])
        norm_ca = np.linalg.norm(coll_freq_a)
        norm_cb = np.linalg.norm(coll_freq_b)
        if norm_ca > 0 and norm_cb > 0:
            collocation_freq_similarity = float(np.dot(coll_freq_a, coll_freq_b) / (norm_ca * norm_cb))
        else:
            collocation_freq_similarity = 0.0
    else:
        collocation_freq_similarity = 0.0

    # 词序一致性
    order_a = lang_a.get_preferred_order()
    order_b = lang_b.get_preferred_order()
    order_match = 1.0 if order_a == order_b else 0.0

    # 形容词层级一致性
    mod_a = tuple(lang_a.get_preferred_modifier_order())
    mod_b = tuple(lang_b.get_preferred_modifier_order())
    modifier_order_match = 1.0 if mod_a == mod_b else 0.0

    # 综合相似度（加权平均）
    overall_similarity = (
        0.4 * vocab_freq_similarity +
        0.3 * collocation_freq_similarity +
        0.2 * order_match +
        0.1 * modifier_order_match
    )

    return {
        'vocab_freq_similarity': vocab_freq_similarity,
        'collocation_freq_similarity': collocation_freq_similarity,
        'order_match': order_match,
        'modifier_order_match': modifier_order_match,
        'overall_similarity': overall_similarity,
    }
