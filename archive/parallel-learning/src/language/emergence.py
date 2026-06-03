"""
语言涌现模块：从符号到语法的自发涌现（精简移植版）

从 mvl/language_emergence.py 精简移植，保留核心功能：
- vocabulary dict, collocations, ngram_patterns, compounds
- record_usage(), record_collocation(), record_ngram()
- check_compound_formation()
- save_state() / load_state()
- mutate() 文化变异
- 常量 COLORS, SHAPES, SIZES, MATERIALS, ACTIONS, _symbol_category()

去掉所有与 CommunicationGame 紧耦合的逻辑。
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations

# ---------------------------------------------------------------------------
# 符号分类常量
# ---------------------------------------------------------------------------

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

ACTIONS = {'push', 'pull', 'grab', 'drop', 'move', 'go', 'stop', 'turn', 'throw'}
ACTION_EFFECTS = {
    'push': 'displacement', 'pull': 'attraction', 'grab': 'attached',
    'drop': 'detached', 'move': 'translation', 'go': 'directional',
    'stop': 'halt', 'turn': 'rotation', 'throw': 'trajectory',
}
ACTION_EFFECT_VALUES = set(ACTION_EFFECTS.values())

EMOTIONS = {'happy', 'calm', 'curious', 'frustrated', 'scared', 'surprised', 'bored', 'confident'}
CAUSAL_MARKERS = {'then', 'because', 'so', 'when', 'if', 'cause', 'result'}
CAUSAL_REASONING_MARKERS = {'because', 'so', 'therefore', 'thus', 'hence'}
TEMPORAL_MARKERS = {'then', 'after', 'before', 'when'}
PERSPECTIVE_MARKERS = {'know', 'think', 'believe', 'see', 'hear', 'feel'}
ABSTRACT_MARKERS = {'like', 'same', 'similar', 'different', 'analogous'}
TOOL_MARKERS = {'use', 'for', 'need', 'can', 'tool', 'help'}
AUDITORY_SYMBOLS = {'loud', 'quiet', 'sharp', 'soft', 'high', 'low', 'buzz', 'click', 'hum', 'crack'}
TACTILE_SYMBOLS = {'rough', 'smooth', 'hard', 'soft_tactile', 'hot', 'cold', 'wet', 'dry', 'sharp_tactile', 'fuzzy'}
NEGATION_MARKERS = {'not', 'no', 'exclude'}
TENSE_MARKERS = {'past', 'present', 'future'}
RELATIVE_MARKERS = {'that', 'which', 'who'}

DEFAULT_MODIFIER_ORDER = ['size', 'color', 'material']


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _symbol_category(sym) -> Optional[str]:
    """推断符号所属的属性类别"""
    if not isinstance(sym, str):
        return None
    for group, cat in [
        (COLORS, 'color'), (SIZES, 'size'), (MATERIALS, 'material'),
        (SHAPES, 'shape'), (TEXTURES, 'texture'), (WEIGHTS, 'weight'),
        (TEMPERATURES, 'temperature'), (BRIGHTNESSES, 'brightness'),
        (PATTERNS, 'pattern'), (ORIGINS, 'origin'), (ACTIONS, 'action'),
        (ACTION_EFFECT_VALUES, 'action_effect'), (EMOTIONS, 'emotion'),
        (CAUSAL_MARKERS, 'causal'), (CAUSAL_REASONING_MARKERS, 'causal_reasoning'),
        (TEMPORAL_MARKERS, 'temporal'), (PERSPECTIVE_MARKERS, 'perspective'),
        (ABSTRACT_MARKERS, 'abstract'), (TOOL_MARKERS, 'tool'),
        (NEGATION_MARKERS, 'negation'), (TENSE_MARKERS, 'tense'),
        (RELATIVE_MARKERS, 'relative'), (AUDITORY_SYMBOLS, 'auditory'),
        (TACTILE_SYMBOLS, 'tactile'),
    ]:
        if sym in group:
            return cat
    return None


def _is_compound(sym) -> bool:
    """判断符号是否为复合符号（如 'red-cube'）"""
    if not isinstance(sym, str):
        return False
    return '-' in sym and _symbol_category(sym) is None


def _expand_compound(sym: str) -> List[str]:
    """展开复合符号为组件列表；非复合符号返回自身"""
    if _is_compound(sym):
        return sym.split('-')
    return [sym]


# ---------------------------------------------------------------------------
# EmergingLanguage 核心类
# ---------------------------------------------------------------------------

class EmergingLanguage:
    """
    涌现语言系统

    从交流经验中自发涌现的语言：
    - 词汇表：符号 -> 使用统计
    - 组合规则：哪些符号经常一起出现
    - 词序偏好：哪种排列更有效
    - 语法规则：从成功模式中提取的规律
    """

    def __init__(self):
        # 词汇表：符号 -> 使用统计
        self.vocabulary: Dict[str, Dict] = {}

        # 组合规则：(sym_a, sym_b) -> 成功率
        self.collocations: Dict[Tuple[str, str], Dict] = {}

        # 词序偏好
        self.word_order_scores: Dict[str, float] = {'modifier_first': 0.0, 'head_first': 0.0}
        self.word_order_counts: Dict[str, int] = {'modifier_first': 0, 'head_first': 0}

        # 语法规则
        self.grammar_rules: List[Dict] = []

        # n-gram 模式
        self.ngram_patterns: Dict[Tuple[str, ...], Dict] = {}

        # 形容词层级偏好
        self.modifier_order: Dict[Tuple[str, str], Dict] = {}

        # 维度级统计
        self.dimension_stats: Dict[str, Dict] = {}

        # 复合符号
        self.compounds: Dict[str, Dict] = {}

        # 复合符号共现追踪
        self.compound_cooccurrence: Dict[Tuple[str, str], Dict] = {}

        # 统计计数
        self.total_games: int = 0
        self.total_successes: int = 0
        self.multi_symbol_games: int = 0
        self.tri_symbol_games: int = 0

    # -------------------------------------------------------------------
    # 序列化
    # -------------------------------------------------------------------

    def save_state(self) -> dict:
        """导出所有学习到的状态"""
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
        }

    def load_state(self, state: dict):
        """从导出的状态恢复"""
        import ast

        self.vocabulary = state.get('vocabulary', {})
        self.collocations = _restore_tuple_keys(state.get('collocations', {}))
        self.word_order_scores = state.get('word_order_scores', {'modifier_first': 0.0, 'head_first': 0.0})
        self.word_order_counts = state.get('word_order_counts', {'modifier_first': 0, 'head_first': 0})
        self.grammar_rules = state.get('grammar_rules', [])
        self.ngram_patterns = _restore_tuple_keys(state.get('ngram_patterns', {}))
        self.modifier_order = _restore_tuple_keys(state.get('modifier_order', {}))
        self.dimension_stats = state.get('dimension_stats', {})
        self.compounds = state.get('compounds', {})
        self.compound_cooccurrence = _restore_tuple_keys(state.get('compound_cooccurrence', {}))
        self.total_games = state.get('total_games', 0)
        self.total_successes = state.get('total_successes', 0)
        self.multi_symbol_games = state.get('multi_symbol_games', 0)
        self.tri_symbol_games = state.get('tri_symbol_games', 0)

    # -------------------------------------------------------------------
    # 记录 API
    # -------------------------------------------------------------------

    def expose_symbol(self, symbol: str):
        """记录符号暴露（像儿童听到/看到一个词）

        感受性词汇（能听懂）先于产出性词汇（能说出）增长。
        """
        if symbol not in self.vocabulary:
            self.vocabulary[symbol] = {
                'frequency': 0, 'successes': 0, 'success_rate': 0.0,
                'exposures': 1,
            }
        else:
            self.vocabulary[symbol]['exposures'] = self.vocabulary[symbol].get('exposures', 0) + 1

    def record_usage(self, symbols: List[str], success: bool):
        """记录一次符号使用"""
        for sym in symbols:
            if sym not in self.vocabulary:
                self.vocabulary[sym] = {
                    'frequency': 0, 'successes': 0, 'success_rate': 0.0,
                    'exposures': 0,
                }
            self.vocabulary[sym]['frequency'] += 1
            if success:
                self.vocabulary[sym]['successes'] += 1
            self.vocabulary[sym]['success_rate'] = (
                self.vocabulary[sym]['successes'] / self.vocabulary[sym]['frequency']
            )
            if 'exposures' not in self.vocabulary[sym]:
                self.vocabulary[sym]['exposures'] = self.vocabulary[sym]['frequency']

            # 更新复合符号的 last_used
            if sym in self.compounds:
                self.compounds[sym]['last_used'] = self.total_games

            # 维度级统计
            dim = _symbol_category(sym)
            if dim:
                if dim not in self.dimension_stats:
                    self.dimension_stats[dim] = {'frequency': 0, 'successes': 0, 'success_rate': 0.0}
                self.dimension_stats[dim]['frequency'] += 1
                if success:
                    self.dimension_stats[dim]['successes'] += 1
                self.dimension_stats[dim]['success_rate'] = (
                    self.dimension_stats[dim]['successes'] / self.dimension_stats[dim]['frequency']
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
        """记录 n-gram 模式"""
        if len(symbols) >= 2:
            key = tuple(symbols)
            if key not in self.ngram_patterns:
                self.ngram_patterns[key] = {'count': 0, 'successes': 0}
            self.ngram_patterns[key]['count'] += 1
            if success:
                self.ngram_patterns[key]['successes'] += 1

    def record_modifier_order(self, cat_a: str, cat_b: str, success: bool):
        """记录形容词层级偏好"""
        key = (cat_a, cat_b)
        if key not in self.modifier_order:
            self.modifier_order[key] = {'count': 0, 'successes': 0}
        self.modifier_order[key]['count'] += 1
        if success:
            self.modifier_order[key]['successes'] += 1

    def record_compound_cooccurrence(self, symbols: List[str], success: bool):
        """记录复合符号与其他符号的共现"""
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

    # -------------------------------------------------------------------
    # 复合符号
    # -------------------------------------------------------------------

    def check_compound_formation(self, symbols: List[str], success: bool):
        """检查符号序列是否应该形成复合符号"""
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

        # 路径 2：递归复合
        for sym in symbols:
            if not _is_compound(sym):
                continue
            comp_data = self.compounds.get(sym)
            if comp_data is None:
                continue
            for other in symbols:
                if other == sym or _is_compound(other):
                    continue
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

    def prune_compounds(self, max_age: int = 100) -> int:
        """淘汰长期不使用的复合符号"""
        to_remove = []
        for sym, data in self.compounds.items():
            last_used = data.get('last_used', 0)
            if self.total_games - last_used > max_age:
                to_remove.append(sym)
        for sym in to_remove:
            del self.compounds[sym]
        return len(to_remove)

    def prune_vocabulary(self, min_usage: int = 3, max_age: int = 200) -> int:
        """剪枝低频/过时词汇，返回移除数量"""
        to_remove = []
        for sym, data in list(self.vocabulary.items()):
            if data.get('usage_count', 0) < min_usage and \
               self.total_games - data.get('last_used', 0) > max_age:
                to_remove.append(sym)
        for sym in to_remove:
            del self.vocabulary[sym]
        return len(to_remove)

    def prune_collocations(self, min_count: int = 2, max_collocations: int = 500) -> int:
        """剪枝低频搭配，限制总数"""
        to_remove = []
        for pair, data in list(self.collocations.items()):
            if data.get('count', 0) < min_count:
                to_remove.append(pair)
        for pair in to_remove:
            del self.collocations[pair]
        # 如果仍超过上限，保留高频的
        if len(self.collocations) > max_collocations:
            sorted_pairs = sorted(
                self.collocations.items(),
                key=lambda x: x[1].get('count', 0),
                reverse=True,
            )
            self.collocations = dict(sorted_pairs[:max_collocations])
        return len(to_remove)

    def prune_ngrams(self, max_patterns: int = 300) -> int:
        """剪枝 n-gram 模式，保留高频的"""
        if len(self.ngram_patterns) <= max_patterns:
            return 0
        sorted_ngrams = sorted(
            self.ngram_patterns.items(),
            key=lambda x: x[1].get('count', 0),
            reverse=True,
        )
        removed = len(self.ngram_patterns) - max_patterns
        self.ngram_patterns = dict(sorted_ngrams[:max_patterns])
        return removed

    def prune_all(self) -> Dict[str, int]:
        """一键剪枝所有可增长数据结构"""
        return {
            'compounds': self.prune_compounds(),
            'vocabulary': self.prune_vocabulary(),
            'collocations': self.prune_collocations(),
            'ngrams': self.prune_ngrams(),
        }

    # -------------------------------------------------------------------
    # 文化变异
    # -------------------------------------------------------------------

    def mutate(self, mutation_rate: float = 0.05,
               borrow_from: Optional['EmergingLanguage'] = None):
        """语言变异算子（用于文化演化模拟）"""
        import random as _random

        # 1. 简化：高频复合符号 -> 缩写
        for sym, data in list(self.compounds.items()):
            if data['frequency'] >= 10 and _random.random() < mutation_rate:
                components = data['components']
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
                data['success_rate'] = max(0.0, min(1.0, data['success_rate'] + drift))

    # -------------------------------------------------------------------
    # 查询 API
    # -------------------------------------------------------------------

    def get_preferred_order(self) -> str:
        """获取偏好词序"""
        if sum(self.word_order_counts.values()) < 5:
            return 'modifier_first'
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
        if self.total_games == 0:
            return 0.0
        return self.multi_symbol_games / self.total_games

    def get_preferred_modifier_order(self) -> List[str]:
        """获取偏好形容词层级排序"""
        if not self.modifier_order:
            return DEFAULT_MODIFIER_ORDER
        rates = {}
        for (cat_a, cat_b), stats in self.modifier_order.items():
            if stats['count'] >= 3:
                rate = stats['successes'] / stats['count']
                rates[(cat_a, cat_b)] = rate
        if not rates:
            return DEFAULT_MODIFIER_ORDER
        cat_wins = defaultdict(float)
        for (cat_a, cat_b), rate in rates.items():
            cat_wins[cat_a] += rate
        sorted_cats = sorted(cat_wins.keys(), key=lambda c: cat_wins[c], reverse=True)
        return sorted_cats

    def extract_grammar_rules(self, min_confidence: float = 0.6) -> List[Dict]:
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

    def get_stats(self) -> Dict:
        return {
            'vocabulary_size': self.get_vocabulary_size(),
            'total_games': self.total_games,
            'total_successes': self.total_successes,
            'success_rate': self.total_successes / max(1, self.total_games),
            'combination_rate': self.get_combination_rate(),
            'tri_symbol_rate': self.tri_symbol_games / max(1, self.total_games),
            'order_consistency': self.get_order_consistency(),
            'preferred_order': self.get_preferred_order(),
            'preferred_modifier_order': self.get_preferred_modifier_order(),
            'grammar_rules': len(self.grammar_rules),
            'ngram_patterns': len(self.ngram_patterns),
        }


# ---------------------------------------------------------------------------
# 辅助函数：序列化 tuple key 恢复
# ---------------------------------------------------------------------------

def _restore_tuple_keys(raw: dict) -> dict:
    """将序列化后的字符串 key 恢复为 tuple key"""
    import ast
    result = {}
    for k, v in raw.items():
        if isinstance(k, str):
            try:
                key = ast.literal_eval(k)
            except (ValueError, SyntaxError):
                key = k
        else:
            key = k
        result[key] = v
    return result
