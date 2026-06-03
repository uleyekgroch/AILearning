"""
语法系统：从交流经验中提取和应用语法规则

从 emergence.py 提取语法相关逻辑，独立为 GrammarSystem 类。
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.language.emergence import _symbol_category


@dataclass
class GrammarRule:
    """一条语法规则"""
    pattern: Tuple[str, ...]   # 符号模式
    category_pattern: Tuple[str, ...]  # 类别模式
    confidence: float = 0.0
    count: int = 0


class GrammarSystem:
    """
    语法系统：管理词序偏好和语法规则

    从成功的交流模式中提取规则，应用于新话语的排列。
    """

    def __init__(self):
        self.rules: List[GrammarRule] = []
        self.category_order_scores: Dict[Tuple[str, str], Dict] = {}
        self.word_order_scores: Dict[str, float] = {'modifier_first': 0.0, 'head_first': 0.0}
        self.word_order_counts: Dict[str, int] = {'modifier_first': 0, 'head_first': 0}

    # -------------------------------------------------------------------
    # 学习
    # -------------------------------------------------------------------

    def learn_pattern(self, symbols: List[str], success: bool):
        """
        从成功/失败的模式中学习语法规则

        记录：
        1. 符号类别的顺序关系
        2. 修饰语-中心词顺序
        """
        if len(symbols) < 2:
            return

        # 记录相邻类别对的顺序
        categories = [_symbol_category(s) or 'other' for s in symbols]

        for i in range(len(categories) - 1):
            cat_a, cat_b = categories[i], categories[i + 1]
            if cat_a != cat_b:
                key = (cat_a, cat_b)
                if key not in self.category_order_scores:
                    self.category_order_scores[key] = {'count': 0, 'successes': 0}
                self.category_order_scores[key]['count'] += 1
                if success:
                    self.category_order_scores[key]['successes'] += 1

        # 检测词序
        order = self._detect_order(categories)
        if order:
            self.word_order_counts[order] += 1
            if success:
                self.word_order_scores[order] += 1.0

        # 从成功模式中提取规则
        if success:
            self._extract_rule(symbols, categories)

    def apply_rules(self, symbols: List[str]) -> List[str]:
        """应用已学规则排列符号"""
        if len(symbols) <= 1:
            return symbols

        # 获取类别排序
        preferred_cats = self._get_preferred_category_order()

        by_cat = defaultdict(list)
        for sym in symbols:
            cat = _symbol_category(sym) or 'other'
            by_cat[cat].append(sym)

        result = []
        for cat in preferred_cats:
            result.extend(by_cat.pop(cat, []))
        # 剩余类别
        for remaining in by_cat.values():
            result.extend(remaining)

        return result

    # -------------------------------------------------------------------
    # 查询
    # -------------------------------------------------------------------

    def get_rules(self) -> List[GrammarRule]:
        """获取所有已学规则"""
        return sorted(self.rules, key=lambda r: r.confidence, reverse=True)

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

    # -------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------

    def _detect_order(self, categories: List[str]) -> Optional[str]:
        """检测词序类型"""
        if 'shape' in categories:
            head_idx = categories.index('shape')
            modifiers_before = head_idx
            modifiers_after = len(categories) - head_idx - 1

            if modifiers_before > 0 and modifiers_after == 0:
                return 'modifier_first'
            if modifiers_after > 0 and modifiers_before == 0:
                return 'head_first'

        if len(categories) == 2:
            cat_0, cat_1 = categories[0], categories[1]
            if cat_0 and cat_1 and cat_0 != cat_1:
                if cat_0 == 'shape':
                    return 'head_first'
                if cat_1 == 'shape':
                    return 'modifier_first'

        return None

    def _extract_rule(self, symbols: List[str], categories: List[str]):
        """从成功模式中提取规则"""
        cat_pattern = tuple(categories)
        sym_pattern = tuple(symbols)

        # 检查是否已有相同类别模式的规则
        for rule in self.rules:
            if rule.category_pattern == cat_pattern:
                rule.count += 1
                rule.confidence = rule.count / (rule.count + 1)  # 简化置信度
                return

        self.rules.append(GrammarRule(
            pattern=sym_pattern,
            category_pattern=cat_pattern,
            confidence=0.5,
            count=1,
        ))

    def _get_preferred_category_order(self) -> List[str]:
        """获取偏好的类别排序"""
        if not self.category_order_scores:
            return ['size', 'color', 'material', 'shape']

        rates = {}
        for (cat_a, cat_b), stats in self.category_order_scores.items():
            if stats['count'] >= 3:
                rate = stats['successes'] / stats['count']
                rates[(cat_a, cat_b)] = rate

        if not rates:
            return ['size', 'color', 'material', 'shape']

        cat_wins = defaultdict(float)
        for (cat_a, cat_b), rate in rates.items():
            cat_wins[cat_a] += rate

        sorted_cats = sorted(cat_wins.keys(), key=lambda c: cat_wins[c], reverse=True)
        return sorted_cats
