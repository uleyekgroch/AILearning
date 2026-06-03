"""学习策略选择器 — 根据难度和缺口选择学习策略"""

from dataclasses import dataclass


@dataclass
class LearningStrategy:
    """学习策略"""
    name: str
    description: str
    effectiveness: float  # 0-1


class StrategySelector:
    """学习策略选择器

    根据当前难度和知识缺口，选择最合适的学习策略，
    并跟踪每种策略的实际效果。
    """

    STRATEGIES = {
        'explore': LearningStrategy('explore', '好奇心驱动探索', 0.5),
        'practice': LearningStrategy('practice', '重复练习已学内容', 0.7),
        'analogize': LearningStrategy('analogize', '用类比学习新概念', 0.6),
        'decompose': LearningStrategy('decompose', '分解为子问题', 0.8),
        'seek_help': LearningStrategy('seek_help', '寻求社会帮助', 0.6),
        'hypothesize': LearningStrategy('hypothesize', '提出并验证假设', 0.7),
    }

    def __init__(self):
        # 跟踪每种策略的使用次数和成功次数
        self._usage: dict = {name: 0 for name in self.STRATEGIES}
        self._success: dict = {name: 0 for name in self.STRATEGIES}

    def choose_strategy(self, difficulty: float, gaps: list = None) -> LearningStrategy:
        """根据难度和缺口选择策略

        决策逻辑：
        - difficulty >= 0.7 → decompose（分解难题）
        - 缺口多（>= 5） → practice（集中练习）
        - difficulty >= 0.5 且缺口 3-4 → analogize（用类比）
        - difficulty < 0.3 → explore（探索新领域）
        - 缺口较少但难度中等 → hypothesize（假设驱动）
        - 默认 → explore
        """
        if gaps is None:
            gaps = []

        gap_count = len(gaps)

        # 高难度：分解
        if difficulty >= 0.7:
            return self._pick('decompose')

        # 缺口多：集中练习
        if gap_count >= 5:
            return self._pick('practice')

        # 中等难度 + 一定缺口：类比
        if difficulty >= 0.5 and gap_count >= 3:
            return self._pick('analogize')

        # 低难度：探索
        if difficulty < 0.3:
            return self._pick('explore')

        # 中等难度，少量缺口：假设驱动
        if gap_count >= 1:
            return self._pick('hypothesize')

        # 默认探索
        return self._pick('explore')

    def evaluate_effectiveness(self, strategy_name: str, result: bool) -> None:
        """根据实际学习结果更新策略效果

        使用指数移动平均 (EMA) 平滑更新 effectiveness。
        """
        if strategy_name not in self.STRATEGIES:
            return

        self._usage[strategy_name] += 1
        if result:
            self._success[strategy_name] += 1

        # 基于 EMA 更新 effectiveness
        alpha = 0.3  # 平滑系数
        old_eff = self.STRATEGIES[strategy_name].effectiveness
        new_signal = 1.0 if result else 0.0
        self.STRATEGIES[strategy_name].effectiveness = round(
            alpha * new_signal + (1 - alpha) * old_eff, 4
        )

    def get_strategy_stats(self) -> dict:
        """返回每种策略的使用统计"""
        stats = {}
        for name in self.STRATEGIES:
            usage = self._usage[name]
            success = self._success[name]
            stats[name] = {
                'usage': usage,
                'success': success,
                'success_rate': round(success / usage, 4) if usage > 0 else 0.0,
                'effectiveness': self.STRATEGIES[name].effectiveness,
            }
        return stats

    def _pick(self, name: str) -> LearningStrategy:
        """选策略并记录使用"""
        self._usage[name] += 1
        return self.STRATEGIES[name]
