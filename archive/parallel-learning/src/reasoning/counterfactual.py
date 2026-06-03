"""反事实推理 — 从替代模拟中涌现 "if/would/instead" 标记

核心思想：
反事实思维是人类独特认知里程碑（4-6 岁出现）。
当环境存在分支结果时（动作 A→效果 X，但如果做 B 则→效果 Y），
agent 模拟替代动作的后果，用反事实标记描述假设场景。

CounterfactualWorld 存储每个动作的多个可能结果及概率，
CounterfactualModule 从中推理替代情景并选择标记词。

源自 mvl/experiment_counterfactual.py，numpy→torch 转换，适配 DDD 架构。
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

# 反事实标记词集合
COUNTERFACTUAL_MARKERS = {'if', 'would', 'instead', 'otherwise'}


class CounterfactualWorld:
    """反事实世界 — 每个动作有多个可能结果

    存储格式：action -> [(effect, probability), ...]
    execute(action) 按概率采样一个结果。
    simulate_alternative(actual, alt) 同时执行两个动作，返回两种效果。
    """

    def __init__(self):
        self.action_effects: Dict[str, List[Tuple[str, float]]] = {}
        self._event_log: List[Dict] = []

    def add_action_effect(self, action: str, effect: str,
                          probability: float) -> None:
        """添加动作-效果对及其概率"""
        if action not in self.action_effects:
            self.action_effects[action] = []
        self.action_effects[action].append((effect, probability))

    def execute(self, action: str) -> Tuple[str, bool]:
        """执行动作，按概率采样返回 (效果, 是否因果发生)"""
        if action not in self.action_effects:
            return ('nothing', False)

        effects = self.action_effects[action]
        r = torch.rand(1).item()
        cumulative = 0.0
        for effect, prob in effects:
            cumulative += prob
            if r < cumulative:
                self._event_log.append({
                    'action': action, 'effect': effect, 'causal': True,
                })
                return (effect, True)

        self._event_log.append({
            'action': action, 'effect': 'nothing', 'causal': False,
        })
        return ('nothing', False)

    def simulate_alternative(self, actual: str,
                             alt: str) -> Tuple[str, str]:
        """模拟替代动作 — 返回 (实际效果, 反事实效果)"""
        actual_effect, _ = self.execute(actual)
        cf_effect, _ = self.execute(alt)
        return (actual_effect, cf_effect)

    def get_all_effects(self, action: str) -> List[str]:
        """获取动作的所有可能效果"""
        if action not in self.action_effects:
            return ['nothing']
        return [e for e, _ in self.action_effects[action]]

    def get_primary_effect(self, action: str) -> Optional[str]:
        """获取动作的主效果（最高概率）"""
        if action not in self.action_effects:
            return None
        effects = sorted(
            self.action_effects[action], key=lambda x: x[1], reverse=True
        )
        return effects[0][0]

    def save_state(self) -> dict:
        serialized = {}
        for action, effects in self.action_effects.items():
            serialized[action] = [
                {'effect': e, 'probability': p} for e, p in effects
            ]
        return {'action_effects': serialized}

    def load_state(self, state: dict) -> None:
        self.action_effects.clear()
        for action, entries in state.get('action_effects', {}).items():
            self.action_effects[action] = [
                (e['effect'], e['probability']) for e in entries
            ]


class CounterfactualModule:
    """反事实推理模块 — 模拟替代结果并选择标记

    核心循环：
    1. reason(actual, effect, world) — 模拟替代动作的结果
    2. choose_marker(actual, counterfactual) — 根据效果差异选择反事实标记
    3. update_marker_success(marker, success) — 更新标记使用统计
    """

    def __init__(self):
        self.marker_stats: Dict[str, Dict] = defaultdict(
            lambda: {'frequency': 0, 'successes': 0, 'success_rate': 0.0}
        )
        self._cf_count: int = 0
        self._then_count: int = 0

    def reason(self, actual_action: str, actual_effect: str,
               world: CounterfactualWorld) -> Optional[str]:
        """反事实推理：模拟替代动作的结果

        从 world 中选择一个与实际动作不同的替代动作，
        执行模拟并返回反事实效果。如果替代效果与实际效果
        相同或为 'nothing'，返回 None。
        """
        available = list(world.action_effects.keys())
        alternatives = [a for a in available if a != actual_action]
        if not alternatives:
            return None

        # 随机选择替代动作
        idx = torch.randint(0, len(alternatives), (1,)).item()
        alt_action = alternatives[idx]

        _, cf_effect = world.simulate_alternative(actual_action, alt_action)

        if cf_effect != actual_effect and cf_effect != 'nothing':
            return cf_effect
        return None

    def choose_marker(self, actual: str, counterfactual: str) -> str:
        """根据效果对比选择反事实标记

        效果相同 → "then"（非反事实，只是时序）
        效果不同 → 从 {if, would, instead} 中按历史成功率加权选择
        """
        self._cf_count += 1

        if actual == counterfactual:
            self._then_count += 1
            return 'then'

        # 按历史成功率加权选择标记
        candidates = ['if', 'would', 'instead']
        weights = []
        for m in candidates:
            stats = self.marker_stats[m]
            # 成功率 + 基础探索权重
            rate = stats['success_rate']
            weight = max(rate, 0.1)  # 最低 0.1 保证探索
            weights.append(weight)

        # 归一化并采样
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        # 使用 torch 做加权随机选择
        idx = torch.multinomial(
            torch.tensor(probs, dtype=torch.float32), 1
        ).item()

        marker = candidates[idx]
        self.marker_stats[marker]['frequency'] += 1
        return marker

    def update_marker_success(self, marker: str, success: bool) -> None:
        """更新标记成功率统计"""
        self.marker_stats[marker]['frequency'] += 1
        if success:
            self.marker_stats[marker]['successes'] += 1
        freq = self.marker_stats[marker]['frequency']
        succ = self.marker_stats[marker]['successes']
        if freq > 0:
            self.marker_stats[marker]['success_rate'] = succ / freq

    def get_stats(self) -> dict:
        return {
            'cf_count': self._cf_count,
            'then_count': self._then_count,
            'marker_stats': {
                m: dict(stats) for m, stats in self.marker_stats.items()
            },
        }

    def save_state(self) -> dict:
        return {
            'marker_stats': {
                m: dict(stats) for m, stats in self.marker_stats.items()
            },
            'cf_count': self._cf_count,
            'then_count': self._then_count,
        }

    def load_state(self, state: dict) -> None:
        self.marker_stats.clear()
        for m, stats in state.get('marker_stats', {}).items():
            self.marker_stats[m] = dict(stats)
        self._cf_count = state.get('cf_count', 0)
        self._then_count = state.get('then_count', 0)
