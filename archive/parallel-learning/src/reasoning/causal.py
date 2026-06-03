"""因果推理 — 贝叶斯因果规则学习

核心思想：
"because" 从区分因果和时序中涌现。系统维护 Beta(α+成功, β+失败) 后验：
当观察到 (cause, effect) 共现时 α 增加，观察到 cause 出现但 effect 未出现时 β 增加。
置信度 = α/(α+β)。高置信度使用 "because"，中置信度使用 "then"，低置信度使用 "if"。

源自 mvl/grounding_causal.py，numpy→torch 转换，适配 DDD 架构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class CausalRule:
    """一条因果规则 — Beta 后验置信度

    置信度计算：alpha / (alpha + beta)
    其中 alpha = 1 + evidence_count (先验 + 成功次数)
         beta  = 1 + (total - evidence_count) (先验 + 失败次数)
    """
    cause: str
    effect: str
    confidence: float   # Beta 后验均值
    evidence_count: int  # 成功次数
    total_observations: int  # 总观察次数

    def expression(self) -> str:
        """根据置信度生成因果表达式

        > 0.8 → "X because Y"（强因果）
        > 0.6 → "X then Y"（时序相关）
        其他  → "X if Y"（弱关联）
        """
        if self.confidence > 0.8:
            return f"{self.cause} because {self.effect}"
        elif self.confidence > 0.6:
            return f"{self.cause} then {self.effect}"
        else:
            return f"{self.cause} if {self.effect}"

    def to_feature_dict(self) -> Dict[str, str]:
        """转换为特征字典（用于语言交流模块）"""
        if self.confidence > 0.8:
            marker = 'because'
        elif self.confidence > 0.6:
            marker = 'then'
        elif self.confidence > 0.4:
            marker = 'so'
        else:
            marker = 'if'
        return {
            'causal_cause': self.cause,
            'causal_effect': self.effect,
            'causal_marker': marker,
        }


class CausalReasoningModule:
    """因果推理模块 — 贝叶斯因果规则学习

    核心循环：
    1. observe(cause, effect) — 观察到因果共现，更新 Beta(alpha+1, beta)
    2. observe_non_occurrence(cause, expected) — cause 出现但 effect 未出现，更新 Beta(alpha, beta+1)
    3. get_confident_rules(threshold) — 获取置信度超过阈值的规则
    4. express_causal(cause, effect) — 用因果标记表达因果关系
    """

    def __init__(self):
        self.rules: Dict[Tuple[str, str], CausalRule] = {}
        self._event_log: List[Tuple[str, str]] = []

    def observe(self, cause: str, effect: str) -> None:
        """观察到 (cause, effect) 共现 — Beta(alpha+success, beta+failure)

        首次观察到该 (cause, effect) 对时创建新规则，先验置信度 0.5。
        每次观察后重新计算 Beta 后验均值。
        """
        self._event_log.append((cause, effect))
        key = (cause, effect)

        if key not in self.rules:
            self.rules[key] = CausalRule(
                cause=cause,
                effect=effect,
                confidence=0.5,  # 先验
                evidence_count=0,
                total_observations=0,
            )

        rule = self.rules[key]
        rule.evidence_count += 1
        rule.total_observations += 1
        self._update_confidence(rule)

    def observe_non_occurrence(self, cause: str, expected: str) -> None:
        """观察到 cause 出现但 expected 未出现

        增加 total_observations 但不增加 evidence_count，
        使 Beta 后验均值下降。
        """
        key = (cause, expected)

        if key not in self.rules:
            # 即使未成功，也创建规则记录观察
            self.rules[key] = CausalRule(
                cause=cause,
                effect=expected,
                confidence=0.5,
                evidence_count=0,
                total_observations=0,
            )

        rule = self.rules[key]
        rule.total_observations += 1
        self._update_confidence(rule)

    def _update_confidence(self, rule: CausalRule) -> None:
        """Beta 后验更新：alpha / (alpha + beta)"""
        alpha = 1 + rule.evidence_count
        beta_param = 1 + (rule.total_observations - rule.evidence_count)
        rule.confidence = alpha / (alpha + beta_param)

    def get_confident_rules(self, threshold: float = 0.6) -> List[CausalRule]:
        """获取置信度超过阈值的规则"""
        return [r for r in self.rules.values() if r.confidence >= threshold]

    def get_strongest_rules(self, n: int = 5) -> List[CausalRule]:
        """获取置信度最高的 n 条规则"""
        sorted_rules = sorted(
            self.rules.values(), key=lambda r: r.confidence, reverse=True
        )
        return sorted_rules[:n]

    def express_causal(self, cause: str, effect: str) -> Optional[str]:
        """用因果标记表达因果关系

        如果该 (cause, effect) 对有记录规则，返回对应表达式；
        否则返回 None。
        """
        key = (cause, effect)
        if key not in self.rules:
            return None
        return self.rules[key].expression()

    def get_confidence(self, cause: str, effect: str) -> float:
        """获取特定因果对的置信度，未知返回 0.5"""
        key = (cause, effect)
        if key not in self.rules:
            return 0.5
        return self.rules[key].confidence

    def get_stats(self) -> dict:
        total = len(self.rules)
        confident = len(self.get_confident_rules())
        strongest = self.get_strongest_rules(3)
        return {
            'total_rules': total,
            'confident_rules': confident,
            'strongest': [
                (r.cause, r.effect, r.confidence) for r in strongest
            ],
            'event_log_length': len(self._event_log),
        }

    def save_state(self) -> dict:
        rules_data = {}
        for (cause, effect), rule in self.rules.items():
            rules_data[f"{cause}|{effect}"] = {
                'cause': rule.cause,
                'effect': rule.effect,
                'confidence': rule.confidence,
                'evidence_count': rule.evidence_count,
                'total_observations': rule.total_observations,
            }
        return {
            'rules': rules_data,
            'event_log': self._event_log[-200:],  # 只保留最近 200 条
        }

    def load_state(self, state: dict) -> None:
        self.rules.clear()
        for _key, rd in state.get('rules', {}).items():
            cause = rd['cause']
            effect = rd['effect']
            rule = CausalRule(
                cause=cause,
                effect=effect,
                confidence=rd['confidence'],
                evidence_count=rd['evidence_count'],
                total_observations=rd['total_observations'],
            )
            self.rules[(cause, effect)] = rule
        self._event_log = [
            tuple(pair) for pair in state.get('event_log', [])
        ]
