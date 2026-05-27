"""
因果接地模块：因果符号从事件序列模式中涌现

核心思想：
因果标记（if, then, because, so）的意义来自事件之间的统计依赖。
"because" = 高置信度的 cause→effect 规则，
"then" = 时间序列中的先后关系。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener,
    CAUSAL_MARKERS, _symbol_category, generate_rich_scene
)


@dataclass
class CausalRule:
    """一条因果规则"""
    cause: str               # 原因特征 (如 'push', 'red')
    effect: str              # 结果特征 (如 'displacement', 'moved')
    confidence: float        # 置信度 (贝叶斯后验)
    evidence_count: int      # 支持证据数
    total_observations: int  # 总观察次数

    def expression(self) -> str:
        """生成因果表达式"""
        if self.confidence > 0.8:
            return f"{self.cause} because {self.effect}"
        elif self.confidence > 0.6:
            return f"{self.cause} then {self.effect}"
        else:
            return f"{self.cause} if {self.effect}"

    def to_feature_dict(self) -> Dict[str, str]:
        """转换为特征字典（用于语言交流）"""
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


class CausalGroundingModule:
    """
    因果接地模块：从事件序列中学习因果规则

    使用贝叶斯更新维护规则置信度。
    每次观察到 (cause, effect) 共现，更新对应规则的置信度。
    """

    def __init__(self):
        self.rules: Dict[Tuple[str, str], CausalRule] = {}
        self.event_history: List[Tuple[str, str]] = []

    def observe(self, cause: str, effect: str):
        """观察一次因果事件"""
        self.event_history.append((cause, effect))
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

        # 贝叶斯更新：P(cause→effect | evidence)
        # 使用 Beta 分布后验
        alpha = 1 + rule.evidence_count  # 成功次数 + 先验
        beta_param = 1 + (rule.total_observations - rule.evidence_count)  # 失败次数 + 先验
        rule.confidence = alpha / (alpha + beta_param)

    def observe_non_occurrence(self, cause: str, expected_effect: str):
        """观察到 cause 发生但 expected_effect 未出现"""
        key = (cause, expected_effect)
        if key in self.rules:
            rule = self.rules[key]
            rule.total_observations += 1
            alpha = 1 + rule.evidence_count
            beta_param = 1 + (rule.total_observations - rule.evidence_count)
            rule.confidence = alpha / (alpha + beta_param)

    def get_confident_rules(self, threshold: float = 0.6) -> List[CausalRule]:
        """获取置信度超过阈值的规则"""
        return [r for r in self.rules.values() if r.confidence >= threshold]

    def get_strongest_rules(self, n: int = 5) -> List[CausalRule]:
        """获取置信度最高的 n 条规则"""
        sorted_rules = sorted(self.rules.values(), key=lambda r: r.confidence, reverse=True)
        return sorted_rules[:n]

    def express_causal(self, cause: str, effect: str) -> Optional[str]:
        """用因果标记表达因果关系"""
        key = (cause, effect)
        if key not in self.rules:
            return None
        rule = self.rules[key]
        return rule.expression()

    def get_stats(self) -> Dict:
        total = len(self.rules)
        confident = len(self.get_confident_rules())
        strongest = self.get_strongest_rules(3)
        return {
            'total_rules': total,
            'confident_rules': confident,
            'strongest': [(r.cause, r.effect, r.confidence) for r in strongest],
            'event_history_length': len(self.event_history),
        }


@dataclass
class CausalExpression:
    """因果表达式：将因果规则编码为可交流的特征"""
    cause_feature: str       # 原因的特征值
    effect_feature: str      # 结果的特征值
    marker: str              # 因果标记 (because/then/so/if/when)

    def to_feature_dict(self) -> Dict[str, str]:
        return {
            'causal_cause': self.cause_feature,
            'causal_effect': self.effect_feature,
            'causal_marker': self.marker,
        }

    @staticmethod
    def from_rule(rule: CausalRule) -> 'CausalExpression':
        if rule.confidence > 0.8:
            marker = 'because'
        elif rule.confidence > 0.6:
            marker = 'then'
        elif rule.confidence > 0.4:
            marker = 'so'
        else:
            marker = 'if'
        return CausalExpression(
            cause_feature=rule.cause,
            effect_feature=rule.effect,
            marker=marker,
        )


def generate_causal_scene(num_objects: int = 6,
                           complexity: str = 'medium') -> Tuple[List[Dict[str, str]], List[CausalRule]]:
    """生成带因果标注的场景"""
    base_scene = generate_rich_scene(complexity)[:num_objects]

    # 生成因果规则
    action_types = ['push', 'pull', 'grab', 'drop', 'move', 'go', 'stop', 'turn']
    effects = ['displacement', 'attraction', 'attached', 'detached',
               'translation', 'directional', 'halt', 'rotation']

    causal_module = CausalGroundingModule()
    causal_expressions = []

    for i, obj_features in enumerate(base_scene):
        if np.random.random() < 0.4:
            action = np.random.choice(action_types)
            effect = np.random.choice(effects)

            # 模拟观察：action 有 70% 概率导致对应 effect
            if np.random.random() < 0.7:
                causal_module.observe(action, effect)
            else:
                causal_module.observe_non_occurrence(action, effect)

            # 获取置信度最高的规则
            key = (action, effect)
            if key in causal_module.rules:
                rule = causal_module.rules[key]
                expr = CausalExpression.from_rule(rule)

                # 将因果信息注入物体特征
                obj_features['causal_cause'] = action
                obj_features['causal_effect'] = effect
                obj_features['causal_marker'] = expr.marker

                causal_expressions.append(rule)

    return base_scene, causal_expressions


class CausalCommunicationGame:
    """因果交流游戏：描述"为什么发生了什么"，listener 识别目标"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.causal_module = CausalGroundingModule()
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   causal_rules: List[CausalRule],
                   target_rule_idx: int) -> bool:
        if target_rule_idx >= len(causal_rules):
            return False

        target_rule = causal_rules[target_rule_idx]

        # 将因果信息注入场景
        enriched_scene = []
        for i, obj in enumerate(scene_features):
            if i < len(causal_rules):
                rule = causal_rules[i]
                expr = CausalExpression.from_rule(rule)
                enriched_scene.append({**obj, **expr.to_feature_dict()})
            else:
                enriched_scene.append(obj)

        # speaker 描述目标
        target_obj_idx = min(target_rule_idx, len(scene_features) - 1)
        target = enriched_scene[target_obj_idx]
        utterance = self.speaker.describe(target, enriched_scene)
        if not utterance:
            return False

        # listener 解释
        chosen_idx = self.listener.interpret(utterance, enriched_scene)
        success = (chosen_idx == target_obj_idx)

        # 更新语言统计
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
            'target_rule': (target_rule.cause, target_rule.effect, target_rule.confidence),
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success


class CausalLanguageAgent:
    """拥有独立因果语言的 agent"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.causal_module = CausalGroundingModule()

    def learn_causal(self, cause: str, effect: str):
        """学习一条因果规则"""
        self.causal_module.observe(cause, effect)

    def speak(self, scene: List[Dict[str, str]], target_idx: int,
              causal_rules: List[CausalRule]) -> List[str]:
        """描述目标（包含因果信息）"""
        enriched = []
        for i, obj in enumerate(scene):
            if i < len(causal_rules):
                rule = causal_rules[i]
                expr = CausalExpression.from_rule(rule)
                enriched.append({**obj, **expr.to_feature_dict()})
            else:
                enriched.append(obj)

        if target_idx < len(enriched):
            return self.speaker.describe(enriched[target_idx], enriched)
        return []

    def listen(self, utterance: List[str], scene: List[Dict[str, str]]) -> Optional[int]:
        return self.listener.interpret(utterance, scene)

    def update(self, utterance: List[str], success: bool):
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                self.language.record_collocation(utterance[i], utterance[i + 1], success)
            self.language.record_ngram(utterance, success)


def cross_causal_round(speaker_agent: CausalLanguageAgent,
                       listener_agent: CausalLanguageAgent,
                       scene: List[Dict[str, str]],
                       target_idx: int,
                       causal_rules: List[CausalRule]) -> bool:
    """跨语言因果交流一轮"""
    utterance = speaker_agent.speak(scene, target_idx, causal_rules)
    if not utterance:
        return False

    # listener 用 speaker 的因果知识来解释
    enriched = []
    for i, obj in enumerate(scene):
        if i < len(causal_rules):
            rule = causal_rules[i]
            expr = CausalExpression.from_rule(rule)
            enriched.append({**obj, **expr.to_feature_dict()})
        else:
            enriched.append(obj)

    chosen_idx = listener_agent.listen(utterance, enriched)
    success = (chosen_idx == target_idx)

    speaker_agent.update(utterance, success)
    listener_agent.update(utterance, success)
    return success
