"""心智理论 — 建模其他 Agent 的知识状态和信念

核心思想：
心智理论 = 理解他人的认知状态（知识、信念、注意焦点）。
当信息不对称时，Speaker 必须建模 Listener 知道什么、不知道什么，
才能选择有效的描述策略。视角标记词（know/think/believe）从这种需要中涌现。

源自 mvl/grounding_theory_of_mind.py，numpy→torch 转换，适配 DDD 架构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

# 视角标记词集合
PERSPECTIVE_MARKERS = {'know', 'think', 'believe', 'sure', 'doubt'}


class Perspective:
    """Agent 的认知状态 — 已知事实、信念置信度、注意焦点

    贝叶斯信念更新：
    posterior = (prior * likelihood) / (prior * likelihood + (1-prior) * (1-likelihood))
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.known_facts: Set[str] = set()
        self.beliefs: Dict[str, float] = {}  # 命题 -> 置信度
        self.attention: Set[str] = set()
        self._observation_count: int = 0

    def observe(self, fact: str, confidence: float = 1.0) -> None:
        """观察一个事实，记录到已知事实集和信念字典"""
        self.known_facts.add(fact)
        self.beliefs[fact] = confidence
        self._observation_count += 1

    def update_belief(self, proposition: str, evidence: float) -> None:
        """贝叶斯信念更新

        evidence 为似然度（当前证据支持命题的程度），
        与先验结合得到后验置信度。
        """
        if proposition in self.beliefs:
            prior = self.beliefs[proposition]
            # 贝叶斯公式：P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|~H)*P(~H)]
            posterior = (prior * evidence) / (
                prior * evidence + (1 - prior) * (1 - evidence) + 1e-12
            )
            self.beliefs[proposition] = posterior
        else:
            self.beliefs[proposition] = evidence

    def knows(self, fact: str) -> bool:
        """是否知道某个事实"""
        return fact in self.known_facts

    def get_confidence(self, proposition: str) -> float:
        """获取对命题的置信度，未知命题返回 0.5（中性）"""
        return self.beliefs.get(proposition, 0.5)

    def forget(self, fact: str) -> None:
        """遗忘一个事实（模拟过时信息）"""
        self.known_facts.discard(fact)
        self.beliefs.pop(fact, None)

    def save_state(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'known_facts': sorted(self.known_facts),
            'beliefs': self.beliefs,
            'attention': sorted(self.attention),
            'observation_count': self._observation_count,
        }

    def load_state(self, state: dict) -> None:
        self.agent_id = state['agent_id']
        self.known_facts = set(state.get('known_facts', []))
        self.beliefs = state.get('beliefs', {})
        self.attention = set(state.get('attention', []))
        self._observation_count = state.get('observation_count', 0)


class TheoryOfMindModule:
    """心智理论模块 — 建模其他 Agent 的知识/信念状态

    核心能力：
    1. model_other — 为其他 Agent 建立认知模型
    2. estimate_other_knowledge — 估计他人是否知道某事实
    3. detect_asymmetry — 检测知识不对称（自己知道但他人不知道）
    4. adjust_description — 根据他人知识调整描述策略
    5. choose_perspective_marker — 根据确定性选择视角标记词
    """

    def __init__(self):
        self.own_perspective = Perspective('self')
        self.other_models: Dict[str, Perspective] = {}
        self._adjustment_count: int = 0
        self._asymmetry_count: int = 0
        self._marker_usage: Dict[str, int] = {}

    def model_other(self, other_id: str, observed_facts: Set[str]) -> None:
        """为另一个 Agent 建立认知模型，记录其可观察到的事实"""
        if other_id not in self.other_models:
            self.other_models[other_id] = Perspective(other_id)
        other_p = self.other_models[other_id]
        for fact in observed_facts:
            other_p.observe(fact)

    def estimate_other_knowledge(self, other_id: str, fact: str) -> float:
        """估计另一个 Agent 是否知道某事实

        返回 1.0（确定知道）、0.0（确定不知道）、0.5（未知）
        """
        if other_id not in self.other_models:
            return 0.5
        other_p = self.other_models[other_id]
        if other_p.knows(fact):
            return 1.0
        return 0.0

    def detect_asymmetry(self, other_id: str, fact: str) -> bool:
        """检测知识不对称：自己知道但他人不知道"""
        self_knows = self.own_perspective.knows(fact)
        other_knows = self.estimate_other_knowledge(other_id, fact) > 0.5
        if self_knows and not other_knows:
            self._asymmetry_count += 1
            return True
        return False

    def adjust_description(self, target_features: Dict[str, str],
                           scene: List[Dict], other_id: str) -> List[str]:
        """根据他人的知识调整描述

        策略：过滤掉他人不知道的特征，只保留对方能理解的部分。
        如果对方完全不了解目标，则回退到原始特征描述。
        """
        target_values = set(target_features.values())

        # 筛选对方知道的特征
        known_to_other = [
            val for val in target_values
            if self.estimate_other_knowledge(other_id, val) > 0.5
        ]

        if known_to_other:
            self._adjustment_count += 1
            return known_to_other[:2]

        # 回退：使用原始特征
        return list(target_values)[:2]

    def choose_perspective_marker(self, certainty: float,
                                   language_experience: int = 0) -> Optional[str]:
        """根据确定性选择视角标记词

        高确定性(>0.8) → "know"
        中确定性(>0.5) → "think"
        低确定性(<=0.5) → "believe"

        language_experience 控制标记词的探索-利用策略：
        经验不足时随机尝试，经验充足后基于确定性选择。
        """
        # 经验不足时不使用标记
        if language_experience < 10:
            return None

        # 探索阶段：30% 概率随机尝试
        if language_experience < 50:
            if torch.rand(1).item() < 0.3:
                return self._pick_marker_by_certainty(certainty)
            return None

        # 利用阶段：基于确定性选择
        return self._pick_marker_by_certainty(certainty)

    def _pick_marker_by_certainty(self, certainty: float) -> str:
        if certainty > 0.8:
            marker = 'know'
        elif certainty > 0.5:
            marker = 'think'
        else:
            marker = 'believe'
        self._marker_usage[marker] = self._marker_usage.get(marker, 0) + 1
        return marker

    def get_stats(self) -> dict:
        return {
            'adjustment_count': self._adjustment_count,
            'asymmetry_count': self._asymmetry_count,
            'other_models_count': len(self.other_models),
            'marker_usage': dict(self._marker_usage),
        }

    def save_state(self) -> dict:
        return {
            'own_perspective': self.own_perspective.save_state(),
            'other_models': {
                oid: p.save_state() for oid, p in self.other_models.items()
            },
            'adjustment_count': self._adjustment_count,
            'asymmetry_count': self._asymmetry_count,
            'marker_usage': self._marker_usage,
        }

    def load_state(self, state: dict) -> None:
        self.own_perspective.load_state(state.get('own_perspective', {}))
        self.other_models.clear()
        for oid, pdata in state.get('other_models', {}).items():
            p = Perspective(oid)
            p.load_state(pdata)
            self.other_models[oid] = p
        self._adjustment_count = state.get('adjustment_count', 0)
        self._asymmetry_count = state.get('asymmetry_count', 0)
        self._marker_usage = state.get('marker_usage', {})
