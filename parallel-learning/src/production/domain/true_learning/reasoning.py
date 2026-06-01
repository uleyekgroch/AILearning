"""
推理引擎 - 真正的推理

不是简单的检索，而是逻辑推理：
1. 演绎推理 - 从一般到特殊
2. 归纳推理 - 从特殊到一般
3. 类比推理 - 基于相似性
4. 因果推理 - 基于因果关系
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_type: str  # deductive, inductive, analogical, causal
    premise: str
    conclusion: str
    confidence: float
    evidence: List[str]


@dataclass
class ReasoningResult:
    """推理结果"""
    query: str
    steps: List[ReasoningStep]
    conclusion: str
    confidence: float
    reasoning_type: str


class ReasoningEngine:
    """推理引擎

    真正的推理：
    - 不只是检索
    - 而是逻辑推理
    - 从已知推未知
    """

    def __init__(self):
        """初始化推理引擎"""
        # 知识库
        self.knowledge: Dict[str, Dict[str, Any]] = {}

        # 规则库
        self.rules: List[Dict[str, Any]] = []

        # 推理历史
        self.reasoning_history: List[ReasoningResult] = []

        # 统计
        self.stats = {
            'deductive_count': 0,
            'inductive_count': 0,
            'analogical_count': 0,
            'causal_count': 0,
        }

    def add_knowledge(self, concept: str, properties: Dict[str, Any],
                     relations: Dict[str, List[str]] = None) -> None:
        """
        添加知识

        Args:
            concept: 概念
            properties: 属性
            relations: 关系
        """
        self.knowledge[concept] = {
            'properties': properties,
            'relations': relations or {},
        }

    def add_rule(self, premise: str, conclusion: str,
                rule_type: str = "implication") -> None:
        """
        添加规则

        Args:
            premise: 前提
            conclusion: 结论
            rule_type: 规则类型
        """
        self.rules.append({
            'premise': premise,
            'conclusion': conclusion,
            'type': rule_type,
        })

    def reason_deductive(self, premise: str) -> ReasoningResult:
        """
        演绎推理

        Args:
            premise: 前提

        Returns:
            推理结果
        """
        self.stats['deductive_count'] += 1

        steps = []
        conclusion = premise

        # 查找匹配的规则
        for rule in self.rules:
            if rule['premise'] in premise:
                # 应用规则
                conclusion = premise.replace(rule['premise'], rule['conclusion'])
                steps.append(ReasoningStep(
                    step_type='deductive',
                    premise=premise,
                    conclusion=conclusion,
                    confidence=0.9,
                    evidence=[rule['premise']]
                ))
                break

        result = ReasoningResult(
            query=premise,
            steps=steps,
            conclusion=conclusion,
            confidence=0.9 if steps else 0.5,
            reasoning_type='deductive'
        )

        self.reasoning_history.append(result)
        return result

    def reason_inductive(self, examples: List[str]) -> ReasoningResult:
        """
        归纳推理

        Args:
            examples: 例子列表

        Returns:
            推理结果
        """
        self.stats['inductive_count'] += 1

        steps = []

        # 查找共同特征
        common_features = self._find_common_features(examples)

        # 生成结论
        if common_features:
            conclusion = f"共同特征: {', '.join(common_features)}"
        else:
            conclusion = "未发现共同特征"

        steps.append(ReasoningStep(
            step_type='inductive',
            premise=str(examples),
            conclusion=conclusion,
            confidence=0.7,
            evidence=examples
        ))

        result = ReasoningResult(
            query=str(examples),
            steps=steps,
            conclusion=conclusion,
            confidence=0.7 if common_features else 0.3,
            reasoning_type='inductive'
        )

        self.reasoning_history.append(result)
        return result

    def reason_analogical(self, source: str, target: str) -> ReasoningResult:
        """
        类比推理

        Args:
            source: 源概念
            target: 目标概念

        Returns:
            推理结果
        """
        self.stats['analogical_count'] += 1

        steps = []

        # 计算相似度
        similarity = self._compute_similarity(source, target)

        # 生成结论
        if similarity > 0.5:
            conclusion = f"{source}和{target}相似，相似度: {similarity:.2f}"
        else:
            conclusion = f"{source}和{target}不相似"

        steps.append(ReasoningStep(
            step_type='analogical',
            premise=f"{source} vs {target}",
            conclusion=conclusion,
            confidence=similarity,
            evidence=[source, target]
        ))

        result = ReasoningResult(
            query=f"{source} vs {target}",
            steps=steps,
            conclusion=conclusion,
            confidence=similarity,
            reasoning_type='analogical'
        )

        self.reasoning_history.append(result)
        return result

    def reason_causal(self, cause: str) -> ReasoningResult:
        """
        因果推理

        Args:
            cause: 原因

        Returns:
            推理结果
        """
        self.stats['causal_count'] += 1

        steps = []

        # 查找因果关系
        effects = []
        for concept, data in self.knowledge.items():
            if 'relations' in data:
                for rel_type, targets in data['relations'].items():
                    if rel_type in ['导致', 'cause', 'result_in']:
                        if concept == cause:
                            effects.extend(targets)

        # 生成结论
        if effects:
            conclusion = f"{cause}导致: {', '.join(effects)}"
        else:
            conclusion = f"未找到{cause}的因果关系"

        steps.append(ReasoningStep(
            step_type='causal',
            premise=cause,
            conclusion=conclusion,
            confidence=0.8 if effects else 0.3,
            evidence=effects
        ))

        result = ReasoningResult(
            query=cause,
            steps=steps,
            conclusion=conclusion,
            confidence=0.8 if effects else 0.3,
            reasoning_type='causal'
        )

        self.reasoning_history.append(result)
        return result

    def _find_common_features(self, examples: List[str]) -> List[str]:
        """
        查找共同特征

        Args:
            examples: 例子列表

        Returns:
            共同特征列表
        """
        if not examples:
            return []

        # 提取每个例子的特征
        features_per_example = []
        for example in examples:
            features = set()
            # 从知识库查找
            if example in self.knowledge:
                features.update(self.knowledge[example].get('properties', {}).keys())
            # 从文本提取
            features.update(example.split())
            features_per_example.append(features)

        # 查找交集
        if features_per_example:
            common = features_per_example[0]
            for features in features_per_example[1:]:
                common = common & features
            return list(common)

        return []

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度 [0, 1]
        """
        # 简单的字符重叠计算
        set1 = set(text1)
        set2 = set(text2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
