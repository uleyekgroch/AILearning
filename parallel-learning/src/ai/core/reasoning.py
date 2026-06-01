"""
推理模块 - 从第一性原理出发

人类推理的本质：
1. 演绎推理：从一般到特殊
2. 归纳推理：从特殊到一般
3. 类比推理：基于相似性
4. 因果推理：基于因果关系

设计原则：
- 单一职责：只负责推理
- 简洁清晰：代码易于理解
- 可测试：接口明确，易于测试
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class ReasoningResult:
    """推理结果"""
    premise: str                # 前提
    conclusion: str             # 结论
    reasoning_type: str         # 推理类型
    confidence: float           # 置信度
    steps: List[str]            # 推理步骤


class ReasoningModule:
    """推理模块

    职责：
    - 演绎推理
    - 归纳推理
    - 类比推理
    - 因果推理

    设计原则：
    - 单一职责
    - 简洁清晰
    - 可测试
    """

    def __init__(self):
        """初始化推理模块"""
        # 推理规则
        self.rules: List[Dict[str, Any]] = []

        # 知识库
        self.knowledge: Dict[str, Dict[str, Any]] = {}

    def add_rule(self, premise: str, conclusion: str, rule_type: str = "implication") -> None:
        """
        添加推理规则

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

    def add_knowledge(self, concept: str, properties: Dict[str, Any]) -> None:
        """
        添加知识

        Args:
            concept: 概念
            properties: 属性
        """
        self.knowledge[concept] = properties

    def reason_deductive(self, premise: str) -> ReasoningResult:
        """
        演绎推理

        Args:
            premise: 前提

        Returns:
            推理结果
        """
        steps = [f"前提: {premise}"]

        # 查找匹配的规则
        conclusion = premise
        for rule in self.rules:
            if rule['premise'] in premise:
                conclusion = premise.replace(rule['premise'], rule['conclusion'])
                steps.append(f"应用规则: {rule['premise']} -> {rule['conclusion']}")
                break

        steps.append(f"结论: {conclusion}")

        return ReasoningResult(
            premise=premise,
            conclusion=conclusion,
            reasoning_type='deductive',
            confidence=0.9,
            steps=steps
        )

    def reason_inductive(self, examples: List[str]) -> ReasoningResult:
        """
        归纳推理

        Args:
            examples: 例子列表

        Returns:
            推理结果
        """
        steps = [f"例子: {examples}"]

        # 查找共同特征
        common_features = self._find_common_features(examples)

        if common_features:
            conclusion = f"共同特征: {', '.join(common_features)}"
            steps.append(f"发现共同特征: {common_features}")
        else:
            conclusion = "未发现共同特征"
            steps.append("未发现共同特征")

        steps.append(f"结论: {conclusion}")

        return ReasoningResult(
            premise=str(examples),
            conclusion=conclusion,
            reasoning_type='inductive',
            confidence=0.7 if common_features else 0.3,
            steps=steps
        )

    def reason_analogical(self, source: str, target: str) -> ReasoningResult:
        """
        类比推理

        Args:
            source: 源概念
            target: 目标概念

        Returns:
            推理结果
        """
        steps = [f"源: {source}, 目标: {target}"]

        # 计算相似度
        similarity = self._compute_similarity(source, target)

        if similarity > 0.5:
            conclusion = f"{source}和{target}相似，相似度: {similarity:.2f}"
            steps.append(f"计算相似度: {similarity:.2f}")
        else:
            conclusion = f"{source}和{target}不相似"
            steps.append(f"相似度低: {similarity:.2f}")

        steps.append(f"结论: {conclusion}")

        return ReasoningResult(
            premise=f"{source} vs {target}",
            conclusion=conclusion,
            reasoning_type='analogical',
            confidence=similarity,
            steps=steps
        )

    def reason_causal(self, cause: str) -> ReasoningResult:
        """
        因果推理

        Args:
            cause: 原因

        Returns:
            推理结果
        """
        steps = [f"原因: {cause}"]

        # 查找因果关系
        effects = []
        for concept, data in self.knowledge.items():
            if 'effects' in data:
                if concept == cause:
                    effects.extend(data['effects'])

        if effects:
            conclusion = f"{cause}导致: {', '.join(effects)}"
            steps.append(f"找到结果: {effects}")
        else:
            conclusion = f"未找到{cause}的结果"
            steps.append("未找到因果关系")

        steps.append(f"结论: {conclusion}")

        return ReasoningResult(
            premise=cause,
            conclusion=conclusion,
            reasoning_type='causal',
            confidence=0.8 if effects else 0.3,
            steps=steps
        )

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
                features.update(self.knowledge[example].keys())
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
