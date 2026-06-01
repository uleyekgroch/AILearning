"""
理解模块 - 从第一性原理出发

人类理解的本质：
1. 理解含义
2. 理解关系
3. 理解上下文

设计原则：
- 单一职责：只负责理解
- 简洁清晰：代码易于理解
- 可测试：接口明确，易于测试
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class UnderstandingResult:
    """理解结果"""
    text: str                       # 输入文本
    concepts: List[str]             # 识别的概念
    relations: Dict[str, List[str]] # 概念关系
    meaning: str                    # 理解的含义
    confidence: float               # 置信度


class UnderstandingModule:
    """理解模块

    职责：
    - 语义理解
    - 关系理解
    - 上下文理解

    设计原则：
    - 单一职责
    - 简洁清晰
    - 可测试
    """

    def __init__(self):
        """初始化理解模块"""
        # 概念知识库
        self.concept_kb: Dict[str, Dict[str, Any]] = {}

        # 关系知识库
        self.relation_kb: Dict[str, List[str]] = {}

    def learn_concept(self, concept: str, definition: str,
                     properties: Dict[str, Any] = None) -> None:
        """
        学习概念

        Args:
            concept: 概念名称
            definition: 定义
            properties: 属性
        """
        self.concept_kb[concept] = {
            'definition': definition,
            'properties': properties or {},
        }

    def learn_relation(self, subject: str, relation: str, obj: str) -> None:
        """
        学习关系

        Args:
            subject: 主语
            relation: 关系
            obj: 宾语
        """
        key = f"{subject}_{relation}"
        if key not in self.relation_kb:
            self.relation_kb[key] = []
        self.relation_kb[key].append(obj)

    def understand(self, text: str, context: Dict[str, Any] = None) -> UnderstandingResult:
        """
        理解文本

        Args:
            text: 输入文本
            context: 上下文

        Returns:
            理解结果
        """
        # 识别概念
        concepts = self._identify_concepts(text)

        # 理解关系
        relations = self._understand_relations(text, concepts)

        # 理解含义
        meaning = self._understand_meaning(text, concepts, relations)

        # 计算置信度
        confidence = self._compute_confidence(concepts, relations)

        return UnderstandingResult(
            text=text,
            concepts=concepts,
            relations=relations,
            meaning=meaning,
            confidence=confidence
        )

    def _identify_concepts(self, text: str) -> List[str]:
        """
        识别概念

        Args:
            text: 文本

        Returns:
            概念列表
        """
        concepts = []

        # 从知识库中匹配概念
        for concept in self.concept_kb.keys():
            if concept in text:
                concepts.append(concept)

        # 如果没有匹配，提取关键词
        if not concepts:
            words = text.split()
            concepts = [w for w in words if len(w) > 1]

        return concepts

    def _understand_relations(self, text: str, concepts: List[str]) -> Dict[str, List[str]]:
        """
        理解关系

        Args:
            text: 文本
            concepts: 概念列表

        Returns:
            关系字典
        """
        relations = {}

        # 查找已知关系
        for concept in concepts:
            for key, targets in self.relation_kb.items():
                if concept in key:
                    relations[concept] = targets

        return relations

    def _understand_meaning(self, text: str, concepts: List[str],
                           relations: Dict[str, List[str]]) -> str:
        """
        理解含义

        Args:
            text: 文本
            concepts: 概念列表
            relations: 关系字典

        Returns:
            含义描述
        """
        if not concepts:
            return text

        # 构建含义
        meaning_parts = []
        for concept in concepts:
            if concept in self.concept_kb:
                definition = self.concept_kb[concept]['definition']
                meaning_parts.append(f"{concept}: {definition}")

        if meaning_parts:
            return "; ".join(meaning_parts)

        return text

    def _compute_confidence(self, concepts: List[str],
                           relations: Dict[str, List[str]]) -> float:
        """
        计算置信度

        Args:
            concepts: 概念列表
            relations: 关系字典

        Returns:
            置信度 [0, 1]
        """
        confidence = 0.5

        # 根据概念数量调整
        if concepts:
            confidence += 0.1

        # 根据关系数量调整
        if relations:
            confidence += 0.1

        return min(1.0, confidence)
