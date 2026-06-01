"""
真正的AI系统 - 从第一性原理出发

整合所有模块，提供统一接口

设计原则：
- 简洁清晰
- 易于使用
- 可测试
"""

from typing import Dict, List, Any
from dataclasses import dataclass

from .core.perception import PerceptionModule, PerceptionResult
from .core.understanding import UnderstandingModule, UnderstandingResult
from .core.reasoning import ReasoningModule, ReasoningResult
from .core.creation import CreationModule, CreationRequest, CreationResult
from .core.improvement import ImprovementModule, Problem, ImprovementResult


@dataclass
class AIResponse:
    """AI响应"""
    input_text: str                 # 输入文本
    perception: PerceptionResult    # 感知结果
    understanding: UnderstandingResult  # 理解结果
    reasoning: ReasoningResult      # 推理结果
    creation: CreationResult        # 创造结果
    confidence: float               # 总体置信度


class TrueAISystem:
    """真正的AI系统

    从第一性原理出发，实现真正的理解、推理、创造、改进能力
    """

    def __init__(self):
        """初始化AI系统"""
        # 初始化各模块
        self.perception = PerceptionModule()
        self.understanding = UnderstandingModule()
        self.reasoning = ReasoningModule()
        self.creation = CreationModule()
        self.improvement = ImprovementModule()

        # 统计
        self.stats = {
            'total_queries': 0,
            'total_learnings': 0,
            'total_improvements': 0,
        }

    def learn(self, topic: str, facts: List[str]) -> None:
        """
        学习知识

        Args:
            topic: 主题
            facts: 事实列表
        """
        # 学习概念
        for fact in facts:
            self.understanding.learn_concept(topic, fact)

        # 学习知识
        self.creation.learn_knowledge(topic, facts)

        self.stats['total_learnings'] += 1

    def think(self, query: str) -> AIResponse:
        """
        思考问题

        Args:
            query: 查询

        Returns:
            AI响应
        """
        self.stats['total_queries'] += 1

        # 1. 感知
        perception = self.perception.perceive(query)

        # 2. 理解
        understanding = self.understanding.understand(query)

        # 3. 推理
        reasoning = self.reasoning.reason_deductive(query)

        # 4. 创造
        creation_request = CreationRequest(
            topic=query,
            style='expository',
            requirements=[]
        )
        creation = self.creation.create(creation_request)

        # 5. 计算总体置信度
        confidence = (
            perception.confidence +
            understanding.confidence +
            reasoning.confidence +
            creation.quality_score
        ) / 4

        return AIResponse(
            input_text=query,
            perception=perception,
            understanding=understanding,
            reasoning=reasoning,
            creation=creation,
            confidence=confidence
        )

    def improve(self, problem_description: str, severity: str = 'medium') -> ImprovementResult:
        """
        改进

        Args:
            problem_description: 问题描述
            severity: 严重程度

        Returns:
            改进结果
        """
        # 识别问题
        problem = self.improvement.identify_problem(
            description=problem_description,
            severity=severity,
            location='system'
        )

        # 生成计划
        plan = self.improvement.generate_plan(problem)

        # 实施计划
        result = self.improvement.implement_plan(plan)

        self.stats['total_improvements'] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
