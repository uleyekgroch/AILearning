"""
真正的学习器 - 整合所有组件

从第一性原理出发，实现真正的学习能力：
1. 理解 - 理解含义
2. 推理 - 逻辑推理
3. 创造 - 创造新内容
4. 改进 - 自我改进
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .understanding import UnderstandingEngine, UnderstandingResult
from .reasoning import ReasoningEngine, ReasoningResult
from .creation import CreationEngine, CreationRequest, CreationResult
from .improvement import ImprovementEngine, Problem, ImprovementResult

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """学习结果"""
    topic: str
    understanding: UnderstandingResult
    reasoning: ReasoningResult
    creation: CreationResult
    improvement: ImprovementResult
    overall_score: float


class TrueLearner:
    """真正的学习器

    整合所有组件，实现真正的学习能力
    """

    def __init__(self, embedding_dim: int = 128):
        """
        初始化学习器

        Args:
            embedding_dim: 嵌入维度
        """
        # 初始化各引擎
        self.understanding_engine = UnderstandingEngine(embedding_dim)
        self.reasoning_engine = ReasoningEngine()
        self.creation_engine = CreationEngine()
        self.improvement_engine = ImprovementEngine()

        # 学习历史
        self.learning_history: List[LearningResult] = []

        # 统计
        self.stats = {
            'topics_learned': 0,
            'creations': 0,
            'improvements': 0,
        }

    def learn_topic(self, topic: str, facts: List[str],
                   relations: Dict[str, List[str]] = None) -> LearningResult:
        """
        学习主题

        Args:
            topic: 主题
            facts: 事实列表
            relations: 关系

        Returns:
            学习结果
        """
        logger.info(f"Learning topic: {topic}")

        # 1. 理解
        understanding = self._understand_topic(topic, facts, relations)

        # 2. 推理
        reasoning = self._reason_about_topic(topic, facts)

        # 3. 创造
        creation = self._create_about_topic(topic, facts)

        # 4. 改进
        improvement = self._improve_learning(topic, understanding, reasoning, creation)

        # 计算总分
        overall_score = self._calculate_overall_score(
            understanding, reasoning, creation, improvement
        )

        result = LearningResult(
            topic=topic,
            understanding=understanding,
            reasoning=reasoning,
            creation=creation,
            improvement=improvement,
            overall_score=overall_score
        )

        self.learning_history.append(result)
        self.stats['topics_learned'] += 1

        return result

    def create_content(self, topic: str, style: str,
                      requirements: List[str] = None) -> CreationResult:
        """
        创造内容

        Args:
            topic: 主题
            style: 风格
            requirements: 要求

        Returns:
            创造结果
        """
        request = CreationRequest(
            topic=topic,
            style=style,
            length='medium',
            requirements=requirements or [],
            context={}
        )

        result = self.creation_engine.create(request)
        self.stats['creations'] += 1

        return result

    def improve_system(self, problem_description: str,
                      severity: str = 'medium') -> ImprovementResult:
        """
        改进系统

        Args:
            problem_description: 问题描述
            severity: 严重程度

        Returns:
            改进结果
        """
        # 识别问题
        problem = self.improvement_engine.identify_problem(
            description=problem_description,
            severity=severity,
            location='system',
            impact='performance'
        )

        # 生成计划
        plan = self.improvement_engine.generate_plan(problem)

        # 实施计划
        result = self.improvement_engine.implement_plan(plan)
        self.stats['improvements'] += 1

        return result

    def reason(self, query: str, reasoning_type: str = 'deductive') -> ReasoningResult:
        """
        推理

        Args:
            query: 查询
            reasoning_type: 推理类型

        Returns:
            推理结果
        """
        if reasoning_type == 'deductive':
            return self.reasoning_engine.reason_deductive(query)
        elif reasoning_type == 'inductive':
            return self.reasoning_engine.reason_inductive([query])
        elif reasoning_type == 'causal':
            return self.reasoning_engine.reason_causal(query)
        else:
            return self.reasoning_engine.reason_deductive(query)

    def understand(self, text: str) -> UnderstandingResult:
        """
        理解

        Args:
            text: 文本

        Returns:
            理解结果
        """
        return self.understanding_engine.understand(text)

    def _understand_topic(self, topic: str, facts: List[str],
                         relations: Dict[str, List[str]] = None) -> UnderstandingResult:
        """
        理解主题

        Args:
            topic: 主题
            facts: 事实
            relations: 关系

        Returns:
            理解结果
        """
        # 学习概念
        self.understanding_engine.learn_concept(
            concept=topic,
            definition=f"{topic}的相关知识",
            properties={'facts': facts},
            relations=relations
        )

        # 理解
        text = f"{topic}: {', '.join(facts)}"
        return self.understanding_engine.understand(text)

    def _reason_about_topic(self, topic: str, facts: List[str]) -> ReasoningResult:
        """
        推理主题

        Args:
            topic: 主题
            facts: 事实

        Returns:
            推理结果
        """
        # 添加知识
        self.reasoning_engine.add_knowledge(topic, {'facts': facts})

        # 归纳推理
        return self.reasoning_engine.reason_inductive(facts)

    def _create_about_topic(self, topic: str, facts: List[str]) -> CreationResult:
        """
        创造关于主题的内容

        Args:
            topic: 主题
            facts: 事实

        Returns:
            创造结果
        """
        # 学习知识
        self.creation_engine.learn_knowledge(topic, facts)

        # 创造
        request = CreationRequest(
            topic=topic,
            style='expository',
            length='medium',
            requirements=[],
            context={'facts': facts}
        )

        return self.creation_engine.create(request)

    def _improve_learning(self, topic: str,
                         understanding: UnderstandingResult,
                         reasoning: ReasoningResult,
                         creation: CreationResult) -> ImprovementResult:
        """
        改进学习

        Args:
            topic: 主题
            understanding: 理解结果
            reasoning: 推理结果
            creation: 创造结果

        Returns:
            改进结果
        """
        # 评估学习质量
        quality_score = (
            understanding.confidence +
            reasoning.confidence +
            creation.quality_score
        ) / 3

        # 如果质量不高，识别问题
        if quality_score < 0.7:
            problem = self.improvement_engine.identify_problem(
                description=f"{topic}学习质量不高",
                severity='medium',
                location='learning',
                impact='knowledge'
            )

            plan = self.improvement_engine.generate_plan(problem)
            return self.improvement_engine.implement_plan(plan)
        else:
            # 创建虚拟成功结果
            from .improvement import ImprovementPlan
            dummy_plan = ImprovementPlan(
                plan_id='dummy',
                problem=Problem('dummy', 'dummy', 'low', 'dummy', 'dummy'),
                solution='学习质量良好',
                steps=[],
                expected_outcome='继续保持',
                confidence=quality_score
            )

            return ImprovementResult(
                plan=dummy_plan,
                success=True,
                actual_outcome='学习质量良好',
                lessons_learned=[f'{topic}学习成功'],
                metrics={'quality_score': quality_score}
            )

    def _calculate_overall_score(self, understanding: UnderstandingResult,
                                reasoning: ReasoningResult,
                                creation: CreationResult,
                                improvement: ImprovementResult) -> float:
        """
        计算总分

        Args:
            understanding: 理解结果
            reasoning: 推理结果
            creation: 创造结果
            improvement: 改进结果

        Returns:
            总分 [0, 1]
        """
        scores = [
            understanding.confidence,
            reasoning.confidence,
            creation.quality_score,
            improvement.plan.confidence if improvement.plan else 0.5,
        ]

        return sum(scores) / len(scores)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'understanding_stats': self.understanding_engine.get_stats(),
            'reasoning_stats': self.reasoning_engine.get_stats(),
            'creation_stats': self.creation_engine.get_stats(),
            'improvement_stats': self.improvement_engine.get_stats(),
        }
