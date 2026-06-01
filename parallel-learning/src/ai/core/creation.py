"""
创造模块 - 从第一性原理出发

人类创造的本质：
1. 理解需求
2. 组合知识
3. 生成内容
4. 评估质量

设计原则：
- 单一职责：只负责创造
- 简洁清晰：代码易于理解
- 可测试：接口明确，易于测试
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import random


@dataclass
class CreationRequest:
    """创造请求"""
    topic: str                  # 主题
    style: str                  # 风格
    requirements: List[str] = field(default_factory=list)  # 要求


@dataclass
class CreationResult:
    """创造结果"""
    content: str                # 生成内容
    style: str                  # 风格
    quality_score: float        # 质量分数
    creation_process: List[str] = field(default_factory=list)  # 创造过程


class CreationModule:
    """创造模块

    职责：
    - 内容生成
    - 问题解决
    - 方案设计

    设计原则：
    - 单一职责
    - 简洁清晰
    - 可测试
    """

    def __init__(self):
        """初始化创造模块"""
        # 知识库
        self.knowledge: Dict[str, List[str]] = {}

        # 创作模式
        self.creation_patterns = {
            'narrative': self._create_narrative,
            'descriptive': self._create_descriptive,
            'argumentative': self._create_argumentative,
            'expository': self._create_expository,
        }

    def learn_knowledge(self, topic: str, facts: List[str]) -> None:
        """
        学习知识

        Args:
            topic: 主题
            facts: 事实列表
        """
        self.knowledge[topic] = facts

    def create(self, request: CreationRequest) -> CreationResult:
        """
        创造内容

        Args:
            request: 创造请求

        Returns:
            创造结果
        """
        # 选择创作方法
        creation_method = self.creation_patterns.get(
            request.style,
            self._create_expository
        )

        # 执行创作
        result = creation_method(request)

        return result

    def _create_narrative(self, request: CreationRequest) -> CreationResult:
        """
        创作叙述文

        Args:
            request: 创造请求

        Returns:
            创造结果
        """
        process = []

        # 获取知识
        facts = self.knowledge.get(request.topic, [])
        process.append(f"获取{request.topic}的知识: {len(facts)}条")

        # 生成内容
        content = f"关于{request.topic}的故事：\n\n"
        if facts:
            content += f"从前，有一个{request.topic}。"
            content += f"它具有以下特点：{facts[0]}。"
            content += f"随着时间的推移，{request.topic}变得越来越重要。"
        else:
            content += f"从前，有一个{request.topic}。它是一个有趣的话题。"

        process.append("生成叙述内容")

        # 计算质量分数
        quality_score = 0.7 if facts else 0.5
        process.append(f"质量分数: {quality_score}")

        return CreationResult(
            content=content,
            style='narrative',
            quality_score=quality_score,
            creation_process=process
        )

    def _create_descriptive(self, request: CreationRequest) -> CreationResult:
        """
        创作描写文

        Args:
            request: 创造请求

        Returns:
            创造结果
        """
        process = []

        # 获取知识
        facts = self.knowledge.get(request.topic, [])
        process.append(f"获取{request.topic}的知识: {len(facts)}条")

        # 生成内容
        content = f"描写{request.topic}：\n\n"
        if facts:
            content += f"{request.topic}是一个重要的概念。"
            content += f"它的特点包括：{', '.join(facts[:3])}。"
            content += f"这些特点使得{request.topic}与众不同。"
        else:
            content += f"{request.topic}是一个值得探索的主题。"

        process.append("生成描写内容")

        # 计算质量分数
        quality_score = 0.7 if facts else 0.5
        process.append(f"质量分数: {quality_score}")

        return CreationResult(
            content=content,
            style='descriptive',
            quality_score=quality_score,
            creation_process=process
        )

    def _create_argumentative(self, request: CreationRequest) -> CreationResult:
        """
        创作议论文

        Args:
            request: 创造请求

        Returns:
            创造结果
        """
        process = []

        # 获取知识
        facts = self.knowledge.get(request.topic, [])
        process.append(f"获取{request.topic}的知识: {len(facts)}条")

        # 生成内容
        content = f"论{request.topic}：\n\n"
        content += f"我认为{request.topic}非常重要。\n\n"

        if facts:
            content += f"首先，{facts[0]}。\n"
            if len(facts) > 1:
                content += f"其次，{facts[1]}。\n"
            content += f"\n综上所述，{request.topic}具有重要意义。"
        else:
            content += f"{request.topic}是一个值得深入研究的话题。"

        process.append("生成议论内容")

        # 计算质量分数
        quality_score = 0.7 if facts else 0.5
        process.append(f"质量分数: {quality_score}")

        return CreationResult(
            content=content,
            style='argumentative',
            quality_score=quality_score,
            creation_process=process
        )

    def _create_expository(self, request: CreationRequest) -> CreationResult:
        """
        创作说明文

        Args:
            request: 创造请求

        Returns:
            创造结果
        """
        process = []

        # 获取知识
        facts = self.knowledge.get(request.topic, [])
        process.append(f"获取{request.topic}的知识: {len(facts)}条")

        # 生成内容
        content = f"{request.topic}介绍：\n\n"
        content += f"{request.topic}是一个重要的概念。\n\n"

        if facts:
            content += f"关于{request.topic}，有以下几个要点：\n"
            for i, fact in enumerate(facts[:5], 1):
                content += f"{i}. {fact}\n"
            content += f"\n以上就是关于{request.topic}的介绍。"
        else:
            content += f"关于{request.topic}的详细信息有待补充。"

        process.append("生成说明内容")

        # 计算质量分数
        quality_score = 0.7 if facts else 0.5
        process.append(f"质量分数: {quality_score}")

        return CreationResult(
            content=content,
            style='expository',
            quality_score=quality_score,
            creation_process=process
        )
