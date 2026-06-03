#!/usr/bin/env python3
"""社会反馈学习 + 知识蒸馏传递 (Social Feedback Learning + Knowledge Distillation)

基于论文:
- Cornell 2025: "Power of Babble" — 婴儿咿呀声引导照料者简化语言
  → 社会反馈闭环：婴儿发声 → 照料者简化 → 婴儿学到更多
- PNAS 2025: "A Simple Threshold Captures the Social Learning of Conventions"
  → 阈值模型的社会学习比模仿和优化更准确
- MIT-IBM HMAT: "Hierarchical Multiagent Teaching"
  → 分层教学：高能力智能体教低能力智能体
- IJCAI 2024: "Social Learning through Interactions with Other Agents"
  → 综合综述：模仿、互动、知识传递

核心思想:
  社会反馈闭环:
  1. 学习者表达（咿呀/提问/输出） → 2. 环境/教师反馈（简化/纠正）
  → 3. 学习者调整 → 4. 再次表达 → 循环

  知识蒸馏传递:
  1. "专家"智能体(已掌握领域A) → 2. 提取核心知识 → 3. 教给"学生"智能体
  → 4. 学生在领域B应用（跨域迁移）

  对学习系统的意义:
  - 社会反馈: 系统提问/表达后，环境(语料/用户)提供有针对性的反馈
  - 知识蒸馏: 将学到的核心知识压缩提炼，用于更高效的学习
  - 两者结合: 学习→表达→反馈→提炼→教学的完整闭环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class FeedbackEvent:
    """反馈事件"""
    learner_output: str         # 学习者的输出/表达
    environment_feedback: str   # 环境的反馈
    feedback_type: str          # 'positive', 'corrective', 'simplification'
    learning_gain: float        # 这次反馈带来的学习增益


@dataclass
class DistilledKnowledge:
    """蒸馏后的核心知识"""
    domain: str                 # 知识领域
    core_patterns: List[str]    # 核心模式（最频繁/最重要的关系）
    abstraction: str            # 抽象描述
    confidence: float           # 置信度
    source_count: int           # 来源数量


class SocialFeedbackSystem:
    """社会反馈学习系统

    模拟婴儿-照料者反馈闭环:
    1. 学习者表达（生成文本/提问）
    2. 评估表达质量
    3. 生成反馈（纠正/简化/肯定）
    4. 从反馈中学习
    """

    def __init__(self):
        # 反馈历史
        self.feedback_history: deque = deque(maxlen=1000)

        # 学习者表达质量追踪
        self.expression_quality: deque = deque(maxlen=100)

        # 反馈-学习效果映射
        self.feedback_effectiveness: Dict[str, float] = {
            'positive': 0.1,     # 肯定反馈 → 小幅强化
            'corrective': 0.3,   # 纠正反馈 → 中等学习
            'simplification': 0.2, # 简化反馈 → 理解加深
        }

        # 统计
        self.stats = {
            'feedback_events': 0,
            'positive_feedback': 0,
            'corrective_feedback': 0,
            'simplification_feedback': 0,
            'avg_learning_gain': 0.0,
        }

    def process_feedback(self, learner_output: str,
                         expected_output: str = '') -> FeedbackEvent:
        """处理反馈事件

        学习者输出后，系统评估并生成反馈。
        """
        # 评估输出质量
        if expected_output:
            quality = self._compute_output_quality(learner_output, expected_output)
        else:
            quality = 0.5  # 无参考时默认中等

        self.expression_quality.append(quality)

        # 确定反馈类型
        if quality > 0.8:
            feedback_type = 'positive'
            feedback = f"正确: {learner_output}"
            gain = self.feedback_effectiveness['positive']
        elif quality > 0.5:
            feedback_type = 'simplification'
            feedback = f"接近正确，可以更精确: {learner_output}"
            gain = self.feedback_effectiveness['simplification']
        else:
            feedback_type = 'corrective'
            feedback = f"需要纠正: {learner_output} → {expected_output}"
            gain = self.feedback_effectiveness['corrective']

        event = FeedbackEvent(
            learner_output=learner_output,
            environment_feedback=feedback,
            feedback_type=feedback_type,
            learning_gain=gain,
        )

        self.feedback_history.append(event)
        self.stats['feedback_events'] += 1
        self.stats[f'{feedback_type}_feedback'] += 1

        # 更新平均学习增益
        recent = [e.learning_gain for e in list(self.feedback_history)[-50:]]
        self.stats['avg_learning_gain'] = sum(recent) / len(recent)

        return event

    def _compute_output_quality(self, output: str, expected: str) -> float:
        """计算输出质量（字符级匹配）"""
        if not output or not expected:
            return 0.0

        # 字符级重叠
        output_chars = set(output)
        expected_chars = set(expected)
        if not expected_chars:
            return 0.0

        overlap = len(output_chars & expected_chars) / len(expected_chars)

        # 长度相似度
        len_ratio = min(len(output), len(expected)) / max(len(output), len(expected), 1)

        return overlap * 0.7 + len_ratio * 0.3

    def should_ask_for_help(self) -> bool:
        """判断是否应该寻求帮助

        基于最近的表达质量：
        - 质量持续低 → 需要帮助
        - 质量持续高 → 可以独立学习
        """
        if len(self.expression_quality) < 5:
            return True  # 初期总是需要帮助

        recent = list(self.expression_quality)[-5:]
        avg_quality = sum(recent) / len(recent)
        return avg_quality < 0.5

    def get_feedback_pattern(self) -> str:
        """获取当前反馈模式（用于调整学习策略）"""
        if not self.feedback_history:
            return 'exploring'  # 还没有反馈

        recent = list(self.feedback_history)[-10:]
        corrective_ratio = sum(1 for e in recent if e.feedback_type == 'corrective') / len(recent)

        if corrective_ratio > 0.6:
            return 'struggling'  # 需要简化内容
        elif corrective_ratio < 0.2:
            return 'mastering'   # 可以增加难度
        else:
            return 'learning'    # 正常学习

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'current_pattern': self.get_feedback_pattern(),
            'needs_help': self.should_ask_for_help(),
        }


class KnowledgeDistillationSystem:
    """知识蒸馏系统

    将学到的海量具体知识压缩提炼为核心模式，
    供更高效的学习和跨域迁移使用。
    """

    def __init__(self):
        # 蒸馏后的知识
        self.distilled: Dict[str, DistilledKnowledge] = {}

        # 领域知识积累（蒸馏前）
        self.domain_knowledge: Dict[str, List[Tuple]] = {}

        # 蒸馏阈值
        self.distill_threshold = 5  # 积累5条同领域知识后触发蒸馏

        # 统计
        self.stats = {
            'knowledge_accumulated': 0,
            'distillations_performed': 0,
            'patterns_extracted': 0,
            'domains_covered': 0,
        }

    def accumulate(self, domain: str, triple: Tuple[str, str, str]):
        """积累原始知识

        学到的每条三元组都先积累，达到阈值后蒸馏。
        """
        if domain not in self.domain_knowledge:
            self.domain_knowledge[domain] = []

        self.domain_knowledge[domain].append(triple)
        self.stats['knowledge_accumulated'] += 1

        # 达到阈值时自动蒸馏
        if len(self.domain_knowledge[domain]) >= self.distill_threshold:
            if domain not in self.distilled:
                self._distill_domain(domain)

    def _distill_domain(self, domain: str):
        """蒸馏一个领域的知识

        从大量具体三元组中提取：
        1. 最频繁的关系类型
        2. 核心实体
        3. 抽象模式
        """
        triples = self.domain_knowledge.get(domain, [])
        if len(triples) < 3:
            return

        # 提取最频繁的关系类型
        relation_counts: Dict[str, int] = {}
        entity_counts: Dict[str, int] = {}

        for triple in triples:
            if len(triple) >= 3:
                subj, rel, obj = triple[0], triple[1], triple[2]
                relation_counts[rel] = relation_counts.get(rel, 0) + 1
                entity_counts[subj] = entity_counts.get(subj, 0) + 1
                entity_counts[obj] = entity_counts.get(obj, 0) + 1

        # 核心模式：最频繁的关系
        sorted_rels = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
        core_patterns = [f"{rel}({count})" for rel, count in sorted_rels[:5]]

        # 核心实体
        sorted_ents = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        top_entities = [ent for ent, _ in sorted_ents[:5]]

        # 抽象描述
        abstraction = f"{domain}: {', '.join(top_entities[:3])} → {', '.join([r for r, _ in sorted_rels[:3]])}"

        # 置信度
        confidence = min(1.0, len(triples) / 20.0)

        dk = DistilledKnowledge(
            domain=domain,
            core_patterns=core_patterns,
            abstraction=abstraction,
            confidence=confidence,
            source_count=len(triples),
        )

        self.distilled[domain] = dk
        self.stats['distillations_performed'] += 1
        self.stats['patterns_extracted'] += len(core_patterns)
        self.stats['domains_covered'] = len(self.distilled)

    def get_distilled_knowledge(self, domain: str) -> Optional[DistilledKnowledge]:
        """获取蒸馏后的领域知识"""
        return self.distilled.get(domain)

    def get_teaching_material(self, target_domain: str,
                              source_domain: str = '') -> List[str]:
        """生成教学内容

        将蒸馏知识转化为可教学的材料。
        用于知识从A领域迁移到B领域。
        """
        if source_domain and source_domain in self.distilled:
            dk = self.distilled[source_domain]
            material = []
            for pattern in dk.core_patterns:
                material.append(f"在{source_domain}中学到的核心模式: {pattern}")
            return material

        if target_domain in self.distilled:
            dk = self.distilled[target_domain]
            return [f"已有知识: {dk.abstraction}"]

        return []

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'undistilled_domains': sum(
                1 for d, triples in self.domain_knowledge.items()
                if d not in self.distilled
            ),
        }
