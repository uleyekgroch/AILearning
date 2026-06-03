#!/usr/bin/env python3
"""注意力门控 (Attention Gating)

基于论文:
- Trends in Cognitive Sciences 2025: "A Hierarchical Model of Early Brain
  Functional Network Development"
  幼儿大脑从感觉→情感→认知的层次注意力发展。
- Treisman & Gelade 1980: Feature Integration Theory
  注意力从特征到物体的整合。
- Desimone & Duncan 1995: Neural Mechanisms of Selective Visual Attention
  竞争式注意力选择。

核心思想:
  不是所有输入都同等重要。注意力机制决定哪些信息进入学习系统：

  1. 底-Up (刺激驱动): 新奇/意外的输入自动吸引注意力
  2. 顶-Down (目标驱动): 当前学习目标决定关注什么
  3. 竞争选择: 多个输入竞争有限的注意力资源
  4. 门控放大: 被选中的输入获得更强的学习信号

  对学习系统的意义:
  - 避免被无用信息淹没（百科语料中有大量低质量内容）
  - 聚焦在当前最需要学习的领域
  - 新奇信息获得更强的学习信号（与好奇心驱动互补）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class AttentionTarget:
    """注意力目标"""
    content: str                    # 内容标识
    embedding: torch.Tensor         # 嵌入
    novelty_score: float            # 新奇度(0-1)
    relevance_score: float          # 相关度(0-1)
    attention_weight: float         # 最终注意力权重(0-1)


class AttentionGate:
    """注意力门控系统

    模拟人类注意力的三层机制:
    1. 新奇检测 (Novelty Detection): 从未见过的模式获得高注意力
    2. 目标相关 (Goal Relevance): 与当前学习目标匹配的获得高注意力
    3. 竞争选择 (Competitive Selection): 归一化所有注意力权重
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 已见过的模式（用于新奇检测）
        self.seen_patterns: Dict[str, torch.Tensor] = {}
        self.max_seen = 5000

        # 当前学习目标（顶-Down注意力）
        self.current_goals: List[str] = []
        self.goal_embedding: Optional[torch.Tensor] = None

        # 注意力历史
        self.attention_history: List[float] = []

        # 门控参数
        self.novelty_weight = 0.4    # 新奇度权重
        self.relevance_weight = 0.4  # 相关度权重
        self.surprise_weight = 0.2   # 意外度权重

        # 统计
        self.stats = {
            'inputs_gated': 0,
            'inputs_passed': 0,
            'inputs_filtered': 0,
            'avg_attention': 0.0,
            'novel_inputs': 0,
        }

    def set_goal(self, goal: str, goal_embedding: torch.Tensor):
        """设置当前学习目标（顶-Down注意力）"""
        self.current_goals.append(goal)
        if len(self.current_goals) > 5:
            self.current_goals = self.current_goals[-5:]
        self.goal_embedding = goal_embedding.detach().to(self.device)

    def compute_attention(self, content: str, embedding: torch.Tensor,
                          prediction_error: float = 0.0) -> AttentionTarget:
        """计算输入的注意力权重

        综合考虑:
        1. 新奇度: 从未见过的模式
        2. 目标相关: 与当前学习目标的匹配度
        3. 预测误差: 意外的输入

        Returns:
            AttentionTarget 包含注意力和各维度分数
        """
        emb = embedding.to(self.device)
        self.stats['inputs_gated'] += 1

        # 1. 新奇度计算
        novelty = self._compute_novelty(content, emb)

        # 2. 目标相关度计算
        relevance = self._compute_relevance(emb)

        # 3. 意外度（来自时序预测的误差）
        surprise = min(1.0, prediction_error)

        # 4. 综合注意力权重
        attention = (
            self.novelty_weight * novelty +
            self.relevance_weight * relevance +
            self.surprise_weight * surprise
        )
        attention = min(1.0, attention)

        # 记录
        target = AttentionTarget(
            content=content,
            embedding=emb.detach(),
            novelty_score=novelty,
            relevance_score=relevance,
            attention_weight=attention,
        )

        self.attention_history.append(attention)
        if len(self.attention_history) > 1000:
            self.attention_history = self.attention_history[-500:]

        # 更新统计
        if attention > 0.5:
            self.stats['inputs_passed'] += 1
        else:
            self.stats['inputs_filtered'] += 1

        if novelty > 0.7:
            self.stats['novel_inputs'] += 1

        if self.attention_history:
            recent = self.attention_history[-100:]
            self.stats['avg_attention'] = sum(recent) / len(recent)

        # 注册为已见模式
        self._register_seen(content, emb)

        return target

    def _compute_novelty(self, content: str, embedding: torch.Tensor) -> float:
        """计算新奇度

        新奇度 = 1 - max(与所有已见模式的相似度)
        """
        if not self.seen_patterns:
            return 1.0  # 第一个输入完全新奇

        if content in self.seen_patterns:
            return 0.1  # 完全相同的已见内容

        emb = embedding
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        max_sim = 0.0
        for seen_emb in list(self.seen_patterns.values())[:200]:  # 限制比较数量
            s_emb = seen_emb.to(self.device)
            if s_emb.dim() == 1:
                s_emb = s_emb.unsqueeze(0)
            sim = F.cosine_similarity(emb, s_emb).item()
            if sim > max_sim:
                max_sim = sim

        return max(0.0, 1.0 - max_sim)

    def _compute_relevance(self, embedding: torch.Tensor) -> float:
        """计算与当前学习目标的匹配度"""
        if self.goal_embedding is None:
            return 0.5  # 没有目标时，中等相关度

        emb = embedding
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        goal = self.goal_embedding
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)

        return F.cosine_similarity(emb, goal).item()

    def _register_seen(self, content: str, embedding: torch.Tensor):
        """注册为已见模式"""
        self.seen_patterns[content] = embedding.detach().cpu()

        # 超容量时清理
        if len(self.seen_patterns) > self.max_seen:
            # 删除最旧的一半
            keys = list(self.seen_patterns.keys())
            for k in keys[:len(keys) // 2]:
                del self.seen_patterns[k]

    def gate_learning(self, content: str, embedding: torch.Tensor,
                      prediction_error: float = 0.0) -> Tuple[bool, float]:
        """门控决策：是否学习这个输入？

        Returns:
            (should_learn, attention_weight)
        """
        target = self.compute_attention(content, embedding, prediction_error)

        # 自适应阈值：基于注意力历史
        if len(self.attention_history) < 10:
            threshold = 0.3  # 初始低阈值（多学习）
        else:
            recent = self.attention_history[-50:]
            threshold = sum(recent) / len(recent) * 0.7

        should_learn = target.attention_weight >= threshold

        return should_learn, target.attention_weight

    def get_stats(self) -> Dict:
        total = max(1, self.stats['inputs_gated'])
        return {
            **self.stats,
            'pass_rate': self.stats['inputs_passed'] / total,
            'filter_rate': self.stats['inputs_filtered'] / total,
            'seen_patterns': len(self.seen_patterns),
        }
