"""Empowerment驱动探索 — 从好奇心到能力感

替代单纯的预测误差好奇心，引入Empowerment指标。

核心概念：
- 好奇心：预测误差（新颖性）
- Empowerment：agent对环境的控制力
- 能力感：agent在某区域的熟练度

关键发现（Mantiuk 2025）：
- 不是所有新颖性都值得探索
- Empowerment与人类探索进度持续正相关
- 世界模型的质量决定哪种动机更有效

设计原则：
- 优先探索agent能控制的区域
- 避免在随机噪声区域浪费时间
- 平衡好奇心和能力感
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ExplorationState:
    """探索状态"""
    position: Tuple[float, ...]
    visit_count: int = 0
    prediction_error: float = 0.0
    empowerment: float = 0.0
    competence: float = 0.0


class EmpowermentCalculator:
    """Empowerment计算器

    Empowerment = agent对环境的控制力
    给定当前状态，agent能产生多少不同的未来状态

    简化实现：使用动作空间的熵来近似
    """

    def __init__(self, action_dim: int = 10, horizon: int = 3):
        self.action_dim = action_dim
        self.horizon = horizon

    def compute(self, state: torch.Tensor, world_model=None) -> float:
        """计算当前状态的Empowerment

        简化：使用随机采样估计可达状态的多样性
        """
        if world_model is None:
            # 无世界模型时，使用状态的新颖性作为代理
            return self._novelty_based_empowerment(state)

        # 使用世界模型估计可达状态
        reachable_states = []
        for _ in range(10):
            action = torch.randn(self.action_dim)
            try:
                next_state = world_model.predict_next_state_single(state, action)
                reachable_states.append(next_state)
            except Exception:
                pass

        if len(reachable_states) < 2:
            return 0.0

        # 计算可达状态的多样性（方差）
        stacked = torch.stack(reachable_states)
        diversity = stacked.var(dim=0).mean().item()

        return min(1.0, diversity)

    def _novelty_based_empowerment(self, state: torch.Tensor) -> float:
        """基于新颖性的Empowerment代理"""
        # 使用向量的方差作为新颖性的粗略估计
        return min(1.0, state.var().item() * 10)


class CuriosityCompetenceBalance:
    """好奇心-能力感平衡器

    平衡两种探索动机：
    - 好奇心：预测误差（新颖性）
    - 能力感：Empowerment（控制力）
    """

    def __init__(self, curiosity_weight: float = 0.5, competence_weight: float = 0.5):
        self.curiosity_weight = curiosity_weight
        self.competence_weight = competence_weight

        # 历史记录
        self.curiosity_history = deque(maxlen=100)
        self.competence_history = deque(maxlen=100)

    def exploration_bonus(self, curiosity: float, competence: float) -> float:
        """计算探索奖励"""
        self.curiosity_history.append(curiosity)
        self.competence_history.append(competence)

        # 自适应权重
        if len(self.curiosity_history) > 10:
            curiosity_var = sum((c - sum(self.curiosity_history)/len(self.curiosity_history))**2
                              for c in self.curiosity_history) / len(self.curiosity_history)
            competence_var = sum((c - sum(self.competence_history)/len(self.competence_history))**2
                               for c in self.competence_history) / len(self.competence_history)

            # 方差大的维度需要更多探索
            total_var = curiosity_var + competence_var + 1e-6
            self.curiosity_weight = curiosity_var / total_var
            self.competence_weight = competence_var / total_var

        return (self.curiosity_weight * curiosity +
                self.competence_weight * competence)


class MultiScaleCuriosity:
    """多尺度好奇心（基于人类学习研究）

    - 短期好奇心：下一个状态的新颖性
    - 中期好奇心：区域的新颖性
    - 长期好奇心：世界规则的变化
    - 学习进度好奇心：选择"够得着的挑战"（PMC 2021）
    """

    def __init__(self, short_window: int = 10, medium_window: int = 100):
        self.short_window = short_window
        self.medium_window = medium_window

        # 历史
        self.state_history = deque(maxlen=medium_window)
        self.error_history = deque(maxlen=medium_window)

        # 学习进度监控（人类好奇心研究的核心发现）
        self.learning_progress_history = deque(maxlen=medium_window)
        self.domain_progress = {}  # 每个领域的学习进度

    def update(self, state: torch.Tensor, prediction_error: float):
        """更新历史"""
        self.state_history.append(state.detach())
        self.error_history.append(prediction_error)

        # 计算学习进度（误差变化率）
        if len(self.error_history) >= 2:
            recent_errors = list(self.error_history)[-5:]
            if len(recent_errors) >= 2:
                progress = recent_errors[0] - recent_errors[-1]  # 正值=进步
                self.learning_progress_history.append(progress)

    def short_term_curiosity(self) -> float:
        """短期好奇心：最近状态的新颖性"""
        if len(self.state_history) < 2:
            return 1.0

        # 与最近状态的平均距离
        recent = list(self.state_history)[-self.short_window:]
        if len(recent) < 2:
            return 1.0

        distances = []
        for i in range(1, len(recent)):
            dist = torch.cosine_similarity(
                recent[-1].unsqueeze(0), recent[i].unsqueeze(0)
            ).item()
            distances.append(1.0 - dist)  # 距离 = 1 - 相似度

        return sum(distances) / len(distances)

    def medium_term_curiosity(self) -> float:
        """中期好奇心：区域的探索程度"""
        if len(self.state_history) < self.short_window:
            return 1.0

        # 未访问状态的比例（简化：使用向量量化）
        states = list(self.state_history)
        unique_count = 0
        seen = set()

        for state in states:
            # 简化：将向量量化为字符串作为key
            key = str(state[:10].round(decimals=1).tolist())
            if key not in seen:
                seen.add(key)
                unique_count += 1

        return unique_count / len(states)

    def long_term_curiosity(self) -> float:
        """长期好奇心：预测误差的变化趋势"""
        if len(self.error_history) < self.medium_window // 2:
            return 0.5

        errors = list(self.error_history)
        mid = len(errors) // 2

        early_avg = sum(errors[:mid]) / max(mid, 1)
        late_avg = sum(errors[mid:]) / max(len(errors) - mid, 1)

        # 如果误差在增加，说明有新东西要学
        return max(0.0, late_avg - early_avg)

    def learning_progress_curiosity(self) -> float:
        """学习进度好奇心（基于PMC 2021研究）

        人类在好奇心驱动的探索中监控自己的学习进度，
        选择那些能带来最大学习进展的探索方向。
        """
        if len(self.learning_progress_history) < 3:
            return 0.5

        # 计算平均学习进度
        recent_progress = list(self.learning_progress_history)[-10:]
        avg_progress = sum(recent_progress) / len(recent_progress)

        # 正进度 = 学习中，应继续探索
        # 负进度 = 退步，应改变策略
        # 零进度 = 饱和，应探索新领域
        if avg_progress > 0.01:
            return 0.7  # 有进步，继续
        elif avg_progress < -0.01:
            return 0.3  # 退步，需要改变
        else:
            return 0.9  # 饱和，探索新领域

    def combined_curiosity(self) -> float:
        """组合好奇心"""
        short = self.short_term_curiosity()
        medium = self.medium_term_curiosity()
        long = self.long_term_curiosity()

        # 加入学习进度好奇心（人类研究的核心发现）
        progress = self.learning_progress_curiosity()
        return 0.4 * short + 0.25 * medium + 0.15 * long + 0.2 * progress


class EmpowermentExplorationSystem:
    """Empowerment驱动探索系统

    整合Empowerment、好奇心-能力平衡、多尺度好奇心。
    """

    def __init__(self):
        # Empowerment计算器
        self.empowerment = EmpowermentCalculator()

        # 好奇心-能力平衡
        self.balance = CuriosityCompetenceBalance()

        # 多尺度好奇心
        self.curiosity = MultiScaleCuriosity()

        # 探索统计
        self.exploration_stats = {
            'total_steps': 0,
            'empowerment_sum': 0.0,
            'curiosity_sum': 0.0,
        }

    def compute_exploration_bonus(self, state: torch.Tensor,
                                  prediction_error: float,
                                  world_model=None) -> Dict:
        """计算探索奖励

        Returns:
            {
                'empowerment': float,
                'curiosity': float,
                'competence': float,
                'bonus': float,
            }
        """
        # Empowerment
        emp = self.empowerment.compute(state, world_model)

        # 好奇心
        self.curiosity.update(state, prediction_error)
        cur = self.curiosity.combined_curiosity()

        # 能力感（基于历史成功率）
        competence = 1.0 - prediction_error  # 简化

        # 平衡奖励
        bonus = self.balance.exploration_bonus(cur, emp)

        # 更新统计
        self.exploration_stats['total_steps'] += 1
        self.exploration_stats['empowerment_sum'] += emp
        self.exploration_stats['curiosity_sum'] += cur

        return {
            'empowerment': emp,
            'curiosity': cur,
            'competence': competence,
            'bonus': bonus,
        }

    def should_explore(self, state: torch.Tensor, threshold: float = 0.5) -> bool:
        """判断是否值得探索"""
        emp = self.empowerment.compute(state)
        cur = self.curiosity.short_term_curiosity()

        # 高Empowerment或高新颖性都值得探索
        return emp > threshold or cur > threshold

    def get_stats(self) -> Dict:
        """获取探索统计"""
        steps = max(self.exploration_stats['total_steps'], 1)
        return {
            'total_steps': self.exploration_stats['total_steps'],
            'avg_empowerment': self.exploration_stats['empowerment_sum'] / steps,
            'avg_curiosity': self.exploration_stats['curiosity_sum'] / steps,
        }
