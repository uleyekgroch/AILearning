"""自改进系统 — 统一脚手架+权重更新

基于 SIA (2605.27276) 论文的核心思想：
- 脚手架更新：调整策略、阈值、模式选择
- 权重更新：训练模型参数
- 两者在同一个循环中统一进行

关键洞见：
"脚手架更新塑造Agent的搜索和行为方式，
 权重更新构建prompt无法注入的领域直觉。"

设计原则：
- 每次学习后评估性能
- 性能低→调整策略（脚手架更新）
- 性能高→强化当前策略（权重更新）
- 记录改进历史，避免重复失败
"""

import torch
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PerformanceRecord:
    """性能记录"""
    task: str
    score: float
    timestamp: float
    parameters: Dict = field(default_factory=dict)


@dataclass
class StrategyAdjustment:
    """策略调整记录"""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    timestamp: float


class SelfImprovementSystem:
    """自改进系统

    统一脚手架更新和权重更新的自改进循环。
    """

    def __init__(self):
        # 性能历史
        self.performance_history: List[PerformanceRecord] = []
        self.strategy_adjustments: List[StrategyAdjustment] = []

        # 可调参数
        self.tunable_params = {
            'hebbian_lr': 0.1,           # Hebbian学习率
            'negative_threshold': 0.5,    # 负样本阈值
            'confidence_threshold': 0.3,  # 置信度阈值
            'exploration_weight': 0.5,    # 探索权重
            'consolidation_interval': 10, # 巩固间隔
        }

        # 改进统计
        self.improvement_stats = {
            'total_evaluations': 0,
            'harness_updates': 0,
            'weight_updates': 0,
            'performance_trend': 0.0,
        }

        # 窗口大小
        self.window_size = 20

    def evaluate_performance(self, task: str, score: float) -> Dict:
        """评估性能并决定改进策略

        Returns:
            {
                'should_adjust_harness': bool,
                'should_adjust_weights': bool,
                'adjustments': list,
            }
        """
        # 记录性能
        record = PerformanceRecord(
            task=task,
            score=score,
            timestamp=time.time(),
            parameters=self.tunable_params.copy(),
        )
        self.performance_history.append(record)
        self.improvement_stats['total_evaluations'] += 1

        # 计算趋势
        recent = self.performance_history[-self.window_size:]
        if len(recent) >= 2:
            first_half = sum(r.score for r in recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(r.score for r in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            trend = second_half - first_half
            self.improvement_stats['performance_trend'] = trend
        else:
            trend = 0.0

        # 决定改进策略
        adjustments = []
        should_adjust_harness = False
        should_adjust_weights = False

        # 性能下降 → 调整脚手架
        if trend < -0.1:
            adjustments.extend(self._suggest_harness_adjustments(score, trend))
            should_adjust_harness = True

        # 性能稳定但低 → 调整权重
        if score < 0.5 and abs(trend) < 0.05:
            should_adjust_weights = True

        # 性能高 → 强化当前策略
        if score > 0.8 and trend >= 0:
            adjustments.extend(self._reinforce_current_strategy())

        return {
            'should_adjust_harness': should_adjust_harness,
            'should_adjust_weights': should_adjust_weights,
            'adjustments': adjustments,
            'trend': trend,
            'score': score,
        }

    def _suggest_harness_adjustments(self, score: float, trend: float) -> List[Dict]:
        """建议脚手架调整"""
        adjustments = []

        # 降低置信度阈值，让更多结果通过
        if score < 0.3:
            old_val = self.tunable_params['confidence_threshold']
            new_val = max(0.1, old_val - 0.1)
            adjustments.append({
                'parameter': 'confidence_threshold',
                'old_value': old_val,
                'new_value': new_val,
                'reason': f'性能过低({score:.2f})，降低置信度阈值让更多结果通过',
            })
            self.tunable_params['confidence_threshold'] = new_val

        # 降低负样本阈值，更积极防止嵌入坍缩
        if score < 0.5:
            old_val = self.tunable_params['negative_threshold']
            new_val = max(0.2, old_val - 0.1)
            adjustments.append({
                'parameter': 'negative_threshold',
                'old_value': old_val,
                'new_value': new_val,
                'reason': f'性能中等({score:.2f})，降低负样本阈值防止嵌入坍缩',
            })
            self.tunable_params['negative_threshold'] = new_val

        # 增加探索权重
        if trend < -0.2:
            old_val = self.tunable_params['exploration_weight']
            new_val = min(0.8, old_val + 0.1)
            adjustments.append({
                'parameter': 'exploration_weight',
                'old_value': old_val,
                'new_value': new_val,
                'reason': f'性能下降({trend:.2f})，增加探索权重',
            })
            self.tunable_params['exploration_weight'] = new_val

        self.improvement_stats['harness_updates'] += 1

        # 记录调整
        for adj in adjustments:
            self.strategy_adjustments.append(StrategyAdjustment(
                parameter=adj['parameter'],
                old_value=adj['old_value'],
                new_value=adj['new_value'],
                reason=adj['reason'],
                timestamp=time.time(),
            ))

        return adjustments

    def _reinforce_current_strategy(self) -> List[Dict]:
        """强化当前策略"""
        adjustments = []

        # 微调学习率（稍微增加以加速收敛）
        if self.tunable_params['hebbian_lr'] < 0.2:
            old_val = self.tunable_params['hebbian_lr']
            new_val = min(0.2, old_val + 0.01)
            adjustments.append({
                'parameter': 'hebbian_lr',
                'old_value': old_val,
                'new_value': new_val,
                'reason': '性能良好，微调学习率加速收敛',
            })
            self.tunable_params['hebbian_lr'] = new_val

        return adjustments

    def get_parameter(self, name: str) -> float:
        """获取可调参数"""
        return self.tunable_params.get(name, 0.0)

    def get_stats(self) -> Dict:
        """获取改进统计"""
        return {
            **self.improvement_stats,
            'total_adjustments': len(self.strategy_adjustments),
            'current_params': self.tunable_params.copy(),
        }

    def get_report(self) -> str:
        """获取改进报告"""
        stats = self.get_stats()
        lines = [
            "=== 自改进系统报告 ===",
            f"总评估次数: {stats['total_evaluations']}",
            f"脚手架更新: {stats['harness_updates']}",
            f"权重更新: {stats['weight_updates']}",
            f"性能趋势: {stats['performance_trend']:.3f}",
            "",
            "当前参数:",
        ]
        for param, value in stats['current_params'].items():
            lines.append(f"  {param}: {value:.3f}")

        if self.strategy_adjustments:
            lines.append("")
            lines.append("最近调整:")
            for adj in self.strategy_adjustments[-5:]:
                lines.append(f"  {adj.parameter}: {adj.old_value:.3f} → {adj.new_value:.3f} ({adj.reason})")

        return '\n'.join(lines)
