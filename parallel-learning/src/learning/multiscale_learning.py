"""多时间尺度学习 — 解决灾难性遗忘

不同模块以不同速率学习，模拟大脑的分层学习机制。

核心组件：
1. 分层学习率 — 感知层快，知识层慢
2. 选择性巩固 — 高不确定性记忆优先
3. 遗忘曲线 — Ebbinghaus曲线
4. 元学习信号 — 记录适应速度

设计原则：
- 感知层（分钟级）：快速适应新输入
- 知识层（天级）：缓慢固化长期知识
- 元层（周级）：学习如何学习
"""

import torch
import torch.nn as nn
import math
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MemoryTrace:
    """记忆痕迹"""
    content: str
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    uncertainty: float = 1.0
    importance: float = 0.5


class ForgettingCurve:
    """Ebbinghaus 遗忘曲线

    R = e^(-t/S)
    R: 保留率
    t: 时间间隔
    S: 记忆强度
    """

    def __init__(self, decay_rate: float = 0.5):
        self.decay_rate = decay_rate

    def retention(self, time_elapsed: float, strength: float = 1.0) -> float:
        """计算保留率"""
        return math.exp(-time_elapsed / (strength * 3600))  # 1小时为单位

    def should_review(self, trace: MemoryTrace, current_time: float) -> bool:
        """判断是否需要复习"""
        time_since_access = current_time - trace.last_access
        retention = self.retention(time_since_access, trace.access_count + 1)

        # 保留率低于阈值时需要复习
        return retention < 0.6


class SelectiveConsolidation:
    """选择性巩固

    优先巩固高不确定性的记忆。
    """

    def __init__(self, max_consolidations: int = 10):
        self.max_consolidations = max_consolidations
        self.forgetting_curve = ForgettingCurve()

    def select_memories(self, memories: Dict[str, MemoryTrace],
                       current_time: float) -> List[str]:
        """选择需要巩固的记忆"""
        candidates = []

        for key, trace in memories.items():
            # 计算巩固优先级
            priority = self._compute_priority(trace, current_time)
            candidates.append((key, priority))

        # 按优先级排序
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [key for key, _ in candidates[:self.max_consolidations]]

    def _compute_priority(self, trace: MemoryTrace, current_time: float) -> float:
        """计算巩固优先级"""
        # 不确定性权重
        uncertainty_weight = trace.uncertainty

        # 遗忘风险
        time_since_access = current_time - trace.last_access
        retention = self.forgetting_curve.retention(time_since_access, trace.access_count + 1)
        forgetting_risk = 1.0 - retention

        # 重要性权重
        importance_weight = trace.importance

        return uncertainty_weight * 0.4 + forgetting_risk * 0.4 + importance_weight * 0.2


class MultiTimescaleOptimizer:
    """多时间尺度优化器

    不同模块使用不同的学习率。
    """

    def __init__(self):
        # 学习率层次
        self.learning_rates = {
            'perception': 1e-3,    # 感知层：快速
            'knowledge': 1e-5,     # 知识层：缓慢
            'meta': 1e-7,          # 元层：极慢
        }

        # 优化器
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

    def register_module(self, name: str, module: nn.Module, layer: str = 'knowledge'):
        """注册模块到对应的学习率层"""
        lr = self.learning_rates.get(layer, 1e-5)
        self.optimizers[name] = torch.optim.Adam(module.parameters(), lr=lr)

    def step(self, name: str):
        """更新指定模块"""
        if name in self.optimizers:
            self.optimizers[name].step()

    def zero_grad(self, name: str):
        """清零指定模块的梯度"""
        if name in self.optimizers:
            self.optimizers[name].zero_grad()


class MetaLearningSignal:
    """元学习信号

    记录适应速度，强化有效的学习策略。
    """

    def __init__(self):
        self.adaptation_history: List[Dict] = []
        self.transfer_scores: Dict[str, float] = {}

    def record_adaptation(self, task: str, initial_performance: float,
                         final_performance: float, time_elapsed: float):
        """记录适应过程"""
        adaptation_speed = (final_performance - initial_performance) / max(time_elapsed, 0.001)

        self.adaptation_history.append({
            'task': task,
            'initial': initial_performance,
            'final': final_performance,
            'speed': adaptation_speed,
            'time': time_elapsed,
        })

    def record_transfer(self, source_task: str, target_task: str, benefit: float):
        """记录迁移效果"""
        key = f"{source_task}→{target_task}"
        self.transfer_scores[key] = benefit

    def get_best_transfer(self, target_task: str) -> Optional[str]:
        """找到对目标任务最有益的源任务"""
        best_source = None
        best_benefit = 0.0

        for key, benefit in self.transfer_scores.items():
            if key.endswith(f"→{target_task}") and benefit > best_benefit:
                best_source = key.split("→")[0]
                best_benefit = benefit

        return best_source


class MultiTimescaleLearningSystem:
    """多时间尺度学习系统

    整合分层学习率、选择性巩固、遗忘曲线和元学习信号。
    """

    def __init__(self):
        # 多时间尺度优化器
        self.optimizer = MultiTimescaleOptimizer()

        # 选择性巩固
        self.consolidation = SelectiveConsolidation()

        # 元学习信号
        self.meta_signal = MetaLearningSignal()

        # 记忆库
        self.memory_traces: Dict[str, MemoryTrace] = {}

        # 学习统计
        self.learning_stats = {
            'total_consolidations': 0,
            'total_forgets': 0,
            'adaptation_speeds': [],
        }

    def register_module(self, name: str, module: nn.Module, layer: str = 'knowledge'):
        """注册学习模块"""
        self.optimizer.register_module(name, module, layer)

    def learn(self, module_name: str, loss: torch.Tensor):
        """学习一步"""
        self.optimizer.zero_grad(module_name)
        loss.backward()
        self.optimizer.step(module_name)

    def store_memory(self, key: str, content: str, importance: float = 0.5):
        """存储记忆"""
        current_time = time.time()

        if key in self.memory_traces:
            # 更新已有记忆
            trace = self.memory_traces[key]
            trace.access_count += 1
            trace.last_access = current_time
            trace.uncertainty *= 0.9  # 每次访问降低不确定性
        else:
            # 创建新记忆
            self.memory_traces[key] = MemoryTrace(
                content=content,
                timestamp=current_time,
                access_count=1,
                last_access=current_time,
                uncertainty=1.0,
                importance=importance,
            )

    def consolidate(self) -> List[str]:
        """执行选择性巩固"""
        current_time = time.time()

        # 选择需要巩固的记忆
        to_consolidate = self.consolidation.select_memories(
            self.memory_traces, current_time
        )

        # 巩固
        consolidated = []
        for key in to_consolidate:
            if key in self.memory_traces:
                trace = self.memory_traces[key]
                # 降低不确定性
                trace.uncertainty *= 0.5
                # 增加重要性
                trace.importance = min(1.0, trace.importance + 0.1)
                consolidated.append(key)

        self.learning_stats['total_consolidations'] += len(consolidated)

        return consolidated

    def forget(self, threshold: float = 0.1) -> List[str]:
        """遗忘低保留率的记忆"""
        current_time = time.time()
        to_forget = []

        for key, trace in self.memory_traces.items():
            retention = self.consolidation.forgetting_curve.retention(
                current_time - trace.last_access,
                trace.access_count + 1
            )
            if retention < threshold:
                to_forget.append(key)

        # 删除
        for key in to_forget:
            del self.memory_traces[key]

        self.learning_stats['total_forgets'] += len(to_forget)

        return to_forget

    def get_stats(self) -> Dict:
        """获取学习统计"""
        return {
            'total_memories': len(self.memory_traces),
            'total_consolidations': self.learning_stats['total_consolidations'],
            'total_forgets': self.learning_stats['total_forgets'],
            'avg_uncertainty': self._avg_uncertainty(),
            'meta_signals': len(self.meta_signal.adaptation_history),
        }

    def _avg_uncertainty(self) -> float:
        """平均不确定性"""
        if not self.memory_traces:
            return 0.0
        return sum(t.uncertainty for t in self.memory_traces.values()) / len(self.memory_traces)
