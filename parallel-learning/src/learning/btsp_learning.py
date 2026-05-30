"""BTSP启发的单次学习 — 行为时间尺度突触可塑性

基于 Quanta Magazine 2026年4月报道的突破性发现：
- 传统赫布学习：毫秒级，需多次重复
- BTSP：秒级，单次体验即可完成学习

核心机制：
1. 资格痕迹（Eligibility Traces）：遇到实体时标记为"可学习"
2. 平台电位（Plateau Potential）：验证通过或高预测误差时触发
3. 一次性强化：所有带标记的连接同时增强

关键数据：
- 单次树突平台电位后，位置细胞放电率达99.5%
- 解决了信用分配问题：哪些神经元应该编码特定经验

对当前系统的应用：
- learn_from_text时：遇到的实体标记资格痕迹
- 验证通过时：触发平台电位，一次性强化所有相关实体
- 实现真正的one-shot learning
"""

import torch
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field
import time


@dataclass
class EligibilityTrace:
    """资格痕迹"""
    entity: str
    embedding: torch.Tensor
    timestamp: float
    strength: float = 1.0
    decay_rate: float = 0.1  # 每秒衰减


class BTSPLearningSystem:
    """BTSP启发的学习系统

    实现行为时间尺度的突触可塑性：
    1. 资格痕迹：遇到实体时标记
    2. 平台电位：验证通过时触发
    3. 一次性强化：所有带标记的连接同时增强
    """

    def __init__(self, decay_time: float = 5.0):
        # 资格痕迹池
        self.traces: Dict[str, EligibilityTrace] = {}

        # 衰减时间（秒）
        self.decay_time = decay_time

        # 平台电位历史
        self.plateau_history: List[Dict] = []

        # 统计
        self.stats = {
            'total_traces': 0,
            'total_plateaus': 0,
            'total_strengthened': 0,
        }

    def mark_eligible(self, entity: str, embedding: torch.Tensor):
        """标记实体为可学习（创建资格痕迹）

        当遇到实体时调用，标记其嵌入为"可学习"。
        """
        self.traces[entity] = EligibilityTrace(
            entity=entity,
            embedding=embedding.detach().clone(),
            timestamp=time.time(),
            strength=1.0,
        )
        self.stats['total_traces'] += 1

    def trigger_plateau(self, trigger_strength: float = 1.0,
                       reason: str = '') -> Dict[str, torch.Tensor]:
        """触发平台电位（一次性强化所有带标记的连接）

        当验证通过或高预测误差时调用。

        Returns:
            更新后的嵌入字典 {entity: new_embedding}
        """
        current_time = time.time()
        updated = {}

        # 清理过期的痕迹
        expired = []
        for entity, trace in self.traces.items():
            age = current_time - trace.timestamp
            if age > self.decay_time:
                expired.append(entity)
        for entity in expired:
            del self.traces[entity]

        # 触发所有有效痕迹
        for entity, trace in self.traces.items():
            # 计算衰减后的强度
            age = current_time - trace.timestamp
            decayed_strength = trace.strength * torch.exp(
                torch.tensor(-age / self.decay_time)
            ).item()

            # 平台电位强化：增强嵌入
            if decayed_strength > 0.1:
                # 计算强化方向（向原始嵌入的中心靠拢）
                center = torch.stack([
                    t.embedding for t in self.traces.values()
                ]).mean(dim=0)

                # 强化：向中心靠拢 + 保持独特性
                reinforcement = (center - trace.embedding) * 0.1 * trigger_strength * decayed_strength
                new_embedding = trace.embedding + reinforcement

                # 归一化
                new_embedding = torch.nn.functional.normalize(
                    new_embedding.unsqueeze(0), p=2, dim=1
                ).squeeze(0)

                updated[entity] = new_embedding
                self.stats['total_strengthened'] += 1

        # 记录平台电位
        self.plateau_history.append({
            'timestamp': current_time,
            'trigger_strength': trigger_strength,
            'reason': reason,
            'entities_strengthened': len(updated),
        })
        self.stats['total_plateaus'] += 1

        # 清空痕迹池（已触发）
        self.traces.clear()

        return updated

    def get_active_traces(self) -> List[str]:
        """获取当前活跃的资格痕迹"""
        current_time = time.time()
        active = []
        for entity, trace in self.traces.items():
            age = current_time - trace.timestamp
            if age <= self.decay_time:
                active.append(entity)
        return active

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'active_traces': len(self.traces),
            'recent_plateaus': len(self.plateau_history),
        }

    def get_report(self) -> str:
        """获取报告"""
        stats = self.get_stats()
        lines = [
            "=== BTSP学习系统报告 ===",
            f"总资格痕迹: {stats['total_traces']}",
            f"总平台电位: {stats['total_plateaus']}",
            f"总强化连接: {stats['total_strengthened']}",
            f"当前活跃痕迹: {stats['active_traces']}",
        ]

        if self.plateau_history:
            lines.append("")
            lines.append("最近平台电位:")
            for p in self.plateau_history[-3:]:
                lines.append(f"  [{p['reason']}] 强化了{p['entities_strengthened']}个实体")

        return '\n'.join(lines)
